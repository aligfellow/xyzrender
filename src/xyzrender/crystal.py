"""Crystal structure support (optional — requires xyzrender[crystal] / phonopy).

This module contains all phonopy-dependent functionality for loading periodic
crystal structures and generating periodic image atoms for rendering.  It is
intentionally separated from ``io.py`` so that the optional ``phonopy``
dependency is not imported at all unless crystal loading is actually requested.

Public API
----------
load_crystal
    Load a VASP/QE/... crystal structure file and return a molecular graph
    together with its ``CellData`` (lattice matrix + cell origin).
build_supercell
    Expand a unit cell into a supercell by integer repetition.
add_crystal_images
    Populate a crystal graph with ghost atoms from the 26 neighbouring unit
    cells so that bonds crossing cell boundaries are visible.
"""

from __future__ import annotations

import itertools
import logging
from itertools import product as _product
from typing import TYPE_CHECKING

import numpy as np
from xyzgraph import DATA, build_graph
from xyzgraph.parameters import BondThresholds

from xyzrender.types import CellData

_bond_thresholds = BondThresholds()

if TYPE_CHECKING:
    from pathlib import Path

    import networkx as nx

logger = logging.getLogger(__name__)

__all__ = ["add_crystal_images", "build_supercell", "load_crystal"]


def _is_bonded(sym_i: str, sym_j: str, dist: float) -> bool:
    """Return True if two atoms at *dist* Å apart are likely bonded.

    Uses xyzgraph's VDW radii (DATA.vdw) and the same type-specific distance
    thresholds as xyzgraph's BondThresholds defaults, so ghost-bond detection
    is consistent with main-cell bond detection.  Note: xyzgraph also applies
    geometric pruning (bond angles, valence) which is not replicated here.
    """
    ri = DATA.vdw.get(sym_i, 2.0)
    rj = DATA.vdw.get(sym_j, 2.0)
    metals = DATA.metals
    hi, hj = sym_i == "H", sym_j == "H"
    mi, mj = sym_i in metals, sym_j in metals
    if hi and hj:
        t = _bond_thresholds.threshold_h_h
    elif hi or hj:
        t = _bond_thresholds.threshold_h_metal if (mi or mj) else _bond_thresholds.threshold_h_nonmetal
    elif mi and mj:
        t = _bond_thresholds.threshold_metal_metal_self
    elif mi or mj:
        t = _bond_thresholds.threshold_metal_ligand
    else:
        t = _bond_thresholds.threshold_nonmetal_nonmetal
    return dist < t * (ri + rj)


def load_crystal(
    path: str | Path,
    interface_mode: str,
) -> tuple[nx.Graph, CellData]:
    """Load a periodic crystal structure using phonopy.

    Parameters
    ----------
    path:
        Path to the crystal structure input file (POSCAR/CONTCAR for VASP,
        ``*.in`` / ``pw.in`` for Quantum ESPRESSO, etc.).
    interface_mode:
        Phonopy interface identifier: ``"vasp"``, ``"qe"``, ``"abinit"``, etc.

    Returns
    -------
    tuple[nx.Graph, CellData]
        Molecular graph with atoms as nodes and ``CellData`` containing the
        3x3 lattice matrix (rows = a, b, c in Å).
    """
    logger.info("Loading %s", path)
    try:
        from phonopy.interface.calculator import (
            get_calculator_physical_units,
            read_crystal_structure,
        )
    except ImportError:
        msg = "Crystal structure loading requires phonopy: pip install 'xyzrender[crystal]'"
        raise ImportError(msg) from None

    unitcell, _ = read_crystal_structure(str(path), interface_mode=interface_mode)
    if unitcell is None:
        msg = f"Failed to read crystal structure from {path!r} (interface_mode={interface_mode!r})"
        raise ValueError(msg)
    # Convert native units → Angstrom.
    factor: float = get_calculator_physical_units(interface_mode).distance_to_A
    symbols: list[str] = list(unitcell.symbols)
    positions = unitcell.positions * factor  # ndarray, shape (N, 3), in Å
    lattice = np.array(unitcell.cell) * factor  # shape (3, 3), rows = a, b, c in Å

    atoms: list[tuple[str, tuple[float, float, float]]] = [
        (sym, (float(pos[0]), float(pos[1]), float(pos[2]))) for sym, pos in zip(symbols, positions, strict=True)
    ]
    graph = build_graph(atoms, charge=0, multiplicity=None, kekule=False, quick=True)
    logger.info(
        "Crystal graph: %d atoms, %d bonds, lattice=%s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        lattice.diagonal().round(3),
    )
    graph.graph["lattice"] = lattice
    graph.graph["lattice_origin"] = np.zeros(3)
    return graph, CellData(lattice=lattice)


def build_supercell(graph: "nx.Graph", cell_data: CellData, repeats: tuple[int, int, int]) -> "nx.Graph":
    """Return a new graph representing a repeated supercell.

    The unit-cell graph is replicated *m x n x l* times.  Intra-replica edges
    are copied verbatim (preserving bond orders and all edge attributes).
    Cross-boundary bonds between adjacent replicas are detected with the same
    ``_is_bonded`` distance logic used by :func:`add_crystal_images`.

    Parameters
    ----------
    graph:
        Base-cell graph. Must not already contain periodic image atoms
        (nodes with ``image=True``).
    cell_data:
        Cell lattice/origin describing the base cell.
    repeats:
        Integer repetition counts ``(m, n, l)`` along lattice vectors
        ``a, b, c``. Each must be >= 1.

    Returns
    -------
    nx.Graph
        Supercell graph.  Graph-level metadata (including ``lattice``) is
        copied from the input — the lattice remains the **unit-cell** lattice
        so that the cell-box overlay shows the original unit cell.
    """
    import networkx as nx

    m, n, l_rep = repeats
    if m < 1 or n < 1 or l_rep < 1:
        raise ValueError(f"supercell repeats must be >= 1, got {repeats!r}")

    if any(bool(graph.nodes[nid].get("image", False)) for nid in graph.nodes()):
        raise ValueError("build_supercell: graph already contains image atoms (apply before add_crystal_images)")

    a = np.array(cell_data.lattice[0], dtype=float)
    b = np.array(cell_data.lattice[1], dtype=float)
    c = np.array(cell_data.lattice[2], dtype=float)

    base_nodes = list(graph.nodes())
    n_base = len(base_nodes)

    if n_base == 0:
        empty = nx.Graph()
        empty.graph.update(dict(graph.graph))
        return empty

    nid_to_idx = {nid: idx for idx, nid in enumerate(base_nodes)}

    # -- 1. Replicate nodes ------------------------------------------------
    # Deterministic mapping: replica (ii,jj,kk) atom idx →
    #   (ii * n * l_rep + jj * l_rep + kk) * n_base + idx
    new_g = nx.Graph()
    for ii, jj, kk in _product(range(m), range(n), range(l_rep)):
        offset = ii * a + jj * b + kk * c
        base = (ii * n * l_rep + jj * l_rep + kk) * n_base
        for idx, nid in enumerate(base_nodes):
            attrs = dict(graph.nodes[nid])
            pos = np.array(attrs["position"], dtype=float) + offset
            attrs["position"] = (float(pos[0]), float(pos[1]), float(pos[2]))
            attrs.pop("image", None)
            attrs.pop("source", None)
            new_g.add_node(base + idx, **attrs)

    # -- 2. Copy intra-replica edges (preserves bond_order etc.) -----------
    edges = [(nid_to_idx[u], nid_to_idx[v], dict(d)) for u, v, d in graph.edges(data=True)]
    for ii, jj, kk in _product(range(m), range(n), range(l_rep)):
        base = (ii * n * l_rep + jj * l_rep + kk) * n_base
        for ui, vi, data in edges:
            new_g.add_edge(base + ui, base + vi, **data)

    # -- 3. Stitch cross-boundary bonds (same logic as add_crystal_images) -
    # Only the 13 lexicographically-forward shifts so each pair is visited once.
    forward_shifts = [(dx, dy, dz) for dx, dy, dz in _product((-1, 0, 1), repeat=3) if (dx, dy, dz) > (0, 0, 0)]
    for dx, dy, dz in forward_shifts:
        for ii, jj, kk in _product(range(m), range(n), range(l_rep)):
            ni, nj, nk = ii + dx, jj + dy, kk + dz
            if not (0 <= ni < m and 0 <= nj < n and 0 <= nk < l_rep):
                continue
            src_base = (ii * n * l_rep + jj * l_rep + kk) * n_base
            tgt_base = (ni * n * l_rep + nj * l_rep + nk) * n_base
            for u in range(n_base):
                sid = src_base + u
                spos = np.array(new_g.nodes[sid]["position"])
                ssym = new_g.nodes[sid]["symbol"]
                for v in range(n_base):
                    tid = tgt_base + v
                    tpos = np.array(new_g.nodes[tid]["position"])
                    tsym = new_g.nodes[tid]["symbol"]
                    if _is_bonded(ssym, tsym, float(np.linalg.norm(spos - tpos))):
                        new_g.add_edge(sid, tid, bond_order=1.0)

    # -- 4. Graph-level metadata (lattice stays as unit cell for cell box) --
    new_g.graph.update(dict(graph.graph))
    return new_g


def add_crystal_images(graph: nx.Graph, crystal_data: CellData) -> int:
    """Add periodic image atoms that are bonded to cell atoms.

    For each of the 26 neighbouring unit cells, adds image copies of cell
    atoms that form at least one bond with an atom inside the cell.  Image
    nodes carry ``image=True`` and ``source=<cell_atom_id>`` attributes;
    image bonds carry ``image_bond=True``.

    Returns the number of image atoms added.
    """
    lattice = crystal_data.lattice  # (3, 3)
    a, b, c = lattice[0], lattice[1], lattice[2]

    cell_ids = list(graph.nodes())
    if not cell_ids:
        return 0

    cell_syms = {i: graph.nodes[i]["symbol"] for i in cell_ids}
    cell_pos = {i: np.array(graph.nodes[i]["position"]) for i in cell_ids}

    next_id = max(cell_ids) + 1
    n_added = 0

    shifts = [(dx, dy, dz) for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3) if (dx, dy, dz) != (0, 0, 0)]

    for dx, dy, dz in shifts:
        offset = dx * a + dy * b + dz * c
        for src_id in cell_ids:
            sym_i = cell_syms[src_id]
            img_pos = cell_pos[src_id] + offset

            bonded_to: list[int] = [
                j for j in cell_ids if _is_bonded(sym_i, cell_syms[j], float(np.linalg.norm(img_pos - cell_pos[j])))
            ]

            # Skip ghost hydrogens that only bond to C (C-H across boundary
            # is not chemically interesting and clutters the image).
            if sym_i == "H":
                bonded_to = [j for j in bonded_to if cell_syms[j] != "C"]

            if not bonded_to:
                continue

            img_id = next_id
            next_id += 1
            n_added += 1
            graph.add_node(
                img_id,
                symbol=sym_i,
                position=(float(img_pos[0]), float(img_pos[1]), float(img_pos[2])),
                image=True,
                source=src_id,
            )
            for j in bonded_to:
                graph.add_edge(img_id, j, bond_order=1.0, image_bond=True)

    logger.debug("Added %d image atoms", n_added)
    return n_added
