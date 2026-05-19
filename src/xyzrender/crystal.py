"""Crystal structure support.

Loading periodic crystal structures (VASP, QE, SIESTA, ABINIT) and generating
periodic image atoms for rendering.

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
from collections import defaultdict
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


def _build_threshold_matrix(syms: list[str]) -> np.ndarray:
    """Return an (n, n) matrix of bond-distance cutoffs for a list of element symbols.

    Groups atoms by unique element so the inner work is O(E²) where E is the
    number of distinct elements, not O(n²) in Python.
    """
    unique = sorted(set(syms))
    elem_to_idx = {s: i for i, s in enumerate(unique)}

    # Build a small (E, E) cutoff matrix for unique elements only, then
    # expand to (n, n) by mapping each atom to its element row/col.
    vdw = np.array([DATA.vdw.get(s, 2.0) for s in unique])
    tf = np.array([[_bond_threshold_factor(si, sj) for sj in unique] for si in unique])
    elem_thresh = tf * (vdw[:, None] + vdw[None, :])  # (E, E)

    idx = np.array([elem_to_idx[s] for s in syms])  # (n,)
    return elem_thresh[idx[:, None], idx[None, :]]  # (n, n)


def _build_elem_thresh(syms: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (elem_thresh, elem_idx, max_cutoff) for cell-list distance queries.

    *elem_thresh* is a small (E, E) matrix of bond-distance cutoffs for unique
    elements.  *elem_idx* maps each atom index to its element row in that matrix.
    *max_cutoff* is the global maximum cutoff (cell size for spatial hashing).
    """
    unique = sorted(set(syms))
    elem_to_idx = {s: i for i, s in enumerate(unique)}
    vdw = np.array([DATA.vdw.get(s, 2.0) for s in unique])
    tf = np.array([[_bond_threshold_factor(si, sj) for sj in unique] for si in unique])
    elem_thresh = tf * (vdw[:, None] + vdw[None, :])  # (E, E)
    elem_idx = np.array([elem_to_idx[s] for s in syms])  # (n,)
    max_cutoff = float(elem_thresh.max())
    return elem_thresh, elem_idx, max_cutoff


def _find_bonded_pairs(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    eidx_a: np.ndarray,
    eidx_b: np.ndarray,
    elem_thresh: np.ndarray,
    max_cutoff: float,
) -> list[tuple[int, int]]:
    """Find bonded atom pairs between two position arrays using cell-list spatial hashing.

    Returns a list of (i, j) index pairs where atom i in *pos_a* is bonded to
    atom j in *pos_b*.  Pure numpy, O(n·k) where k is avg neighbors per cell.
    """
    cell_size = max_cutoff
    if cell_size < 1e-6:
        return []

    na, nb = len(pos_a), len(pos_b)

    # For small arrays, vectorized all-pairs is faster than cell-list overhead
    if na * nb <= 50_000:
        dists = np.linalg.norm(pos_a[:, None, :] - pos_b[None, :, :], axis=2)
        thresh_mat = elem_thresh[eidx_a[:, None], eidx_b[None, :]]
        mask = dists < thresh_mat
        ii, jj = np.where(mask)
        return list(zip(ii.tolist(), jj.tolist(), strict=False))

    # Cell-list spatial hashing for large arrays
    min_corner = np.minimum(pos_a.min(0), pos_b.min(0)) - cell_size
    cidx_b = ((pos_b - min_corner) / cell_size).astype(np.intp)
    cidx_a = ((pos_a - min_corner) / cell_size).astype(np.intp)

    # Build B cell lookup: group B atoms by cell key
    _p1, _p2 = np.intp(73856093), np.intp(19349669)
    b_keys = cidx_b[:, 0] * _p1 ^ cidx_b[:, 1] * _p2 ^ cidx_b[:, 2]
    b_order = np.argsort(b_keys)
    b_keys_sorted = b_keys[b_order]
    uniq_keys, starts, counts = np.unique(b_keys_sorted, return_index=True, return_counts=True)
    cell_start = dict(zip(uniq_keys.tolist(), starts.tolist(), strict=True))
    cell_count = dict(zip(uniq_keys.tolist(), counts.tolist(), strict=True))

    # For each of 27 offsets, vectorize the key matching and pair generation
    all_ci: list[np.ndarray] = []
    all_cj: list[np.ndarray] = []

    offsets = np.array(list(_product((-1, 0, 1), repeat=3)))  # Shape (27, 3)
    shifted_coords = cidx_a[None, :, :] + offsets[:, None, :]

    flat_keys = ((shifted_coords[:, :, 0] * _p1) ^ (shifted_coords[:, :, 1] * _p2) ^ (shifted_coords[:, :, 2])).ravel()

    sorted_keys = np.array(sorted(cell_start.keys()))
    starts = np.array([cell_start[k] for k in sorted_keys])
    counts = np.array([cell_count[k] for k in sorted_keys])

    idx = np.searchsorted(sorted_keys, flat_keys)
    in_range = idx < len(sorted_keys)
    mask = np.zeros(len(flat_keys), dtype=bool)
    mask[in_range] = sorted_keys[idx[in_range]] == flat_keys[in_range]

    atom_a_indices = np.tile(np.arange(len(cidx_a)), 27)
    valid_a_idx = atom_a_indices[mask]
    valid_starts = starts[idx[mask]]
    valid_counts = counts[idx[mask]]

    repeats = valid_counts
    res_i = np.repeat(valid_a_idx, repeats)

    offsets_in_cell = np.arange(repeats.sum()) - np.repeat(repeats.cumsum() - repeats, repeats)
    res_j = b_order[np.repeat(valid_starts, repeats) + offsets_in_cell]

    all_ci.append(res_i)
    all_cj.append(res_j)

    if not all_ci:
        return []

    ci = np.concatenate(all_ci)
    cj = np.concatenate(all_cj)

    # Vectorized distance + threshold check
    diff = pos_a[ci] - pos_b[cj]
    d2 = (diff * diff).sum(axis=1)
    t = elem_thresh[eidx_a[ci], eidx_b[cj]]
    mask = d2 < t * t
    return list(zip(ci[mask].tolist(), cj[mask].tolist(), strict=False))


def _bond_threshold_factor(sym_i: str, sym_j: str) -> float:
    """Return the bond-distance threshold multiplier for a pair of element symbols."""
    metals = DATA.metals
    hi, hj = sym_i == "H", sym_j == "H"
    mi, mj = sym_i in metals, sym_j in metals
    if hi and hj:
        return _bond_thresholds.threshold_h_h
    if hi or hj:
        return _bond_thresholds.threshold_h_metal if (mi or mj) else _bond_thresholds.threshold_h_nonmetal
    if mi and mj:
        return _bond_thresholds.threshold_metal_metal_self
    if mi or mj:
        metal_sym = sym_i if sym_i in metals else sym_j
        if metal_sym in DATA.sblock_metals:
            return _bond_thresholds.threshold_sblock_ligand
        return _bond_thresholds.threshold_metal_ligand
    return _bond_thresholds.threshold_nonmetal_nonmetal


def _is_bonded(sym_i: str, sym_j: str, dist: float) -> bool:
    """Return True if two atoms at *dist* Å apart are likely bonded.

    Uses xyzgraph's VDW radii (DATA.vdw) and the same type-specific distance
    thresholds as xyzgraph's BondThresholds defaults, so ghost-bond detection
    is consistent with main-cell bond detection.  Note: xyzgraph also applies
    geometric pruning (bond angles, valence) which is not replicated here.
    """
    ri = DATA.vdw.get(sym_i, 2.0)
    rj = DATA.vdw.get(sym_j, 2.0)
    t = _bond_threshold_factor(sym_i, sym_j)
    return dist < t * (ri + rj)


def load_crystal(
    path: str | Path,
    interface_mode: str,
) -> tuple[nx.Graph, CellData]:
    """Load a periodic crystal structure.

    Uses built-in parsers for VASP, QE, SIESTA, and ABINIT.

    Parameters
    ----------
    path:
        Path to the crystal structure input file (POSCAR/CONTCAR for VASP,
        ``*.in`` / ``pw.in`` for Quantum ESPRESSO, ``.fdf`` for SIESTA, etc.).
    interface_mode:
        Interface identifier: ``"vasp"``, ``"qe"``, ``"siesta"``, ``"abinit"``.

    Returns
    -------
    tuple[nx.Graph, CellData]
        Molecular graph with atoms as nodes and ``CellData`` containing the
        3x3 lattice matrix (rows = a, b, c in Å).
    """
    logger.info("Loading %s", path)

    if interface_mode == "vasp":
        from xyzrender.inputs import parse_poscar

        atoms, lattice = parse_poscar(str(path))
    elif interface_mode == "qe":
        from xyzrender.inputs import parse_qe_input

        atoms, lattice, _charge = parse_qe_input(str(path))
    elif interface_mode == "siesta":
        from xyzrender.inputs import parse_siesta_fdf

        atoms, lattice = parse_siesta_fdf(str(path))
    elif interface_mode == "abinit":
        from xyzrender.inputs import parse_abinit_input

        atoms, lattice = parse_abinit_input(str(path))
    else:
        msg = f"Unsupported crystal interface mode: {interface_mode!r}. Supported: vasp, qe, siesta, abinit."
        raise ValueError(msg)

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

    Node IDs are ``replica_index * n_base + base_idx`` (replica_index =
    ``ii * n * l + jj * l + kk``), and nodes are inserted in ascending ID
    order so the first ``n_base`` are the unit cell.
    """
    import networkx as nx

    m, n, l_rep = repeats
    if m < 1 or n < 1 or l_rep < 1:
        raise ValueError(f"supercell repeats must be >= 1, got {repeats!r}")

    if any(graph.nodes[nid].get("image", False) for nid in graph.nodes()):
        raise ValueError("build_supercell: graph already contains image atoms")

    n_base = graph.number_of_nodes()
    if n_base == 0:
        empty = nx.Graph()
        empty.graph.update(dict(graph.graph))
        return empty

    # -- 1. Vectorized Coordinate and Offset Generation --
    ii, jj, kk = np.meshgrid(np.arange(m), np.arange(n), np.arange(l_rep), indexing="ij")
    replica_indices = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=-1)

    lattice = np.array(cell_data.lattice, dtype=float)
    offsets = replica_indices @ lattice

    num_replicas = len(replica_indices)
    replica_ids = replica_indices[:, 0] * n * l_rep + replica_indices[:, 1] * l_rep + replica_indices[:, 2]

    # -- 2. Replicate Nodes (Vectorized) --
    base_nodes = list(graph.nodes(data=True))
    base_pos = np.array([attrs["position"] for _, attrs in base_nodes], dtype=float)

    # Broadcast positions and global IDs across all replicas
    # shape logic: (n_base, 1, 3) + (1, num_replicas, 3) -> flattened to (n_base * num_replicas, 3)
    all_positions = (base_pos[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    global_ids = (np.arange(n_base)[:, None] + replica_ids[None, :] * n_base).ravel()

    clean_attrs = [
        {k: v for k, v in attrs.items() if k not in ("image", "source", "position")} for _, attrs in base_nodes
    ]

    # Sort by id so list(graph.nodes())[:n_base] is the unit cell —
    # _add_crystal_images_supercell depends on this.
    new_nodes = sorted(
        (
            (int(gid), {**clean_attrs[i // num_replicas], "position": tuple(pos)})
            for i, (gid, pos) in enumerate(zip(global_ids, all_positions, strict=True))
        ),
        key=lambda nt: nt[0],
    )

    new_g = nx.Graph()
    new_g.add_nodes_from(new_nodes)

    # -- 3. Replicate Intra-replica Edges (Vectorized) --
    nid_to_idx = {nid: idx for idx, (nid, _) in enumerate(base_nodes)}
    base_edges = [(nid_to_idx[u], nid_to_idx[v], d) for u, v, d in graph.edges(data=True)]

    new_edges = []
    if base_edges:
        u_idxs, v_idxs, edge_datas = zip(*base_edges, strict=True)
        u_arr = np.array(u_idxs)[:, None]
        v_arr = np.array(v_idxs)[:, None]

        # Broadcast edge offsets over replicas
        u_shifted = (replica_ids[None, :] * n_base + u_arr).ravel()
        v_shifted = (replica_ids[None, :] * n_base + v_arr).ravel()

        repeated_data = [d for d in edge_datas for _ in range(num_replicas)]
        new_edges.extend(zip(u_shifted.tolist(), v_shifted.tolist(), repeated_data, strict=True))

    # -- 4. Vectorized Cross-boundary Stitching --
    base_syms = [d["symbol"] for _, d in base_nodes]
    elem_thresh, eidx, max_cutoff = _build_elem_thresh(base_syms)

    forward_shifts = [s for s in _product((-1, 0, 1), repeat=3) if s > (0, 0, 0)]

    for dx, dy, dz in forward_shifts:
        shift_vec = dx * lattice[0] + dy * lattice[1] + dz * lattice[2]
        pairs = _find_bonded_pairs(base_pos + shift_vec, base_pos, eidx, eidx, elem_thresh, max_cutoff)

        if pairs:
            mask = (
                (replica_indices[:, 0] + dx >= 0)
                & (replica_indices[:, 0] + dx < m)
                & (replica_indices[:, 1] + dy >= 0)
                & (replica_indices[:, 1] + dy < n)
                & (replica_indices[:, 2] + dz >= 0)
                & (replica_indices[:, 2] + dz < l_rep)
            )

            valid_orig_ids = replica_ids[mask]
            valid_shifted_ids = (
                (replica_indices[mask, 0] + dx) * n * l_rep
                + (replica_indices[mask, 1] + dy) * l_rep
                + (replica_indices[mask, 2] + dz)
            )

            # Broadcast the valid ID multipliers across all discovered pairs simultaneously
            u_arr, v_arr = np.array(pairs).T
            u_global = (valid_shifted_ids[:, None] * n_base + u_arr[None, :]).ravel()
            v_global = (valid_orig_ids[:, None] * n_base + v_arr[None, :]).ravel()

            new_edges.extend(
                zip(u_global.tolist(), v_global.tolist(), [{"bond_order": 1.0}] * len(u_global), strict=True)
            )

    new_g.add_edges_from(new_edges)
    new_g.graph.update(dict(graph.graph))

    return new_g


def add_crystal_images(
    graph: nx.Graph,
    crystal_data: CellData,
    supercell_repeats: tuple[int, int, int] | None = None,
    unit_cell_data: CellData | None = None,
    n_base: int | None = None,
) -> int:
    """Add periodic image atoms that are bonded to cell atoms.

    For each of the 26 neighbouring unit cells, adds image copies of cell
    atoms that form at least one bond with an atom inside the cell.  Image
    nodes carry ``image=True`` and ``source=<cell_atom_id>`` attributes;
    image bonds carry ``image_bond=True``.

    When *supercell_repeats* and *unit_cell_data* are provided (supercell path),
    cross-boundary bond pairs are computed once on the small unit cell, then
    replayed only for atoms on the relevant outer face of the supercell.
    This avoids O(N²) distance computation on the full supercell.

    Returns the number of image atoms added.
    """
    if supercell_repeats is not None and unit_cell_data is not None and n_base is not None:
        return _add_crystal_images_supercell(
            graph,
            crystal_data,
            supercell_repeats,
            unit_cell_data,
            n_base,
        )
    return _add_crystal_images_generic(graph, crystal_data)


def _add_crystal_images_generic(graph: nx.Graph, crystal_data: CellData) -> int:
    """Generate ghost atoms for any cell (unit cell or supercell)."""
    lattice = crystal_data.lattice  # (3, 3)
    a, b, c = lattice[0], lattice[1], lattice[2]

    cell_ids = list(graph.nodes())
    if not cell_ids:
        return 0

    cell_syms_list = [graph.nodes[i]["symbol"] for i in cell_ids]
    cell_pos_arr = np.array([graph.nodes[i]["position"] for i in cell_ids])  # (n, 3)

    elem_thresh, eidx, max_cutoff = _build_elem_thresh(cell_syms_list)

    # Precompute H and C masks for ghost-H filtering
    is_h = [s == "H" for s in cell_syms_list]
    is_c = [s == "C" for s in cell_syms_list]

    next_id = max(cell_ids) + 1
    n_added = 0

    shifts = [(dx, dy, dz) for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3) if (dx, dy, dz) != (0, 0, 0)]

    for dx, dy, dz in shifts:
        offset = dx * a + dy * b + dz * c
        img_pos_arr = cell_pos_arr + offset  # (n, 3)

        # Cell-list spatial hashing: O(n) instead of O(n²)
        pairs = _find_bonded_pairs(img_pos_arr, cell_pos_arr, eidx, eidx, elem_thresh, max_cutoff)

        # Group bonded cell atoms by source image atom
        src_to_targets: dict[int, list[int]] = defaultdict(list)
        for src_idx, tgt_idx in pairs:
            src_to_targets[src_idx].append(tgt_idx)

        for src_idx, bonded_cols in src_to_targets.items():
            # Ghost H that only bonds to C across boundary is not interesting.
            targets = bonded_cols
            if is_h[src_idx]:
                targets = [j for j in targets if not is_c[j]]
                if not targets:
                    continue

            src_id = cell_ids[src_idx]
            img_pos = img_pos_arr[src_idx]
            img_id = next_id
            next_id += 1
            n_added += 1
            graph.add_node(
                img_id,
                symbol=cell_syms_list[src_idx],
                position=(float(img_pos[0]), float(img_pos[1]), float(img_pos[2])),
                image=True,
                source=src_id,
            )
            for j in targets:
                graph.add_edge(img_id, cell_ids[j], bond_order=1.0, image_bond=True)

    logger.debug("Added %d image atoms", n_added)
    return n_added


def _add_crystal_images_supercell(
    graph: nx.Graph,
    crystal_data: CellData,
    repeats: tuple[int, int, int],
    unit_cell_data: CellData,
    n_base: int,
) -> int:
    """Optimized ghost generation for supercells.

    Uses the unit-cell lattice to precompute which atom pairs bond across each
    shift direction (small n_base x n_base check), then replays those pairs
    only for atoms on the relevant outer face of the supercell.

    For a supercell shift (dx_sc, dy_sc, dz_sc), the image of source atom
    at replica (ii, jj, kk) is at: pos + dx_sc*sc_a + dy_sc*sc_b + dz_sc*sc_c.
    This image bonds to a target at replica (ii', jj', kk') if the displacement
    corresponds to a valid unit-cell shift (di, dj, dk) with known bond pairs:
        ii' = ii + dx_sc*m - di   (and 0 <= ii' < m)
        jj' = jj + dy_sc*n - dj   (and 0 <= jj' < n)
        kk' = kk + dz_sc*l - dk   (and 0 <= kk' < l)

    Assumes ``list(graph.nodes())`` is in ascending-ID order so the first
    ``n_base`` entries are the unit cell — see ``build_supercell``.
    """
    sc_lattice = crystal_data.lattice
    uc_lattice = unit_cell_data.lattice
    uc_a, uc_b, uc_c = uc_lattice[0], uc_lattice[1], uc_lattice[2]
    sc_a, sc_b, sc_c = sc_lattice[0], sc_lattice[1], sc_lattice[2]
    m, n_rep, l_rep = repeats

    cell_ids = list(graph.nodes())
    if not cell_ids:
        return 0

    cell_syms_list = [graph.nodes[i]["symbol"] for i in cell_ids]
    cell_pos_arr = np.array([graph.nodes[i]["position"] for i in cell_ids])

    is_h = [s == "H" for s in cell_syms_list]
    is_c = [s == "C" for s in cell_syms_list]

    # Build element threshold from unit-cell atom types
    base_syms = cell_syms_list[:n_base]
    elem_thresh, eidx_base, max_cutoff = _build_elem_thresh(base_syms)

    # Precompute unit-cell cross-boundary pairs for each of the 26 UC shifts
    base_pos = cell_pos_arr[:n_base]
    uc_shifts = [(dx, dy, dz) for dx, dy, dz in _product((-1, 0, 1), repeat=3) if (dx, dy, dz) != (0, 0, 0)]

    uc_pairs: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for di, dj, dk in uc_shifts:
        offset = di * uc_a + dj * uc_b + dk * uc_c
        pairs = _find_bonded_pairs(base_pos + offset, base_pos, eidx_base, eidx_base, elem_thresh, max_cutoff)
        if pairs:
            uc_pairs[(di, dj, dk)] = pairs

    next_id = max(cell_ids) + 1
    n_added = 0

    # For each supercell shift direction
    sc_shifts = [(dx, dy, dz) for dx, dy, dz in _product((-1, 0, 1), repeat=3) if (dx, dy, dz) != (0, 0, 0)]

    for dx_sc, dy_sc, dz_sc in sc_shifts:
        sc_offset = dx_sc * sc_a + dy_sc * sc_b + dz_sc * sc_c

        # Collect all ghost sources across all contributing UC shift directions
        # ghost_key = src_global_idx → list of tgt_global_idx
        ghost_sources: dict[int, list[int]] = defaultdict(list)

        for (di, dj, dk), pairs in uc_pairs.items():
            # Determine valid source replica ranges from the constraint:
            #   ii' = ii + dx_sc*m - di,  0 <= ii' < m  →  di - dx_sc*m <= ii < m + di - dx_sc*m
            def _valid_range(d_sc: int, rep: int, delta: int) -> range:
                lo = max(0, delta - d_sc * rep)
                hi = min(rep, rep + delta - d_sc * rep)
                return range(lo, hi)

            i_range = _valid_range(dx_sc, m, di)
            j_range = _valid_range(dy_sc, n_rep, dj)
            k_range = _valid_range(dz_sc, l_rep, dk)

            if not i_range or not j_range or not k_range:
                continue

            for ii in i_range:
                ti = ii + dx_sc * m - di
                for jj in j_range:
                    tj = jj + dy_sc * n_rep - dj
                    for kk in k_range:
                        tk = kk + dz_sc * l_rep - dk
                        src_base = (ii * n_rep * l_rep + jj * l_rep + kk) * n_base
                        tgt_base = (ti * n_rep * l_rep + tj * l_rep + tk) * n_base
                        for src_local, tgt_local in pairs:
                            ghost_sources[src_base + src_local].append(tgt_base + tgt_local)

        # Create ghost nodes
        for src_idx, tgt_indices in ghost_sources.items():
            targets = tgt_indices
            if is_h[src_idx]:
                targets = [t for t in targets if not is_c[t]]
                if not targets:
                    continue

            img_pos = cell_pos_arr[src_idx] + sc_offset
            img_id = next_id
            next_id += 1
            n_added += 1
            graph.add_node(
                img_id,
                symbol=cell_syms_list[src_idx],
                position=(float(img_pos[0]), float(img_pos[1]), float(img_pos[2])),
                image=True,
                source=cell_ids[src_idx],
            )
            for tgt_idx in targets:
                graph.add_edge(img_id, cell_ids[tgt_idx], bond_order=1.0, image_bond=True)

    logger.debug("Added %d image atoms (supercell-optimized)", n_added)
    return n_added
