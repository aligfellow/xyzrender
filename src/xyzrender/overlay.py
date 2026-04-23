"""Molecule overlay: RMSD-minimising structural alignment and combined rendering.

Two molecules are aligned via the Kabsch algorithm so that mol2 is superimposed
onto mol1 in its coordinate frame.  The merged graph is rendered with the overlay
color (default: mediumorchid); mol1 atoms use the standard CPK palette and are
always on top when depths are equal (drawn last in SVG order).

When both molecules have the same atoms in the same order, alignment is direct
(index-based Kabsch).  When atom counts or elements differ, the Maximum Common
Substructure (MCS) is found automatically and used as the alignment basis.

This module also exposes :func:`kabsch_align`, the shared Kabsch helper used by
both overlay and ensemble alignment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from xyzrender.merge import (
    _Z_NUDGE,
    merge_aromatic_rings,
    stamp_structure_edges,
    stamp_structure_nodes,
)
from xyzrender.utils import kabsch_align

if TYPE_CHECKING:
    import networkx as nx

    from xyzrender.types import RenderConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _node_list(graph: nx.Graph) -> list:
    return list(graph.nodes())


def _positions(graph: nx.Graph) -> tuple[np.ndarray, list]:
    nodes = _node_list(graph)
    pos = np.array([graph.nodes[n]["position"] for n in nodes], dtype=float)
    return pos, nodes


def _elements_match(g1: nx.Graph, g2: nx.Graph) -> bool:
    """Check if both graphs have the same element sequence (ignoring ghosts)."""
    syms1 = [g1.nodes[n]["symbol"] for n in g1.nodes() if g1.nodes[n].get("symbol", "") != "*"]
    syms2 = [g2.nodes[n]["symbol"] for n in g2.nodes() if g2.nodes[n].get("symbol", "") != "*"]
    return syms1 == syms2


# kabsch_align is implemented in utils and re-exported here for backward compat.
__all__ = ["align", "kabsch_align", "merge_graphs"]


# ---------------------------------------------------------------------------
# Public API — overlay
# ---------------------------------------------------------------------------


def align(
    mol1_graph: nx.Graph,
    mol2_graph: nx.Graph,
    align_atoms: list[int] | None = None,
) -> np.ndarray:
    """Align mol2 onto mol1; return aligned positions for mol2 nodes.

    When both molecules have the same atoms in the same order, alignment is
    direct (index-based Kabsch).  Otherwise the Maximum Common Substructure
    is found automatically and used as the alignment basis.

    Parameters
    ----------
    mol1_graph, mol2_graph:
        NetworkX graphs.  This function does not mutate them.
    align_atoms:
        Optional 0-indexed atom indices to fit on (min 3).  Only used when
        both molecules have the same number of atoms.

    Returns
    -------
    np.ndarray, shape (n2, 3)
        Aligned 3-D positions for mol2 nodes in their original graph order.
    """
    pos1, nodes1 = _positions(mol1_graph)
    pos2, nodes2 = _positions(mol2_graph)
    n1, n2 = len(nodes1), len(pos2)

    # Fast path: same molecule, same ordering
    if n1 == n2 and (align_atoms is not None or _elements_match(mol1_graph, mol2_graph)):
        return kabsch_align(pos1, pos2, align_atoms=align_atoms)

    # Different molecules — MCS alignment
    import logging

    from xyzrender.mcs import find_mcs_mapping
    from xyzrender.utils import mcs_kabsch_align

    mapping = find_mcs_mapping(mol1_graph, mol2_graph)
    if mapping is None:
        msg = f"overlay: no common substructure (>= 3 atoms) between mol1 ({n1} atoms) and mol2 ({n2} atoms)"
        raise ValueError(msg)

    g1_ids, g2_ids = mapping
    matched_frac = len(g1_ids) / min(n1, n2)
    if matched_frac < 0.25:
        logging.getLogger(__name__).warning(
            "overlay: only %d/%d atoms matched (%.0f%%) — alignment may be poor",
            len(g1_ids),
            min(n1, n2),
            matched_frac * 100,
        )
    g1_idx = [nodes1.index(n) for n in g1_ids]
    g2_idx = [nodes2.index(n) for n in g2_ids]
    return mcs_kabsch_align(pos1, pos2, g1_idx, g2_idx)


def merge_graphs(
    mol1_graph: nx.Graph,
    mol2_graph: nx.Graph,
    aligned_pos2: np.ndarray,
    cfg: RenderConfig,
) -> nx.Graph:
    """Build a merged NetworkX graph containing both molecules.

    mol1 nodes keep their original integer IDs (``0 … n1-1``); mol2 nodes are
    renumbered to ``n1 … n1+n2-1``.  Per-structure attributes
    (``molecule_index``, ``structure_color``, ``structure_opacity``,
    ``bond_color_override``) are stamped by the shared helpers in
    :mod:`xyzrender.merge`.

    The overlay molecule's ``aromatic_rings`` are translated through the
    id_map and merged into ``merged.graph["aromatic_rings"]`` so downstream
    consumers (e.g. ``apply_bond_rules`` for haptic detection) see rings
    from both molecules.

    mol2 z-positions are nudged back by :data:`_Z_NUDGE` so mol1 atoms render
    on top when projected depths coincide.
    """
    import networkx as nx

    ov = cfg.overlay

    n1 = mol1_graph.number_of_nodes()
    merged = nx.Graph()
    merged.graph.update(mol1_graph.graph)
    # Fresh list so merge_aromatic_rings can extend it without mutating mol1_graph.
    if "aromatic_rings" in mol1_graph.graph:
        merged.graph["aromatic_rings"] = [set(r) for r in mol1_graph.graph["aromatic_rings"]]

    mol1_ids = _node_list(mol1_graph)
    mol1_map = {nid: nid for nid in mol1_ids}
    mol1_positions = np.array([mol1_graph.nodes[n]["position"] for n in mol1_ids], dtype=float)
    stamp_structure_nodes(merged, mol1_graph, mol1_map, mol1_positions, molecule_index=0)
    stamp_structure_edges(merged, mol1_graph, mol1_map, molecule_index=0)

    mol2_ids = _node_list(mol2_graph)
    mol2_map = {old: n1 + k for k, old in enumerate(mol2_ids)}
    stamp_structure_nodes(
        merged,
        mol2_graph,
        mol2_map,
        aligned_pos2,
        molecule_index=1,
        color=ov.color,
        opacity=ov.opacity,
        atom_scale=ov.atom_scale,
        stroke_width=ov.atom_stroke_width,
        stroke_color=ov.atom_stroke_color,
        z_offset=_Z_NUDGE,
    )
    stamp_structure_edges(
        merged,
        mol2_graph,
        mol2_map,
        molecule_index=1,
        color=ov.color,
        bond_color=ov.bond_color,
        bond_width=ov.bond_width,
        outline_width=ov.bond_outline_width,
        outline_color=ov.bond_outline_color,
    )
    merge_aromatic_rings(merged, mol2_graph, mol2_map)

    return merged
