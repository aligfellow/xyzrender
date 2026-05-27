"""Molecule overlay: structural alignment + combined rendering.

``align()`` picks the best strategy automatically and logs which one ran at INFO:

1. Explicit ``align_atoms`` selector → :func:`xyzrender.align.align_with_selection`
   (metal-fragment when metals are in the selection, otherwise MCS-on-subgraph
   + K-subset Kabsch, lowest RMSD wins).
2. Auto path: both have metals → metal-fragment overlay (per-M-pairing,
   metal-pivot Kabsch so paired metals coincide exactly).
3. Auto path: same shape + same elements → index-paired Kabsch.
4. Auto path: otherwise → type-aware MCS on full graphs, then PCA + ICP best-fit
   as last resort.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from xyzrender.colors import Color, bond_color_from_atom
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


def _has_metal(graph: nx.Graph) -> bool:
    from xyzgraph import DATA

    return any(d.get("symbol") in DATA.metals for _, d in graph.nodes(data=True))


def _selection_has_metal(graph: nx.Graph, atoms: list[int]) -> bool:
    from xyzgraph import DATA

    return any(graph.nodes[n].get("symbol") in DATA.metals for n in atoms)


def _metal_pair(g1: nx.Graph, a: int, g2: nx.Graph, b: int) -> bool:
    from xyzgraph import DATA

    return g1.nodes[a].get("symbol") in DATA.metals and g2.nodes[b].get("symbol") in DATA.metals


# ---------------------------------------------------------------------------
# Public API — overlay
# ---------------------------------------------------------------------------


def align(
    mol1_graph: nx.Graph,
    mol2_graph: nx.Graph,
    align_atoms: list[int] | str | None = None,
) -> np.ndarray:
    """Align mol2 onto mol1; return aligned 3-D positions for mol2 in mol1's frame.

    Strategy is auto-selected (see module docstring) unless *align_atoms* is set.
    All strategy choices and key results are logged at INFO so users can see
    which path ran.
    """
    import logging

    from xyzrender.align import (
        _align_metal_fragments,
        align_with_selection,
        best_fit_align,
        kabsch_with_pivot,
    )
    from xyzrender.mcs import find_mcs_mapping
    from xyzrender.selectors import resolve_atom_indices
    from xyzrender.utils import mcs_kabsch_align

    log = logging.getLogger(__name__)

    pos1, nodes1 = _positions(mol1_graph)
    pos2, nodes2 = _positions(mol2_graph)
    n1, n2 = len(nodes1), len(nodes2)

    # --- 1. Explicit selector spec from caller ("M,L", "Fe,P", "het", "1-5", …) ---
    if isinstance(align_atoms, str):
        ref_atoms = sorted(resolve_atom_indices(align_atoms, mol1_graph))
        mob_atoms = sorted(resolve_atom_indices(align_atoms, mol2_graph))
        # Warn when the selection excludes metals on organometallic graphs —
        # selector path is literal so we honour it, but the user probably wants
        # metal-fragment overlay (which needs metals in the selection).
        if (
            _has_metal(mol1_graph)
            and _has_metal(mol2_graph)
            and not _selection_has_metal(mol1_graph, ref_atoms)
            and not _selection_has_metal(mol2_graph, mob_atoms)
        ):
            log.warning("overlay: %r excludes metals; consider 'M,%s'", align_atoms, align_atoms)
        aligned, refs, _, mm, rmsd = align_with_selection(
            mol1_graph,
            mol2_graph,
            ref_atoms,
            mob_atoms,
        )
        log.info(
            "overlay: selector %r → %d/%d candidates, %d paired, %d mismatch, full-mol rmsd=%.3fÅ",
            align_atoms,
            len(ref_atoms),
            len(mob_atoms),
            len(refs),
            mm,
            rmsd,
        )
        return aligned

    # --- 2. Explicit atom-index subset ---
    if align_atoms is not None:
        if n1 != n2:
            msg = f"overlay: align_atoms indices require same atom count (got {n1} vs {n2})"
            raise ValueError(msg)
        log.info("overlay: index-paired Kabsch on %d explicit atoms", len(align_atoms))
        return kabsch_align(pos1, pos2, align_atoms=align_atoms)

    # --- 3. Auto: both have metals → metal-fragment overlay ---
    if _has_metal(mol1_graph) and _has_metal(mol2_graph):
        try:
            aligned, refs, _, mm, rmsd = _align_metal_fragments(mol1_graph, mol2_graph)
            log.info(
                "overlay: metal-fragment → %d atoms paired, %d mismatch, full-mol rmsd=%.3fÅ",
                len(refs),
                mm,
                rmsd,
            )
            return aligned
        except ValueError as exc:
            log.warning("overlay: metal-fragment path failed (%s); trying alternatives", exc)

    # --- 4. Same shape + same elements: index-paired Kabsch ---
    if n1 == n2 and _elements_match(mol1_graph, mol2_graph):
        log.info("overlay: same-shape index-paired Kabsch on %d atoms", n1)
        return kabsch_align(pos1, pos2)

    # --- 5. Type-aware MCS on full graphs (metal-pivot when found) ---
    mapping = find_mcs_mapping(mol1_graph, mol2_graph, type_aware=True)
    if mapping is not None:
        g1_ids, g2_ids = mapping
        g1_idx = [nodes1.index(n) for n in g1_ids]
        g2_idx = [nodes2.index(n) for n in g2_ids]
        matched_frac = len(g1_ids) / min(n1, n2)
        if matched_frac < 0.25:
            log.warning(
                "overlay: only %d/%d atoms matched (%.0f%%) — alignment may be poor",
                len(g1_ids),
                min(n1, n2),
                matched_frac * 100,
            )
        metal_pairs = [(a, b) for a, b in zip(g1_ids, g2_ids, strict=True) if _metal_pair(mol1_graph, a, mol2_graph, b)]
        m_pivot_ref = [nodes1.index(a) for a, _ in metal_pairs]
        m_pivot_mob = [nodes2.index(b) for _, b in metal_pairs]
        if m_pivot_ref:
            log.info("overlay: type-aware MCS (%d atoms) with metal pivot", len(g1_ids))
            return kabsch_with_pivot(pos1, pos2, g1_idx, g2_idx, m_pivot_ref, m_pivot_mob)
        log.info("overlay: type-aware MCS (%d atoms, no metals in match)", len(g1_ids))
        return mcs_kabsch_align(pos1, pos2, g1_idx, g2_idx)

    # --- 6. Last resort: spec-free PCA + ICP ---
    log.warning("overlay: no MCS match — falling back to PCA-seeded ICP")
    aligned, rmsd = best_fit_align(mol1_graph, mol2_graph)
    log.info("overlay: best_fit (PCA+ICP) → rmsd=%.3fÅ", rmsd)
    return aligned


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

    # Manual overlay TS bonds — translate overlay-local indices through the
    # id_map and stamp TS=True (add the edge if missing, with overlay colour).
    if ov.ts_bonds:
        n2 = mol2_graph.number_of_nodes()
        ov_bond_color = ov.bond_color
        if ov_bond_color is None and ov.color is not None:
            ov_bond_color = bond_color_from_atom(Color.from_str(ov.color))
        for i, j in ov.ts_bonds:
            if not (0 <= i < n2 and 0 <= j < n2):
                msg = f"--overlay-ts-bond pair ({i + 1}, {j + 1}) out of range for overlay with {n2} atoms"
                raise ValueError(msg)
            u, v = mol2_map[mol2_ids[i]], mol2_map[mol2_ids[j]]
            if merged.has_edge(u, v):
                merged[u][v]["TS"] = True
            else:
                extras: dict = {"molecule_index": 1, "TS": True, "bond_order": 1.0}
                if ov_bond_color is not None:
                    extras["bond_color_override"] = ov_bond_color
                merged.add_edge(u, v, **extras)

    return merged
