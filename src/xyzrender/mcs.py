"""Geometric MCS alignment for cross-molecule overlay and ref-orientation.

Aligns two molecular graphs by shape: PCA-orient both independently, try
multiple initial-alignment candidates (PCA sign-flips + heteroatom seeds),
refine each with ICP, then grow the largest *edge-preserving connected match*
via BFS from each close same-element pair.

The BFS growth ensures the returned match is a contiguous subgraph in both
molecules — a benzene ring in one region cannot match a benzene ring in a
different region.

Returns paired node IDs for Kabsch alignment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from xyzgraph import DATA

if TYPE_CHECKING:
    import networkx as nx


def _type_key(symbol: str, *, type_aware: bool) -> str:
    """Map an element symbol to its MCS-matching class.

    When *type_aware* is ``True``:

    * All metals collapse to ``"M"`` — so Pt↔Mn, Pd↔Fe etc. can match.
    * All heteroatoms (non-C, non-H, non-metal) collapse to ``"het"`` — so
      N↔Cl, P↔O etc. can match when no same-element option fits.
    * ``C`` and ``H`` stay as themselves (alkyl carbons / hydrogens still
      need to match their kind).

    With *type_aware=False* every atom is element-strict (original behaviour).
    """
    if not type_aware:
        return symbol
    if symbol in DATA.metals:
        return "M"
    if symbol not in {"C", "H"}:
        return "het"
    return symbol


def find_mcs_mapping(
    graph1: nx.Graph,
    graph2: nx.Graph,
    *,
    min_atoms: int = 3,
    threshold: float = 1.5,
    type_aware: bool = False,
) -> tuple[list[int], list[int]] | None:
    """Find the largest edge-preserving connected atom match between two graphs.

    Parameters
    ----------
    graph1, graph2:
        Molecular NetworkX graphs with ``"symbol"`` and ``"position"`` attrs.
    min_atoms:
        Minimum matched atoms to accept (default 3 — SVD minimum for Kabsch).
    threshold:
        Distance cutoff (Å) for accepting an atom pair as matched.
    type_aware:
        When ``True``, all metal elements collapse into a single ``"M"`` class
        so e.g. Pt↔Mn or Fe↔Pd can match in the MCS — useful for aligning
        related organometallic complexes whose metal centres differ.  Non-metal
        atoms remain element-strict (P↔P, N↔N, Cl↔Cl).  Default ``False``
        preserves element-strict legacy behaviour.

    Returns
    -------
    (g1_node_ids, g2_node_ids) or None
        Paired node IDs from the *original* input graphs.
    """
    from xyzrender.utils import kabsch_rotation, pca_orient

    nodes1, pos1, sym1 = _extract_atoms(graph1)
    nodes2, pos2, sym2 = _extract_atoms(graph2)

    if len(nodes1) < min_atoms or len(nodes2) < min_atoms:
        return None

    hi1 = [i for i, s in enumerate(sym1) if s != "H"]
    hi2 = [i for i, s in enumerate(sym2) if s != "H"]
    if len(hi1) < min_atoms or len(hi2) < min_atoms:
        return None

    # PCA-orient heavy atoms of each molecule independently
    hp1, hp2 = pos1[hi1], pos2[hi2]
    _, rot1 = pca_orient(hp1, return_matrix=True)
    _, rot2 = pca_orient(hp2, return_matrix=True)
    c1, c2 = hp1.mean(axis=0), hp2.mean(axis=0)
    all1 = (pos1 - c1) @ rot1.T
    all2_base = (pos2 - c2) @ rot2.T

    # --- Build candidate initial alignments ---
    # PCA sign flips in PCA space (4 candidates — shape-based)
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for flip in _FLIPS:
        candidates.append((all1, all2_base * flip))

    # Heteroatom local seeds in original space — crucial when molecules are
    # very different sizes and PCA axes don't correspond.
    for aligned in _local_het_seeds(
        pos1,
        sym1,
        pos2,
        sym2,
        graph1,
        graph2,
        nodes1,
        nodes2,
        kabsch_rotation,
        type_aware=type_aware,
    ):
        candidates.append((pos1, aligned))

    # Ring seeds — align each pair of same-size rings (5- or 6-mem cycles, so
    # benzene/cyclopentadienyl/heterocycle pairs all qualify).  Needed when
    # neither molecule has a unique shared heteroatom for _local_het_seeds.
    # Each ring pair contributes 2 seed orientations (normal sign-flip).
    for aligned in _ring_seeds(graph1, graph2, pos1, pos2, nodes1, nodes2):
        candidates.append((pos1, aligned))

    # --- Evaluate each candidate: ICP → BFS connected match ---
    node_to_idx1 = {n: i for i, n in enumerate(nodes1)}
    node_to_idx2 = {n: i for i, n in enumerate(nodes2)}

    best_match: list[tuple[int, int]] = []
    for ref, init_aligned in candidates:
        aligned = _icp_refine(
            ref,
            sym1,
            init_aligned.copy(),
            sym2,
            kabsch_rotation,
            type_aware=type_aware,
        )
        match = _connected_match(
            ref,
            sym1,
            aligned,
            sym2,
            graph1,
            graph2,
            nodes1,
            nodes2,
            node_to_idx1,
            node_to_idx2,
            threshold,
            type_aware=type_aware,
        )
        if len(match) > len(best_match):
            best_match = match

    if len(best_match) < min_atoms:
        return None

    # Reject small all-C/H matches (methyl, CH2, ethyl, propyl, butyl).  Larger
    # all-C/H matches (e.g. benzene's 6 carbons, cyclopentadiene's 5) are real
    # structural anchors and stay.  The heavy-atom threshold ≥ 5 keeps benzene
    # and cyclopentadienyl rings; a heteroatom in the match (N, O, P, Cl, …)
    # also passes.  Only enforced for type-aware mode (legacy keeps its answer).
    if type_aware:
        has_heteroatom = any(sym1[i] not in {"C", "H"} or sym2[j] not in {"C", "H"} for i, j in best_match)
        heavy_count = sum(1 for i, j in best_match if sym1[i] != "H" and sym2[j] != "H")
        if not has_heteroatom and heavy_count < 5:
            return None

    g1_ids = [nodes1[i] for i, _ in best_match]
    g2_ids = [nodes2[j] for _, j in best_match]
    return (g1_ids, g2_ids)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FLIPS = [
    np.array([1.0, 1.0, 1.0]),
    np.array([-1.0, -1.0, 1.0]),
    np.array([-1.0, 1.0, -1.0]),
    np.array([1.0, -1.0, -1.0]),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_atoms(
    graph: nx.Graph,
) -> tuple[list[int], np.ndarray, list[str]]:
    """Extract non-ghost node IDs, positions, and element symbols."""
    nodes, pos, sym = [], [], []
    for n in graph.nodes():
        s = graph.nodes[n].get("symbol", "")
        if s == "*":
            continue
        nodes.append(n)
        pos.append(graph.nodes[n]["position"])
        sym.append(s)
    return nodes, np.array(pos, dtype=float), sym


def _connected_match(
    pos1: np.ndarray,
    sym1: list[str],
    pos2: np.ndarray,
    sym2: list[str],
    graph1: nx.Graph,
    graph2: nx.Graph,
    nodes1: list[int],
    nodes2: list[int],
    node_to_idx1: dict[int, int],
    node_to_idx2: dict[int, int],
    threshold: float,
    *,
    type_aware: bool = False,
) -> list[tuple[int, int]]:
    """Find the largest edge-preserving connected match via BFS growth.

    For each close same-element pair, grow outward along bonds that exist in
    *both* graphs.  Returns the largest match found across all seeds.
    """
    close_pairs = _close_element_pairs(pos1, sym1, pos2, sym2, threshold, type_aware=type_aware)
    close_set: set[tuple[int, int]] = {(i, j) for _, i, j in close_pairs}
    max_possible = min(len(sym1), len(sym2))

    best: list[tuple[int, int]] = []
    covered1: set[int] = set()
    for _, seed_i, seed_j in close_pairs:
        if seed_i in covered1:
            continue

        matched: dict[int, int] = {seed_i: seed_j}
        used2: set[int] = {seed_j}
        queue = [seed_i]

        while queue:
            i1 = queue.pop(0)
            j1 = matched[i1]
            nid1, nid2 = nodes1[i1], nodes2[j1]

            # Try each neighbor pair bonded in BOTH graphs
            for nb_nid1 in graph1.neighbors(nid1):
                nb_i = node_to_idx1.get(nb_nid1)
                if nb_i is None or nb_i in matched:
                    continue
                for nb_nid2 in graph2.neighbors(nid2):
                    nb_j = node_to_idx2.get(nb_nid2)
                    if nb_j is None or nb_j in used2:
                        continue
                    if (nb_i, nb_j) in close_set:
                        matched[nb_i] = nb_j
                        used2.add(nb_j)
                        queue.append(nb_i)
                        break

        result = list(matched.items())
        if len(result) > len(best):
            best = result
            covered1 = set(matched.keys())
            if len(best) >= max_possible:
                break

    return best


def _icp_refine(
    pos1: np.ndarray,
    sym1: list[str],
    aligned: np.ndarray,
    sym2: list[str],
    kabsch_rotation,
    *,
    iters: int = 5,
    match_threshold: float = 2.0,
    type_aware: bool = False,
) -> np.ndarray:
    """Refine alignment via iterative closest-point element matching."""
    for _ in range(iters):
        pairs = _greedy_element_pairs(
            pos1,
            sym1,
            aligned,
            sym2,
            match_threshold,
            type_aware=type_aware,
        )
        if len(pairs) < 3:
            break
        idx1 = np.array([i for i, _ in pairs])
        idx2 = np.array([j for _, j in pairs])
        ref_pts = pos1[idx1]
        mob_pts = aligned[idx2]
        rot = kabsch_rotation(mob_pts, ref_pts)
        c_ref = ref_pts.mean(axis=0)
        c_mob = mob_pts.mean(axis=0)
        aligned = (aligned - c_mob) @ rot.T + c_ref
    return aligned


def _greedy_element_pairs(
    pos1: np.ndarray,
    sym1: list[str],
    pos2: np.ndarray,
    sym2: list[str],
    threshold: float,
    *,
    type_aware: bool = False,
) -> list[tuple[int, int]]:
    """Greedy closest-first assignment of same-element atom pairs (for ICP)."""
    used1: set[int] = set()
    used2: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in _close_element_pairs(pos1, sym1, pos2, sym2, threshold, type_aware=type_aware):
        if i not in used1 and j not in used2:
            pairs.append((i, j))
            used1.add(i)
            used2.add(j)
    return pairs


def _close_element_pairs(
    pos1: np.ndarray,
    sym1: list[str],
    pos2: np.ndarray,
    sym2: list[str],
    threshold: float,
    *,
    type_aware: bool = False,
) -> list[tuple[float, int, int]]:
    """Find all same-class atom pairs within *threshold*, sorted by distance.

    When *type_aware*, all metals collapse into a single ``"M"`` class so
    e.g. Pt↔Mn pairs are considered; non-metals stay element-strict.
    Vectorised per element class via broadcasting.
    """
    # Group indices by class (element symbol, or "M" for any metal when type-aware)
    elem_to_idx1: dict[str, list[int]] = {}
    for i, s in enumerate(sym1):
        elem_to_idx1.setdefault(_type_key(s, type_aware=type_aware), []).append(i)
    elem_to_idx2: dict[str, list[int]] = {}
    for j, s in enumerate(sym2):
        elem_to_idx2.setdefault(_type_key(s, type_aware=type_aware), []).append(j)

    pairs: list[tuple[float, int, int]] = []
    tsq = threshold * threshold
    for elem, idxs1 in elem_to_idx1.items():
        idxs2 = elem_to_idx2.get(elem)
        if idxs2 is None:
            continue
        # (n1e, 3) vs (n2e, 3) → (n1e, n2e) squared distances
        diff = pos1[idxs1, np.newaxis, :] - pos2[np.newaxis, idxs2, :]
        dsq = np.einsum("ijk,ijk->ij", diff, diff)
        rows, cols = np.nonzero(dsq < tsq)
        for r, c in zip(rows, cols, strict=True):
            pairs.append((float(np.sqrt(dsq[r, c])), idxs1[r], idxs2[c]))

    pairs.sort()
    return pairs


def _ring_seeds(
    graph1: nx.Graph,
    graph2: nx.Graph,
    pos1: np.ndarray,
    pos2: np.ndarray,
    nodes1: list[int],
    nodes2: list[int],
) -> list[np.ndarray]:
    """Seed candidates by aligning each pair of same-size 5/6-membered rings.

    Reuses ``graph.graph['rings']`` (cached by xyzgraph at build time, includes
    all rings not just aromatic) so seeding doesn't depend on aromaticity
    flags — which can be lost when bond orders change e.g. via overlay charge
    inheritance.  Sub-rings outside 5-6 atoms are skipped: real anchors come
    from arene / cyclopentadienyl scale.
    """
    from xyzrender.utils import pca_matrix, rotation_to_align

    def frames(graph, pos, nodes):
        idx_of = {n: i for i, n in enumerate(nodes)}
        for ring in graph.graph.get("rings") or []:
            if 5 <= len(ring) <= 6 and all(n in idx_of for n in ring):
                rp = pos[[idx_of[n] for n in ring]]
                # pca_matrix's least-variance axis = plane normal for a planar ring
                yield len(ring), rp.mean(axis=0), pca_matrix(rp)[-1]

    f1 = list(frames(graph1, pos1, nodes1))
    f2 = list(frames(graph2, pos2, nodes2))
    return [
        (pos2 - c2) @ rotation_to_align(n2, sign * n1).T + c1
        for size1, c1, n1 in f1
        for size2, c2, n2 in f2
        if size1 == size2
        for sign in (1.0, -1.0)
    ]


def _local_het_seeds(
    pos1: np.ndarray,
    sym1: list[str],
    pos2: np.ndarray,
    sym2: list[str],
    graph1: nx.Graph,
    graph2: nx.Graph,
    nodes1: list[int],
    nodes2: list[int],
    kabsch_rotation,
    *,
    min_seed: int = 3,
    type_aware: bool = False,
) -> list[np.ndarray]:
    """Generate alignments seeded from unique shared heteroatoms + neighbors.

    For each element appearing exactly once in both molecules, build a seed
    from that atom + its element-matched bonded neighbors.  When *type_aware*,
    all metals collapse into a single ``"M"`` class so e.g. a Pt complex and a
    Mn complex can seed from their respective single metals.
    """
    from collections import Counter

    c1 = Counter(_type_key(s, type_aware=type_aware) for s in sym1)
    c2 = Counter(_type_key(s, type_aware=type_aware) for s in sym2)
    # Skip ubiquitous classes; "M" stays because metals are typically rare and useful seeds.
    unique = {e for e in (set(c1) & set(c2)) - {"C", "H"} if c1[e] == 1 and c2[e] == 1}
    if not unique:
        return []

    node_to_idx1 = {n: i for i, n in enumerate(nodes1)}
    node_to_idx2 = {n: i for i, n in enumerate(nodes2)}
    g1_idx = {e: next(i for i, s in enumerate(sym1) if _type_key(s, type_aware=type_aware) == e) for e in unique}
    g2_idx = {e: next(i for i, s in enumerate(sym2) if _type_key(s, type_aware=type_aware) == e) for e in unique}

    results: list[np.ndarray] = []
    for e in sorted(unique):
        i1, i2 = g1_idx[e], g2_idx[e]
        seed = [(i1, i2)]

        # Group neighbors by class and pair them (M-classed when type_aware)
        nb1: dict[str, list[int]] = {}
        for nb in graph1.neighbors(nodes1[i1]):
            idx = node_to_idx1.get(nb)
            if idx is not None:
                nb1.setdefault(_type_key(sym1[idx], type_aware=type_aware), []).append(idx)

        nb2: dict[str, list[int]] = {}
        for nb in graph2.neighbors(nodes2[i2]):
            idx = node_to_idx2.get(nb)
            if idx is not None:
                nb2.setdefault(_type_key(sym2[idx], type_aware=type_aware), []).append(idx)

        for elem, idx_list1 in nb1.items():
            idx_list2 = nb2.get(elem)
            if idx_list2 is None:
                continue
            for a, b in zip(idx_list1, idx_list2, strict=False):
                seed.append((a, b))

        if len(seed) < min_seed:
            continue
        ref_pts = pos1[np.array([i for i, _ in seed])]
        mob_pts = pos2[np.array([j for _, j in seed])]
        rot = kabsch_rotation(mob_pts, ref_pts)
        c_ref = ref_pts.mean(axis=0)
        c_mob = mob_pts.mean(axis=0)
        results.append((pos2 - c_mob) @ rot.T + c_ref)

    return results
