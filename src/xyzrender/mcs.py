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

if TYPE_CHECKING:
    import networkx as nx


def find_mcs_mapping(
    graph1: nx.Graph,
    graph2: nx.Graph,
    *,
    min_atoms: int = 3,
    threshold: float = 1.5,
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
    ):
        candidates.append((pos1, aligned))

    # --- Evaluate each candidate: ICP → BFS connected match ---
    node_to_idx1 = {n: i for i, n in enumerate(nodes1)}
    node_to_idx2 = {n: i for i, n in enumerate(nodes2)}

    best_match: list[tuple[int, int]] = []
    for ref, init_aligned in candidates:
        aligned = _icp_refine(ref, sym1, init_aligned.copy(), sym2, kabsch_rotation)
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
        )
        if len(match) > len(best_match):
            best_match = match

    if len(best_match) < min_atoms:
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
) -> list[tuple[int, int]]:
    """Find the largest edge-preserving connected match via BFS growth.

    For each close same-element pair, grow outward along bonds that exist in
    *both* graphs.  Returns the largest match found across all seeds.
    """
    close_pairs = _close_element_pairs(pos1, sym1, pos2, sym2, threshold)
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
) -> np.ndarray:
    """Refine alignment via iterative closest-point element matching."""
    for _ in range(iters):
        pairs = _greedy_element_pairs(pos1, sym1, aligned, sym2, match_threshold)
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
) -> list[tuple[int, int]]:
    """Greedy closest-first assignment of same-element atom pairs (for ICP)."""
    used1: set[int] = set()
    used2: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in _close_element_pairs(pos1, sym1, pos2, sym2, threshold):
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
) -> list[tuple[float, int, int]]:
    """Find all same-element atom pairs within *threshold*, sorted by distance.

    Vectorised per element type: computes full distance matrix for each element
    via broadcasting instead of looping over individual atoms.
    """
    # Group indices by element
    elem_to_idx1: dict[str, list[int]] = {}
    for i, s in enumerate(sym1):
        elem_to_idx1.setdefault(s, []).append(i)
    elem_to_idx2: dict[str, list[int]] = {}
    for j, s in enumerate(sym2):
        elem_to_idx2.setdefault(s, []).append(j)

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
) -> list[np.ndarray]:
    """Generate alignments seeded from unique shared heteroatoms + neighbors.

    For each element appearing exactly once in both molecules, build a seed
    from that atom + its element-matched bonded neighbors.
    """
    from collections import Counter

    c1, c2 = Counter(sym1), Counter(sym2)
    unique = {e for e in (set(c1) & set(c2)) - {"C", "H"} if c1[e] == 1 and c2[e] == 1}
    if not unique:
        return []

    node_to_idx1 = {n: i for i, n in enumerate(nodes1)}
    node_to_idx2 = {n: i for i, n in enumerate(nodes2)}
    g1_idx = {e: next(i for i, s in enumerate(sym1) if s == e) for e in unique}
    g2_idx = {e: next(i for i, s in enumerate(sym2) if s == e) for e in unique}

    results: list[np.ndarray] = []
    for e in sorted(unique):
        i1, i2 = g1_idx[e], g2_idx[e]
        seed = [(i1, i2)]

        # Group neighbors by element and pair them
        nb1: dict[str, list[int]] = {}
        for nb in graph1.neighbors(nodes1[i1]):
            idx = node_to_idx1.get(nb)
            if idx is not None:
                nb1.setdefault(sym1[idx], []).append(idx)

        nb2: dict[str, list[int]] = {}
        for nb in graph2.neighbors(nodes2[i2]):
            idx = node_to_idx2.get(nb)
            if idx is not None:
                nb2.setdefault(sym2[idx], []).append(idx)

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
