"""Structural alignment strategies for cross-molecule overlay.

Public entry points used by :mod:`overlay`:

* :func:`align_with_selection` — selector-driven alignment.
* :func:`best_fit_align` — spec-free PCA + ICP fallback.
* :func:`kabsch_with_pivot` — Kabsch with exact pivot coincidence.

:func:`_align_metal_fragments` powers the organometallic auto-path.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import TYPE_CHECKING

import numpy as np
from xyzgraph import DATA

from xyzrender.utils import kabsch_rotation, pca_matrix

if TYPE_CHECKING:
    import networkx as nx


# Cap anchor-permutation enumeration to bound memory + runtime.
_MAX_ANCHOR_CANDIDATES = 1024


# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------


def _greedy_nn(
    ref_pos: np.ndarray,
    mob_pos: np.ndarray,
) -> tuple[list[int], list[int], float]:
    """Greedy nearest-neighbour pairing. Returns (ref_idx, mob_idx, rmsd)."""
    n_ref, n_mob = len(ref_pos), len(mob_pos)
    if n_ref == 0 or n_mob == 0:
        return [], [], float("inf")
    work = np.linalg.norm(ref_pos[:, None, :] - mob_pos[None, :, :], axis=-1)
    k = min(n_ref, n_mob)
    pr: list[int] = []
    pm: list[int] = []
    total_sq = 0.0
    for _ in range(k):
        flat = int(np.argmin(work))
        i, j = flat // n_mob, flat % n_mob
        pr.append(i)
        pm.append(j)
        total_sq += float(work[i, j]) ** 2
        work[i, :] = np.inf
        work[:, j] = np.inf
    return pr, pm, float(np.sqrt(total_sq / k))


def kabsch_with_pivot(
    ref_pos: np.ndarray,
    mob_pos: np.ndarray,
    ref_indices: list[int],
    mob_indices: list[int],
    pivot_ref_indices: list[int],
    pivot_mob_indices: list[int],
) -> np.ndarray:
    """Kabsch where the pivot centroid coincides exactly; rotation fitted on anchors."""
    pivot_ref = ref_pos[pivot_ref_indices].mean(axis=0)
    pivot_mob = mob_pos[pivot_mob_indices].mean(axis=0)
    anchor_ref = ref_pos[ref_indices] - pivot_ref
    anchor_mob = mob_pos[mob_indices] - pivot_mob
    rot = kabsch_rotation(anchor_mob, anchor_ref)
    return (mob_pos - pivot_mob) @ rot.T + pivot_ref


# ---------------------------------------------------------------------------
# best_fit_align: spec-free PCA + ICP
# ---------------------------------------------------------------------------


def _icp(
    ref_pos: np.ndarray,
    mob_pos: np.ndarray,
    *,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """ICP: greedy-NN pair → Kabsch → repeat until RMSD converges."""
    current = mob_pos.copy()
    last_rmsd = rmsd = float("inf")
    for _ in range(max_iter):
        pr, pm, rmsd = _greedy_nn(ref_pos, current)
        if not pr:
            return current, rmsd
        sub_r = ref_pos[pr]
        sub_m = current[pm]
        t_r = sub_r.mean(axis=0)
        t_m = sub_m.mean(axis=0)
        rot = kabsch_rotation(sub_m - t_m, sub_r - t_r)
        current = (current - t_m) @ rot.T + t_r
        if abs(last_rmsd - rmsd) < tol:
            break
        last_rmsd = rmsd
    return current, rmsd


def best_fit_align(ref_graph: "nx.Graph", mob_graph: "nx.Graph") -> tuple[np.ndarray, float]:
    """Spec-free best-fit: PCA-orient each graph, ICP from 4 sign-flip seeds."""
    ref_pos = np.array([ref_graph.nodes[n]["position"] for n in ref_graph.nodes()], dtype=float)
    mob_pos = np.array([mob_graph.nodes[n]["position"] for n in mob_graph.nodes()], dtype=float)
    ref_centroid = ref_pos.mean(axis=0)
    mob_centroid = mob_pos.mean(axis=0)
    r_pca = pca_matrix(ref_pos - ref_centroid)
    m_pca = pca_matrix(mob_pos - mob_centroid)
    rp = (ref_pos - ref_centroid) @ r_pca.T
    mp = (mob_pos - mob_centroid) @ m_pca.T

    best_rmsd = float("inf")
    best_aligned = mp
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        sz = sx * sy  # det = +1
        seed = mp * np.array([sx, sy, sz], dtype=float)
        aligned, rmsd = _icp(rp, seed)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_aligned = aligned

    return best_aligned @ r_pca + ref_centroid, best_rmsd


# ---------------------------------------------------------------------------
# Anchor enumeration (used by K-subset Kabsch fallback + _align_metal_fragments)
# ---------------------------------------------------------------------------


def _token_pairings(
    ref_atoms: list[int],
    mob_atoms: list[int],
    cap: int,
) -> list[tuple[list[int], list[int]]]:
    """Up to *cap* K-pairings; permutes the larger side, keeps the smaller whole.

    Samples randomly (fixed seed) when the permutation space exceeds *cap*,
    avoiding lexicographic bias toward early indices.
    """
    n_ref, n_mob = len(ref_atoms), len(mob_atoms)
    if n_ref == 0 or n_mob == 0:
        return []
    ref_smaller = n_ref <= n_mob
    n_small, n_large = (n_ref, n_mob) if ref_smaller else (n_mob, n_ref)

    total = math.perm(n_large, n_small)
    if total <= cap:
        perms: list[tuple[int, ...]] = list(itertools.permutations(range(n_large), n_small))
    else:
        rng = random.Random(0)  # deterministic — alignments must reproduce
        seen: set[tuple[int, ...]] = set()
        perms = []
        while len(perms) < cap:
            p = tuple(rng.sample(range(n_large), n_small))
            if p not in seen:
                seen.add(p)
                perms.append(p)

    if ref_smaller:
        return [(list(ref_atoms), [mob_atoms[j] for j in p]) for p in perms]
    return [([ref_atoms[i] for i in p], list(mob_atoms)) for p in perms]


def _candidate_anchors(
    tokens_ref: list[list[int]],
    tokens_mob: list[list[int]],
) -> list[tuple[list[int], list[int]]]:
    """Cartesian product of per-token pairings, capped by _MAX_ANCHOR_CANDIDATES."""
    per_token = [_token_pairings(r, m, _MAX_ANCHOR_CANDIDATES) for r, m in zip(tokens_ref, tokens_mob, strict=True)]
    sizes = [len(opts) for opts in per_token]
    while math.prod(sizes) > _MAX_ANCHOR_CANDIDATES and max(sizes) > 1:
        i = max(range(len(sizes)), key=lambda k: sizes[k])
        sizes[i] = max(1, sizes[i] // 2)
    per_token = [opts[:s] for opts, s in zip(per_token, sizes, strict=True)]

    return [
        ([n for rp, _ in combo for n in rp], [n for _, mp in combo for n in mp])
        for combo in itertools.product(*per_token)
    ]


def _align_anchored(
    ref_graph: "nx.Graph",
    mob_graph: "nx.Graph",
    tokens_ref: list[list[int]],
    tokens_mob: list[list[int]],
    pivot_token_idx: int | None = None,
) -> tuple[np.ndarray, list[int], list[int], int, float]:
    """Pick the lex-min (rmsd, mismatches) over all candidate anchor pairings.

    RMSD rounded to 1 mÅ so ties break by element match.  *pivot_token_idx*,
    when set, anchors that token's centroid exactly.
    """
    ref_nodes = list(ref_graph.nodes())
    mob_nodes = list(mob_graph.nodes())
    pos_ref = np.array([ref_graph.nodes[n]["position"] for n in ref_nodes], dtype=float)
    pos_mob = np.array([mob_graph.nodes[n]["position"] for n in mob_nodes], dtype=float)
    sym_ref = [ref_graph.nodes[n]["symbol"] for n in ref_nodes]
    sym_mob = [mob_graph.nodes[n]["symbol"] for n in mob_nodes]
    ref_row = {n: i for i, n in enumerate(ref_nodes)}
    mob_row = {n: i for i, n in enumerate(mob_nodes)}

    if pivot_token_idx is not None:
        start = sum(len(t) for t in tokens_ref[:pivot_token_idx])
        size = len(tokens_ref[pivot_token_idx])
        pivot_slice = slice(start, start + size)
    else:
        pivot_slice = None

    candidates = _candidate_anchors(tokens_ref, tokens_mob)
    if not candidates:
        msg = "anchored: no candidate pairings from the supplied tokens"
        raise ValueError(msg)

    best_score: tuple[float, int] = (float("inf"), 0)
    best_rows: tuple[list[int], list[int]] = ([], [])
    best_aligned = pos_mob
    best_rmsd = float("inf")

    for ref_anchors, mob_anchors in candidates:
        r_rows = [ref_row[n] for n in ref_anchors]
        m_rows = [mob_row[n] for n in mob_anchors]
        sub_r = pos_ref[r_rows]
        sub_m = pos_mob[m_rows]
        if pivot_slice is not None:
            t_ref = sub_r[pivot_slice].mean(axis=0)
            t_mob = sub_m[pivot_slice].mean(axis=0)
        else:
            t_ref = sub_r.mean(axis=0)
            t_mob = sub_m.mean(axis=0)
        rot = kabsch_rotation(sub_m - t_mob, sub_r - t_ref)
        aligned = (pos_mob - t_mob) @ rot.T + t_ref
        _, _, full_rmsd = _greedy_nn(pos_ref, aligned)
        mismatches = sum(1 for r, m in zip(r_rows, m_rows, strict=True) if sym_ref[r] != sym_mob[m])
        score = (round(full_rmsd, 3), mismatches)
        if score < best_score:
            best_score = score
            best_rows = (r_rows, m_rows)
            best_aligned = aligned
            best_rmsd = full_rmsd

    return best_aligned, best_rows[0], best_rows[1], best_score[1], best_rmsd


# ---------------------------------------------------------------------------
# Public: selector-driven alignment
# ---------------------------------------------------------------------------


def align_with_selection(
    ref_graph: "nx.Graph",
    mob_graph: "nx.Graph",
    ref_atoms: list[int] | set[int],
    mob_atoms: list[int] | set[int],
) -> tuple[np.ndarray, list[int], list[int], int, float]:
    """Align *mob* → *ref* on a selector-resolved candidate atom set.

    Metals in both selections → metal-fragment overlay (paired metals
    coincide exactly).  Otherwise tries MCS-on-induced-subgraph and K-subset
    Kabsch, returns the lowest-RMSD candidate.

    Returns ``(aligned_positions, paired_ref_rows, paired_mob_rows, mismatches, full_mol_rmsd)``.
    """
    import logging

    from xyzrender.mcs import find_mcs_mapping

    log = logging.getLogger(__name__)

    ref_atoms = list(ref_atoms)
    mob_atoms = list(mob_atoms)
    if len(ref_atoms) < 3 or len(mob_atoms) < 3:
        msg = (
            f"alignment: selector resolved {len(ref_atoms)} ref / {len(mob_atoms)} "
            "mobile atoms; need ≥ 3 in each for Kabsch"
        )
        raise ValueError(msg)

    # Metals in selection on both sides → metal-fragment wins by construction.
    metals_ref = [n for n in ref_atoms if ref_graph.nodes[n].get("symbol") in DATA.metals]
    metals_mob = [n for n in mob_atoms if mob_graph.nodes[n].get("symbol") in DATA.metals]
    if metals_ref and metals_mob:
        try:
            return _align_metal_fragments(ref_graph, mob_graph, ref_atoms, mob_atoms)
        except ValueError as exc:
            log.debug("metal-fragment path unavailable (%s); falling through", exc)

    # Otherwise: try every applicable strategy and return the lowest-RMSD.
    candidates: list[tuple[np.ndarray, list[int], list[int], int, float]] = []

    # Strategy A: MCS on induced subgraph (with metal pivot when matched).
    mapping = find_mcs_mapping(
        ref_graph.subgraph(ref_atoms),
        mob_graph.subgraph(mob_atoms),
        type_aware=True,
    )
    if mapping is not None:
        g1_ids, g2_ids = mapping
        m_pairs, rest_pairs = [], []
        for a, b in zip(g1_ids, g2_ids, strict=True):
            sa = ref_graph.nodes[a].get("symbol")
            sb = mob_graph.nodes[b].get("symbol")
            (m_pairs if sa in DATA.metals and sb in DATA.metals else rest_pairs).append((a, b))
        if m_pairs and rest_pairs:
            m_ref, m_mob = (list(x) for x in zip(*m_pairs, strict=True))
            r_ref, r_mob = (list(x) for x in zip(*rest_pairs, strict=True))
            candidates.append(
                _align_anchored(
                    ref_graph,
                    mob_graph,
                    [m_ref, r_ref],
                    [m_mob, r_mob],
                    pivot_token_idx=0,
                )
            )
        else:
            candidates.append(
                _align_anchored(
                    ref_graph,
                    mob_graph,
                    [list(g1_ids)],
                    [list(g2_ids)],
                )
            )

    # Strategy B: K-subset Kabsch over the raw selection (catches the
    # isolated-atom case where MCS finds no edges).  Random sampling in
    # _token_pairings handles permutation bias for large selections.
    candidates.append(_align_anchored(ref_graph, mob_graph, [ref_atoms], [mob_atoms]))

    # Pick the candidate with the lowest full-molecule RMSD (index 4 of tuple).
    return min(candidates, key=lambda r: r[4])


# ---------------------------------------------------------------------------
# Metal-fragment overlay (per-M-pairing + ligand-shell + metal pivot)
# ---------------------------------------------------------------------------


def _coord_shell(
    graph: "nx.Graph",
    metals: set[int],
    restrict_to: set[int] | None = None,
) -> list[int]:
    """Non-metal atoms in *graph* bonded to *metals* (optionally restricted)."""
    return [
        n
        for n, d in graph.nodes(data=True)
        if d.get("symbol") not in DATA.metals
        and (restrict_to is None or n in restrict_to)
        and any(nb in metals for nb in graph.neighbors(n))
    ]


def _align_metal_fragments(
    ref_graph: "nx.Graph",
    mob_graph: "nx.Graph",
    sel_ref: list[int] | None = None,
    sel_mob: list[int] | None = None,
) -> tuple[np.ndarray, list[int], list[int], int, float]:
    """Per-M-pairing organometallic overlay with metal-pivot Kabsch.

    Enumerates metal correspondences, narrows ligands to each pair's coord
    shell, fits with the metal centroid as pivot, keeps lowest-RMSD.
    *sel_ref* / *sel_mob* restrict the search (default: all nodes).
    """
    if sel_ref is None:
        sel_ref = list(ref_graph.nodes())
    if sel_mob is None:
        sel_mob = list(mob_graph.nodes())
    sel_ref_set, sel_mob_set = set(sel_ref), set(sel_mob)
    metals_ref = {n for n in sel_ref if ref_graph.nodes[n].get("symbol") in DATA.metals}
    metals_mob = {n for n in sel_mob if mob_graph.nodes[n].get("symbol") in DATA.metals}
    if not metals_ref or not metals_mob:
        msg = "metal-fragment: both selections must contain metal atoms"
        raise ValueError(msg)

    best_score: tuple[float, int] = (float("inf"), 0)
    best_result: tuple[np.ndarray, list[int], list[int], int, float] | None = None
    for m_ref_ids, m_mob_ids in _token_pairings(
        sorted(metals_ref),
        sorted(metals_mob),
        _MAX_ANCHOR_CANDIDATES,
    ):
        m_ref_set, m_mob_set = set(m_ref_ids), set(m_mob_ids)
        l_ref = _coord_shell(ref_graph, m_ref_set, restrict_to=sel_ref_set)
        l_mob = _coord_shell(mob_graph, m_mob_set, restrict_to=sel_mob_set)
        if not l_ref or not l_mob:
            continue
        # Kabsch needs ≥3 anchors total = |paired metals| + |shared shell|.
        if len(m_ref_ids) + min(len(l_ref), len(l_mob)) < 3:
            continue
        result = _align_anchored(
            ref_graph,
            mob_graph,
            [sorted(m_ref_set), l_ref],
            [sorted(m_mob_set), l_mob],
            pivot_token_idx=0,
        )
        _, _, _, mismatches, rmsd = result
        score = (round(rmsd, 3), mismatches)
        if score < best_score:
            best_score = score
            best_result = result

    if best_result is None:
        msg = "metal-fragment: no viable metal pairing"
        raise ValueError(msg)
    return best_result
