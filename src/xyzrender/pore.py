"""3D pore detection via coarse-grained net topology.

Coarse-grains the molecular graph (prune leaves, contract chains,
merge metal clusters), builds the cluster-net topology, and finds
shortest cycles — each cycle is a pore window.

Usage::

    from xyzrender.pore import find_pores

    pores = find_pores(graph, lattice=lattice)
    # pores = [(centroid, radius, vertex_positions), ...]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    import collections.abc

logger = logging.getLogger(__name__)

PoreData = tuple[tuple[float, float, float], float, list[tuple[float, float, float]]]
"""(centroid_xyz, radius, vertex_positions) for one pore."""


# ---------------------------------------------------------------------------
# Graph coarse-graining
# ---------------------------------------------------------------------------


@dataclass
class _CoarseResult:
    """Result of graph coarse-graining."""

    graph: nx.Graph
    coarse_pos: dict[int, np.ndarray]
    edge_atoms: dict[tuple[int, int], list[int]]
    cluster_atoms: dict[int, list[int]]


def _coarse_grain(
    graph: nx.Graph,
    positions: dict[int, np.ndarray],
    cluster_radius: float = 5.5,
) -> _CoarseResult:
    """Prune leaves → contract degree-2 chains → merge spatial clusters."""
    cg = graph.copy()

    edge_atoms: dict[tuple[int, int], list[int]] = {(min(u, v), max(u, v)): [] for u, v in cg.edges()}

    # Phase 1: iterative leaf pruning + chain contraction.
    changed = True
    while changed:
        changed = False
        for n in [n for n in cg.nodes() if cg.degree(n) <= 1]:
            for nb in list(cg.neighbors(n)):
                edge_atoms.pop((min(n, nb), max(n, nb)), None)
            cg.remove_node(n)
            changed = True
        for n in [n for n in cg.nodes() if cg.degree(n) == 2]:
            nbs = list(cg.neighbors(n))
            if len(nbs) != 2:
                continue
            a, b = nbs
            merged = [*edge_atoms.pop((min(a, n), max(a, n)), []), n, *edge_atoms.pop((min(n, b), max(n, b)), [])]
            if not cg.has_edge(a, b):
                cg.add_edge(a, b)
            edge_atoms.setdefault((min(a, b), max(a, b)), []).extend(merged)
            cg.remove_node(n)
            changed = True

    if cg.number_of_nodes() == 0:
        return _CoarseResult(cg, {}, {}, {})

    # Phase 2: merge high-degree (≥4) spatially close nodes within 2 hops.
    node_pos = {n: positions[n] for n in cg.nodes()}
    high_deg = {n for n in cg.nodes() if cg.degree(n) >= 4}
    prox = nx.Graph()
    prox.add_nodes_from(cg.nodes())
    for u in high_deg:
        for v in high_deg:
            if u >= v or np.linalg.norm(node_pos[u] - node_pos[v]) >= cluster_radius:
                continue
            if cg.has_edge(u, v) or set(cg.neighbors(u)) & set(cg.neighbors(v)):
                prox.add_edge(u, v)

    clusters = list(nx.connected_components(prox))
    if len(clusters) == len(list(cg.nodes())):
        return _CoarseResult(cg, dict(node_pos), edge_atoms, {n: [n] for n in cg.nodes()})

    merged_g = nx.Graph()
    coarse_pos: dict[int, np.ndarray] = {}
    node_to_rep: dict[int, int] = {}
    cluster_map: dict[int, list[int]] = {}
    for cl in clusters:
        rep = min(cl)
        merged_g.add_node(rep)
        coarse_pos[rep] = np.mean([node_pos[n] for n in cl], axis=0)
        cluster_map[rep] = list(cl)
        for n in cl:
            node_to_rep[n] = rep

    merged_ea: dict[tuple[int, int], list[int]] = {}
    for u, v in cg.edges():
        ru, rv = node_to_rep[u], node_to_rep[v]
        if ru == rv:
            continue
        if not merged_g.has_edge(ru, rv):
            merged_g.add_edge(ru, rv)
        key_new = (min(ru, rv), max(ru, rv))
        merged_ea.setdefault(key_new, []).extend(edge_atoms.get((min(u, v), max(u, v)), []))

    return _CoarseResult(merged_g, coarse_pos, merged_ea, cluster_map)


# ---------------------------------------------------------------------------
# Pore detection helpers
# ---------------------------------------------------------------------------


def _build_metal_net(
    cg: nx.Graph,
    cr: _CoarseResult,
    graph: nx.Graph,
) -> nx.Graph | None:
    """Build metal-cluster-only net from coarse graph. Returns None if no metals."""
    from xyzgraph import DATA

    metals = frozenset(DATA.metals)
    if not any(graph.nodes[n].get("symbol", "C") in metals for n in cg.nodes()):
        return None

    cluster_nodes = {
        rep for rep, m in cr.cluster_atoms.items() if len(m) > 1 or graph.nodes[rep].get("symbol", "C") in metals
    }
    linker_set = set(cg.nodes()) - cluster_nodes
    net = nx.Graph()
    net.add_nodes_from(cluster_nodes)
    nodes = list(cluster_nodes)
    for i, r1 in enumerate(nodes):
        for r2 in nodes[i + 1 :]:
            try:
                path = nx.shortest_path(cg, r1, r2)
                if set(path[1:-1]) <= linker_set:
                    net.add_edge(r1, r2)
            except nx.NetworkXNoPath:
                pass
    logger.info("Metal cluster net: %d nodes, %d edges", net.number_of_nodes(), net.number_of_edges())
    return net


def _tile_pbc(
    cycle_graph: nx.Graph,
    cpos_fn: collections.abc.Callable,
    lattice: np.ndarray,
) -> tuple[nx.Graph, list[np.ndarray], list[int]]:
    """3x3x3 PBC tiling of cycle graph. Returns (tiled_graph, positions, source_ids)."""
    lat = np.array(lattice, dtype=float)
    nodes = list(cycle_graph.nodes())
    tiled_pos: list[np.ndarray] = []
    tiled_src: list[int] = []
    for n in nodes:
        c = cpos_fn(n)
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    tiled_pos.append(c + i * lat[0] + j * lat[1] + k * lat[2])
                    tiled_src.append(n)

    max_edge = max((float(np.linalg.norm(cpos_fn(u) - cpos_fn(v))) for u, v in cycle_graph.edges()), default=15.0) + 1.0
    # Build edge set for source-node connectivity lookup.
    src_edges = {(min(u, v), max(u, v)) for u, v in cycle_graph.edges()}
    tg = nx.Graph()
    tg.add_nodes_from(range(len(tiled_pos)))
    # Vectorised pairwise distances for candidate pairs.
    pos_arr = np.array(tiled_pos)
    for i in range(len(tiled_pos)):
        si = tiled_src[i]
        # Only check j > i where source nodes are bonded or identical.
        candidates = [
            j
            for j in range(i + 1, len(tiled_pos))
            if tiled_src[j] == si or (min(si, tiled_src[j]), max(si, tiled_src[j])) in src_edges
        ]
        if not candidates:
            continue
        dists = np.linalg.norm(pos_arr[candidates] - pos_arr[i], axis=1)
        for j, d in zip(candidates, dists, strict=True):
            if 0.5 < d < max_edge:
                tg.add_edge(i, j)
    logger.info("PBC-tiled net: %d nodes, %d edges", tg.number_of_nodes(), tg.number_of_edges())
    return tg, tiled_pos, tiled_src


def _detect_cycles(graph: nx.Graph, max_size: int) -> set[frozenset[int]]:
    """Per-edge shortest-cycle detection."""
    cycles: set[frozenset[int]] = set()
    for u, v in list(graph.edges()):
        graph.remove_edge(u, v)
        try:
            path = nx.shortest_path(graph, u, v)
            if len(path) <= max_size:
                cycles.add(frozenset(path))
        except nx.NetworkXNoPath:
            pass
        graph.add_edge(u, v)
    return cycles


def _cycles_to_pore_data(
    cycles: set[frozenset[int]],
    cpos_fn: collections.abc.Callable,
    tiled_pos: list[np.ndarray] | None,
    tiled_src: list[int] | None,
    lattice: np.ndarray | None,
    min_size: int,
    max_size: int,
    min_radius: float,
) -> list[PoreData]:
    """Convert raw cycles → filtered, deduplicated PoreData list."""
    frac_inv = np.linalg.inv(np.array(lattice, dtype=float)) if lattice is not None else None

    # Convert cycles to position lists, mapping tiled nodes back to source.
    # Collect ALL cycles first (no radius filter yet — small faces merge into cages).
    pos_rings: list[list[tuple[float, float, float]]] = []
    for cc in cycles:
        if tiled_pos is not None and tiled_src is not None:
            seen_src: set[int] = set()
            pts = []
            for n in cc:
                src = tiled_src[n]
                if src not in seen_src:
                    seen_src.add(src)
                    pts.append(cpos_fn(src))
        else:
            pts = [cpos_fn(n) for n in cc]
        if not (min_size <= len(pts) <= max_size):
            continue
        ring = [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
        if frac_inv is not None:
            frac = frac_inv @ np.mean(ring, axis=0)
            if not all(-0.1 <= f < 1.1 for f in frac):
                continue
        pos_rings.append(ring)

    if not pos_rings:
        return []

    # Merge rings that share vertices — cycles sharing nodes are faces
    # of the same cavity (e.g. 5/6-rings in a buckyball → one pore).
    # Use inverted index: vertex → ring indices.  O(total_vertices).
    vert_to_rings: dict[tuple[float, float, float], list[int]] = {}
    for i, ring in enumerate(pos_rings):
        for v in ring:
            key = (round(v[0], 1), round(v[1], 1), round(v[2], 1))
            vert_to_rings.setdefault(key, []).append(i)
    merge_g = nx.Graph()
    merge_g.add_nodes_from(range(len(pos_rings)))
    for ring_ids in vert_to_rings.values():
        for k in range(1, len(ring_ids)):
            merge_g.add_edge(ring_ids[0], ring_ids[k])

    pore_data: list[PoreData] = []
    for comp in nx.connected_components(merge_g):
        # Collect all unique vertices from merged rings.
        all_v: set[tuple[float, float, float]] = set()
        for idx in comp:
            all_v.update(pos_rings[idx])
        verts = list(all_v)
        pts = np.array(verts)
        centroid = pts.mean(axis=0)
        avg_r = float(np.linalg.norm(pts - centroid, axis=1).mean())
        if avg_r < min_radius:
            continue
        radius = float(np.linalg.norm(pts - centroid, axis=1).min()) * 0.7
        pore_data.append(((float(centroid[0]), float(centroid[1]), float(centroid[2])), radius, verts))

    # Deduplicate by centroid proximity.
    if len(pore_data) > 1:
        centroids = np.array([np.array(p[0]) for p in pore_data])
        used = np.zeros(len(pore_data), dtype=bool)
        unique: list[PoreData] = []
        for i in range(len(pore_data)):
            if used[i]:
                continue
            used[np.linalg.norm(centroids - centroids[i], axis=1) < 1.5] = True
            unique.append(pore_data[i])
        pore_data = unique

    return pore_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_pores(
    graph: nx.Graph,
    *,
    max_size: int = 100,
    min_size: int = 0,
    lattice: np.ndarray | None = None,
    min_pore_radius: float = 3.0,
) -> list[PoreData]:
    """Detect 3D pores via coarse-grained net topology.

    Parameters
    ----------
    lattice :
        Optional 3x3 lattice matrix (rows = a, b, c in Å).  When
        provided, cluster centroids are tiled 3x3x3 to detect pores
        that cross periodic boundaries.
    min_pore_radius :
        Minimum average vertex-to-centroid distance (Å) for a cycle
        to qualify as a pore.

    Returns list of ``(centroid, radius, vertex_positions)`` tuples.
    Does not mutate the graph.
    """
    from xyzrender.face import _is_effectively_2d

    heavy = [n for n in graph.nodes() if graph.nodes[n].get("symbol", "C") != "H"]
    if len(heavy) < 3:
        return []
    sub = graph.subgraph(heavy)
    if sub.number_of_edges() == 0:
        return []

    positions_arr = np.array([graph.nodes[n]["position"] for n in sorted(sub.nodes())])
    if _is_effectively_2d(positions_arr):
        logger.info("2D structure — no 3D pores (use --hull faces for ring detection)")
        return []

    logger.info("3D structure detected, using coarse-grained pore detection")

    # Coarse-grain.
    positions = {n: np.array(graph.nodes[n]["position"]) for n in heavy}
    cr = _coarse_grain(sub.copy(), positions)
    cg = cr.graph
    if cg.number_of_edges() == 0:
        return []

    if cr.graph.number_of_nodes() < sub.number_of_nodes():
        logger.info(
            "Coarse-grained %d→%d nodes, %d→%d edges",
            sub.number_of_nodes(),
            cg.number_of_nodes(),
            sub.number_of_edges(),
            cg.number_of_edges(),
        )

    def _cpos(n: int) -> np.ndarray:
        return cr.coarse_pos.get(n, positions[n])

    # Choose cycle graph: metal-cluster net if metals present, else full coarse.
    metal_net = _build_metal_net(cg, cr, graph)
    cycle_graph = metal_net if metal_net is not None else cg

    # PBC tiling.
    if lattice is not None:
        ring_graph, tiled_pos, tiled_src = _tile_pbc(cycle_graph, _cpos, lattice)
    else:
        ring_graph, tiled_pos, tiled_src = cycle_graph, None, None

    # Detect cycles + convert to pore data.
    cycles = _detect_cycles(ring_graph, max_size)
    pore_data = _cycles_to_pore_data(
        cycles,
        _cpos,
        tiled_pos,
        tiled_src,
        lattice,
        min_size,
        max_size,
        min_pore_radius,
    )

    if not pore_data:
        logger.warning("No pores found (min_size=%d, max_size=%d)", min_size, max_size)
    else:
        logger.info("Detected %d pore(s)", len(pore_data))

    return pore_data
