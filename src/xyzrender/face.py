"""2D structural face detection via geometric planar traversal.

Detects inner faces of planar molecular graphs (graphene, COFs) and
roughly-planar rings in 3D structures (MOFs, zeolites, buckyballs) by
projecting atom positions onto multiple viewing directions and running
half-edge face enumeration on the CCW-sorted adjacency.

Usage::

    from xyzrender.face import find_2d_faces

    faces = find_2d_faces(graph)
    # faces = [[0, 1, 5, 4], [1, 2, 6, 5], ...]  (node-ID lists)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import collections.abc

    import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------


def _build_ccw_adjacency(
    graph: nx.Graph,
    positions: np.ndarray,
    node_list: list[int],
    node_to_idx: dict[int, int],
) -> tuple[list[list[int]], list[dict[int, int]]]:
    """Build CCW-sorted adjacency lists + reverse-lookup dicts.

    Returns ``(adj, adj_rev)`` where ``adj[i]`` is the CCW-sorted
    neighbour list and ``adj_rev[i][nb] = index_in_adj[i]`` for O(1)
    lookup during face enumeration.
    """
    n = len(node_list)
    adj: list[list[int]] = [[] for _ in range(n)]
    adj_rev: list[dict[int, int]] = [{} for _ in range(n)]

    for node in node_list:
        i = node_to_idx[node]
        px, py = positions[i, 0], positions[i, 1]
        neighbours = list(graph.neighbors(node))
        if not neighbours:
            continue
        angle_nb = sorted(
            (math.atan2(positions[node_to_idx[nb], 1] - py, positions[node_to_idx[nb], 0] - px), node_to_idx[nb])
            for nb in neighbours
        )
        nbs = [j for _, j in angle_nb]
        adj[i] = nbs
        adj_rev[i] = {nb: idx for idx, nb in enumerate(nbs)}

    return adj, adj_rev


def _enumerate_faces(
    adj: list[list[int]],
    adj_rev: list[dict[int, int]],
    n: int,
) -> list[list[int]]:
    """Enumerate all faces via half-edge traversal on CCW adjacency."""
    seen: set[tuple[int, int]] = set()
    faces: list[list[int]] = []
    for u in range(n):
        for v in adj[u]:
            if (u, v) in seen:
                continue
            face: list[int] = []
            cu, cv = u, v
            while True:
                seen.add((cu, cv))
                face.append(cu)
                nbs = adj[cv]
                idx = adj_rev[cv][cu]
                cu, cv = cv, nbs[(idx + 1) % len(nbs)]
                if cu == u and cv == v:
                    break
            faces.append(face)
    return faces


def _signed_area(positions: np.ndarray, face: list[int]) -> float:
    """Signed area via shoelace formula (positive = CCW)."""
    c = positions[face]
    x, y = c[:, 0], c[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _is_effectively_2d(positions: np.ndarray, tol: float = 0.1) -> bool:
    """Check coplanarity via SVD ratio (< *tol* → flat)."""
    centred = positions - positions.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    if sv[0] < 1e-10:
        return True
    return float(sv[2] / sv[0]) < tol


# 6 icosahedron-vertex normals — uniform sphere coverage.
_phi = (1 + math.sqrt(5)) / 2
_ICO_NORMALS = np.array(
    [
        [0, 1, _phi],
        [0, 1, -_phi],
        [1, _phi, 0],
        [1, -_phi, 0],
        [_phi, 0, 1],
        [_phi, 0, -1],
    ]
)
_ICO_NORMALS = _ICO_NORMALS / np.linalg.norm(_ICO_NORMALS, axis=1, keepdims=True)


def _rotation_to_align_z(normal: np.ndarray) -> np.ndarray:
    """3x3 rotation mapping *normal* onto the z-axis (Rodrigues)."""
    n = normal / np.linalg.norm(normal)
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(n, z):
        return np.eye(3)
    if np.allclose(n, -z):
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z, n)
    s = np.linalg.norm(v)
    c = float(np.dot(z, n))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _run_multi_projection(
    sub: nx.Graph,
    positions: np.ndarray,
    node_list: list[int],
    node_to_idx: dict[int, int],
    collector: collections.abc.Callable[[np.ndarray], None],
) -> None:
    """Run face traversal from 6 icosahedron-vertex projections."""
    for normal in _ICO_NORMALS:
        rot = _rotation_to_align_z(normal)
        collector(positions @ rot.T)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_2d_faces(
    graph: nx.Graph,
    *,
    max_size: int = 100,
    min_size: int = 3,
    cell_data: object | None = None,
    face_planarity: float = 0.25,
) -> list[list[int]]:
    """Detect structural faces via geometric planar face traversal.

    For 2D structures a single xy projection suffices.  For 3D, the
    traversal runs from 6 uniformly spaced viewing directions and
    unions results.  Non-planar artefact faces are rejected.

    Parameters
    ----------
    cell_data :
        Optional ``CellData`` with lattice.  Ghost atoms already in
        the graph are used to close cycles crossing cell boundaries.
    face_planarity :
        SVD planarity tolerance for 3D face filtering (0 = strict,
        1 = permissive).
    """
    work_graph = graph
    cell_atom_set: set[int] | None = None
    if cell_data is not None:
        cell_atom_set = {n for n in graph.nodes() if not graph.nodes[n].get("image", False)}

    heavy = [n for n in work_graph.nodes() if work_graph.nodes[n].get("symbol", "C") != "H"]
    if len(heavy) < 3:
        return []
    sub = work_graph.subgraph(heavy)
    if sub.number_of_edges() == 0:
        return []

    node_list = sorted(sub.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    positions = np.array([work_graph.nodes[n]["position"] for n in node_list])
    is_2d = _is_effectively_2d(positions)

    seen: set[frozenset[int]] = set()
    result: list[list[int]] = []

    def _remap(ids: list[int]) -> list[int] | None:
        if cell_atom_set is None or all(n in cell_atom_set for n in ids):
            return ids
        out = []
        for n in ids:
            if n in cell_atom_set:
                out.append(n)
            else:
                src = work_graph.nodes[n].get("source")
                if src is None:
                    return None
                out.append(src)
        return out

    def _planar(ids: list[int]) -> bool:
        pts = np.array([work_graph.nodes[n]["position"] for n in ids])
        c = pts - pts.mean(axis=0)
        sv = np.linalg.svd(c, compute_uv=False)
        return sv[0] < 1e-10 or float(sv[2] / sv[0]) < face_planarity

    def _collect(proj: np.ndarray) -> None:
        adj, adj_rev = _build_ccw_adjacency(sub, proj, node_list, node_to_idx)
        for face in _enumerate_faces(adj, adj_rev, len(node_list)):
            if _signed_area(proj, face) >= 0 or not (min_size <= len(face) <= max_size):
                continue
            ids = _remap([node_list[i] for i in face])
            if ids is None:
                continue
            key = frozenset(ids)
            if key in seen:
                continue
            if not is_2d and not _planar(ids):
                continue
            seen.add(key)
            result.append(ids)

    if is_2d:
        _collect(positions)
    else:
        _run_multi_projection(sub, positions, node_list, node_to_idx, _collect)

    # Auto-filter oversized artefacts for 3D structures.
    if result and not is_2d:
        sizes = sorted(len(f) for f in result)
        median = sizes[len(sizes) // 2]
        if sizes[0] <= 8 and sizes[-1] > median * 3:
            result = [f for f in result if len(f) <= max(median * 2, 8)]

    if not result:
        logger.warning("hull='faces' found no faces (max_size=%d)", max_size)
    else:
        from collections import Counter

        sc = Counter(len(f) for f in result)
        logger.info("Detected %d face(s): %s", len(result), ", ".join(f"{c}x{s}-ring" for s, c in sorted(sc.items())))

    return result
