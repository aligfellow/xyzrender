"""Convex hull facet computation and SVG rendering for molecular visualization.

In ``render()`` pass ``hull=True`` (all heavy atoms), a flat list of 1-indexed
atom indices (one hull, e.g. ``[1, 2, 3, 4, 5, 6]``), or a list of lists
(multiple hulls, e.g. ``[[1, 2, 3], [4, 5, 6]]``).  Per-subset colours are
set via ``hull_color``.  Facets from all subsets are depth-sorted together
for correct occlusion.

Hull edges (the 1-skeleton of the convex hull) that do not coincide with a
molecular bond can be drawn as thin lines; toggle with ``hull_edge=False``.

Requires ``scipy``: ``pip install scipy`` or ``pip install 'xyzrender[hull]'``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import networkx as nx
    from scipy.spatial import ConvexHull

    from xyzrender.types import RenderConfig

logger = logging.getLogger(__name__)


def _convex_hull(points: np.ndarray, *, qhull_options: str | None = None) -> ConvexHull:
    """Lazy-import scipy and build a ConvexHull, raising a clear error if missing."""
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        msg = "scipy is required for convex hull rendering — pip install scipy  or  pip install 'xyzrender[hull]'"
        raise ImportError(msg) from None
    if qhull_options is not None:
        return ConvexHull(points, qhull_options=qhull_options)
    return ConvexHull(points)


def hull_indices_to_0indexed(
    hull: list[int] | list[list[int]],
) -> list[int] | list[list[int]]:
    """Convert 1-indexed hull indices to 0-indexed (internal).

    Handles both flat ``[1, 2, 3]`` → ``[0, 1, 2]`` and nested
    ``[[1, 2], [3, 4]]`` → ``[[0, 1], [2, 3]]``.
    """
    if hull and isinstance(hull[0], list):
        subs = cast("list[list[int]]", hull)
        return [[i - 1 for i in sub] for sub in subs]
    flat = cast("list[int]", hull)
    return [i - 1 for i in flat]


def normalize_hull_subsets(
    raw: list[int] | list[list[int]],
) -> list[list[int]]:
    """Normalize hull_atom_indices to a list of subsets.

    A flat ``[0, 1, 2]`` becomes ``[[0, 1, 2]]``; a nested ``[[0, 1], [2, 3]]``
    passes through unchanged. Empty list returns ``[]``.
    """
    if not raw:
        return []
    if isinstance(raw[0], list):
        return cast("list[list[int]]", raw)
    return [cast("list[int]", raw)]


def get_convex_hull_facets(
    pos_3d: np.ndarray,
    include_mask: np.ndarray | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Compute convex hull facets from 3D positions.

    Parameters
    ----------
    pos_3d :
        Shape (N, 3) array of 3D positions (e.g. oriented atom positions).
    include_mask :
        Optional boolean array of length N. If provided, only positions where
        True are used for the hull (e.g. exclude NCI dummy nodes).

    Returns
    -------
    list of (face_vertices_3d, centroid_z)
        Each facet is a triangle: face_vertices_3d has shape (3, 3).
        centroid_z is the z-coordinate of the facet centroid for back-to-front sorting.
        Empty list if fewer than 4 points (no 3D hull).
    """
    if include_mask is not None:
        points = pos_3d[np.asarray(include_mask, dtype=bool)]
    else:
        points = np.asarray(pos_3d, dtype=float)

    if points.shape[0] < 4:
        return []

    try:
        hull = _convex_hull(points)
    except Exception:
        # Coplanar or degenerate points (e.g. ring atoms); try QJ to joggle into 3D
        try:
            hull = _convex_hull(points, qhull_options="QJ")
        except Exception:
            return []

    out: list[tuple[np.ndarray, float]] = []
    for simplex in hull.simplices:
        # simplex: indices (3,) into points
        face = points[simplex]  # (3, 3)
        centroid_z = float(face[:, 2].mean())
        out.append((face, centroid_z))
    return out


def get_convex_hull_edges(
    pos_3d: np.ndarray,
    include_mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Compute convex hull edges (1-skeleton) as graph node index pairs.

    Hull edges are the unique vertex pairs from each facet. Use with the
    molecule's bond set to draw only edges that do not overlap a bond
    (e.g. as thin gray lines for better 3D visualization).

    Parameters
    ----------
    pos_3d :
        Shape (N, 3) array of 3D positions (e.g. oriented atom positions).
    include_mask :
        Optional boolean array of length N. If provided, only positions where
        True are used for the hull; returned indices are into the full pos_3d
        (graph node indices).

    Returns
    -------
    list of (node_i, node_j)
        Unique edges with node_i < node_j, in graph (full) index space.
        Empty list if fewer than 4 points or hull construction fails.
    """
    if include_mask is not None:
        points = pos_3d[np.asarray(include_mask, dtype=bool)]
        graph_indices = np.flatnonzero(include_mask)
    else:
        points = np.asarray(pos_3d, dtype=float)
        graph_indices = np.arange(pos_3d.shape[0], dtype=np.intp)

    if points.shape[0] < 4:
        return []

    try:
        hull = _convex_hull(points)
    except Exception:
        try:
            hull = _convex_hull(points, qhull_options="QJ")
        except Exception:
            return []

    seen: set[frozenset[int]] = set()
    out: list[tuple[int, int]] = []
    for simplex in hull.simplices:
        # simplex: 3 vertex indices into points
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            if a > b:
                a, b = b, a
            key = frozenset((a, b))
            if key in seen:
                continue
            seen.add(key)
            ni, nj = int(graph_indices[a]), int(graph_indices[b])
            if ni > nj:
                ni, nj = nj, ni
            out.append((ni, nj))
    return out


def get_convex_hull_edges_silhouette(
    pos_3d: np.ndarray,
    include_mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Return only hull edges that lie on the 2D silhouette (boundary) of the hull.

    Projects hull vertices to the viewing plane (x, y) and returns only edges
    that are on the boundary of that 2D convex hull. This avoids drawing
    diagonals or other edges that would cross the interior (e.g. inside
    benzene or anthracene rings).

    Parameters
    ----------
    pos_3d :
        Shape (N, 3) array of 3D positions (viewing axis is z).
    include_mask :
        Optional boolean array of length N; same as :func:`get_convex_hull_edges`.

    Returns
    -------
    list of (node_i, node_j)
        Edges on the 2D silhouette with node_i < node_j, in graph index space.
    """
    if include_mask is not None:
        points = pos_3d[np.asarray(include_mask, dtype=bool)]
        graph_indices = np.flatnonzero(include_mask)
    else:
        points = np.asarray(pos_3d, dtype=float)
        graph_indices = np.arange(pos_3d.shape[0], dtype=np.intp)

    if points.shape[0] < 3:
        return []

    points_2d = points[:, :2]
    try:
        hull_2d = _convex_hull(points_2d)
    except Exception:
        return []

    # hull_2d.vertices: indices of boundary points in counterclockwise order
    verts = hull_2d.vertices
    out: list[tuple[int, int]] = []
    for k in range(len(verts)):
        a, b = verts[k], verts[(k + 1) % len(verts)]
        ni, nj = int(graph_indices[a]), int(graph_indices[b])
        if ni > nj:
            ni, nj = nj, ni
        out.append((ni, nj))
    return out


def hull_facets_svg(
    facets: list[tuple[np.ndarray, float]],
    color_hex: str,
    opacity: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
    *,
    per_facet_color_hex: list[str] | None = None,
) -> list[str]:
    """Produce SVG polygon elements for hull facets.

    Parameters
    ----------
    facets :
        List of (face_vertices_3d, centroid_z) from get_convex_hull_facets.
    color_hex :
        Default fill color as CSS hex (e.g. '#4682b4').
    opacity :
        Fill opacity in [0, 1].
    scale, cx, cy, canvas_w, canvas_h :
        Same convention as renderer _proj: x_svg = canvas_w/2 + scale*(x - cx),
        y_svg = canvas_h/2 - scale*(y - cy).
    per_facet_color_hex :
        Optional list of hex colors, one per facet (after sort). Overrides color_hex when given.

    Returns
    -------
    list of str
        SVG fragment strings (one <polygon> per facet), back-to-front order.
    """
    sorted_facets = sorted(facets, key=lambda item: item[1])  # ascending centroid_z
    n_facets = len(sorted_facets)
    use_per_color = per_facet_color_hex is not None and len(per_facet_color_hex) >= n_facets
    colors = (per_facet_color_hex or []) if use_per_color else []
    svg: list[str] = []
    for k, (face_vertices_3d, _) in enumerate(sorted_facets):
        c = colors[k] if use_per_color and k < len(colors) else color_hex
        xs = canvas_w / 2 + scale * (face_vertices_3d[:, 0] - cx)
        ys = canvas_h / 2 - scale * (face_vertices_3d[:, 1] - cy)
        points_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
        svg.append(f'  <polygon points="{points_str}" fill="{c}" fill-opacity="{opacity:.2f}" stroke="none"/>')
    return svg


# ---------------------------------------------------------------------------
# Hull resolution helpers (moved from api.py)
# ---------------------------------------------------------------------------


def resolve_hull_rings(graph: nx.Graph) -> list[list[int]]:
    """Return aromatic ring atom indices from the molecular graph.

    Reads ``graph.graph["aromatic_rings"]`` if present (set by xyzgraph).
    Otherwise runs ``xyzgraph.build_graph`` on demand for Hückel detection
    from 3D geometry — this avoids the cost of a full rebuild at load time
    for molecules that never use ``hull="rings"``.

    Each ring is a list of 0-indexed atom indices.  If no aromatic rings are
    found, logs a warning and returns an empty list.
    """
    rings = graph.graph.get("aromatic_rings", [])
    if not rings and "aromatic_rings" not in graph.graph:
        from xyzgraph import build_graph

        atoms = [(graph.nodes[i]["symbol"], tuple(graph.nodes[i]["position"])) for i in graph.nodes()]
        charge = graph.graph.get("total_charge", 0)
        mult = graph.graph.get("multiplicity")
        tmp = build_graph(atoms, charge=charge, multiplicity=mult)
        rings = tmp.graph.get("aromatic_rings", [])
    if not rings:
        logger.warning("hull='rings' requested but no aromatic rings detected in the molecular graph")
        return []
    return [list(r) for r in rings]


def resolve_hull_flag_and_indices(
    hull: bool | str | list[int] | list[list[int]] | None,
    graph: nx.Graph | None,
) -> tuple[bool | None, list[int] | list[list[int]] | None]:
    r"""Resolve hull option to (show_convex_hull, hull_atom_indices) for config.

    Returns (None, None) when hull is None or when hull="rings" but graph has
    no aromatic rings. Used by both render() and render_gif() to avoid duplicating
    hull resolution logic.
    """
    if hull is None:
        return None, None
    if hull == "rings":
        if graph is None:
            return None, None
        ring_indices = resolve_hull_rings(graph)
        if not ring_indices:
            return None, None
        return True, ring_indices
    if isinstance(hull, list):
        return True, hull_indices_to_0indexed(hull)
    if isinstance(hull, bool):
        return hull, None
    return None, None


def apply_hull_to_config(
    cfg: RenderConfig,
    hull: bool | str | list[int] | list[list[int]] | None,
    hull_color: str | list[str] | None,
    hull_opacity: float | None,
    hull_edge: bool | None,
    hull_edge_width_ratio: float | None,
    graph: nx.Graph | None,
) -> None:
    """Apply hull-related options to *cfg*. Single place for hull semantics."""
    show_hull, hull_idx = resolve_hull_flag_and_indices(hull, graph)
    if show_hull is not None:
        cfg.show_convex_hull = show_hull
    if hull_idx is not None:
        cfg.hull_atom_indices = hull_idx
    if hull_color is not None:
        cfg.hull_colors = [hull_color] if isinstance(hull_color, str) else hull_color
    if hull_opacity is not None:
        cfg.hull_opacity = hull_opacity
    if hull_edge is not None:
        cfg.show_hull_edges = hull_edge
    if hull_edge_width_ratio is not None:
        cfg.hull_edge_width_ratio = hull_edge_width_ratio
