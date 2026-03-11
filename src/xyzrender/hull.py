"""Convex hull facet computation and SVG rendering for molecular visualization.

When using :attr:`~xyzrender.types.RenderConfig.hull_atom_indices`, you can pass either:

- A single subset: ``[0, 1, 2, 3, 4, 5]`` (0-based atom indices for one hull, e.g. ring carbons).
- Multiple subsets: ``[[0, 1, 2], [3, 4, 5]]`` — each inner list defines a separate hull,
  drawn with :attr:`~xyzrender.types.RenderConfig.hull_color` and
  :attr:`~xyzrender.types.RenderConfig.hull_opacity` by default. For per-subset hue and opacity,
  set :attr:`~xyzrender.types.RenderConfig.hull_colors` and/or
  :attr:`~xyzrender.types.RenderConfig.hull_opacities` (lists, one entry per subset).
  Facets from all subsets are depth-sorted together for correct occlusion.

Hull edges (the 1-skeleton of the convex hull) that do not coincide with a molecular
bond can be drawn as thin lines (e.g. gray) via :attr:`~xyzrender.types.RenderConfig.show_hull_edges`
and :attr:`~xyzrender.types.RenderConfig.hull_edge_color` for better 3D perception.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


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
        hull = ConvexHull(points)
    except Exception:
        # Coplanar or degenerate points (e.g. ring atoms); try QJ to joggle into 3D
        try:
            hull = ConvexHull(points, qhull_options="QJ")
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
        hull = ConvexHull(points)
    except Exception:
        try:
            hull = ConvexHull(points, qhull_options="QJ")
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


def get_convex_hull_edges_visible(
    pos_3d: np.ndarray,
    include_mask: np.ndarray | None = None,
    view_dir: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> list[tuple[int, int]]:
    """Return only hull edges on the visible (front-facing) side of the hull.

    Edges that belong to at least one facet facing the viewer are returned;
    edges entirely on the back of the hull are omitted so only outermost
    (silhouette) edges are drawn, never internal-looking lines.

    Parameters
    ----------
    pos_3d :
        Shape (N, 3) array of 3D positions.
    include_mask :
        Optional boolean array of length N; same as :func:`get_convex_hull_edges`.
    view_dir :
        View direction vector (default +z). A facet is front-facing if its
        outward normal has positive dot product with view_dir.

    Returns
    -------
    list of (node_i, node_j)
        Subset of hull edges with node_i < node_j, in graph index space.
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
        hull = ConvexHull(points)
    except Exception:
        try:
            hull = ConvexHull(points, qhull_options="QJ")
        except Exception:
            return []

    v = np.asarray(view_dir, dtype=float)
    v = v / (np.linalg.norm(v) + 1e-12)
    visible_edges: set[tuple[int, int]] = set()
    for simplex in hull.simplices:
        i, j, k = simplex[0], simplex[1], simplex[2]
        pi, pj, pk = points[i], points[j], points[k]
        normal = np.cross(pj - pi, pk - pi)
        if np.dot(normal, v) > 0:
            for a, b in ((i, j), (j, k), (k, i)):
                lo, hi = (a, b) if a <= b else (b, a)
                visible_edges.add((lo, hi))
    out: list[tuple[int, int]] = []
    for a, b in visible_edges:
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
        hull_2d = ConvexHull(points_2d)
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
    per_facet_opacity: list[float] | None = None,
) -> list[str]:
    """Produce SVG polygon elements for hull facets.

    Parameters
    ----------
    facets :
        List of (face_vertices_3d, centroid_z) from get_convex_hull_facets.
    color_hex :
        Default fill color as CSS hex (e.g. '#4682b4').
    opacity :
        Default fill opacity in [0, 1].
    scale, cx, cy, canvas_w, canvas_h :
        Same convention as renderer _proj: x_svg = canvas_w/2 + scale*(x - cx),
        y_svg = canvas_h/2 - scale*(y - cy).
    per_facet_color_hex :
        Optional list of hex colors, one per facet (after sort). Overrides color_hex when given.
    per_facet_opacity :
        Optional list of opacities, one per facet (after sort). Overrides opacity when given.

    Returns
    -------
    list of str
        SVG fragment strings (one <polygon> per facet), back-to-front order.
    """
    sorted_facets = sorted(facets, key=lambda item: item[1])  # ascending centroid_z
    n_facets = len(sorted_facets)
    use_per_color = per_facet_color_hex is not None and len(per_facet_color_hex) >= n_facets
    use_per_opacity = per_facet_opacity is not None and len(per_facet_opacity) >= n_facets
    colors = (per_facet_color_hex or []) if use_per_color else []
    opacities = (per_facet_opacity or []) if use_per_opacity else []
    svg: list[str] = []
    for k, (face_vertices_3d, _) in enumerate(sorted_facets):
        c = colors[k] if use_per_color and k < len(colors) else color_hex
        o = opacities[k] if use_per_opacity and k < len(opacities) else opacity
        xs = canvas_w / 2 + scale * (face_vertices_3d[:, 0] - cx)
        ys = canvas_h / 2 - scale * (face_vertices_3d[:, 1] - cy)
        points_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
        svg.append(f'  <polygon points="{points_str}" fill="{c}" fill-opacity="{o:.2f}" stroke="none"/>')
    return svg
