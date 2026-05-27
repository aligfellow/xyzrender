"""Convex hull facet computation and SVG rendering for molecular visualization.

In ``render()`` pass ``hull=True`` (all heavy atoms), a flat list of 1-indexed
atom indices (one hull, e.g. ``[1, 2, 3, 4, 5, 6]``), or a list of lists
(multiple hulls, e.g. ``[[1, 2, 3], [4, 5, 6]]``).  Per-subset colours are
set via ``hull_color``.  Facets from all subsets are depth-sorted together
for correct occlusion.

Hull edges (the 1-skeleton of the convex hull) that do not coincide with a
molecular bond can be drawn as thin lines; toggle with ``hull_edge=False``.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

    from xyzrender.types import RenderConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-numpy convex hull algorithms
# ---------------------------------------------------------------------------


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Compute 2D convex hull via Andrew's monotone chain.

    Parameters
    ----------
    points :
        Shape (N, 2) array of 2D points.  N >= 2.

    Returns
    -------
    np.ndarray
        1-D array of vertex indices into *points* in counter-clockwise order,
        Ordered boundary vertex indices in counter-clockwise order.
    """
    n = points.shape[0]
    if n <= 1:
        return np.arange(n, dtype=np.intp)

    # Lexicographic sort by (x, y) — np.lexsort sorts by last key first.
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]
    # Andrew's monotone chain is sequential; the inner work dominates and
    # numpy scalar indexing (pts[i]) costs ~1us each.  Working from plain
    # Python lists of floats is ~5x faster on the typical 60-200 point sets.
    xs = pts[:, 0].tolist()
    ys = pts[:, 1].tolist()

    def _half_hull(seq: range) -> list[int]:
        hull: list[int] = []
        for i in seq:
            bx, by = xs[i], ys[i]
            while len(hull) >= 2:
                h1 = hull[-1]
                h0 = hull[-2]
                ox, oy = xs[h0], ys[h0]
                # Cross product (a-o) x (b-o); pop if non-left turn.
                if (xs[h1] - ox) * (by - oy) - (ys[h1] - oy) * (bx - ox) <= 0:
                    hull.pop()
                else:
                    break
            hull.append(i)
        return hull

    lower = _half_hull(range(n))
    upper = _half_hull(range(n - 1, -1, -1))

    # Concatenate, removing duplicate join points.
    hull_local = lower[:-1] + upper[:-1]
    return order[hull_local]


def _convex_hull_3d(points: np.ndarray) -> np.ndarray:
    """Return (F, 3) simplices for the 3D convex hull of *points* (N, 3)."""
    n = points.shape[0]
    _empty = np.empty((0, 3), dtype=np.intp)

    # --- Special case: exactly 3 points → single triangle -----------------
    if n == 3:
        return np.array([[0, 1, 2]], dtype=np.intp)

    # --- Dimensionality check: coplanar shortcut --------------------------
    centered = points - points.mean(axis=0)
    rank = np.linalg.matrix_rank(centered, tol=0.1)
    if rank < 3:
        return _coplanar_hull(points)

    # --- Full 3D incremental hull -----------------------------------------
    return _incremental_3d(points, _empty)


def _coplanar_hull(points: np.ndarray) -> np.ndarray:
    """Triangulate coplanar points: project to best-fit plane, 2D hull, fan."""
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    # Project onto the two principal axes (rows 0 and 1 of vt).
    proj = centered @ vt[:2].T  # (N, 2)
    hull_idx = _convex_hull_2d(proj)
    if len(hull_idx) < 3:
        return np.empty((0, 3), dtype=np.intp)
    # Fan triangulation from hull_idx[0].
    fan = np.column_stack(
        [
            np.full(len(hull_idx) - 2, hull_idx[0], dtype=np.intp),
            hull_idx[1:-1],
            hull_idx[2:],
        ]
    )
    return fan


def _incremental_3d(points: np.ndarray, _empty: np.ndarray) -> np.ndarray:
    """Core incremental 3D convex hull algorithm."""
    n = points.shape[0]

    # --- Initial tetrahedron: 4 well-separated non-coplanar points --------
    p0 = 0
    dists = np.linalg.norm(points - points[p0], axis=1)
    p1 = int(np.argmax(dists))
    if dists[p1] < 1e-12:
        return _empty

    edge_dir = points[p1] - points[p0]
    edge_dir /= np.linalg.norm(edge_dir)
    vecs = points - points[p0]
    perp = vecs - np.outer(vecs @ edge_dir, edge_dir)
    p2 = int(np.argmax(np.linalg.norm(perp, axis=1)))
    if np.linalg.norm(perp[p2]) < 1e-12:
        return _empty

    normal = np.cross(points[p1] - points[p0], points[p2] - points[p0])
    normal /= np.linalg.norm(normal)
    p3 = int(np.argmax(np.abs(vecs @ normal)))
    if abs(vecs[p3] @ normal) < 1e-12:
        return _empty

    tet = [p0, p1, p2, p3]
    centroid = points[tet].mean(axis=0)

    # 4 faces, oriented so normals point away from centroid.
    faces: list[list[int]] = [
        [tet[0], tet[1], tet[2]],
        [tet[0], tet[1], tet[3]],
        [tet[0], tet[2], tet[3]],
        [tet[1], tet[2], tet[3]],
    ]
    for f in faces:
        fn = np.cross(points[f[1]] - points[f[0]], points[f[2]] - points[f[0]])
        if fn @ (points[f[0]] - centroid) < 0:
            f[1], f[2] = f[2], f[1]

    live = set(range(4))
    in_hull = set(tet)

    # --- Add remaining points one at a time -------------------------------
    for pi in range(n):
        if pi in in_hull:
            continue
        pt = points[pi]

        # Vectorized visibility test.
        live_list = sorted(live)
        face_arr = np.array([faces[fi] for fi in live_list], dtype=np.intp)
        v0 = points[face_arr[:, 0]]
        normals = np.cross(points[face_arr[:, 1]] - v0, points[face_arr[:, 2]] - v0)
        dots = np.einsum("ij,ij->i", normals, pt - v0)
        visible = {live_list[i] for i in np.flatnonzero(dots > 1e-15)}
        if not visible:
            continue

        # Horizon = edges of visible faces that appear exactly once.
        edge_counts: Counter[tuple[int, int]] = Counter()
        for fi in visible:
            f = faces[fi]
            for k in range(3):
                a, b = f[k], f[(k + 1) % 3]
                edge_counts[(min(a, b), max(a, b))] += 1
        horizon = [e for e, c in edge_counts.items() if c == 1]

        # Remove visible faces, add new faces from pi to horizon edges.
        live -= visible
        for ea, eb in horizon:
            new_face = [pi, ea, eb]
            fn = np.cross(points[ea] - points[pi], points[eb] - points[pi])
            if fn @ (points[pi] - centroid) < 0:
                new_face = [pi, eb, ea]
            live.add(len(faces))
            faces.append(new_face)
        in_hull.add(pi)

    if not live:
        return _empty
    return np.array([faces[fi] for fi in sorted(live)], dtype=np.intp)


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
        Empty list if fewer than 3 points.
    """
    if include_mask is not None:
        points = pos_3d[np.asarray(include_mask, dtype=bool)]
    else:
        points = np.asarray(pos_3d, dtype=float)

    if points.shape[0] < 3:
        return []

    simplices = _convex_hull_3d(points)
    if simplices.shape[0] == 0:
        return []

    out: list[tuple[np.ndarray, float]] = []
    for simplex in simplices:
        face = points[simplex]  # (3, 3)
        centroid_z = float(face[:, 2].mean())
        out.append((face, centroid_z))
    return out


def get_ring_facets(
    pos_3d: np.ndarray,
    ring_indices: list[int],
) -> list[tuple[np.ndarray, float]]:
    """Single polygon facet for an ordered ring.

    Returns the ring vertices as one facet (not a triangle fan), avoiding
    opacity buildup at the centroid.  The z-depth is the mean z of all
    ring vertices.
    """
    pts = pos_3d[ring_indices]
    if len(pts) < 3:
        return []
    z = float(pts[:, 2].mean())
    return [(pts, z)]


def get_ring_edges(
    ring_indices: list[int],
) -> list[tuple[int, int]]:
    """Ring polygon edges as sorted (i, j) pairs."""
    out: list[tuple[int, int]] = []
    k = len(ring_indices)
    for i in range(k):
        a, b = ring_indices[i], ring_indices[(i + 1) % k]
        out.append((min(a, b), max(a, b)))
    return out


def get_silhouette_polygon(
    pos_3d: np.ndarray,
    include_mask: np.ndarray | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Return the 2D convex hull silhouette as triangle-fan facets.

    Projects to xy, computes the 2D convex hull boundary, then returns
    a triangle fan from the centroid — giving a single smooth filled
    polygon instead of many overlapping 3D triangles.
    """
    if include_mask is not None:
        points = pos_3d[np.asarray(include_mask, dtype=bool)]
    else:
        points = np.asarray(pos_3d, dtype=float)

    if points.shape[0] < 3:
        return []

    hull_idx = _convex_hull_2d(points[:, :2])
    if len(hull_idx) < 3:
        return []

    hull_pts = points[hull_idx]
    z = float(hull_pts[:, 2].mean())
    return [(hull_pts, z)]


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
    verts = _convex_hull_2d(points_2d)
    if len(verts) < 2:
        return []

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
# Ring colouring (fingerprint-based palette assignment)
# ---------------------------------------------------------------------------


def ring_fingerprint(
    ring: list[int],
    graph: nx.Graph | None = None,
    *,
    mode: str = "type",
    shared_atoms: set[int] | None = None,
) -> tuple[int, tuple]:
    """Ring fingerprint for colour assignment.

    *mode* controls what distinguishes rings:

    - ``"size"``: ring size only.
    - ``"type"``: ring size + sorted (element, degree) per atom.
    - ``"env"``: like ``"type"`` but also marks atoms shared with other
      rings (fused vs terminal), distinguishing e.g. annulated benzene
      from isolated phenyl.
    """
    size = len(ring)
    if graph is None or mode == "size":
        return (size, ())
    if mode == "env" and shared_atoms is not None:
        sig = sorted((graph.nodes[n].get("symbol", "C"), graph.degree(n), n in shared_atoms) for n in ring)
    else:
        sig = sorted((graph.nodes[n].get("symbol", "C"), graph.degree(n)) for n in ring)
    return (size, tuple(sig))


def _ring_colors(
    subsets: list[list[int]],
    graph: nx.Graph | None = None,
    palette: list[str] | None = None,
    *,
    mode: str = "type",
) -> list[str]:
    """Map ring subsets to hex colours via fingerprint."""
    pal = palette or ["steelblue", "firebrick", "mediumseagreen", "mediumpurple", "darkgoldenrod", "cadetblue"]
    # For "env" mode: precompute atoms shared between rings.
    shared: set[int] | None = None
    if mode == "env":
        from collections import Counter

        atom_counts: Counter[int] = Counter()
        for s in subsets:
            atom_counts.update(s)
        shared = {a for a, c in atom_counts.items() if c > 1}
    fps = [ring_fingerprint(s, graph, mode=mode, shared_atoms=shared) for s in subsets]
    unique_fps = sorted(set(fps))
    fp_map = {fp: pal[i % len(pal)] for i, fp in enumerate(unique_fps)}
    return [fp_map[fp] for fp in fps]


# Keep old name as public alias for external callers.
pore_size_colors = _ring_colors


# ---------------------------------------------------------------------------
# Hull resolution helpers
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


def resolve_hull_pores(
    graph: nx.Graph,
    cfg: RenderConfig,
    *,
    max_size: int = 100,
    min_size: int = 0,
    cell_data: object | None = None,
) -> list[list[int]]:
    """Detect 3D pores and return real atom node IDs for hull drawing.

    Maps each pore vertex position back to the nearest real graph node.
    Stores pore geometry on *cfg* for sphere rendering.
    """
    from xyzrender.pore import find_pores

    _lat_arr = getattr(cell_data, "lattice", None) if cell_data else None
    pore_data = find_pores(graph, max_size=max_size, min_size=min_size, lattice=_lat_arr)
    if not pore_data:
        return []

    # Map pore vertex positions to nearest real graph nodes (for hull drawing).
    # Store true centroids + radii separately (for accurate sphere placement).
    import numpy as np

    all_nodes = list(graph.nodes())
    all_pos = np.array([graph.nodes[n]["position"] for n in all_nodes])

    hull_subsets: list[list[int]] = []
    centroids: list[tuple[float, float, float]] = []
    radii: list[float] = []
    for centroid, coarse_radius, verts in pore_data:
        subset: list[int] = []
        seen: set[int] = set()
        for v in verts:
            dists = np.linalg.norm(all_pos - np.array(v), axis=1)
            nearest = all_nodes[int(np.argmin(dists))]
            if nearest not in seen:
                seen.add(nearest)
                subset.append(nearest)
        hull_subsets.append(subset)
        centroids.append(centroid)
        # Use coarse-grained radius (0.7 x min vertex-centroid distance).
        # The mapped atoms are cluster representatives, not pore-boundary
        # atoms — linkers are edges in the coarse graph, so their atoms
        # aren't in the vertex set.  VdW subtraction on cluster atoms
        # would overestimate the visual pore size.
        radii.append(coarse_radius)

    cfg.pore_node_ids = hull_subsets
    cfg.pore_centroids = centroids
    cfg.pore_radii = radii
    return hull_subsets


def resolve_hull_faces(
    graph: nx.Graph,
    *,
    max_size: int = 100,
    min_size: int = 0,
    cell_data: object | None = None,
    face_planarity: float = 0.25,
) -> list[list[int]]:
    """Return structural face indices via geometric face traversal.

    When *cell_data* is provided, ghost atoms are used to close
    boundary-crossing cycles.  *face_planarity* controls how planar
    a face must be in 3D (0 = strict, 1 = permissive).
    """
    from xyzrender.face import find_2d_faces

    return find_2d_faces(
        graph,
        max_size=max_size,
        min_size=min_size,
        cell_data=cell_data,
        face_planarity=face_planarity,
    )


def resolve_hull_flag_and_indices(
    hull: bool | str | list[int] | list[list[int]] | None,
    graph: nx.Graph | None,
    cfg: RenderConfig | None = None,
    *,
    ring_max_size: int = 100,
    ring_min_size: int = 0,
    face_planarity: float = 0.25,
) -> tuple[bool | None, list[int] | list[list[int]] | None]:
    r"""Resolve hull option to (show_convex_hull, hull_atom_indices) for config."""
    if hull is None:
        return None, None
    if isinstance(hull, str):
        if hull in {"rings", "ring"}:
            if graph is None:
                return None, None
            ring_indices = resolve_hull_rings(graph)
            if not ring_indices:
                return None, None
            return True, ring_indices
        if hull in {"pores", "pore"}:
            if graph is None:
                return None, None
            from xyzrender.types import RenderConfig

            _cfg = cfg if cfg is not None else RenderConfig()
            _cd = getattr(_cfg, "cell_data", None)
            pore_indices = resolve_hull_pores(
                graph,
                _cfg,
                max_size=ring_max_size,
                min_size=ring_min_size,
                cell_data=_cd,
            )
            if not pore_indices:
                return None, None
            return True, pore_indices
        if hull in {"faces", "face"}:
            if graph is None:
                return None, None
            _cd = getattr(cfg, "cell_data", None) if cfg is not None else None
            face_indices = resolve_hull_faces(
                graph,
                max_size=ring_max_size,
                min_size=ring_min_size,
                cell_data=_cd,
                face_planarity=face_planarity,
            )
            if not face_indices:
                return None, None
            return True, face_indices
        return None, None
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
    *,
    ring_max_size: int = 100,
    ring_min_size: int = 0,
    face_planarity: float = 0.25,
    precomputed_indices: list[list[int]] | None = None,
    hull_color_type: str = "type",
) -> None:
    """Apply hull-related options to *cfg*. Single place for hull semantics.

    When *precomputed_indices* is provided (e.g. pre-tiled supercell indices),
    detection is skipped and those indices are used directly.
    """
    if precomputed_indices is not None:
        show_hull = True
        hull_idx = precomputed_indices
    else:
        show_hull, hull_idx = resolve_hull_flag_and_indices(
            hull,
            graph,
            cfg,
            ring_max_size=ring_max_size,
            ring_min_size=ring_min_size,
            face_planarity=face_planarity,
        )
    if show_hull is not None:
        cfg.show_convex_hull = show_hull
    if hull_idx is not None:
        cfg.hull_atom_indices = hull_idx
    # Ordered-ring hull modes — render as actual polygons, not convex hulls.
    _ring_modes = {"pores", "pore", "faces", "face", "rings", "ring"}
    _is_ring_mode = isinstance(hull, str) and hull in _ring_modes
    if _is_ring_mode and hull_idx is not None:
        cfg.hull_ordered = True
    # Auto-set per-size colours via ring fingerprint.
    if _is_ring_mode and hull_idx is not None:
        subsets = normalize_hull_subsets(hull_idx)
        if hull_color is None and subsets:
            cfg.hull_colors = _ring_colors(
                subsets,
                graph,
                palette=cfg.hull_colors,
                mode=hull_color_type,
            )
    if hull_color is not None:
        cfg.hull_colors = [hull_color] if isinstance(hull_color, str) else hull_color
    if hull_opacity is not None:
        cfg.hull_opacity = hull_opacity
    if hull_edge is not None:
        cfg.show_hull_edges = hull_edge
    if hull_edge_width_ratio is not None:
        cfg.hull_edge_width_ratio = hull_edge_width_ratio
