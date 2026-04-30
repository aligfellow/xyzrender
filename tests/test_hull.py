"""Tests for convex hull, face detection, and pore detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xyzrender.hull import (
    _convex_hull_2d,
    _convex_hull_3d,
    get_convex_hull_facets,
    hull_facets_svg,
    normalize_hull_subsets,
)
from xyzrender.renderer import render_svg
from xyzrender.types import RenderConfig


def test_get_convex_hull_facets_tetrahedron():
    """Four points in a tetrahedron yield 4 triangular facets."""
    # Regular tetrahedron
    t = 1.0
    pos = np.array(
        [
            [t, t, t],
            [t, -t, -t],
            [-t, t, -t],
            [-t, -t, t],
        ],
        dtype=float,
    )
    facets = get_convex_hull_facets(pos)
    assert len(facets) == 4
    for face_vertices_3d, centroid_z in facets:
        assert face_vertices_3d.shape == (3, 3)
        assert isinstance(centroid_z, float)


def test_get_convex_hull_facets_fewer_than_three_points():
    """Fewer than 3 points return empty list; exactly 3 returns one triangle."""
    assert get_convex_hull_facets(np.array([[0, 0, 0], [1, 0, 0]])) == []
    facets = get_convex_hull_facets(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    assert len(facets) == 1
    assert facets[0][0].shape == (3, 3)


def test_get_convex_hull_facets_with_include_mask():
    """include_mask restricts which points are used for the hull."""
    # 5 points: 4 in tetrahedron + 1 inside; mask excludes one so we get 4 points -> tetrahedron
    pos = np.array(
        [
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [0, 0, 0],
        ],
        dtype=float,
    )
    # Use only first 4 points
    mask = np.array([True, True, True, True, False])
    facets = get_convex_hull_facets(pos, include_mask=mask)
    assert len(facets) == 4
    # All 5 points -> hull has more simplices (e.g. 4 for this configuration)
    facets_all = get_convex_hull_facets(pos)
    assert len(facets_all) >= 4
    # Mask with only 3 points -> single triangle facet
    mask3 = np.array([True, True, True, False, False])
    facets3 = get_convex_hull_facets(pos, include_mask=mask3)
    assert len(facets3) == 1


def test_hull_facets_svg_produces_polygons():
    """hull_facets_svg returns one polygon per facet with correct attributes."""
    # One triangle
    face = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=float)
    facets = [(face, 0.0)]
    svg = hull_facets_svg(facets, "#4682b4", 0.2, scale=100.0, cx=0.5, cy=0.5, canvas_w=800, canvas_h=600)
    assert len(svg) == 1
    assert "<polygon" in svg[0]
    assert "fill-opacity" in svg[0]
    assert "#4682b4" in svg[0]
    assert 'stroke="none"' in svg[0]


def test_render_svg_with_convex_hull():
    """Rendering with show_convex_hull=True produces SVG containing hull polygons."""
    import networkx as nx

    # Minimal graph: 4 atoms in a tetrahedron
    g = nx.Graph()
    pos = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    )
    for i in range(4):
        g.add_node(i, symbol="C", position=pos[i].tolist())
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j)

    cfg = RenderConfig(show_convex_hull=True, hull_colors=["steelblue"], hull_opacity=0.2)
    svg = render_svg(g, cfg)
    assert "<polygon" in svg
    assert "fill-opacity" in svg


def test_render_svg_with_single_subset_indices():
    """Single subset as flat list of indices (backward compatible) draws one hull."""
    import networkx as nx

    g = nx.Graph()
    pos = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    )
    for i in range(4):
        g.add_node(i, symbol="C", position=pos[i].tolist())
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j)

    cfg = RenderConfig(
        show_convex_hull=True,
        hull_atom_indices=[0, 1, 2, 3],
        hull_colors=["steelblue"],
        hull_opacity=0.2,
    )
    svg = render_svg(g, cfg)
    assert "<polygon" in svg
    assert svg.count("fill-opacity") >= 1  # single silhouette polygon


def test_render_svg_with_multiple_subsets():
    """Multiple subsets as list of index lists draw multiple hulls, depth-sorted."""
    import networkx as nx

    # 8 atoms: two tetrahedra (0,1,2,3) and (4,5,6,7)
    g = nx.Graph()
    t1 = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    t2 = np.array([[5, 5, 5], [5, 3, 3], [3, 5, 3], [3, 3, 5]], dtype=float)
    pos = np.vstack([t1, t2])
    for i in range(8):
        g.add_node(i, symbol="C", position=pos[i].tolist())
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j)
    for i in range(4, 8):
        for j in range(i + 1, 8):
            g.add_edge(i, j)

    cfg = RenderConfig(
        show_convex_hull=True,
        hull_atom_indices=[[0, 1, 2, 3], [4, 5, 6, 7]],
        hull_colors=["#4682b4"],
        hull_opacity=0.3,
    )
    svg = render_svg(g, cfg)
    assert "<polygon" in svg
    # Two subsets → 2 silhouette polygons
    assert svg.count("<polygon") == 2
    assert svg.count("fill-opacity") == 2


def test_render_svg_with_per_subset_colors():
    """Per-subset hull_colors apply correct hue per hull."""
    import networkx as nx

    g = nx.Graph()
    t1 = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    t2 = np.array([[5, 5, 5], [5, 3, 3], [3, 5, 3], [3, 3, 5]], dtype=float)
    pos = np.vstack([t1, t2])
    for i in range(8):
        g.add_node(i, symbol="C", position=pos[i].tolist())
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j)
    for i in range(4, 8):
        for j in range(i + 1, 8):
            g.add_edge(i, j)

    cfg = RenderConfig(
        show_convex_hull=True,
        hull_atom_indices=[[0, 1, 2, 3], [4, 5, 6, 7]],
        hull_colors=["red", "blue"],
        hull_opacity=0.3,
    )
    svg = render_svg(g, cfg)
    assert "<polygon" in svg
    assert "fill-opacity" in svg
    # Single opacity applied to all facets
    assert "0.30" in svg
    # Per-subset colors (resolved to hex): red -> #ff0000, blue -> #0000ff
    assert "#ff0000" in svg
    assert "#0000ff" in svg


def test_render_svg_with_empty_subset_list():
    """Empty list of subsets draws no hull (no crash)."""
    import networkx as nx

    g = nx.Graph()
    for i in range(4):
        g.add_node(i, symbol="C", position=[float(i), 0.0, 0.0])
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])

    cfg = RenderConfig(
        show_convex_hull=True,
        hull_atom_indices=[],
        hull_colors=["steelblue"],
        hull_opacity=0.2,
    )
    svg = render_svg(g, cfg)
    assert "<polygon" not in svg


def test_render_svg_hull_edges_not_drawn_when_all_bonds():
    """Tetrahedron with all 6 bonds: no non-bond hull edges, so no hull-edge lines."""
    import networkx as nx

    g = nx.Graph()
    pos = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    )
    for i in range(4):
        g.add_node(i, symbol="C", position=pos[i].tolist())
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j)
    cfg = RenderConfig(
        show_convex_hull=True,
        show_hull_edges=True,
        hull_atom_indices=[0, 1, 2, 3],
        hull_colors=["steelblue"],
        hull_opacity=0.2,
    )
    svg = render_svg(g, cfg)
    # All 6 hull edges are bonds → no extra hull-edge <line> elements beyond bonds.
    # Bonds use bond_color (black by default); hull edges use fill color (#4682b4).
    # Count <line> elements with the hull fill color as stroke.
    import re

    hull_edge_lines = re.findall(r'<line [^>]*stroke="#4682b4"', svg)
    assert len(hull_edge_lines) == 0


def test_render_svg_hull_edges_drawn_for_non_bond():
    """Tetrahedron with one bond missing: one hull edge is drawn as thin line."""
    import networkx as nx

    g = nx.Graph()
    pos = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    )
    for i in range(4):
        g.add_node(i, symbol="C", position=pos[i].tolist())
    # Only 5 bonds: omit (0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(0, 3)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    cfg = RenderConfig(
        show_convex_hull=True,
        show_hull_edges=True,
        hull_atom_indices=[0, 1, 2, 3],
        hull_colors=["steelblue"],
        hull_opacity=0.2,
    )
    svg = render_svg(g, cfg)
    # Edge color = fill color (#4682b4); at least one non-bond edge drawn
    import re

    hull_edge_lines = re.findall(r'<line [^>]*stroke="#4682b4"', svg)
    assert len(hull_edge_lines) >= 1


def test_normalize_hull_subsets():
    """normalize_hull_subsets converts flat lists and nested lists correctly."""
    assert normalize_hull_subsets([]) == []
    assert normalize_hull_subsets([0, 1, 2]) == [[0, 1, 2]]
    assert normalize_hull_subsets([[0, 1], [2, 3]]) == [[0, 1], [2, 3]]


# ---------------------------------------------------------------------------
# Algorithm-level tests for numpy convex hull internals
# ---------------------------------------------------------------------------


def test_convex_hull_2d_square():
    """4 corners of a square yield 4 boundary vertices."""
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    verts = _convex_hull_2d(pts)
    assert len(verts) == 4
    assert set(verts) == {0, 1, 2, 3}


def test_convex_hull_2d_with_interior():
    """Interior point is excluded from the hull."""
    pts = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]], dtype=float)
    verts = _convex_hull_2d(pts)
    assert len(verts) == 4
    assert 4 not in verts


def test_convex_hull_3d_cube():
    """8 corners of a cube yield 12 triangular facets (6 faces x 2 triangles)."""
    pts = np.array(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=float,
    )
    simplices = _convex_hull_3d(pts)
    assert simplices.shape[1] == 3
    assert simplices.shape[0] == 12


def test_convex_hull_3d_coplanar_hexagon():
    """6 coplanar points (benzene-like hexagon) produce facets via fan triangulation."""
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(6)])
    simplices = _convex_hull_3d(pts)
    # Fan triangulation of a hexagon: 4 triangles
    assert simplices.shape[0] == 4
    assert simplices.shape[1] == 3


def test_convex_hull_3d_three_points():
    """3 points yield a single triangular facet."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    simplices = _convex_hull_3d(pts)
    assert simplices.shape == (1, 3)


# ---------------------------------------------------------------------------
# Pore / face detection tests
# ---------------------------------------------------------------------------


def _hex_lattice_graph(rows: int = 2, cols: int = 3):
    """Build a small planar hexagonal lattice graph (like graphene)."""
    import networkx as nx

    g = nx.Graph()
    node_id = 0
    id_map: dict[tuple[int, int, int], int] = {}
    a = 1.42

    for r in range(rows):
        for c in range(cols):
            for sub in (0, 1):
                x = c * a * 3 + sub * a * 1.5
                y = r * a * np.sqrt(3) + sub * a * np.sqrt(3) / 2
                g.add_node(node_id, symbol="C", position=(x, y, 0.0))
                id_map[(r, c, sub)] = node_id
                node_id += 1

    for r in range(rows):
        for c in range(cols):
            g.add_edge(id_map[(r, c, 0)], id_map[(r, c, 1)])
    for r in range(rows):
        for c in range(cols - 1):
            g.add_edge(id_map[(r, c, 1)], id_map[(r, c + 1, 0)])
    for r in range(rows - 1):
        for c in range(cols):
            g.add_edge(id_map[(r + 1, c, 0)], id_map[(r, c, 1)])

    return g


def test_find_2d_faces_hexagonal():
    """Hexagonal lattice yields hexagonal faces."""
    from xyzrender.face import find_2d_faces

    g = _hex_lattice_graph(rows=2, cols=3)
    faces = find_2d_faces(g, max_size=30)
    assert len(faces) > 0
    assert 6 in {len(f) for f in faces}


def test_find_2d_faces_triangle():
    """Three nodes forming a triangle yield one face of size 3."""
    import networkx as nx

    from xyzrender.face import find_2d_faces

    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))
    g.add_node(1, symbol="C", position=(1.0, 0.0, 0.0))
    g.add_node(2, symbol="C", position=(0.5, 0.866, 0.0))
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(0, 2)

    faces = find_2d_faces(g, max_size=30, min_size=3)
    assert len(faces) == 1
    assert len(faces[0]) == 3


def test_find_2d_faces_empty_graph():
    """Empty graph returns no faces."""
    import networkx as nx

    from xyzrender.face import find_2d_faces

    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))
    assert find_2d_faces(g) == []


def test_pore_size_colors():
    """pore_size_colors assigns same color to same-size rings."""
    from xyzrender.hull import pore_size_colors

    subsets = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
    colors = pore_size_colors(subsets)
    assert len(colors) == 3
    assert colors[0] == colors[1]  # both size-3
    assert colors[0] != colors[2]  # size-3 != size-4


def test_resolve_hull_faces():
    """hull='faces' on a 2D graph returns face indices."""
    import networkx as nx

    from xyzrender.hull import resolve_hull_flag_and_indices

    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))
    g.add_node(1, symbol="C", position=(1.0, 0.0, 0.0))
    g.add_node(2, symbol="C", position=(0.5, 0.866, 0.0))
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(0, 2)

    show, indices = resolve_hull_flag_and_indices("faces", g)
    assert show is True
    assert isinstance(indices, list)
    assert len(indices) == 1


def test_resolve_hull_pores_no_graph():
    """hull='pores' with no graph returns (None, None)."""
    from xyzrender.hull import resolve_hull_flag_and_indices

    show, indices = resolve_hull_flag_and_indices("pores", None)
    assert show is None
    assert indices is None


def test_ring_fingerprint_modes():
    """ring_fingerprint returns different signatures for size/type/env modes."""
    import networkx as nx

    from xyzrender.hull import ring_fingerprint

    g = nx.Graph()
    for i in range(6):
        g.add_node(i, symbol="C" if i < 5 else "N", position=(0, 0, 0))
    for i in range(6):
        g.add_edge(i, (i + 1) % 6)

    ring = list(range(6))
    fp_size = ring_fingerprint(ring, g, mode="size")
    fp_type = ring_fingerprint(ring, g, mode="type")
    fp_env = ring_fingerprint(ring, g, mode="env", shared_atoms={0, 1})

    # size: only size matters
    assert fp_size == (6, ())
    # type: includes atom types
    assert fp_size != fp_type
    assert len(fp_type[1]) == 6  # 6 atom signatures
    # env: shared_atoms flag differs from type
    assert fp_env != fp_type


def test_ring_colors_size_vs_type():
    """_ring_colors produces fewer distinct colours in size mode."""
    import networkx as nx

    from xyzrender.hull import _ring_colors

    g = nx.Graph()
    # Two 6-rings: one all-C, one with N — same size but different type.
    for i in range(12):
        g.add_node(i, symbol="C" if i < 6 else ("C" if i < 11 else "N"), position=(0, 0, 0))
    for i in range(6):
        g.add_edge(i, (i + 1) % 6)
    for i in range(6, 12):
        g.add_edge(i, 6 + (i - 6 + 1) % 6)

    subsets = [list(range(6)), list(range(6, 12))]
    colors_type = _ring_colors(subsets, g, mode="type")
    colors_size = _ring_colors(subsets, g, mode="size")

    # type: different atom compositions → different colours
    assert colors_type[0] != colors_type[1]
    # size: same size → same colour
    assert colors_size[0] == colors_size[1]


def test_silhouette_polygon_single_facet():
    """get_silhouette_polygon returns a single polygon, not triangle fan."""
    from xyzrender.hull import get_silhouette_polygon

    pos = np.array(
        [
            [1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [-1, 1, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    facets = get_silhouette_polygon(pos)
    assert len(facets) == 1  # single polygon, not multiple triangles
    verts, _z = facets[0]
    assert verts.shape[0] >= 4  # convex hull of 5 points projected to xy


def test_ring_facets_single_polygon():
    """get_ring_facets returns one polygon per ring, not triangle fan."""
    from xyzrender.hull import get_ring_facets

    pos = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1.5, 0.87, 0],
            [1, 1.73, 0],
            [0, 1.73, 0],
            [-0.5, 0.87, 0],
        ],
        dtype=float,
    )
    facets = get_ring_facets(pos, [0, 1, 2, 3, 4, 5])
    assert len(facets) == 1  # single polygon
    assert facets[0][0].shape == (6, 3)  # 6 vertices, not 3 (triangle)


def test_tile_supercell_indices():
    """_tile_supercell_indices replicates subsets correctly."""
    from xyzrender.api import _tile_supercell_indices

    subsets = [[0, 1], [2, 3]]
    tiled = _tile_supercell_indices(subsets, (2, 1, 1), n_base=10)
    # 2x1x1 = 2 replicas x 2 subsets = 4 subsets
    assert len(tiled) == 4
    assert tiled[0] == [0, 1]  # replica (0,0,0)
    assert tiled[1] == [2, 3]
    assert tiled[2] == [10, 11]  # replica (1,0,0), offset = 10
    assert tiled[3] == [12, 13]


def test_tile_pore_centroids_radii():
    """_tile_pore_centroids_radii tiles centroids and radii across supercell replicas."""
    import numpy as np

    from xyzrender.api import _tile_pore_centroids_radii

    # Lattice vectors (simple cubic)
    lattice = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])

    # Two pores in unit cell
    centroids = [(1.0, 2.0, 3.0), (2.0, 3.0, 4.0)]
    radii = [1.5, 2.0]

    # 2x2x1 supercell
    tiled_c, tiled_r = _tile_pore_centroids_radii(centroids, radii, (2, 2, 1), lattice)

    # Should have 4 replicas x 2 pores = 8 centroids
    assert len(tiled_c) == 8
    assert tiled_r is not None
    assert len(tiled_r) == 8

    # Check first replica (0,0,0) — no shift
    assert tiled_c[0] == centroids[0]
    assert tiled_c[1] == centroids[1]

    # Check second replica (0,1,0) — shifted by lattice[1] = [0,5,0]
    assert tiled_c[2] == (1.0, 7.0, 3.0)  # [1,2,3] + [0,5,0]
    assert tiled_c[3] == (2.0, 8.0, 4.0)  # [2,3,4] + [0,5,0]

    # Check third replica (1,0,0) — shifted by lattice[0] = [5,0,0]
    assert tiled_c[4] == (6.0, 2.0, 3.0)  # [1,2,3] + [5,0,0]
    assert tiled_c[5] == (7.0, 3.0, 4.0)  # [2,3,4] + [5,0,0]

    # Check fourth replica (1,1,0) — shifted by [5,5,0]
    assert tiled_c[6] == (6.0, 7.0, 3.0)  # [1,2,3] + [5,5,0]
    assert tiled_c[7] == (7.0, 8.0, 4.0)  # [2,3,4] + [5,5,0]

    # Radii should be replicated (not shifted)
    assert tiled_r == [1.5, 2.0] * 4


def test_tile_pore_centroids_radii_no_radii():
    """_tile_pore_centroids_radii handles None radii gracefully."""
    import numpy as np

    from xyzrender.api import _tile_pore_centroids_radii

    lattice = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    centroids = [(1.0, 2.0, 3.0)]

    # No radii
    tiled_c, tiled_r = _tile_pore_centroids_radii(centroids, None, (2, 1, 1), lattice)

    assert len(tiled_c) == 2
    assert tiled_r is None


def test_hull_list_input_no_crash():
    """hull=[[0,1,2,3,4]] (list input) should not crash with unhashable type."""
    import networkx as nx

    from xyzrender.hull import apply_hull_to_config
    from xyzrender.types import RenderConfig

    g = nx.Graph()
    for i in range(6):
        g.add_node(i, symbol="C", position=(float(i), 0.0, 0.0))
    for i in range(5):
        g.add_edge(i, i + 1)

    cfg = RenderConfig()
    # This previously crashed with TypeError: unhashable type: 'list'
    apply_hull_to_config(cfg, [[0, 1, 2, 3, 4]], None, None, None, None, g)
    assert cfg.show_convex_hull is True
    assert cfg.hull_atom_indices is not None


# ---------------------------------------------------------------------------
# Integration tests (real structures)
# ---------------------------------------------------------------------------

_STRUCTURES = Path(__file__).resolve().parent.parent / "examples" / "structures"


def _load(name: str):
    import xyzrender as xr

    path = _STRUCTURES / name
    if not path.exists():
        pytest.skip(f"{name} not in examples/structures/")
    return xr.load(str(path))


def test_buckyball_32_faces():
    """C60: exactly 12 pentagons + 20 hexagons."""
    from collections import Counter

    from xyzrender.face import find_2d_faces

    faces = find_2d_faces(_load("buckyball.xyz").graph)
    sizes = Counter(len(f) for f in faces)
    assert sizes[5] == 12
    assert sizes[6] == 20


def test_buckyball_cage_pore():
    """C60 small cycles merge into 1 cage pore."""
    from xyzrender.pore import find_pores

    pores = find_pores(_load("buckyball.xyz").graph)
    assert len(pores) == 1
    assert pores[0][1] > 1.5  # radius
    assert len(pores[0][2]) > 10  # wall vertices


def test_mof5_periodic_pore():
    """MOF-5 with lattice: 1 pore near cell centre."""
    from xyzrender.pore import find_pores

    mol = _load("MOF-5.xyz")
    lat = np.array(mol.cell_data.lattice)
    pores = find_pores(mol.graph, lattice=lat)
    assert len(pores) == 1
    frac = np.linalg.inv(lat) @ np.array(pores[0][0])
    assert all(0.3 < f < 0.7 for f in frac)


def test_render_faces_svg():
    """render(hull='faces') produces 32 hull polygons for C60."""
    import re
    import tempfile

    import xyzrender as xr

    with tempfile.NamedTemporaryFile(suffix=".svg") as f:
        xr.render(_load("buckyball.xyz"), hull="faces", output=f.name)
        svg = Path(f.name).read_text()
    assert len(re.findall(r"<polygon[^>]*fill-opacity", svg)) == 32


def test_render_pore_svg():
    """render(pore=True) produces 1 pore circle for C60."""
    import re
    import tempfile

    import xyzrender as xr

    with tempfile.NamedTemporaryFile(suffix=".svg") as f:
        xr.render(_load("buckyball.xyz"), pore=True, output=f.name)
        svg = Path(f.name).read_text()
    assert len(re.findall(r"circle[^>]*pore", svg)) == 1
