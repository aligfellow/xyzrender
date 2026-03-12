"""Tests for convex hull facet computation and rendering."""

from __future__ import annotations

import numpy as np

from xyzrender.hull import get_convex_hull_edges, get_convex_hull_facets, hull_facets_svg, normalize_hull_subsets
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


def test_get_convex_hull_facets_fewer_than_four_points():
    """Fewer than 4 points return empty list."""
    assert get_convex_hull_facets(np.array([[0, 0, 0], [1, 0, 0]])) == []
    assert get_convex_hull_facets(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])) == []


def test_get_convex_hull_edges_tetrahedron():
    """Tetrahedron has 6 hull edges (all pairs of 4 vertices)."""
    pos = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    )
    edges = get_convex_hull_edges(pos)
    assert len(edges) == 6
    for i, j in edges:
        assert i < j
        assert 0 <= i < 4
        assert 0 <= j < 4
    assert len(set(edges)) == 6


def test_get_convex_hull_edges_fewer_than_four_points():
    """Fewer than 4 points return empty list."""
    assert get_convex_hull_edges(np.array([[0, 0, 0], [1, 0, 0]])) == []
    assert get_convex_hull_edges(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])) == []


def test_get_convex_hull_edges_with_include_mask():
    """Edges are returned in graph (full) index space when include_mask is used."""
    # 5 points: indices 0,1,2,3 form tetrahedron; index 4 is inside
    pos = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    mask = np.array([True, True, True, True, False])
    edges = get_convex_hull_edges(pos, include_mask=mask)
    assert len(edges) == 6
    # All indices must be in 0..3 (graph indices of the tetrahedron)
    for i, j in edges:
        assert i < j
        assert 0 <= i <= 3
        assert 0 <= j <= 3


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
    # Mask with only 3 points -> empty
    mask3 = np.array([True, True, True, False, False])
    assert get_convex_hull_facets(pos, include_mask=mask3) == []


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
    assert svg.count("fill-opacity") >= 4


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
    # Two tetrahedra -> 4 + 4 = 8 facets
    assert svg.count("<polygon") == 8
    assert svg.count("fill-opacity") == 8


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
