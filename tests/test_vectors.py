"""Tests for vector arrow loading and rendering."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from xyzrender.annotations import load_vectors
from xyzrender.readers import load_molecule
from xyzrender.renderer import render_svg
from xyzrender.types import RenderConfig, VectorArrow

EXAMPLES = Path(__file__).parent.parent / "examples" / "structures"


def _write_json(data, suffix=".json") -> Path:
    """Write *data* to a temporary JSON file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    json.dump(data, f)
    f.flush()
    return Path(f.name)


# ---------------------------------------------------------------------------
# load_vectors — parsing
# ---------------------------------------------------------------------------


def test_load_vectors_com_origin(tmp_path):
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    jf = _write_json([{"vector": [1.0, 0.0, 0.0]}])
    arrows = load_vectors(jf, graph)
    assert len(arrows) == 1
    va = arrows[0]
    assert np.allclose(va.vector, [1.0, 0.0, 0.0])
    # Origin should equal the centroid of all atom positions
    positions = np.array([graph.nodes[i]["position"] for i in graph.nodes()])
    assert np.allclose(va.origin, positions.mean(axis=0))


def test_load_vectors_atom_origin(tmp_path):
    graph, _ = load_molecule(EXAMPLES / "ethanol.xyz")
    jf = _write_json([{"origin": 1, "vector": [0.0, 1.0, 0.0]}])
    arrows = load_vectors(jf, graph)
    assert len(arrows) == 1
    expected = np.array(graph.nodes[next(iter(graph.nodes()))]["position"])
    assert np.allclose(arrows[0].origin, expected)


def test_load_vectors_explicit_origin(tmp_path):
    graph, _ = load_molecule(EXAMPLES / "ethanol.xyz")
    jf = _write_json([{"origin": [1.5, 2.5, 3.5], "vector": [0.0, 0.0, 1.0]}])
    arrows = load_vectors(jf, graph)
    assert np.allclose(arrows[0].origin, [1.5, 2.5, 3.5])


def test_load_vectors_color_and_label(tmp_path):
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    jf = _write_json([{"vector": [1.0, 0.0, 0.0], "color": "red", "label": "μ", "scale": 2.5}])
    arrows = load_vectors(jf, graph)
    va = arrows[0]
    assert va.color == "#ff0000"
    assert va.label == "μ"
    assert va.scale == pytest.approx(2.5)


def test_load_vectors_multiple(tmp_path):
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    jf = _write_json(
        [
            {"vector": [1.0, 0.0, 0.0], "color": "#cc0000"},
            {"origin": 2, "vector": [0.0, 1.0, 0.0]},
            {"origin": [0, 0, 0], "vector": [0.0, 0.0, 1.0], "label": "z"},
        ]
    )
    arrows = load_vectors(jf, graph)
    assert len(arrows) == 3
    assert arrows[0].color == "#cc0000"
    assert arrows[2].label == "z"


# ---------------------------------------------------------------------------
# load_vectors — error handling
# ---------------------------------------------------------------------------


def test_load_vectors_missing_vector_key():
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    jf = _write_json([{"origin": "com"}])
    with pytest.raises(ValueError, match="missing required key 'vector'"):
        load_vectors(jf, graph)


def test_load_vectors_atom_index_out_of_range():
    graph, _ = load_molecule(EXAMPLES / "ethanol.xyz")
    n = graph.number_of_nodes()
    jf = _write_json([{"origin": n + 100, "vector": [1.0, 0.0, 0.0]}])
    with pytest.raises(ValueError, match="out of range"):
        load_vectors(jf, graph)


def test_load_vectors_bad_color():
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    jf = _write_json([{"vector": [1.0, 0.0, 0.0], "color": "notacolor"}])
    with pytest.raises(ValueError, match="color"):
        load_vectors(jf, graph)


def test_load_vectors_not_an_array():
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    # A plain object without a 'vectors' key is not valid
    jf = _write_json({"color": "red"})  # no 'vectors' key → empty list
    arrows = load_vectors(jf, graph)
    assert arrows == []


def test_load_vectors_anchor_per_entry():
    """Per-entry anchor overrides the file-level default."""
    graph, _ = load_molecule(EXAMPLES / "ethanol.xyz")
    # File default is 'tail', but first entry overrides to 'center'
    jf = _write_json(
        {
            "anchor": "tail",
            "vectors": [
                {"origin": "com", "vector": [1.0, 0.0, 0.0], "anchor": "center"},
                {"origin": "com", "vector": [0.0, 1.0, 0.0]},
            ],
        }
    )
    arrows = load_vectors(jf, graph)
    assert arrows[0].anchor == "center"
    assert arrows[1].anchor == "tail"


def test_load_vectors_anchor_invalid():
    graph, _ = load_molecule(EXAMPLES / "ethanol.xyz")
    jf = _write_json({"anchor": "middle", "vectors": [{"origin": "com", "vector": [1.0, 0.0, 0.0]}]})
    with pytest.raises(ValueError, match="anchor"):
        load_vectors(jf, graph)


def test_render_anchor_center_differs_from_tail():
    """With anchor=center the rendered tip is offset relative to anchor=tail."""
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    positions = np.array([graph.nodes[i]["position"] for i in graph.nodes()])
    centroid = positions.mean(axis=0)
    va_tail = VectorArrow(vector=np.array([2.0, 0.0, 0.0]), origin=centroid.copy(), anchor="tail")
    va_center = VectorArrow(vector=np.array([2.0, 0.0, 0.0]), origin=centroid.copy(), anchor="center")
    svg_tail = render_svg(graph, RenderConfig(vectors=[va_tail]))
    svg_center = render_svg(graph, RenderConfig(vectors=[va_center]))
    # The SVGs should differ because the arrow positions differ
    assert svg_tail != svg_center


# ---------------------------------------------------------------------------
# Rendering — vector arrows appear in SVG
# ---------------------------------------------------------------------------


def test_render_with_vector_arrows():
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    positions = np.array([graph.nodes[i]["position"] for i in graph.nodes()])
    centroid = positions.mean(axis=0)
    va = VectorArrow(vector=np.array([1.0, 0.0, 0.0]), origin=centroid, color="#cc0000", label="μ")
    cfg = RenderConfig(vectors=[va])
    svg = render_svg(graph, cfg)
    assert "#cc0000" in svg
    assert "μ" in svg


def test_render_zero_length_vector_no_crash():
    """A zero-length vector should render without errors (shaft only, no arrowhead)."""
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    positions = np.array([graph.nodes[i]["position"] for i in graph.nodes()])
    centroid = positions.mean(axis=0)
    va = VectorArrow(vector=np.array([0.0, 0.0, 0.0]), origin=centroid)
    svg = render_svg(graph, RenderConfig(vectors=[va]))
    assert "</svg>" in svg


# ---------------------------------------------------------------------------
# load_vectors — dict input (same as JSON file but in-memory)
# ---------------------------------------------------------------------------


def test_load_vectors_dict_input():
    """Passing a dict is equivalent to the equivalent JSON file."""
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    data = {"anchor": "center", "vectors": [{"origin": "com", "vector": [1.0, 0.0, 0.0], "label": "μ"}]}
    arrows_dict = load_vectors(data, graph)
    jf = _write_json(data)
    arrows_file = load_vectors(jf, graph)
    assert len(arrows_dict) == 1
    assert arrows_dict[0].anchor == arrows_file[0].anchor
    assert np.allclose(arrows_dict[0].vector, arrows_file[0].vector)
    assert arrows_dict[0].label == arrows_file[0].label


# ---------------------------------------------------------------------------
# API — vector= kwarg
# ---------------------------------------------------------------------------


def test_api_render_vectors_path(tmp_path):
    """render() vector= accepts a file path."""
    from xyzrender.api import load as api_load
    from xyzrender.api import render

    mol = api_load(EXAMPLES / "ethanol.xyz")
    jf = _write_json([{"vector": [1.0, 0.0, 0.0], "color": "#cc0000", "label": "F"}])
    result = render(mol, vector=jf)
    assert "#cc0000" in str(result)
    assert "F" in str(result)


def test_api_render_vectors_dict():
    """render() vector= accepts an inline dict."""
    from xyzrender.api import load as api_load
    from xyzrender.api import render

    mol = api_load(EXAMPLES / "ethanol.xyz")
    data = {"vectors": [{"vector": [1.0, 0.0, 0.0], "color": "#aabbcc"}]}
    result = render(mol, vector=data)
    assert "#aabbcc" in str(result)


# ---------------------------------------------------------------------------
# Short-projection rendering: dot and 'x' symbols
# ---------------------------------------------------------------------------


def _make_near_z_arrow(centroid, color, label, z_sign=1):
    """VectorArrow that is nearly along Z so its 2D projected length is << arr.

    z_sign=+1  → tip closer to viewer (dot rendered)
    z_sign=-1  → tip farther from viewer (x rendered)
    """
    return VectorArrow(
        vector=np.array([0.001, 0.0, z_sign * 2.0]),
        origin=centroid.copy(),
        color=color,
        label=label,
        draw_on_top=True,
    )


def test_arrow_dot_drawn_when_short_facing_viewer():
    """Arrow whose 2D projection is shorter than the arrowhead threshold and whose
    tip is closer to the viewer should render as a filled circle (dot), not a polygon."""
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    positions = np.array([graph.nodes[i]["position"] for i in graph.nodes()])
    centroid = positions.mean(axis=0)
    va = _make_near_z_arrow(centroid, "#bb2200", "Z", z_sign=1)
    svg = render_svg(graph, RenderConfig(vectors=[va]))
    # The dot is drawn as a <circle> with the arrow colour.
    assert re.search(r'<circle[^>]*fill="#bb2200"', svg), "expected a dot circle for viewer-facing short arrow"
    # A polygon arrowhead must NOT appear (no other element in a plain caffeine render uses polygon).
    assert "<polygon" not in svg, "unexpected arrowhead polygon for too-short arrow"
    # Label is suppressed for dot/x symbols.
    assert "Z" not in svg.split("<text", 1)[-1] if "<text" in svg else True


def test_arrow_x_drawn_when_short_facing_away():
    """Arrow whose 2D projection is shorter than the arrowhead threshold and whose
    tip is farther from the viewer should render as an 'x' (two crossed lines), not a polygon."""
    graph, _ = load_molecule(EXAMPLES / "caffeine.xyz")
    positions = np.array([graph.nodes[i]["position"] for i in graph.nodes()])
    centroid = positions.mean(axis=0)
    va = _make_near_z_arrow(centroid, "#0033cc", "W", z_sign=-1)
    svg = render_svg(graph, RenderConfig(vectors=[va]))
    # The x is drawn as two <line> elements stroked with the arrow colour.
    x_lines = re.findall(r'<line[^>]*stroke="#0033cc"', svg)
    assert len(x_lines) == 2, f"expected 2 lines for the 'x' symbol, got {len(x_lines)}"
    # No polygon arrowhead.
    assert "<polygon" not in svg, "unexpected arrowhead polygon for too-short arrow"
    # Label is suppressed for dot/x symbols.
    assert "W" not in svg.split("<text", 1)[-1] if "<text" in svg else True


# ---------------------------------------------------------------------------
# API render() — vectors appear when combined with crystal axes
# ---------------------------------------------------------------------------


def test_render_user_vector_appears_plain_mol():
    """render() draws both the shaft/head and label for a vector on a plain molecule."""
    from xyzrender.api import load as api_load
    from xyzrender.api import render

    mol = api_load(EXAMPLES / "caffeine.xyz")
    jf = _write_json([{"origin": "com", "vector": [2.0, 0.0, 0.0], "color": "#ab1234", "label": "testvec"}])
    svg = str(render(mol, vector=str(jf)))
    assert "#ab1234" in svg, "user vector color must appear in SVG"
    assert "testvec" in svg, "user vector label must appear in SVG"
    # A filled arrowhead polygon should be present.
    assert re.search(r'<polygon[^>]*fill="#ab1234"', svg), "arrowhead polygon with user color must appear in SVG"


def test_render_user_vector_appears_with_crystal_axes():
    """render() with cell_data + axes=True must include BOTH axis colors AND the
    user-supplied vector color in the SVG output.  This is the main regression
    guard for the bug where cfg.vectors was overwritten instead of extended."""
    from xyzrender.api import load as api_load
    from xyzrender.api import render

    mol = api_load(EXAMPLES / "caffeine_cell.xyz", cell=True)
    # Use a color that differs from all three axis colors (firebrick/forestgreen/royalblue)
    user_color = "#cd5c5c"  # indianred — resolves to a distinct hex from axis colors
    jf = _write_json([{"origin": "com", "vector": [2.0, 0.0, 0.0], "color": user_color, "label": "dipole"}])
    svg = str(render(mol, vector=str(jf), axes=True))
    # Axis colors must be present (firebrick → #b22222, forestgreen → #228b22, royalblue → #4169e1)
    from xyzrender.types import resolve_color

    assert resolve_color("firebrick") in svg, "a-axis (firebrick) must appear"
    assert resolve_color("forestgreen") in svg, "b-axis (forestgreen) must appear"
    assert resolve_color("royalblue") in svg, "c-axis (royalblue) must appear"
    # User vector must also be present
    assert user_color in svg, "user vector color must appear in SVG alongside axis arrows"
    assert "dipole" in svg, "user vector label must appear in SVG"


def test_render_user_vector_appears_with_crystal_axes_no_double_loading():
    """Calling render() twice with the same pre-built config must NOT accumulate
    vectors across calls (shallow-copy list aliasing regression)."""
    from xyzrender.api import load as api_load
    from xyzrender.api import render
    from xyzrender.config import build_config

    mol = api_load(EXAMPLES / "caffeine_cell.xyz", cell=True)
    user_color = "#cd5c5c"
    jf = _write_json([{"origin": "com", "vector": [2.0, 0.0, 0.0], "color": user_color, "label": "dipole"}])

    cfg = build_config("default")
    svg1 = str(render(mol, config=cfg, vector=str(jf), axes=True))
    svg2 = str(render(mol, config=cfg, vector=str(jf), axes=True))

    # Count polygon arrowheads with user color in each SVG — should be exactly 1 each
    count1 = len(re.findall(rf'<polygon[^>]*fill="{re.escape(user_color)}"', svg1))
    count2 = len(re.findall(rf'<polygon[^>]*fill="{re.escape(user_color)}"', svg2))
    assert count1 == 1, f"first render: expected 1 user-vector arrowhead, got {count1}"
    assert count2 == 1, f"second render: expected 1 user-vector arrowhead, got {count2}"


# ---------------------------------------------------------------------------
# orient_hkl_to_view — user vectors co-rotate with crystal
# ---------------------------------------------------------------------------


def _cubic_graph_and_cell(a: float = 4.0):
    """Return a minimal 2-atom graph and CellData for a simple cubic cell."""
    import networkx as nx

    from xyzrender.types import CellData

    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))
    g.add_node(1, symbol="C", position=(a, a, a))
    lattice = np.diag([a, a, a]).astype(float)
    return g, CellData(lattice=lattice, cell_origin=np.zeros(3))


def test_orient_hkl_vectors_001_no_rotation():
    """Axis [001] on a cubic cell is already +z — vector direction and origin are unchanged."""
    from xyzrender.types import RenderConfig
    from xyzrender.viewer import orient_hkl_to_view

    g, cell_data = _cubic_graph_and_cell()
    va = VectorArrow(vector=np.array([1.0, 2.0, 3.0]), origin=np.array([1.0, 1.0, 1.0]))
    cfg = RenderConfig(vectors=[va])

    orient_hkl_to_view(g, cell_data, "001", cfg)

    np.testing.assert_allclose(np.array(cfg.vectors[0].vector), [1.0, 2.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(np.array(cfg.vectors[0].origin), [1.0, 1.0, 1.0], atol=1e-9)


def test_orient_hkl_vectors_100_direction():
    """Axis [100] on a cubic cell: a vector along lattice-a must point along +z after rotation."""
    from xyzrender.types import RenderConfig
    from xyzrender.viewer import orient_hkl_to_view

    g, cell_data = _cubic_graph_and_cell()
    # The a-axis is [1,0,0]; viewing along [100] must align it with +z.
    va = VectorArrow(vector=np.array([1.0, 0.0, 0.0]), origin=np.array([2.0, 2.0, 2.0]))
    cfg = RenderConfig(vectors=[va])

    orient_hkl_to_view(g, cell_data, "100", cfg)

    np.testing.assert_allclose(np.array(cfg.vectors[0].vector), [0.0, 0.0, 1.0], atol=1e-9)


def test_orient_hkl_vectors_100_origin():
    """After [100] rotation the vector origin is rotated around the atom centroid."""
    from xyzrender.types import RenderConfig
    from xyzrender.viewer import orient_hkl_to_view

    a = 4.0
    g, cell_data = _cubic_graph_and_cell(a)
    pos = np.array([g.nodes[i]["position"] for i in g.nodes()])
    centroid = pos.mean(axis=0)  # (2, 2, 2)

    # Place the origin displaced by a along the a-axis from the centroid.
    origin = centroid + np.array([a, 0.0, 0.0])
    va = VectorArrow(vector=np.array([1.0, 0.0, 0.0]), origin=origin.copy())
    cfg = RenderConfig(vectors=[va])

    orient_hkl_to_view(g, cell_data, "100", cfg)

    # The rotation maps displacement [a, 0, 0] → [0, 0, a].
    expected_origin = centroid + np.array([0.0, 0.0, a])
    np.testing.assert_allclose(np.array(cfg.vectors[0].origin), expected_origin, atol=1e-9)


def test_orient_hkl_vectors_110_direction():
    """Axis [110] on a cubic cell: a vector along the [110] diagonal must point along +z."""
    from xyzrender.types import RenderConfig
    from xyzrender.viewer import orient_hkl_to_view

    g, cell_data = _cubic_graph_and_cell()
    # The [110] lattice direction is [1, 1, 0] / sqrt(2).
    va = VectorArrow(
        vector=np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        origin=np.array([2.0, 2.0, 2.0]),  # at centroid → origin is unchanged
    )
    cfg = RenderConfig(vectors=[va])

    orient_hkl_to_view(g, cell_data, "110", cfg)

    np.testing.assert_allclose(np.array(cfg.vectors[0].vector), [0.0, 0.0, 1.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Integration — user vectors via render() with axis= on a real crystal file
# ---------------------------------------------------------------------------


def test_render_crystal_axis_vector_projects_as_dot_when_parallel_to_view():
    """render() with axis='100' on an orthorhombic cell: a user vector aligned
    with the a-axis must project as a dot (<circle>) in the SVG because the
    arrow is now pointing directly along the viewing axis (+z)."""
    from xyzrender.api import Molecule, render
    from xyzrender.types import CellData

    graph, _ = load_molecule(EXAMPLES / "caffeine_cell.xyz")
    cell_data = CellData(
        lattice=np.array(graph.graph["lattice"], dtype=float),
        cell_origin=np.zeros(3),
    )
    mol = Molecule(graph=graph, cell_data=cell_data)

    # Unit vector along the a-axis (first lattice row)
    a_hat = cell_data.lattice[0] / np.linalg.norm(cell_data.lattice[0])
    positions = np.array([graph.nodes[i]["position"] for i in graph.nodes()])
    centroid = positions.mean(axis=0)

    # Distinct color so we can identify this arrow's elements in the SVG.
    user_color = "#7f1234"
    va = VectorArrow(vector=a_hat.copy(), origin=centroid.copy(), color=user_color)

    # View along [100]: the user vector is now parallel to the viewing axis →
    # projected length ≈ 0 → _draw_arrow_svg renders it as a filled circle, not a polygon.
    svg = str(render(mol, vector=[va], axes=False, ghosts=False, axis="100"))

    assert f'fill="{user_color}"' in svg, "user vector must appear in the rotated SVG"
    # In the short-projection code path the arrow renders as a circle, not a polygon arrowhead.
    assert re.search(rf'<circle[^>]*fill="{re.escape(user_color)}"', svg), (
        "user vector aligned with view axis should render as a dot (circle), not a line+polygon"
    )
    assert not re.search(rf'<polygon[^>]*fill="{re.escape(user_color)}"', svg), (
        "user vector aligned with view axis must NOT render as a polygon arrowhead"
    )
