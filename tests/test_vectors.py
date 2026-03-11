"""Tests for vector arrow loading and rendering."""

from __future__ import annotations

import json
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
    assert "<line" in svg


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
# API — vectors= kwarg
# ---------------------------------------------------------------------------


def test_api_render_vectors_path(tmp_path):
    """render() vectors= accepts a file path."""
    from xyzrender.api import load as api_load
    from xyzrender.api import render

    mol = api_load(EXAMPLES / "ethanol.xyz")
    jf = _write_json([{"vector": [1.0, 0.0, 0.0], "color": "#cc0000", "label": "F"}])
    result = render(mol, vectors=jf)
    assert "#cc0000" in str(result)
    assert "F" in str(result)


def test_api_render_vectors_dict():
    """render() vectors= accepts an inline dict."""
    from xyzrender.api import load as api_load
    from xyzrender.api import render

    mol = api_load(EXAMPLES / "ethanol.xyz")
    data = {"vectors": [{"vector": [1.0, 0.0, 0.0], "color": "#aabbcc"}]}
    result = render(mol, vectors=data)
    assert "#aabbcc" in str(result)
