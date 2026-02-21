"""Integration tests for end-to-end rendering."""

from pathlib import Path

from xyzrender import load_molecule, render_svg
from xyzrender.types import RenderConfig

EXAMPLES = Path(__file__).parent.parent / "examples" / "structures"


def test_caffeine_renders():
    graph = load_molecule(EXAMPLES / "caffeine.xyz")
    svg = render_svg(graph)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert "<circle" in svg


def test_caffeine_gradient():
    graph = load_molecule(EXAMPLES / "caffeine.xyz")
    svg = render_svg(graph, RenderConfig(gradient=True))
    assert "<use" in svg
    assert "<defs>" in svg


def test_ethanol_fog_mode():
    graph = load_molecule(EXAMPLES / "ethanol.xyz")
    svg = render_svg(graph, RenderConfig(fog=True))
    assert "<circle" in svg
    assert "<use" not in svg


def test_gradient_and_fog():
    graph = load_molecule(EXAMPLES / "caffeine.xyz")
    svg = render_svg(graph, RenderConfig(gradient=True, fog=True))
    assert "<use" in svg
    assert "<defs>" in svg


def test_hide_h():
    graph = load_molecule(EXAMPLES / "caffeine.xyz")
    svg_show = render_svg(graph, RenderConfig(hide_h=False))
    svg_hide = render_svg(graph, RenderConfig(hide_h=True))
    assert svg_hide.count("<circle") < svg_show.count("<circle")


def test_bond_orders_off():
    graph = load_molecule(EXAMPLES / "caffeine.xyz")
    svg = render_svg(graph, RenderConfig(bond_orders=False))
    assert "<svg" in svg
    assert "</svg>" in svg


def test_auto_orient():
    graph = load_molecule(EXAMPLES / "caffeine.xyz")
    svg_orient = render_svg(graph, RenderConfig(auto_orient=True))
    svg_raw = render_svg(graph, RenderConfig(auto_orient=False))
    assert "<svg" in svg_orient
    assert "<svg" in svg_raw


def test_custom_canvas_size():
    graph = load_molecule(EXAMPLES / "ethanol.xyz")
    svg = render_svg(graph, RenderConfig(canvas_size=400))
    # Canvas fits to molecule aspect ratio; larger dimension ≤ canvas_size
    assert 'width="' in svg
    assert 'height="' in svg
    import re

    m_w = re.search(r'width="(\d+)"', svg)
    m_h = re.search(r'height="(\d+)"', svg)
    assert m_w is not None
    assert m_h is not None
    w = int(m_w.group(1))
    h = int(m_h.group(1))
    assert max(w, h) <= 400
    assert min(w, h) > 0


def test_custom_background():
    graph = load_molecule(EXAMPLES / "ethanol.xyz")
    svg = render_svg(graph, RenderConfig(background="#000000"))
    assert "#000000" in svg


def test_color_overrides():
    graph = load_molecule(EXAMPLES / "ethanol.xyz")
    svg = render_svg(graph, RenderConfig(color_overrides={"O": "#00ff00"}))
    assert "#00ff00" in svg


def test_vdw_spheres():
    graph = load_molecule(EXAMPLES / "ethanol.xyz")
    svg = render_svg(graph, RenderConfig(vdw_indices=[]))
    assert "vg" in svg


def test_log_suppression():
    graph = load_molecule(EXAMPLES / "ethanol.xyz")
    svg = render_svg(graph, _log=False)
    assert "<svg" in svg


def test_benzene_aromatic():
    graph = load_molecule(EXAMPLES / "benzene.xyz")
    svg = render_svg(graph, RenderConfig(bond_orders=True, hide_h=False))
    assert "<line" in svg
    assert "<svg" in svg


def test_asparagine_renders():
    graph = load_molecule(EXAMPLES / "asparagine.xyz")
    svg = render_svg(graph)
    assert "<svg" in svg
    assert "</svg>" in svg
