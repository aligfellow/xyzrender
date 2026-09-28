"""Tests for export.py — SVG → PNG/PDF conversion via cairosvg."""

import pytest

SIMPLE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'
)


@pytest.fixture
def cairosvg():
    return pytest.importorskip("cairosvg", reason="cairosvg required")


def test_svg_raster_dimensions_wide_viewbox():
    from xyzrender.export import svg_raster_dimensions

    wide = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 800" width="1000" height="800"></svg>'
    w, h = svg_raster_dimensions(wide, size=800)
    assert w == 800
    assert h == 640


def test_svg_to_png_bytes_default_is_square(cairosvg, monkeypatch):
    from io import BytesIO

    from PIL import Image

    from xyzrender.export import svg_to_png_bytes

    monkeypatch.setattr("xyzrender.export._has_resvg", lambda: False)

    wide = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 800" width="1000" height="800">'
        '<rect x="950" y="10" width="40" height="40" fill="blue"/></svg>'
    )
    png = svg_to_png_bytes(wide, size=800)
    img = Image.open(BytesIO(png))
    assert img.width == 800
    assert img.height == 800


def test_svg_to_png_bytes_fit_viewbox(cairosvg, monkeypatch):
    from io import BytesIO

    from PIL import Image

    from xyzrender.export import svg_to_png_bytes

    monkeypatch.setattr("xyzrender.export._has_resvg", lambda: False)

    wide = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 800" width="1000" height="800">'
        '<rect x="950" y="10" width="40" height="40" fill="blue"/></svg>'
    )
    png = svg_to_png_bytes(wide, size=800, fit_viewbox=True)
    img = Image.open(BytesIO(png))
    assert img.width == 800
    assert img.height == 640


def test_svg_to_png_writes_file(cairosvg, tmp_path):
    from xyzrender.export import svg_to_png

    out = tmp_path / "out.png"
    svg_to_png(SIMPLE_SVG, str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    # PNG magic bytes
    assert out.read_bytes()[:4] == b"\x89PNG"


def test_svg_to_pdf_writes_file(cairosvg, tmp_path):
    from xyzrender.export import svg_to_pdf

    out = tmp_path / "out.pdf"
    svg_to_pdf(SIMPLE_SVG, str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:4] == b"%PDF"
