"""Tests for export.py — SVG → PNG/PDF conversion via cairosvg."""

import pytest

SIMPLE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'
)


@pytest.fixture
def cairosvg():
    return pytest.importorskip("cairosvg", reason="cairosvg required")


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
