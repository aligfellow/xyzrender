"""Tests for export.py — SVG → PNG/PDF conversion via cairosvg."""

import pytest

SIMPLE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'
)


@pytest.fixture
def cairosvg():
    try:
        import cairosvg as _cairosvg
    except (ImportError, OSError) as exc:
        pytest.skip(f"cairosvg/libcairo unavailable: {exc}")
    return _cairosvg


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


def test_svg_to_pdf_preserve_filters_writes_file(cairosvg, tmp_path):
    pytest.importorskip("PIL", reason="Pillow required for raster PDF fallback")
    from xyzrender.export import svg_to_pdf

    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="120" height="120">
      <defs>
        <filter id="blur"><feGaussianBlur stdDeviation="4"/></filter>
      </defs>
      <circle cx="60" cy="60" r="35" fill="red" filter="url(#blur)"/>
    </svg>
    """
    out = tmp_path / "out_blur.pdf"
    svg_to_pdf(svg, str(out), preserve_filters=True)
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:4] == b"%PDF"
