"""Export SVG to raster/vector formats.

PNG rendering prefers **resvg-py** — it supports SVG filter primitives
(``feGaussianBlur``, ``feTurbulence``, etc.) that cairosvg silently ignores.
PDF rendering uses cairosvg (resvg-py has no PDF support).
"""

from __future__ import annotations

import re
from functools import lru_cache

_SVG_BASE_DPI = 96  # CSS/SVG spec: 1px = 1/96 inch
_VIEWBOX_RE = re.compile(r'viewBox="\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*"')


@lru_cache(maxsize=1)
def _has_resvg() -> bool:
    try:
        from resvg_py import svg_to_bytes  # noqa: F401
    except ImportError:
        return False
    return True


def svg_raster_dimensions(svg: str, *, size: int = 800) -> tuple[int, int]:
    """Return PNG width/height that fit the SVG viewBox inside a *size*×*size* box.

    Used when ``fit_viewbox=True`` so wide SVGs (e.g. molecule + colorbar) are not
    cropped in square GIF/PNG frames.
    """
    m = _VIEWBOX_RE.search(svg)
    if not m:
        return size, size
    vb_w = max(float(m.group(1)), 1.0)
    vb_h = max(float(m.group(2)), 1.0)
    if vb_w >= vb_h:
        out_w = size
        out_h = max(1, round(size * vb_h / vb_w))
    else:
        out_h = size
        out_w = max(1, round(size * vb_w / vb_h))
    return out_w, out_h


def svg_to_png_bytes(svg: str, *, size: int = 800, fit_viewbox: bool = False) -> bytes:
    """Convert SVG string to PNG bytes at a fixed pixel size.

    Used by GIF frame rendering where exact pixel dimensions matter.
    Default is a square *size*×*size* raster; set ``fit_viewbox=True`` to preserve
    the SVG viewBox aspect ratio within that box.
    """
    if fit_viewbox:
        out_w, out_h = svg_raster_dimensions(svg, size=size)
    else:
        out_w, out_h = size, size
    if _has_resvg():
        from resvg_py import svg_to_bytes

        return svg_to_bytes(svg_string=svg, width=out_w, height=out_h)

    import cairosvg

    return cairosvg.svg2png(bytestring=svg.encode(), output_width=out_w, output_height=out_h)


def svg_to_png(svg: str, output: str, *, size: int = 800, dpi: int = 300, fit_viewbox: bool = False) -> None:
    """Convert SVG string to a PNG file.

    Higher *dpi* produces a larger image with more detail (dpi/96 zoom factor).
    """
    if _has_resvg():
        from resvg_py import svg_to_bytes

        zoom = max(1, round(dpi / _SVG_BASE_DPI))
        if fit_viewbox:
            out_w, out_h = svg_raster_dimensions(svg, size=size)
            data = svg_to_bytes(svg_string=svg, width=out_w, height=out_h)
        else:
            data = svg_to_bytes(svg_string=svg, zoom=zoom)
    else:
        import cairosvg

        if fit_viewbox:
            out_w, out_h = svg_raster_dimensions(svg, size=size)
            data = cairosvg.svg2png(bytestring=svg.encode(), output_width=out_w, output_height=out_h)
        else:
            data = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size, dpi=dpi)

    with open(output, "wb") as f:
        f.write(data)


def svg_to_pdf(svg: str, output: str) -> None:
    """Convert SVG string to PDF file via cairosvg."""
    import cairosvg

    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=output)


def svg_to_tiff(svg: str, output: str, *, size: int = 800, dpi: int = 300, fit_viewbox: bool = False) -> None:
    """Render SVG at target DPI and save as TIFF.

    Rasterises via resvg-py (preferred) or cairosvg, then converts the
    lossless PNG bytes to LZW-compressed TIFF via Pillow.
    """
    from io import BytesIO

    from PIL import Image

    if _has_resvg():
        from resvg_py import svg_to_bytes

        zoom = max(1, round(dpi / _SVG_BASE_DPI))
        if fit_viewbox:
            out_w, out_h = svg_raster_dimensions(svg, size=size)
            png_data = svg_to_bytes(svg_string=svg, width=out_w, height=out_h)
        else:
            png_data = svg_to_bytes(svg_string=svg, zoom=zoom)
    else:
        import cairosvg

        if fit_viewbox:
            out_w, out_h = svg_raster_dimensions(svg, size=size)
            png_data = cairosvg.svg2png(bytestring=svg.encode(), output_width=out_w, output_height=out_h)
        else:
            png_data = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size, dpi=dpi)

    img = Image.open(BytesIO(png_data))
    img.save(output, format="TIFF", dpi=(dpi, dpi), compression="tiff_deflate")
