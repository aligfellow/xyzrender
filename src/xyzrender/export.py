"""Export SVG to raster/vector formats.

PNG rendering prefers **resvg-py** — it supports SVG filter primitives
(``feGaussianBlur``, ``feTurbulence``, etc.) that cairosvg silently ignores.
PDF rendering defaults to cairosvg (vector output). For filtered SVGs (for
example ``--dof``), a PNG->PDF fallback is available to preserve filters.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO

_SVG_BASE_DPI = 96  # CSS/SVG spec: 1px = 1/96 inch


@lru_cache(maxsize=1)
def _has_resvg() -> bool:
    try:
        from resvg_py import svg_to_bytes  # noqa: F401
    except ImportError:
        return False
    return True


def svg_to_png_bytes(svg: str, *, size: int = 800) -> bytes:
    """Convert SVG string to PNG bytes at a fixed pixel size.

    Used by GIF frame rendering where exact pixel dimensions matter.
    """
    if _has_resvg():
        from resvg_py import svg_to_bytes

        return svg_to_bytes(svg_string=svg, width=size, height=size)

    import cairosvg

    return cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)


def svg_to_png(svg: str, output: str, *, size: int = 800, dpi: int = 300) -> None:
    """Convert SVG string to a PNG file.

    Higher *dpi* produces a larger image with more detail (dpi/96 zoom factor).
    """
    if _has_resvg():
        from resvg_py import svg_to_bytes

        zoom = max(1, round(dpi / _SVG_BASE_DPI))
        data = svg_to_bytes(svg_string=svg, zoom=zoom)
    else:
        import cairosvg

        data = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size, dpi=dpi)

    with open(output, "wb") as f:
        f.write(data)


def svg_to_pdf(
    svg: str, output: str, *, preserve_filters: bool = False, size: int = 800, dpi: int = 300
) -> None:
    """Convert SVG string to a PDF file.

    Parameters
    ----------
    svg:
        SVG source.
    output:
        Output PDF path.
    preserve_filters:
        If ``True``, rasterize SVG to PNG first (via :func:`svg_to_png`) and
        then embed into PDF so SVG filter effects (e.g. Gaussian blur) are kept.
        This produces a raster PDF page instead of vector graphics.
    size:
        Canvas size in pixels for the raster fallback.
    dpi:
        Rasterization resolution for the fallback path.
    """
    if not preserve_filters:
        import cairosvg

        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=output)
        return

    # Preserve SVG filters by rasterizing first, then writing a single-page PDF.
    if _has_resvg():
        from resvg_py import svg_to_bytes

        zoom = max(1, round(dpi / _SVG_BASE_DPI))
        png_data = svg_to_bytes(svg_string=svg, zoom=zoom)
    else:
        import cairosvg

        png_data = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size, dpi=dpi)
    from PIL import Image

    with Image.open(BytesIO(png_data)) as im:
        if "A" in im.getbands():
            bg = Image.new("RGB", im.size, "white")
            bg.paste(im, mask=im.getchannel("A"))
            rgb = bg
        else:
            rgb = im.convert("RGB")
        rgb.save(output, "PDF", resolution=float(dpi))
