"""Export SVG to raster/vector formats."""

from __future__ import annotations


def svg_to_png(svg: str, output: str, *, size: int = 800, dpi: int = 300) -> None:
    """Convert SVG string to PNG file via cairosvg."""
    try:
        import cairosvg
    except ImportError:
        msg = "PNG export requires cairosvg"
        raise ImportError(msg) from None

    cairosvg.svg2png(
        bytestring=svg.encode(),
        write_to=output,
        output_width=size,
        dpi=dpi,
    )


def svg_to_pdf(svg: str, output: str) -> None:
    """Convert SVG string to PDF file via cairosvg."""
    try:
        import cairosvg
    except ImportError:
        msg = "PDF export requires cairosvg"
        raise ImportError(msg) from None

    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=output)
