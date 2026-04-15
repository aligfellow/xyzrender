"""Colormap utilities for scalar color legends and ``--cmap`` atom coloring."""

from __future__ import annotations

import numpy as np

from xyzrender.colors import PALETTES, Color, palette_color


def build_palette_lut(palette: str, size: int = 256) -> np.ndarray:
    """Return an RGB LUT sampled from a named palette."""
    lut = np.zeros((size, 3), dtype=np.uint8)
    scale = max(size - 1, 1)
    for i in range(size):
        c = palette_color(palette, i / scale)
        lut[i] = (c.r, c.g, c.b)
    return lut


# ---------------------------------------------------------------------------
# Atom color list
# ---------------------------------------------------------------------------


def atom_colors(
    atom_cmap: dict[int, float],
    n: int,
    palette: str,
    vmin: float,
    vmax: float,
    unlabeled_hex: str,
) -> list[Color]:
    """Return per-atom Color list; atoms absent from atom_cmap get unlabeled_hex."""
    unlabeled = Color.from_hex(unlabeled_hex)
    vrange = max(vmax - vmin, 1e-10)
    return [
        palette_color(palette, (atom_cmap[ai] - vmin) / vrange) if ai in atom_cmap else unlabeled for ai in range(n)
    ]


# ---------------------------------------------------------------------------
# Colorbar SVG
# ---------------------------------------------------------------------------

_BAR_W = 30.0
_MARGIN = 16.0
_TICK_GAP = 16.0


def colorbar_extra_width(vmin: float, vmax: float, fs: float) -> int:
    """Extra SVG canvas width needed to fit the colorbar + labels."""
    fs = min(fs, 40.0)
    char_w = fs * 0.62
    mid = (vmin + vmax) / 2
    max_int_chars = max(len(f"{v:.3f}".replace("-", "\u2212").split(".")[0]) for v in (vmin, mid, vmax))
    return int(_MARGIN + _BAR_W + _TICK_GAP + 3 + (max_int_chars + 4) * char_w + 10)


def colorbar_svg(
    vmin: float,
    vmax: float,
    palette: str,
    mol_canvas_w: float,
    canvas_h: float,
    font_size: float,
    label_color: str,
) -> list[str]:
    """Return SVG element strings for a vertical colorbar to the right of the molecule."""
    stops = PALETTES[palette]

    bar_x = mol_canvas_w + _MARGIN
    bar_h = max(min(canvas_h * 0.80, 400.0), 60.0)
    bar_top = (canvas_h - bar_h) / 2
    bar_bot = bar_top + bar_h

    # Gradient: top = vmax, bottom = vmin.
    n = len(stops)
    grad_stops = "".join(
        f'<stop offset="{int(i / (n - 1) * 100)}%" stop-color="{c.hex}"/>' for i, c in enumerate(reversed(stops))
    )
    elems = [
        f'  <defs><linearGradient id="_cbg" x1="0" y1="0" x2="0" y2="1">{grad_stops}</linearGradient></defs>',
        f'  <rect x="{bar_x:.1f}" y="{bar_top:.1f}" width="{_BAR_W:.1f}" height="{bar_h:.1f}" '
        f'fill="url(#_cbg)" stroke="{label_color}" stroke-width="5"/>',
    ]

    tick_x1 = bar_x + _BAR_W
    label_x = tick_x1 + _TICK_GAP + 3
    fs = min(font_size, 40.0)
    char_w = fs * 0.62

    ticks = [
        (bar_top, vmax),
        ((bar_top + bar_bot) / 2, (vmin + vmax) / 2),
        (bar_bot, vmin),
    ]
    max_int_chars = max(len(f"{val:.3f}".replace("-", "\u2212").split(".")[0]) for _, val in ticks)
    decimal_x = label_x + max_int_chars * char_w
    text_attrs = f'font-family="monospace" font-size="{fs:.1f}px" fill="{label_color}" dominant-baseline="central"'

    for ty, val in ticks:
        s = f"{val:.3f}".replace("-", "\u2212")
        int_part, frac_part = s.split(".", 1)
        elems.append(
            f'  <line x1="{tick_x1:.1f}" y1="{ty:.1f}" x2="{tick_x1 + _TICK_GAP:.1f}" y2="{ty:.1f}" '
            f'stroke="{label_color}" stroke-width="5"/>'
        )
        elems.append(f'  <text x="{decimal_x:.1f}" y="{ty:.1f}" {text_attrs} text-anchor="end">{int_part}</text>')
        elems.append(f'  <text x="{decimal_x:.1f}" y="{ty:.1f}" {text_attrs} text-anchor="start">.{frac_part}</text>')

    return elems
