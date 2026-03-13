"""Skeletal formula rendering helpers.

Provides bond and atom drawing functions for skeletal (line-diagram) style.
Called from ``renderer.render_svg`` when ``cfg.skeletal_style`` is enabled.
"""

from __future__ import annotations

import numpy as np

from xyzrender.renderer import BondStyle, _proj, _ring_side, blend_fog

# ---------------------------------------------------------------------------
# Bond drawing
# ---------------------------------------------------------------------------


def skeletal_bond_radii(
    ai: int,
    aj: int,
    symbols: list[str],
    radii: np.ndarray,
    fs_label: float,
    scale: float,
) -> tuple[float, float]:
    """Return effective radii for bond endpoints in skeletal mode.

    Carbon atoms get radius 0 (bonds meet at the vertex); non-carbon atoms
    get a margin derived from the label font size so bonds don't overlap text.
    """
    margin_3d = (fs_label * 0.7) / max(scale, 1e-6)
    ri = 0.0 if symbols[ai] == "C" else max(radii[ai], margin_3d)
    rj = 0.0 if symbols[aj] == "C" else max(radii[aj], margin_3d)
    return ri, rj


def skeletal_bond_svg(
    svg: list[str],
    ai: int,
    aj: int,
    bo: float,
    style: BondStyle,
    opacity: float,
    *,
    pos: np.ndarray,
    symbols: list[str],
    radii: np.ndarray,
    bw: float,
    gap: float,
    fs_label: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
    fog_f: np.ndarray,
    fog_rgb: np.ndarray,
    fog_enabled: bool,
    bond_color: str,
    color_override: str | None,
    aromatic_rings: list[set[int]],
) -> None:
    """Append SVG elements for a single bond in skeletal style."""
    rij = pos[aj] - pos[ai]
    dist = np.linalg.norm(rij)
    if dist < 1e-6:
        return
    d = rij / dist

    ri, rj = skeletal_bond_radii(ai, aj, symbols, radii, fs_label, scale)

    start = pos[ai] + d * ri * 0.9
    end = pos[aj] - d * rj * 0.9
    if np.dot(end - start, d) <= 0:
        return

    x1, y1 = _proj(start, scale, cx, cy, canvas_w, canvas_h)
    x2, y2 = _proj(end, scale, cx, cy, canvas_w, canvas_h)
    dx, dy = x2 - x1, y2 - y1
    ln = (dx * dx + dy * dy) ** 0.5
    if ln < 1:
        return
    px, py = -dy / ln, dx / ln

    color = color_override if color_override is not None else bond_color
    if fog_enabled:
        avg_fog = (fog_f[ai] + fog_f[aj]) / 2 * 0.75
        color = blend_fog(color, fog_rgb, avg_fog)

    op_attr = f' opacity="{opacity:.2f}"' if opacity < 1.0 else ""

    # Skeletal base width
    _bw = bw * 0.6

    # TS/NCI dashed/dotted bonds
    if style == BondStyle.DASHED:
        d_dash, g = _bw * 1.2, _bw * 2.2
        w = _bw * 1.2
        svg.append(
            f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round" '
            f'stroke-dasharray="{d_dash:.1f},{g:.1f}"{op_attr}/>'
        )
        return
    if style == BondStyle.DOTTED:
        d_dot, g = _bw * 0.08, _bw * 2
        svg.append(
            f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{_bw:.1f}" stroke-linecap="round" '
            f'stroke-dasharray="{d_dot:.1f},{g:.1f}"{op_attr}/>'
        )
        return

    is_aromatic = 1.3 < bo < 1.7
    if is_aromatic:
        side = _ring_side(pos, ai, aj, aromatic_rings, x1, y1, x2, y2, px, py, scale, cx, cy, canvas_w, canvas_h)
        w = _bw
        # Solid component: centre line
        svg.append(
            f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round"{op_attr}/>'
        )
        # Dashed component: offset toward ring centre, trimmed at ends
        vx, vy = dx / ln, dy / ln
        trim = min(ln * 0.2, w * 2.5)
        dx_d, dy_d = vx * trim, vy * trim
        x1d, y1d = x1 + dx_d, y1 + dy_d
        x2d, y2d = x2 - dx_d, y2 - dy_d
        ox, oy = px * 2 * gap * side, py * 2 * gap * side
        d_dash, g_dash = w * 1.0, w * 2.0
        svg.append(
            f'  <line x1="{x1d + ox:.1f}" y1="{y1d + oy:.1f}" x2="{x2d + ox:.1f}" y2="{y2d + oy:.1f}" '
            f'stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round" '
            f'stroke-dasharray="{d_dash:.1f},{g_dash:.1f}"{op_attr}/>'
        )
    else:
        nb = max(1, round(bo))
        w = _bw
        for ib in range(-nb + 1, nb, 2):
            ox, oy = px * ib * gap, py * ib * gap
            svg.append(
                f'  <line x1="{x1 + ox:.1f}" y1="{y1 + oy:.1f}" x2="{x2 + ox:.1f}" y2="{y2 + oy:.1f}" '
                f'stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round"{op_attr}/>'
            )


# ---------------------------------------------------------------------------
# Atom drawing
# ---------------------------------------------------------------------------


def skeletal_atom_svg(
    svg: list[str],
    ai: int,
    xi: float,
    yi: float,
    *,
    symbols: list[str],
    colors: list,
    fs_label: float,
    fog_enabled: bool,
    fog_rgb: np.ndarray,
    fog_f: np.ndarray,
    label_color_override: str | None,
) -> None:
    """Append SVG text label for a non-carbon atom in skeletal style.

    Carbon atoms are not labelled (implicit vertices). If
    *label_color_override* is set, all labels use that colour; otherwise
    per-element CPK colours are used (with optional fog blending).
    """
    sym = symbols[ai]
    if sym == "C":
        return

    if label_color_override:
        fill = label_color_override
    else:
        fill = colors[ai].hex
        if fog_enabled:
            fill = blend_fog(fill, fog_rgb, fog_f[ai])

    svg.append(
        f'  <text x="{xi:.1f}" y="{yi:.1f}" '
        f'font-family="Helvetica,Arial,sans-serif" '
        f'font-size="{fs_label:.1f}px" font-weight="bold" '
        f'text-anchor="middle" dominant-baseline="central" '
        f'fill="{fill}">{sym}</text>'
    )
