"""SVG renderer for molecular structures."""

from __future__ import annotations

import logging

import numpy as np
from xyzgraph import DATA

from xyzrender.colors import _FOG_NEAR, WHITE, blend_fog, get_color, get_gradient_colors
from xyzrender.types import BondStyle, RenderConfig

logger = logging.getLogger(__name__)

_RADIUS_SCALE = 0.075  # VdW → display radius
_REF_SPAN = 6.0  # reference molecular span (Å) for proportional bond/stroke scaling


def render_svg(graph, config: RenderConfig | None = None, *, _log: bool = True) -> str:
    """Render molecular graph to SVG string."""
    cfg = config or RenderConfig()
    n = graph.number_of_nodes()
    symbols = [graph.nodes[i]["symbol"] for i in range(n)]
    pos = np.array([graph.nodes[i]["position"] for i in range(n)], dtype=float)
    a_nums = [DATA.s2n[s] for s in symbols]

    if cfg.auto_orient and n > 1:
        # Collect TS bond pairs to prioritize in orientation
        ts_pairs = list(cfg.ts_bonds) if cfg.ts_bonds else []
        for i, j, d in graph.edges(data=True):
            if d.get("TS", False) or d.get("bond_type", "") == "TS":
                ts_pairs.append((i, j))
        pos = _pca_orient(pos, ts_pairs or None)

    raw_vdw = np.array([DATA.vdw.get(s, 1.5) * (0.6 if s == "H" else 1.0) for s in symbols])
    radii = raw_vdw * cfg.atom_scale * _RADIUS_SCALE

    # Use VdW radii for canvas fitting when VdW spheres are active
    if cfg.vdw_indices is not None:
        vdw_active = set(range(n)) if len(cfg.vdw_indices) == 0 else set(cfg.vdw_indices)
        fit_radii = np.array([raw_vdw[i] * cfg.vdw_scale if i in vdw_active else radii[i] for i in range(n)])
    else:
        fit_radii = radii
    scale, cx, cy, canvas_w, canvas_h = _fit_canvas(pos, fit_radii, cfg)

    # Scale bond width and stroke proportionally with zoom so ratios stay constant
    ref_scale = (cfg.canvas_size - 2 * cfg.padding) / _REF_SPAN
    scale_ratio = scale / ref_scale
    bw = cfg.bond_width * scale_ratio
    sw = cfg.atom_stroke_width * scale_ratio

    if _log:
        logger.debug(
            "Render: %d atoms, %d bonds, scale=%.2f, center=(%.2f, %.2f)", n, graph.number_of_edges(), scale, cx, cy
        )
    z_order = np.argsort(pos[:, 2])
    colors = [get_color(a, cfg.color_overrides) for a in a_nums]

    # Bond lookup: (bond_order, style)
    bonds: dict[tuple[int, int], tuple[float, BondStyle]] = {}
    for i, j, d in graph.edges(data=True):
        bo = d.get("bond_order", 1.0) if cfg.bond_orders else 1.0
        bt = d.get("bond_type", "")
        if bt == "TS" or d.get("TS", False):
            style = BondStyle.DASHED
        elif bt == "NCI" or d.get("NCI", False):
            style = BondStyle.DOTTED
        else:
            style = BondStyle.SOLID
        bonds[(i, j)] = bonds[(j, i)] = (bo, style)
    # Manual overrides (add or restyle)
    for i, j in cfg.ts_bonds:
        bonds[(i, j)] = bonds[(j, i)] = (bonds.get((i, j), (1.0, BondStyle.SOLID))[0], BondStyle.DASHED)
    for i, j in cfg.nci_bonds:
        bonds[(i, j)] = bonds[(j, i)] = (bonds.get((i, j), (1.0, BondStyle.SOLID))[0], BondStyle.DOTTED)

    # Only hide C-H hydrogens (not O-H, N-H, etc.)
    hidden = set()
    if cfg.hide_h:
        show = set(cfg.show_h_indices)
        for ai in range(n):
            if symbols[ai] == "H" and ai not in show:
                if all(symbols[nb] == "C" for nb in graph.neighbors(ai)):
                    hidden.add(ai)

    aromatic_rings = [set(r) for r in graph.graph.get("aromatic_rings", [])]

    # Ensure all aromatic bonds are covered by ring data — auto-detect missing rings
    aromatic_ring_edges = set()
    for ring in aromatic_rings:
        rl = list(ring)
        for ii in range(len(rl)):
            for jj in range(ii + 1, len(rl)):
                if (rl[ii], rl[jj]) in bonds or (rl[jj], rl[ii]) in bonds:
                    aromatic_ring_edges.add((min(rl[ii], rl[jj]), max(rl[ii], rl[jj])))
    missing = False
    for (i, j), (bo, _style) in bonds.items():
        if i < j and 1.3 < bo < 1.7 and (i, j) not in aromatic_ring_edges:
            missing = True
            break
    if missing:
        import networkx as nx

        arom_g = nx.Graph()
        for (i, j), (bo, _style) in bonds.items():
            if i < j and 1.3 < bo < 1.7:
                arom_g.add_edge(i, j)
        if arom_g.number_of_edges() > 0:
            aromatic_rings = [set(c) for c in nx.minimum_cycle_basis(arom_g)]

    # Fog factors — normalized across depth range, with a dead-zone near the front
    fog_f = np.zeros(n)
    fog_rgb = np.array([255, 255, 255])
    if cfg.fog:
        zr = max(pos[:, 2].max() - pos[:, 2].min(), 1e-6)
        depth = pos[:, 2].max() - pos[:, 2]  # distance from front atom
        fog_f = cfg.fog_strength * np.clip((depth - _FOG_NEAR) / zr, 0.0, 1.0)

    # --- Build SVG ---
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{canvas_w}" height="{canvas_h}">'
    ]
    svg.append(f'  <rect width="100%" height="100%" fill="{cfg.background}"/>')

    use_grad = cfg.gradient
    if use_grad:
        svg.append("  <defs>")
        if cfg.fog:
            # Per-atom gradients: fog at half strength preserves gradient pop
            for ai in range(n):
                if ai in hidden:
                    continue
                t = min(fog_f[ai] ** 2 * 0.7, 0.70)
                hi, lo = get_gradient_colors(colors[ai], cfg.gradient_strength)
                hi, lo = hi.blend(WHITE, t), lo.blend(WHITE, t)
                fs = blend_fog(cfg.atom_stroke_color, fog_rgb, fog_f[ai])
                r = radii[ai] * scale
                sa = f' stroke="{fs}" stroke-width="{sw:.1f}"'
                svg.append(
                    f'    <g id="a{ai}"><radialGradient id="g{ai}" cx=".5" cy=".5" fx=".33" fy=".33" r=".66">'
                    f'<stop offset="0%" stop-color="{hi.hex}"/><stop offset="100%" stop-color="{lo.hex}"/>'
                    f'</radialGradient><circle cx="0" cy="0" r="{r:.1f}" fill="url(#g{ai})"{sa}/></g>'
                )
        else:
            seen = set()
            for ai in range(n):
                an = a_nums[ai]
                if an in seen or ai in hidden:
                    continue
                seen.add(an)
                hi, lo = get_gradient_colors(colors[ai], cfg.gradient_strength)
                r = radii[ai] * scale
                sa = f' stroke="{cfg.atom_stroke_color}" stroke-width="{sw:.1f}"'
                svg.append(
                    f'    <g id="a{an}"><radialGradient id="g{an}" cx=".5" cy=".5" fx=".33" fy=".33" r=".66">'
                    f'<stop offset="0%" stop-color="{hi.hex}"/><stop offset="100%" stop-color="{lo.hex}"/>'
                    f'</radialGradient><circle cx="0" cy="0" r="{r:.1f}" fill="url(#g{an})"{sa}/></g>'
                )
        svg.append("  </defs>")

    # VdW surface defs
    vdw_set = None
    if cfg.vdw_indices is not None:
        vdw_set = set(range(n)) if len(cfg.vdw_indices) == 0 else set(cfg.vdw_indices)
        svg.append("  <defs>")
        seen_vdw = set()
        for ai in z_order:
            if ai not in vdw_set:
                continue
            an = a_nums[ai]
            if an not in seen_vdw:
                seen_vdw.add(an)
                hi = colors[ai]  # true atom color at center
                lo = colors[ai].darken(0.845 * cfg.vdw_gradient_strength)
                svg.append(
                    f'    <radialGradient id="vg{an}" cx=".5" cy=".5" fx=".33" fy=".33" r=".66">'
                    f'<stop offset="0%" stop-color="{hi.hex}"/><stop offset="100%" stop-color="{lo.hex}"/>'
                    f"</radialGradient>"
                )
        svg.append("  </defs>")

    # Interleaved z-order: for each atom, render it then its bonds to deeper atoms
    gap = cfg.bond_gap * bw  # pixel gap scales with bond width

    def add_bond(ai, aj, bo, style):
        """Render bond — closure captures shared rendering state."""
        rij = pos[aj] - pos[ai]
        dist = np.linalg.norm(rij)
        if dist < 1e-6:
            return
        d = rij / dist

        start = pos[ai] + d * radii[ai] * 0.9
        end = pos[aj] - d * radii[aj] * 0.9
        if np.dot(end - start, d) <= 0:
            return

        x1, y1 = _proj(start, scale, cx, cy, canvas_w, canvas_h)
        x2, y2 = _proj(end, scale, cx, cy, canvas_w, canvas_h)
        dx, dy = x2 - x1, y2 - y1
        ln = (dx * dx + dy * dy) ** 0.5
        if ln < 1:
            return
        px, py = -dy / ln, dx / ln

        color = cfg.bond_color
        if cfg.fog:
            avg_fog = (fog_f[ai] + fog_f[aj]) / 2 * 0.75  # bonds fog less than atoms
            color = blend_fog(color, fog_rgb, avg_fog)

        # TS/NCI override: single line with dash pattern (scaled to bond width)
        if style == BondStyle.DASHED:
            d, g = bw * 1.2, bw * 2.2
            w = bw * 1.2
            svg.append(
                f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round" '
                f'stroke-dasharray="{d:.1f},{g:.1f}"/>'
            )
            return
        if style == BondStyle.DOTTED:
            d, g = bw * 0.1, bw * 2.0
            svg.append(
                f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-width="{bw:.1f}" stroke-linecap="round" stroke-dasharray="{d:.1f},{g:.1f}"/>'
            )
            return

        is_aromatic = 1.3 < bo < 1.7
        if is_aromatic:
            # Solid + dashed parallel lines, dashed toward ring center
            side = _ring_side(pos, ai, aj, aromatic_rings, x1, y1, x2, y2, px, py, scale, cx, cy, canvas_w, canvas_h)
            w = bw * 0.7
            for ib in [-1, 1]:
                ox, oy = px * ib * gap, py * ib * gap
                dash = f' stroke-dasharray="{w * 1.0:.1f},{w * 2.0:.1f}"' if ib == side else ""
                svg.append(
                    f'  <line x1="{x1 + ox:.1f}" y1="{y1 + oy:.1f}" x2="{x2 + ox:.1f}" y2="{y2 + oy:.1f}" '
                    f'stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round"{dash}/>'
                )
        else:
            nb = max(1, round(bo))
            w = bw if nb == 1 else bw * 0.7
            for ib in range(-nb + 1, nb, 2):
                ox, oy = px * ib * gap, py * ib * gap
                svg.append(
                    f'  <line x1="{x1 + ox:.1f}" y1="{y1 + oy:.1f}" x2="{x2 + ox:.1f}" y2="{y2 + oy:.1f}" '
                    f'stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round"/>'
                )

    for idx, ai in enumerate(z_order):
        if ai in hidden:
            continue
        xi, yi = _proj(pos[ai], scale, cx, cy, canvas_w, canvas_h)

        # Atom
        if use_grad:
            ref = f"#a{ai}" if cfg.fog else f"#a{a_nums[ai]}"
            svg.append(f'  <use x="{xi:.1f}" y="{yi:.1f}" xlink:href="{ref}"/>')
        else:
            fill, stroke = colors[ai].hex, cfg.atom_stroke_color
            if cfg.fog:
                fill = blend_fog(fill, fog_rgb, fog_f[ai])
                stroke = blend_fog(stroke, fog_rgb, fog_f[ai])
            svg.append(
                f'  <circle cx="{xi:.1f}" cy="{yi:.1f}" r="{radii[ai] * scale:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'
            )

        # Bonds to deeper atoms
        for aj in z_order[idx + 1 :]:
            if aj in hidden or (ai, aj) not in bonds:
                continue
            bo, style = bonds[(ai, aj)]
            add_bond(ai, aj, bo, style)

    # VdW surface overlay — on top of molecule, group opacity for proper occlusion
    if vdw_set is not None:
        svg.append(f'  <g opacity="{cfg.vdw_opacity}">')
        for ai in z_order:
            if ai in vdw_set:
                vr = raw_vdw[ai] * cfg.vdw_scale * scale
                xi, yi = _proj(pos[ai], scale, cx, cy, canvas_w, canvas_h)
                svg.append(f'    <circle cx="{xi:.1f}" cy="{yi:.1f}" r="{vr:.1f}" fill="url(#vg{a_nums[ai]})"/>')
        svg.append("  </g>")

    svg.append("</svg>")
    return "\n".join(svg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pca_orient(pos: np.ndarray, priority_pairs: list[tuple[int, int]] | None = None) -> np.ndarray:
    from xyzrender.utils import pca_orient

    return pca_orient(pos, priority_pairs=priority_pairs)


def _fit_canvas(pos, radii, cfg):
    """Scale + center so molecule fits canvas with tight aspect ratio."""
    pad = radii.max() if len(radii) else 0
    lo = pos[:, :2].min(axis=0) - pad
    hi = pos[:, :2].max(axis=0) + pad
    spans = hi - lo  # [x_span, y_span]
    if cfg.fixed_span is not None:
        max_span = cfg.fixed_span
    else:
        max_span = max(spans.max(), 1e-6)
    scale = (cfg.canvas_size - 2 * cfg.padding) / max_span
    if cfg.fixed_span is not None:
        # GIF mode: keep canvas square for consistent framing
        w = h = cfg.canvas_size
    else:
        # Static: crop to molecule aspect ratio
        w = int(spans[0] * scale + 2 * cfg.padding)
        h = int(spans[1] * scale + 2 * cfg.padding)
    if cfg.fixed_center is not None:
        return scale, cfg.fixed_center[0], cfg.fixed_center[1], w, h
    center = (lo + hi) / 2
    return scale, center[0], center[1], w, h


def _proj(p, scale, cx, cy, cw, ch):
    """3D position → 2D pixel coordinates (y-flipped for SVG)."""
    return cw / 2 + scale * (p[0] - cx), ch / 2 - scale * (p[1] - cy)


def _ring_side(pos, ai, aj, aromatic_rings, x1, y1, x2, y2, px, py, scale, cx, cy, canvas_w, canvas_h):
    """Which perpendicular side (+1/-1) of the bond faces the aromatic ring center."""
    for ring in aromatic_rings:
        if ai in ring and aj in ring:
            centroid = pos[list(ring)].mean(axis=0)
            rcx, rcy = _proj(centroid, scale, cx, cy, canvas_w, canvas_h)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            return 1 if px * (rcx - mx) + py * (rcy - my) > 0 else -1
    return 1
