"""SVG renderer for molecular structures."""

from __future__ import annotations

import itertools
import logging

import numpy as np
from xyzgraph import DATA

from xyzrender.colors import _FOG_NEAR, WHITE, blend_fog, cmap_viridis, get_color, get_gradient_colors
from xyzrender.dens import dens_layers_svg
from xyzrender.mo import (
    classify_mo_lobes,
    mo_back_lobes_svg,
    mo_front_lobes_svg,
    mo_gradient_defs_svg,
)
from xyzrender.types import BondStyle, Color, RenderConfig

logger = logging.getLogger(__name__)

_render_counter = itertools.count()  # unique ID prefix per render call (SVG ids are global in Jupyter HTML)
_RADIUS_SCALE = 0.075  # VdW → display radius
_REF_SPAN = 6.0  # reference molecular span (Å) for proportional bond/stroke scaling
_REF_CANVAS = 800  # reference canvas size (px) — bond/label widths are defined at this size
_CENTROID_VDW = 0.5  # VdW radius (Å) for NCI pi-system centroid dummy nodes
_H_ATOM_SCALE = 0.6  # display-radius shrink factor for H atoms (ball-and-stick)
_H_VDW_SCALE = 0.8  # VdW-sphere shrink factor for H atoms


def render_svg(graph, config: RenderConfig | None = None, *, _log: bool = True) -> str:
    """Render molecular graph to SVG string."""
    cfg = config or RenderConfig()
    node_ids = list(graph.nodes())
    n = len(node_ids)
    symbols = [graph.nodes[i]["symbol"] for i in node_ids]
    pos = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
    a_nums = [DATA.s2n.get(s, 0) for s in symbols]  # 0 for NCI centroid nodes ("*")

    # Pre-compute local vector origins/directions so we can rotate them with auto_orient
    _vec_origins = np.array([va.origin for va in cfg.vectors], dtype=float) if cfg.vectors else np.full((0, 3), np.nan)
    _vec_dirs = np.array([va.vector for va in cfg.vectors], dtype=float) if cfg.vectors else np.full((0, 3), np.nan)

    if cfg.auto_orient and n > 1:
        # Collect TS bond pairs to prioritize in orientation
        ts_pairs = list(cfg.ts_bonds) if cfg.ts_bonds else []
        for i, j, d in graph.edges(data=True):
            if d.get("TS", False) or d.get("bond_type", "") == "TS":
                ts_pairs.append((i, j))
        # Exclude NCI centroid dummy nodes from PCA fitting
        atom_mask = np.array([s != "*" for s in symbols])
        fit_mask = atom_mask if not atom_mask.all() else None
        from xyzrender.utils import pca_orient

        if _vec_origins is not None:
            # Capture rotation matrix so vector origins/directions transform with the molecule
            _fit = pos[fit_mask] if fit_mask is not None else pos
            _centroid = _fit.mean(axis=0)
            pos, _orient_rot = pca_orient(pos, ts_pairs or None, fit_mask=fit_mask, return_matrix=True)
            _vec_origins = (_vec_origins - _centroid) @ _orient_rot.T
            _vec_dirs = _vec_dirs @ _orient_rot.T
            logger.debug("render_svg PCA centroid: %s", _centroid)
            for _vi, _vo in enumerate(_vec_origins):
                logger.debug("  vector[%d] origin after PCA: %s (should be ~0 for COM origins)", _vi, _vo)
            if cfg.cell_data is not None:
                cfg.cell_data.lattice = (_orient_rot @ cfg.cell_data.lattice.T).T
                cfg.cell_data.cell_origin = _orient_rot @ (cfg.cell_data.cell_origin - _centroid)
        elif cfg.cell_data is not None:
            pre_centroid = pos.mean(axis=0)
            pos, _rot_mat = pca_orient(pos, ts_pairs, fit_mask=fit_mask, return_matrix=True)
            cfg.cell_data.lattice = (_rot_mat @ cfg.cell_data.lattice.T).T
            cfg.cell_data.cell_origin = _rot_mat @ (cfg.cell_data.cell_origin - pre_centroid)
        else:
            pos = pca_orient(pos, ts_pairs or None, fit_mask=fit_mask)

    raw_vdw = np.array(
        [_CENTROID_VDW if s == "*" else DATA.vdw.get(s, 1.5) * (_H_ATOM_SCALE if s == "H" else 1.0) for s in symbols]
    )
    radii = raw_vdw * cfg.atom_scale * _RADIUS_SCALE

    # VdW sphere radii use a separate (larger) H scaling
    raw_vdw_sphere = np.array(
        [_CENTROID_VDW if s == "*" else DATA.vdw.get(s, 1.5) * (_H_VDW_SCALE if s == "H" else 1.0) for s in symbols]
    )

    # Use VdW radii for canvas fitting when VdW spheres are active
    if cfg.vdw_indices is not None:
        vdw_active = set(range(n)) if len(cfg.vdw_indices) == 0 else set(cfg.vdw_indices)
        fit_radii = np.array([raw_vdw_sphere[i] * cfg.vdw_scale if i in vdw_active else radii[i] for i in range(n)])
    else:
        fit_radii = radii

    # Expand canvas for surface bounds (MO / density / ESP are mutually exclusive)
    extra_lo = extra_hi = None
    if cfg.mo_contours is not None:
        mo = cfg.mo_contours
        if mo.lobe_x_min is not None:
            extra_lo = np.array([mo.lobe_x_min, mo.lobe_y_min])
            extra_hi = np.array([mo.lobe_x_max, mo.lobe_y_max])
    elif cfg.dens_contours is not None:
        extra_lo = np.array([cfg.dens_contours.x_min, cfg.dens_contours.y_min])
        extra_hi = np.array([cfg.dens_contours.x_max, cfg.dens_contours.y_max])
    elif cfg.nci_contours is not None:
        extra_lo = np.array([cfg.nci_contours.x_min, cfg.nci_contours.y_min])
        extra_hi = np.array([cfg.nci_contours.x_max, cfg.nci_contours.y_max])
    if cfg.esp_surface is not None:
        extra_lo = np.array([cfg.esp_surface.x_min, cfg.esp_surface.y_min])
        extra_hi = np.array([cfg.esp_surface.x_max, cfg.esp_surface.y_max])
    # Expand canvas to encompass the unit cell box when crystal mode is active
    if cfg.cell_data is not None and cfg.show_cell:
        lat = cfg.cell_data.lattice
        a_vec, b_vec, c_vec = lat[0], lat[1], lat[2]
        orig3d = cfg.cell_data.cell_origin
        box_verts = np.array(
            [orig3d + i * a_vec + j * b_vec + k * c_vec for i, j, k in itertools.product((0, 1), repeat=3)]
        )
        box_lo = box_verts[:, :2].min(axis=0)
        box_hi = box_verts[:, :2].max(axis=0)
        extra_lo = np.minimum(extra_lo, box_lo) if extra_lo is not None else box_lo
        extra_hi = np.maximum(extra_hi, box_hi) if extra_hi is not None else box_hi
    # Expand canvas to encompass vector arrow tips, tails, and labels
    if cfg.vectors:
        _ref_px_per_ang = (_REF_CANVAS - 2 * cfg.padding) / _REF_SPAN
        _vec_tips = []
        for vi, va in enumerate(cfg.vectors):
            scaled_vec = _vec_dirs[vi] * va.scale * cfg.vector_scale
            tail3d = _vec_origins[vi] - scaled_vec / 2 if va.anchor == "center" else _vec_origins[vi]
            tip3d = tail3d + scaled_vec
            _vec_tips.append(tip3d)
            for pt in (tail3d, tip3d):
                pt2d = pt[:2]
                extra_lo = np.minimum(extra_lo, pt2d) if extra_lo is not None else pt2d.copy()
                extra_hi = np.maximum(extra_hi, pt2d) if extra_hi is not None else pt2d.copy()
        for vi, va in enumerate(cfg.vectors):
            if not va.label:
                continue
            tip2d = _vec_tips[vi][:2]
            label_half_w = len(va.label) * cfg.label_font_size * 1.2 * 0.35 / _ref_px_per_ang
            label_h = cfg.label_font_size * 1.2 / _ref_px_per_ang
            lo = tip2d - np.array([label_half_w, label_h])
            hi = tip2d + np.array([label_half_w, label_h])
            extra_lo = np.minimum(extra_lo, lo) if extra_lo is not None else lo
            extra_hi = np.maximum(extra_hi, hi) if extra_hi is not None else hi
    scale, cx, cy, canvas_w, canvas_h = _fit_canvas(pos, fit_radii, cfg, extra_lo=extra_lo, extra_hi=extra_hi)

    # scale_ratio: encodes both molecule complexity AND canvas size so that
    # bond/label widths defined at _REF_CANVAS grow proportionally on larger canvases.
    ref_scale = (_REF_CANVAS - 2 * cfg.padding) / _REF_SPAN
    scale_ratio = scale / ref_scale
    bw = cfg.bond_width * scale_ratio
    sw = cfg.atom_stroke_width * scale_ratio
    fs_label = cfg.label_font_size * scale_ratio

    if _log:
        logger.debug(
            "Render: %d atoms, %d bonds, scale=%.2f, center=(%.2f, %.2f)", n, graph.number_of_edges(), scale, cx, cy
        )
    z_order = np.argsort(pos[:, 2])

    # Atom base colors — CPK by default, Viridis cmap when --cmap is active
    if cfg.atom_cmap is not None:
        cmap_vals = cfg.atom_cmap
        if cfg.cmap_range is not None:
            vmin, vmax = cfg.cmap_range
        else:
            vmin = min(cmap_vals.values())
            vmax = max(cmap_vals.values())
        vrange = max(vmax - vmin, 1e-10)
        unlabeled = Color.from_hex(cfg.cmap_unlabeled)
        colors = [cmap_viridis((cmap_vals[ai] - vmin) / vrange) if ai in cmap_vals else unlabeled for ai in range(n)]
    else:
        colors = [get_color(a, cfg.color_overrides) for a in a_nums]

    # Override atom colors for overlay (mol2) atoms — must happen before gradient defs
    has_overlay = any(graph.nodes[nid].get("overlay", False) for nid in node_ids)
    if has_overlay:
        overlay_atom_color = Color.from_str(cfg.overlay_color)
        for ai in range(n):
            if graph.nodes[node_ids[ai]].get("overlay", False):
                colors[ai] = overlay_atom_color

    # Bond lookup: (bond_order, style, color_override)
    bonds: dict[tuple[int, int], tuple[float, BondStyle, str | None]] = {}
    for i, j, d in graph.edges(data=True):
        bo = d.get("bond_order", 1.0) if cfg.bond_orders else 1.0
        bt = d.get("bond_type", "")
        if bt == "TS" or d.get("TS", False):
            style = BondStyle.DASHED
        elif bt == "NCI" or d.get("NCI", False):
            style = BondStyle.DOTTED
        else:
            style = BondStyle.SOLID
        color_ov: str | None = d.get("bond_color_override")
        bonds[(i, j)] = bonds[(j, i)] = (bo, style, color_ov)
    # Manual overrides (add or restyle)
    for i, j in cfg.ts_bonds:
        existing = bonds.get((i, j), (1.0, BondStyle.SOLID, None))
        bonds[(i, j)] = bonds[(j, i)] = (existing[0], BondStyle.DASHED, existing[2])
    for i, j in cfg.nci_bonds:
        existing = bonds.get((i, j), (1.0, BondStyle.SOLID, None))
        bonds[(i, j)] = bonds[(j, i)] = (existing[0], BondStyle.DOTTED, existing[2])

    # Only hide C-H hydrogens (not O-H, N-H, free H, etc.)
    hidden = set()
    if cfg.hide_h:
        show = set(cfg.show_h_indices)
        for ai in range(n):
            if symbols[ai] == "H" and ai not in show and not graph.nodes[ai].get("image", False):
                neighbours = list(graph.neighbors(ai))
                if neighbours and all(symbols[nb] == "C" for nb in neighbours):
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
    for (i, j), (bo, _style, _col) in bonds.items():
        if i < j and 1.3 < bo < 1.7 and (i, j) not in aromatic_ring_edges:
            missing = True
            break
    if missing:
        import networkx as nx

        arom_g = nx.Graph()
        for (i, j), (bo, _style, _col) in bonds.items():
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
        f'viewBox="0 0 {canvas_w} {canvas_h}" width="{canvas_w}" height="{canvas_h}"'
        + (' style="background:transparent"' if cfg.transparent else "")
        + ">"
    ]
    if not cfg.transparent:
        svg.append(f'  <rect width="100%" height="100%" fill="{cfg.background}"/>')

    use_grad = cfg.gradient
    # Cmap/fog/overlay all require per-atom gradient defs (each atom may have a distinct colour)
    use_per_atom_grad = cfg.fog or cfg.atom_cmap is not None or has_overlay
    if use_grad:
        svg.append("  <defs>")
        if use_per_atom_grad:
            # Per-atom gradient defs
            for ai in range(n):
                if ai in hidden:
                    continue
                hi, lo = get_gradient_colors(colors[ai], cfg.gradient_strength)
                if cfg.fog:
                    t = min(fog_f[ai] ** 2 * 0.7, 0.70)
                    hi, lo = hi.blend(WHITE, t), lo.blend(WHITE, t)
                    fs = blend_fog(cfg.atom_stroke_color, fog_rgb, fog_f[ai])
                else:
                    fs = cfg.atom_stroke_color
                r = radii[ai] * scale
                sa = f' stroke="{fs}" stroke-width="{sw:.1f}"'
                svg.append(
                    f'    <g id="a{ai}"><radialGradient id="g{ai}" cx=".5" cy=".5" fx=".33" fy=".33" r=".66">'
                    f'<stop offset="0%" stop-color="{hi.hex}"/><stop offset="100%" stop-color="{lo.hex}"/>'
                    f'</radialGradient><circle cx="0" cy="0" r="{r:.1f}" fill="url(#g{ai})"{sa}/></g>'
                )
        else:
            # Shared gradient defs keyed by atomic number (no fog, no cmap)
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

    # MO lobe gradient defs + front/back classification
    mo_is_front = None
    if cfg.mo_contours is not None:
        mo = cfg.mo_contours
        if cfg.flat_mo:
            mo_is_front = [True] * len(mo.lobes)
        else:
            mo_is_front = classify_mo_lobes(mo.lobes, float(pos[:, 2].mean()))
        svg.append("  <defs>")
        svg.extend(mo_gradient_defs_svg(mo))
        svg.append("  </defs>")

    # --- Back MO orbital lobes (behind molecule) — flat faded fill ---
    if cfg.mo_contours is not None:
        assert mo_is_front is not None
        svg.extend(
            mo_back_lobes_svg(cfg.mo_contours, mo_is_front, cfg.surface_opacity, scale, cx, cy, canvas_w, canvas_h)
        )

    # --- Vector arrows: prepare for z-interleaved drawing ---
    # Vectors are drawn just before the atom at their depth in the back-to-front
    # loop below, so each shaft is covered by its own atom while still being
    # occluded by any atoms closer to the viewer.
    # However, when an arrow points toward the viewer the tip or
    # tail may protrude in front of the host atom. To keep
    # those elements visible they are placed in ``_vec_front_heads`` / ``_vec_front_tails``
    # and redrawn on top of relevant atoms in a second pass below.
    _vec_lw = max(bw * 0.6, 1.5) if cfg.vectors else 0.0
    _fs_vec = fs_label * 1.2 if cfg.vectors else 0.0
    # Back-to-front order (ascending z, matching z_order convention)
    _pending_vecs = sorted(range(len(cfg.vectors)), key=lambda vi: _vec_origins[vi][2]) if cfg.vectors else []
    _pv_pos = 0  # pointer into _pending_vecs

    # Calculate whether a vector tip/tail protrudes beyond the atom sphere.
    _atom_r3d = raw_vdw * cfg.atom_scale * _RADIUS_SCALE  # shape (n,)

    # A vector endpoint "protrudes in front" when its z exceeds the z of the
    # nearest atom plus that atom's 3D radius.
    _vec_tip3d: list = []
    _vec_tail3d: list = []
    _vec_head_front: list[bool] = []
    _vec_tail_front: list[bool] = []
    if cfg.vectors:
        for vi in range(len(cfg.vectors)):
            va = cfg.vectors[vi]
            scaled_vec = _vec_dirs[vi] * va.scale * cfg.vector_scale
            if va.anchor == "center":
                tail3d = _vec_origins[vi] - scaled_vec / 2
            else:
                tail3d = _vec_origins[vi]
            tip3d = tail3d + scaled_vec
            _vec_tip3d.append(tip3d)
            _vec_tail3d.append(tail3d)
            # Resolve host atom: use the prescribed index when available (atom-index
            # origin from JSON), otherwise fall back to a nearest-neighbour search.
            if va.host_atom is not None:
                host_ai = va.host_atom
            else:
                host_ai = int(np.argmin(np.linalg.norm(pos - _vec_origins[vi], axis=1)))
            host_z = pos[host_ai][2]
            host_r = _atom_r3d[host_ai]
            # Tip protrudes in front when tip_z > host_z + host_r
            _vec_head_front.append(bool(tip3d[2] > host_z + host_r))
            # Tail protrudes in front when tail_z > host_z + host_r (rare but symmetric)
            _vec_tail_front.append(bool(tail3d[2] > host_z + host_r))

    def _draw_vector_shaft(vi: int) -> None:
        """Draw only the line shaft of vector vi (called before the host atom)."""
        tail3d = _vec_tail3d[vi]
        tip3d = _vec_tip3d[vi]
        ox, oy = _proj(tail3d, scale, cx, cy, canvas_w, canvas_h)
        tx, ty = _proj(tip3d, scale, cx, cy, canvas_w, canvas_h)
        color = cfg.vectors[vi].color
        dx, dy = tx - ox, ty - oy
        px_len = (dx * dx + dy * dy) ** 0.5
        # Stop shaft at arrowhead base so the round linecap doesn't poke through the head
        if px_len > 4:
            arr = max(_vec_lw * 3.5, 7.0)
            frac = max(0.0, 1.0 - arr / px_len)
            sx, sy = ox + dx * frac, oy + dy * frac
        else:
            sx, sy = tx, ty
        svg.append(
            f'  <line x1="{ox:.1f}" y1="{oy:.1f}" x2="{sx:.1f}" y2="{sy:.1f}" '
            f'stroke="{color}" stroke-width="{_vec_lw:.1f}" stroke-linecap="round"/>'
        )

    def _draw_vector_head(vi: int) -> None:
        """Draw the arrowhead polygon and label of vector vi."""
        tail3d = _vec_tail3d[vi]
        tip3d = _vec_tip3d[vi]
        ox, oy = _proj(tail3d, scale, cx, cy, canvas_w, canvas_h)
        tx, ty = _proj(tip3d, scale, cx, cy, canvas_w, canvas_h)
        color = cfg.vectors[vi].color
        dx, dy = tx - ox, ty - oy
        px_len = (dx * dx + dy * dy) ** 0.5
        if px_len > 4:
            nvx, nvy = dx / px_len, dy / px_len
            pvx, pvy = -nvy, nvx
            arr = max(_vec_lw * 3.5, 7.0)
            p1x = tx - nvx * arr + pvx * arr * 0.38
            p1y = ty - nvy * arr + pvy * arr * 0.38
            p2x = tx - nvx * arr - pvx * arr * 0.38
            p2y = ty - nvy * arr - pvy * arr * 0.38
            svg.append(
                f'  <polygon points="{tx:.1f},{ty:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" fill="{color}"/>'
            )
            lx = tx + nvx * (arr * 0.6 + _fs_vec * 0.5)
            ly = ty + nvy * (arr * 0.6 + _fs_vec * 0.5) + _fs_vec * 0.35
        else:
            lx, ly = tx + 4, ty
        va = cfg.vectors[vi]
        if va.label:
            svg.append(
                f'  <text x="{lx:.1f}" y="{ly:.1f}" font-size="{_fs_vec:.1f}" fill="{color}" '
                f'font-family="Arial,sans-serif" text-anchor="middle" font-weight="bold">{va.label}</text>'
            )

    def _draw_vector_arrow(vi: int) -> None:
        """Draw shaft + (if not front-protruding) head for vector vi."""
        _draw_vector_shaft(vi)
        if not _vec_head_front[vi]:
            _draw_vector_head(vi)

    # --- Unit cell box (12 edges, drawn before atoms so bonds/atoms render on top) ---
    if cfg.cell_data is not None and cfg.show_cell:
        lat = cfg.cell_data.lattice
        a_vec, b_vec, c_vec = lat[0], lat[1], lat[2]
        orig3d = cfg.cell_data.cell_origin
        # 8 vertices indexed by (i,j,k)
        verts: dict[tuple[int, int, int], tuple[float, float]] = {}
        for i, j, k in itertools.product((0, 1), repeat=3):
            p3d = orig3d + i * a_vec + j * b_vec + k * c_vec
            verts[(i, j, k)] = _proj(p3d, scale, cx, cy, canvas_w, canvas_h)
        # 12 edges: 4 along each axis direction
        cell_lw = cfg.cell_line_width * scale_ratio
        cell_dash = f"{cell_lw * 2.5:.1f},{cell_lw * 3.0:.1f}"
        svg.append("  <!-- cell box -->")
        # Edges along a (vary i, fix j,k)
        for j, k in itertools.product((0, 1), repeat=2):
            x1, y1 = verts[(0, j, k)]
            x2, y2 = verts[(1, j, k)]
            svg.append(
                f'  <line class="cell-edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{cfg.cell_color}" stroke-width="{cell_lw:.1f}" '
                f'stroke-dasharray="{cell_dash}" stroke-linecap="round"/>'
            )
        # Edges along b (vary j, fix i,k)
        for i, k in itertools.product((0, 1), repeat=2):
            x1, y1 = verts[(i, 0, k)]
            x2, y2 = verts[(i, 1, k)]
            svg.append(
                f'  <line class="cell-edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{cfg.cell_color}" stroke-width="{cell_lw:.1f}" '
                f'stroke-dasharray="{cell_dash}" stroke-linecap="round"/>'
            )
        # Edges along c (vary k, fix i,j)
        for i, j in itertools.product((0, 1), repeat=2):
            x1, y1 = verts[(i, j, 0)]
            x2, y2 = verts[(i, j, 1)]
            svg.append(
                f'  <line class="cell-edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{cfg.cell_color}" stroke-width="{cell_lw:.1f}" '
                f'stroke-dasharray="{cell_dash}" stroke-linecap="round"/>'
            )

    # NCI patches are z-sorted into the atom/bond loop so they appear at the correct
    # depth (in the interstitial space) rather than covering the whole molecule.
    nci_lobes_flat: list[tuple[float, list[str]]] = []
    nci_lobe_idx = 0
    if cfg.nci_contours is not None:
        from xyzrender.nci import nci_lobe_svg_items, nci_static_svg_defs

        if cfg.nci_contours.raster_png:
            svg.extend(nci_static_svg_defs(cfg.nci_contours, scale, cx, cy, canvas_w, canvas_h))
        nci_lobes_flat = nci_lobe_svg_items(cfg.nci_contours, cfg.surface_opacity, scale, cx, cy, canvas_w, canvas_h)

    def _drain_nci(next_z: float) -> None:
        nonlocal nci_lobe_idx
        while nci_lobe_idx < len(nci_lobes_flat) and nci_lobes_flat[nci_lobe_idx][0] < next_z:
            svg.extend(nci_lobes_flat[nci_lobe_idx][1])
            nci_lobe_idx += 1

    # Interleaved z-order: for each atom, render it then its bonds to deeper atoms
    gap = cfg.bond_gap * bw  # pixel gap scales with bond width

    def add_bond(ai, aj, bo, style, opacity: float = 1.0, color_override: str | None = None):
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

        color = color_override if color_override is not None else cfg.bond_color
        if cfg.fog:
            avg_fog = (fog_f[ai] + fog_f[aj]) / 2 * 0.75  # bonds fog less than atoms
            color = blend_fog(color, fog_rgb, avg_fog)

        op_attr = f' opacity="{opacity:.2f}"' if opacity < 1.0 else ""

        # TS/NCI override: single line with dash pattern (scaled to bond width)
        if style == BondStyle.DASHED:
            d, g = bw * 1.2, bw * 2.2
            w = bw * 1.2
            svg.append(
                f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round" '
                f'stroke-dasharray="{d:.1f},{g:.1f}"{op_attr}/>'
            )
            return
        if style == BondStyle.DOTTED:
            d, g = bw * 0.08, bw * 2
            svg.append(
                f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-width="{bw:.1f}" stroke-linecap="round" '
                f'stroke-dasharray="{d:.1f},{g:.1f}"{op_attr}/>'
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
                    f'stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round"{dash}{op_attr}/>'
                )
        else:
            nb = max(1, round(bo))
            w = bw if nb == 1 else bw * 0.7
            for ib in range(-nb + 1, nb, 2):
                ox, oy = px * ib * gap, py * ib * gap
                svg.append(
                    f'  <line x1="{x1 + ox:.1f}" y1="{y1 + oy:.1f}" x2="{x2 + ox:.1f}" y2="{y2 + oy:.1f}" '
                    f'stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round"{op_attr}/>'
                )

    for idx, ai in enumerate(z_order):
        # Flush all vectors whose origin depth <= this atom's depth.  The hidden
        # check is intentionally after the flush so hidden atoms still act as
        # depth markers, keeping vector z-ordering correct.
        while _pv_pos < len(_pending_vecs) and _vec_origins[_pending_vecs[_pv_pos]][2] <= pos[ai][2]:
            _draw_vector_arrow(_pending_vecs[_pv_pos])
            _pv_pos += 1

        if ai in hidden:
            continue

        # Drain NCI patches that belong behind this atom (before drawing it or its bonds)
        if nci_lobes_flat:
            _drain_nci(float(pos[ai][2]))

        xi, yi = _proj(pos[ai], scale, cx, cy, canvas_w, canvas_h)
        is_image = graph.nodes[ai].get("image", False)
        atom_op = cfg.periodic_image_opacity if is_image else 1.0
        op_attr_atom = f' opacity="{atom_op:.2f}"' if atom_op < 1.0 else ""

        # Atom
        if use_grad:
            ref = f"#a{ai}" if use_per_atom_grad else f"#a{a_nums[ai]}"
            svg.append(f'  <use x="{xi:.1f}" y="{yi:.1f}" xlink:href="{ref}"{op_attr_atom}/>')
        else:
            fill, stroke = colors[ai].hex, cfg.atom_stroke_color
            if cfg.fog:
                fill = blend_fog(fill, fog_rgb, fog_f[ai])
                stroke = blend_fog(stroke, fog_rgb, fog_f[ai])
            svg.append(
                f'  <circle cx="{xi:.1f}" cy="{yi:.1f}" r="{radii[ai] * scale:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"{op_attr_atom}/>'
            )

        # Atom index label — depth-sorted with atom so nearer atoms occlude it
        # (skip for image atoms — labels would be confusing)
        if cfg.show_indices and not is_image:
            fmt = cfg.idx_format
            sym = symbols[ai]
            if fmt == "sn":
                idx_text = f"{sym}{ai + 1}"
            elif fmt == "s":
                idx_text = sym
            else:  # "n"
                idx_text = str(ai + 1)
            svg.append(_text_svg(xi, yi, idx_text, fs_label, cfg.label_color, halo=False))

        # Bonds to deeper atoms
        for aj in z_order[idx + 1 :]:
            if aj in hidden or (ai, aj) not in bonds:
                continue
            bo, style, color_ov = bonds[(ai, aj)]
            # Use periodic_image_opacity if either endpoint is an image atom
            bond_op = cfg.periodic_image_opacity if (is_image or graph.nodes[aj].get("image", False)) else 1.0
            add_bond(ai, aj, bo, style, opacity=bond_op, color_override=color_ov)

    # NCI patches in front of all atoms (z_depth > frontmost atom)
    while nci_lobe_idx < len(nci_lobes_flat):
        svg.extend(nci_lobes_flat[nci_lobe_idx][1])
        nci_lobe_idx += 1

    # Flush any vectors whose origin is in front of all atoms
    while _pv_pos < len(_pending_vecs):
        _draw_vector_arrow(_pending_vecs[_pv_pos])
        _pv_pos += 1

    # --- Second pass: redraw arrowheads that protrude in front of their host atom ---
    # These were skipped in the first pass (_draw_vector_arrow) so that the shaft
    # is still painter-sorted correctly, but the head must appear on top of the atom.
    if cfg.vectors:
        for vi in range(len(cfg.vectors)):
            if _vec_head_front[vi]:
                _draw_vector_head(vi)

    # --- Front MO orbital lobes (on top of molecule) ---
    if cfg.mo_contours is not None:
        assert mo_is_front is not None
        svg.extend(
            mo_front_lobes_svg(cfg.mo_contours, mo_is_front, cfg.surface_opacity, scale, cx, cy, canvas_w, canvas_h)
        )

    # --- Density surface (stacked z-layers on top of molecule) ---
    if cfg.dens_contours is not None:
        svg.extend(dens_layers_svg(cfg.dens_contours, cfg.surface_opacity, scale, cx, cy, canvas_w, canvas_h))

    # --- ESP surface (embedded heatmap on top of molecule) ---
    if cfg.esp_surface is not None:
        from xyzrender.esp import esp_surface_svg

        svg.extend(esp_surface_svg(cfg.esp_surface, scale, cx, cy, canvas_w, canvas_h, cfg.surface_opacity))

    # VdW surface overlay — on top of molecule, group opacity for proper occlusion
    if vdw_set is not None:
        svg.append(f'  <g opacity="{cfg.vdw_opacity}">')
        for ai in z_order:
            if ai in vdw_set:
                vr = raw_vdw_sphere[ai] * cfg.vdw_scale * scale
                xi, yi = _proj(pos[ai], scale, cx, cy, canvas_w, canvas_h)
                svg.append(f'    <circle cx="{xi:.1f}" cy="{yi:.1f}" r="{vr:.1f}" fill="url(#vg{a_nums[ai]})"/>')
        svg.append("  </g>")

    # --- Annotations (bond/angle/dihedral/custom labels, always on top) ---
    has_annotations = bool(cfg.annotations)
    if has_annotations:
        svg.extend(
            _annotations_svg(
                graph, cfg, pos, hidden, scale, cx, cy, canvas_w, canvas_h, fog_f, fog_rgb, bw, fs_label, radii
            )
        )

    # --- Crystallographic axis arrows (a=red, b=green, c=blue) ---
    # Drawn last so they are always on top of atoms, bonds, and image atoms.
    if cfg.cell_data is not None and cfg.show_crystal_axes:
        lat = cfg.cell_data.lattice
        orig3d = cfg.cell_data.cell_origin
        axis_lw = cfg.cell_line_width * scale_ratio * cfg.axis_width_scale
        fs_axis = fs_label * 1.6  # larger than atom index labels
        _axis_labels = ("a", "b", "c")
        svg.append("  <!-- crystal axes -->")
        for vec, color, label in zip(lat, cfg.axis_colors, _axis_labels, strict=True):
            length = float(np.linalg.norm(vec))
            if length < 1e-6:
                continue
            # Arrow spans 25% of the cell edge (max 2 Å) from the origin corner
            frac = min(0.25, 2.0 / length)
            tip3d = orig3d + frac * vec
            ox, oy = _proj(orig3d, scale, cx, cy, canvas_w, canvas_h)
            tx, ty = _proj(tip3d, scale, cx, cy, canvas_w, canvas_h)
            # Shaft
            svg.append(
                f'  <line x1="{ox:.1f}" y1="{oy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" '
                f'stroke="{color}" stroke-width="{axis_lw:.1f}" stroke-linecap="round"/>'
            )
            dx, dy = tx - ox, ty - oy
            px_len = (dx * dx + dy * dy) ** 0.5
            if px_len > 4:
                nvx, nvy = dx / px_len, dy / px_len  # shaft direction (unit)
                pvx, pvy = -nvy, nvx  # perpendicular
                arr = max(axis_lw * 3.5, 8.0)  # arrowhead size (px)
                p1x = tx - nvx * arr + pvx * arr * 0.38
                p1y = ty - nvy * arr + pvy * arr * 0.38
                p2x = tx - nvx * arr - pvx * arr * 0.38
                p2y = ty - nvy * arr - pvy * arr * 0.38
                svg.append(
                    f'  <polygon points="{tx:.1f},{ty:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" fill="{color}"/>'
                )
                lx = tx + nvx * (arr * 0.6 + fs_axis * 0.5)
                ly = ty + nvy * (arr * 0.6 + fs_axis * 0.5) + fs_axis * 0.35
            else:
                lx, ly = tx + 4, ty
            svg.append(
                f'  <text x="{lx:.1f}" y="{ly:.1f}" font-size="{fs_axis:.1f}" fill="{color}" '
                f'font-family="Arial,sans-serif" text-anchor="middle" font-weight="bold">{label}</text>'
            )

    svg.append("</svg>")
    raw = "\n".join(svg)
    # SVG id= values are global in an HTML document — multiple renders in the same
    # Jupyter notebook page collide, causing atoms/gradients from the first render to
    # appear in all subsequent ones.  Prefix every id, href, and url() reference with
    # a unique token so each SVG is self-contained regardless of embedding context.
    p = f"x{next(_render_counter)}"
    raw = raw.replace('id="', f'id="{p}')
    raw = raw.replace('href="#', f'href="#{p}')
    raw = raw.replace("url(#", f"url(#{p}")
    return raw


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fit_canvas(pos, radii, cfg, extra_lo=None, extra_hi=None):
    """Scale + center so molecule fits canvas with tight aspect ratio."""
    pad = radii.max() if len(radii) else 0
    lo = pos[:, :2].min(axis=0) - pad
    hi = pos[:, :2].max(axis=0) + pad
    if extra_lo is not None:
        lo = np.minimum(lo, extra_lo)
    if extra_hi is not None:
        hi = np.maximum(hi, extra_hi)
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


def _text_svg(x: float, y: float, text: str, font_size: float, color: str, *, halo: bool = True) -> str:
    """SVG <text> element, bold, with optional white halo for legibility over bond lines.

    Halo is rendered as a separate stroke-only element underneath rather than via
    ``paint-order:stroke`` which is unsupported by CairoSVG (breaks PNG/PDF export).
    """
    attrs = (
        f'x="{x:.1f}" y="{y:.1f}" font-family="monospace" font-size="{font_size:.1f}px" '
        f'font-weight="bold" text-anchor="middle" dominant-baseline="central"'
    )
    if halo:
        sw = font_size * 0.35
        return (
            f'  <text {attrs} fill="#ffffff" stroke="#ffffff" '
            f'stroke-width="{sw:.1f}" stroke-linejoin="round">{text}</text>\n'
            f'  <text {attrs} fill="{color}">{text}</text>'
        )
    return f'  <text {attrs} fill="{color}">{text}</text>'


# Palette for dihedral path segments — distinct, never white
_DIHEDRAL_PALETTE = ["#984ea3", "#458f41", "#3177b0", "#a72d2f", "#A46424"]


def _annotations_svg(
    graph,
    cfg: RenderConfig,
    pos: np.ndarray,
    hidden: set,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
    fog_f: np.ndarray,
    fog_rgb: np.ndarray,
    bw: float,
    fs: float,
    radii: np.ndarray,
) -> list[str]:
    """Render all annotation elements as a flat list of SVG strings."""
    from xyzrender.annotations import AngleLabel, AtomValueLabel, BondLabel, DihedralLabel

    svg: list[str] = []
    col = cfg.label_color

    # Separate passes for each annotation type
    dihedral_idx = 0
    for ann in cfg.annotations:
        if isinstance(ann, AtomValueLabel):
            xi, yi = _proj(pos[ann.index], scale, cx, cy, canvas_w, canvas_h)
            svg.append(_text_svg(xi, yi + fs * cfg.label_offset, ann.text, fs, col))

        elif isinstance(ann, BondLabel):
            mi = (pos[ann.i] + pos[ann.j]) / 2
            mx, my = _proj(mi, scale, cx, cy, canvas_w, canvas_h)
            # Perpendicular offset so label doesn't overlap bond line
            xi, yi = _proj(pos[ann.i], scale, cx, cy, canvas_w, canvas_h)
            xj, yj = _proj(pos[ann.j], scale, cx, cy, canvas_w, canvas_h)
            dx, dy = xj - xi, yj - yi
            ln = (dx * dx + dy * dy) ** 0.5
            bl_off = fs * cfg.label_offset
            if ln > 1e-3:
                px_off, py_off = dy / ln * bl_off, -dx / ln * bl_off
            else:
                px_off, py_off = 0.0, bl_off
            svg.append(_text_svg(mx + px_off, my + py_off, ann.text, fs, col))

        elif isinstance(ann, AngleLabel):
            xi, yi = _proj(pos[ann.i], scale, cx, cy, canvas_w, canvas_h)
            xj, yj = _proj(pos[ann.j], scale, cx, cy, canvas_w, canvas_h)
            xk, yk = _proj(pos[ann.k], scale, cx, cy, canvas_w, canvas_h)

            # 2D vectors from center j toward i and k
            vi = np.array([xi - xj, yi - yj])
            vk = np.array([xk - xj, yk - yj])
            li, lk = np.linalg.norm(vi), np.linalg.norm(vk)
            if li < 1e-3 or lk < 1e-3:
                continue
            vi_hat = vi / li
            vk_hat = vk / lk

            arc_r = radii[ann.j] * scale * 1.5  # scaled with the vertex atom radius

            # Arc endpoints on the unit circle around j
            sx = xj + arc_r * vi_hat[0]
            sy = yj + arc_r * vi_hat[1]
            ex = xj + arc_r * vk_hat[0]
            ey = yj + arc_r * vk_hat[1]

            # Sweep direction: go from vi to vk the short way (inside of angle)
            cross = vi_hat[0] * vk_hat[1] - vi_hat[1] * vk_hat[0]
            sweep = 1 if cross > 0 else 0

            arc = f"M {sx:.1f},{sy:.1f} A {arc_r:.1f},{arc_r:.1f} 0 0,{sweep} {ex:.1f},{ey:.1f}"
            svg.append(
                f'  <path d="{arc}" fill="none" stroke="{col}"'
                f' stroke-width="{bw * 0.5:.1f}"'
                f' stroke-dasharray="{bw * 0.8:.1f},{bw * 1.0:.1f}" stroke-linecap="round"/>'
            )

            # Text at bisector, beyond the arc; distance scales with label_offset
            mid = vi_hat + vk_hat
            mid_len = np.linalg.norm(mid)
            if mid_len > 1e-6:
                mid_hat = mid / mid_len
            else:
                mid_hat = np.array([-vi_hat[1], vi_hat[0]])
            tx = xj + (arc_r + fs * cfg.label_offset * 0.5) * mid_hat[0]
            ty = yj + (arc_r + fs * cfg.label_offset * 0.75) * mid_hat[1]
            svg.append(_text_svg(tx, ty, ann.text, fs, col))

        elif isinstance(ann, DihedralLabel):
            seg_color = _DIHEDRAL_PALETTE[dihedral_idx % len(_DIHEDRAL_PALETTE)]
            dihedral_idx += 1

            # Draw 3 segments: i-j, j-k, k-m, each fog-blended by segment midpoint depth
            atoms_seq = [ann.i, ann.j, ann.k, ann.m]
            for seg_a, seg_b in itertools.pairwise(atoms_seq):
                xa, ya = _proj(pos[seg_a], scale, cx, cy, canvas_w, canvas_h)
                xb, yb = _proj(pos[seg_b], scale, cx, cy, canvas_w, canvas_h)
                seg_col = seg_color
                if cfg.fog:
                    avg_fog = (fog_f[seg_a] + fog_f[seg_b]) / 2 * 0.75
                    seg_col = blend_fog(seg_color, fog_rgb, avg_fog)
                svg.append(
                    f'  <line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" '
                    f'stroke="{seg_col}" stroke-width="{bw * 0.5:.1f}" stroke-linecap="round" '
                    f'stroke-dasharray="{bw * 1.0:.1f},{bw * 1.25:.1f}"/>'
                )

            # Text near j-k midpoint, perpendicular offset opposite to BondLabel
            mid_jk = (pos[ann.j] + pos[ann.k]) / 2
            mx, my = _proj(mid_jk, scale, cx, cy, canvas_w, canvas_h)
            xj2, yj2 = _proj(pos[ann.j], scale, cx, cy, canvas_w, canvas_h)
            xk2, yk2 = _proj(pos[ann.k], scale, cx, cy, canvas_w, canvas_h)
            ddx, ddy = xk2 - xj2, yk2 - yj2
            dln = (ddx * ddx + ddy * ddy) ** 0.5
            doff = fs * cfg.label_offset * 0.5
            if dln > 1e-3:
                dpx, dpy = -ddy / dln * doff, ddx / dln * doff
            else:
                dpx, dpy = 0.0, -doff
            svg.append(_text_svg(mx + dpx, my + dpy, ann.text, fs, col))

    return svg


def _ring_side(pos, ai, aj, aromatic_rings, x1, y1, x2, y2, px, py, scale, cx, cy, canvas_w, canvas_h):
    """Which perpendicular side (+1/-1) of the bond faces the aromatic ring center."""
    for ring in aromatic_rings:
        if ai in ring and aj in ring:
            centroid = pos[list(ring)].mean(axis=0)
            rcx, rcy = _proj(centroid, scale, cx, cy, canvas_w, canvas_h)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            return 1 if px * (rcx - mx) + py * (rcy - my) > 0 else -1
    return 1
