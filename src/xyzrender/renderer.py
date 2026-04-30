"""SVG renderer for molecular structures."""

from __future__ import annotations

import itertools
import logging
from typing import NamedTuple

import networkx as nx
import numpy as np
from xyzgraph import DATA

from xyzrender.cmap import atom_colors as cmap_atom_colors
from xyzrender.cmap import colorbar_extra_width, colorbar_svg
from xyzrender.colors import (
    _FOG_NEAR,
    DEFAULT_CMAP_PALETTE,
    DEFAULT_ESP_PALETTE,
    WHITE,
    Color,
    blend_fog,
    bond_color_from_atom,
    get_color,
    get_gradient_colors,
    resolve_color,
)
from xyzrender.dens import dens_layers_svg
from xyzrender.hull import (
    get_convex_hull_edges_silhouette,
    get_ring_edges,
    get_ring_facets,
    get_silhouette_polygon,
    hull_facets_svg,
    normalize_hull_subsets,
)
from xyzrender.mo import (
    classify_mo_lobes,
    mo_back_lobes_svg,
    mo_front_lobes_svg,
)
from xyzrender.types import BondStyle, RenderConfig
from xyzrender.utils import pca_orient

logger = logging.getLogger(__name__)

_render_counter = itertools.count()  # unique ID prefix per render call (SVG ids are global in Jupyter HTML)
_RADIUS_SCALE = 0.075  # VdW → atoms display radius
_REF_SPAN = 6.0  # reference molecular span (Å) for proportional bond/stroke scaling
_REF_CANVAS = 800  # reference canvas size (px) — bond/label widths are defined at this size
_CENTROID_VDW = 0.5  # VdW radius (Å) for NCI pi-system centroid dummy nodes
_H_ATOM_SCALE = 0.6  # display-radius shrink factor for H atoms (ball-and-stick)
_H_VDW_SCALE = 0.65  # VdW-sphere shrink factor for H atoms


class _BondAttrs(NamedTuple):
    """Per-bond lookup-cache entry.

    ``order`` and ``style`` come from the edge; the four ``*_override`` fields
    are stamped by :mod:`xyzrender.merge` (overlay / ensemble) and consumed by
    ``add_bond`` — each ``None`` means "use the primary/style-region config".
    """

    order: float
    style: BondStyle
    color: str | None = None
    width: float | None = None
    outline_width: float | None = None
    outline_color: str | None = None


def render_svg(graph, config: RenderConfig | None = None, *, _log: bool = True, _unique_ids: bool = True) -> str:
    """Render molecular graph to SVG string."""
    cfg = config or RenderConfig()
    node_ids = list(graph.nodes())
    n = len(node_ids)
    symbols = [graph.nodes[i]["symbol"] for i in node_ids]
    pos = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
    a_nums = [DATA.s2n.get(s, 0) for s in symbols]  # 0 for NCI centroid nodes ("*")

    # Per-atom config resolution for style regions (None = no regions, zero overhead)
    _acfg: list[RenderConfig] | None = None
    if cfg.style_regions:
        _rmap: dict[int, RenderConfig] = {}
        for region in cfg.style_regions:
            for ai in region._index_set:
                _rmap[ai] = region.config
        # NCI centroid nodes ("*") are structural overlays — always base config
        _acfg = [cfg if symbols[ai] == "*" else _rmap.get(ai, cfg) for ai in range(n)]

    # Pre-compute local vector origins/directions so we can rotate them with auto_orient
    _vec_origins = np.array([va.origin for va in cfg.vectors], dtype=float) if cfg.vectors else np.full((0, 3), np.nan)
    _vec_dirs = np.array([va.vector for va in cfg.vectors], dtype=float) if cfg.vectors else np.full((0, 3), np.nan)

    if cfg.auto_orient and n > 1:
        # Collect TS bond pairs to prioritize in orientation
        ts_pairs = list(cfg.ts_bonds) if cfg.ts_bonds else []
        for i, j, d in graph.edges(data=True):
            if d.get("TS", False):
                ts_pairs.append((i, j))
        # Exclude NCI centroid dummy nodes from PCA fitting
        atom_mask = np.array([s != "*" for s in symbols])
        fit_mask = atom_mask if not atom_mask.all() else None
        # Always capture rotation matrix when pore centroids or vectors need transforming.
        _pca_rot: np.ndarray | None = None
        _pca_centroid: np.ndarray | None = None
        if cfg.vectors:
            _fit = pos[fit_mask] if fit_mask is not None else pos
            _pca_centroid = _fit.mean(axis=0)
            pos, _pca_rot = pca_orient(pos, ts_pairs or None, fit_mask=fit_mask, return_matrix=True)
            _vec_origins = (_vec_origins - _pca_centroid) @ _pca_rot.T
            _vec_dirs = _vec_dirs @ _pca_rot.T
            logger.debug("render_svg PCA centroid: %s", _pca_centroid)
            for _vi, _vo in enumerate(_vec_origins):
                logger.debug("  vector[%d] origin after PCA: %s (should be ~0 for COM origins)", _vi, _vo)
            if cfg.cell_data is not None:
                cfg.cell_data.lattice = (_pca_rot @ cfg.cell_data.lattice.T).T
                cfg.cell_data.cell_origin = _pca_rot @ (cfg.cell_data.cell_origin - _pca_centroid)
        elif cfg.cell_data is not None or cfg.pore_centroids:
            _fit = pos[fit_mask] if fit_mask is not None else pos
            _pca_centroid = _fit.mean(axis=0)
            pos, _pca_rot = pca_orient(pos, ts_pairs, fit_mask=fit_mask, return_matrix=True)
            if cfg.cell_data is not None:
                cfg.cell_data.lattice = (_pca_rot @ cfg.cell_data.lattice.T).T
                cfg.cell_data.cell_origin = _pca_rot @ (cfg.cell_data.cell_origin - _pca_centroid)
        else:
            pos = pca_orient(pos, ts_pairs or None, fit_mask=fit_mask)

        # Transform pore centroids with the same PCA rotation.
        if cfg.pore_centroids and _pca_rot is not None and _pca_centroid is not None:
            _rotated = [_pca_rot @ (np.array(c) - _pca_centroid) for c in cfg.pore_centroids]
            cfg.pore_centroids = [(float(r[0]), float(r[1]), float(r[2])) for r in _rotated]

    raw_vdw = np.array(
        [_CENTROID_VDW if s == "*" else DATA.vdw.get(s, 1.5) * (_H_ATOM_SCALE if s == "H" else 1.0) for s in symbols]
    )
    # Per-atom absolute scale: start from cfg (or style-region _acfg), then
    # overlay / ensemble extras replace it when structure_atom_scale is set.
    if _acfg is not None:
        _atom_scale_per = np.array([_acfg[ai].atom_scale for ai in range(n)])
    else:
        _atom_scale_per = np.full(n, cfg.atom_scale)
    for ai, nid in enumerate(node_ids):
        sa = graph.nodes[nid].get("structure_atom_scale")
        if sa is not None:
            _atom_scale_per[ai] = sa
    radii = raw_vdw * _atom_scale_per * _RADIUS_SCALE

    # Per-atom scale multipliers (--scale "N,M" 2.0 or API scale=[("N,M", 2.0)])
    _per_atom_mult: np.ndarray | None = None
    if cfg.radius_scale:
        from xyzrender.selectors import resolve_atom_indices
        from xyzrender.utils import parse_atom_indices

        _per_atom_mult = np.ones(n)
        for spec, factor in cfg.radius_scale:
            if isinstance(spec, str):
                indices = resolve_atom_indices(spec, graph)
            else:
                indices = set(parse_atom_indices(spec))  # 1-indexed list → 0-indexed
            for idx in indices:
                if 0 <= idx < n:
                    _per_atom_mult[idx] *= factor
        radii = radii * _per_atom_mult

    # VdW sphere radii use a separate (larger) H scaling
    raw_vdw_sphere = np.array(
        [_CENTROID_VDW if s == "*" else DATA.vdw.get(s, 1.5) * (_H_VDW_SCALE if s == "H" else 1.0) for s in symbols]
    )
    if _per_atom_mult is not None:
        raw_vdw_sphere = raw_vdw_sphere * _per_atom_mult

    # Use VdW radii for canvas fitting when VdW spheres are active
    if cfg.vdw_indices is not None:
        vdw_active = set(range(n)) if len(cfg.vdw_indices) == 0 else set(cfg.vdw_indices)
        fit_radii = np.array([raw_vdw_sphere[i] * cfg.vdw_scale if i in vdw_active else radii[i] for i in range(n)])
    else:
        fit_radii = radii

    ref_scale = (_REF_CANVAS - 2 * cfg.padding) / _REF_SPAN
    # Pad fit_radii by atom stroke overshoot so the bounding box accounts for it
    fit_radii = fit_radii + cfg.atom_stroke_width / (2 * ref_scale)
    # Ensure fit_radii covers at least half the bond width (+ stroke) so thick
    # bonds don't extend past the canvas when atom_scale is 0.
    _min_bond_r = (cfg.bond_width + 2 * cfg.bond_outline_width) / (2 * ref_scale)
    fit_radii = np.maximum(fit_radii, _min_bond_r)
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
        _vec_tips = []
        for vi, va in enumerate(cfg.vectors):
            _vec_scale = 1.0 if va.is_axis else cfg.vector_scale
            scaled_vec = _vec_dirs[vi] * va.scale * _vec_scale
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
            label_half_w = len(va.label) * cfg.label_font_size * 1.2 * 0.35 / ref_scale
            label_h = cfg.label_font_size * 1.2 / ref_scale
            lo = tip2d - np.array([label_half_w, label_h])
            hi = tip2d + np.array([label_half_w, label_h])
            extra_lo = np.minimum(extra_lo, lo) if extra_lo is not None else lo
            extra_hi = np.maximum(extra_hi, hi) if extra_hi is not None else hi
    scale, cx, cy, canvas_w, canvas_h = _fit_canvas(pos, fit_radii, cfg, extra_lo=extra_lo, extra_hi=extra_hi)

    # scale_ratio: encodes both molecule complexity AND canvas size so that
    # bond/label widths defined at _REF_CANVAS grow proportionally on larger canvases.
    scale_ratio = scale / ref_scale
    bw = cfg.bond_width * scale_ratio
    sw = cfg.atom_stroke_width * scale_ratio
    fs_label = cfg.label_font_size * scale_ratio
    # Mesh/wire surface stroke widths (scaled with canvas like bond widths)
    _mesh_sw = max(2.0, 6.0 * scale_ratio)
    _mesh_inner_sw = max(1.5, 3.5 * scale_ratio)

    # Per-atom stroke overrides (read here so the _atom_sw block below sees them).
    struct_stroke_widths: list[float | None] = [graph.nodes[nid].get("structure_atom_stroke_width") for nid in node_ids]
    struct_stroke_colors: list[str | None] = [graph.nodes[nid].get("structure_atom_stroke_color") for nid in node_ids]

    # Per-atom stroke width overrides: style regions first, then structure
    # (overlay / ensemble) absolute overrides replace those values.
    _atom_sw: np.ndarray | None = None
    if _acfg is not None:
        _atom_sw = np.array([_acfg[ai].atom_stroke_width * scale_ratio for ai in range(n)])
    if any(v is not None for v in struct_stroke_widths):
        if _atom_sw is None:
            _atom_sw = np.full(n, sw)
        for ai, sv in enumerate(struct_stroke_widths):
            if sv is not None:
                _atom_sw[ai] = sv * scale_ratio

    if _log:
        logger.debug(
            "Render: %d atoms, %d bonds, scale=%.2f, center=(%.2f, %.2f)", n, graph.number_of_edges(), scale, cx, cy
        )
    z_order = np.argsort(pos[:, 2])
    _z_rank = np.empty(n, dtype=int)  # atom_idx → position in z_order
    _z_rank[z_order] = np.arange(n)

    # Pre-project all atom positions to 2D (vectorized)
    _px = canvas_w / 2 + scale * (pos[:, 0] - cx)  # (n,) projected x
    _py = canvas_h / 2 - scale * (pos[:, 1] - cy)  # (n,) projected y

    # Pre-extract per-atom flags to avoid repeated NetworkX lookups in the render loop
    _is_image = [graph.nodes[nid].get("image", False) for nid in node_ids]
    # Pre-extract diffuse_opacity for bonds (GIF diffuse animation)
    _diffuse_op: dict[tuple[int, int], float] = {}
    for u, v, d in graph.edges(data=True):
        dop = d.get("diffuse_opacity")
        if dop is not None:
            _diffuse_op[(u, v)] = _diffuse_op[(v, u)] = dop

    # Atom base colors — CPK by default, palette cmap when --cmap is active
    if cfg.atom_cmap is not None:
        cmap_vals = cfg.atom_cmap
        if cfg.cmap_range is not None and cfg.cmap_symm:
            msg = "--cmap-range and --cmap-symm are mutually exclusive"
            raise ValueError(msg)
        if cfg.cmap_range is not None:
            vmin, vmax = cfg.cmap_range
        elif cfg.cmap_symm:
            vmax = max(abs(v) for v in cmap_vals.values())
            vmin = -vmax
        else:
            vmin = min(cmap_vals.values())
            vmax = max(cmap_vals.values())
        colors = cmap_atom_colors(
            cmap_vals,
            n,
            cfg.cmap_palette or DEFAULT_CMAP_PALETTE,
            vmin,
            vmax,
            cfg.cmap_unlabeled,
        )
    elif _acfg is not None:
        colors = [get_color(a_nums[ai], _acfg[ai].color_overrides) for ai in range(n)]
    else:
        colors = [get_color(a, cfg.color_overrides) for a in a_nums]

    cbar_vmin: float | None = None
    cbar_vmax: float | None = None
    cbar_palette: str | None = None
    if cfg.cbar and cfg.atom_cmap is not None:
        cbar_vmin = vmin
        cbar_vmax = vmax
        cbar_palette = cfg.cmap_palette or DEFAULT_CMAP_PALETTE
    elif cfg.cbar and cfg.esp_surface is not None:
        cbar_vmin = cfg.esp_surface.esp_vmin
        cbar_vmax = cfg.esp_surface.esp_vmax
        cbar_palette = cfg.cmap_palette or DEFAULT_ESP_PALETTE

    # Reserve space on the right for the cmap colorbar.
    # canvas_w stays at the molecule width so _proj() keeps the molecule centred there.
    # _cb_svg_w is the full SVG width used only in the viewBox / width attribute.
    cb_extra_w = (
        colorbar_extra_width(cbar_vmin, cbar_vmax, fs_label) if cbar_vmin is not None and cbar_vmax is not None else 0
    )
    _cb_svg_w = canvas_w + cb_extra_w

    # Per-structure colour override (overlay mol2 atoms and ensemble non-reference
    # conformers both stamp `structure_color` on their nodes — single code path).
    struct_colors: list[str | None] = [graph.nodes[nid].get("structure_color") for nid in node_ids]
    for ai in range(n):
        sc = struct_colors[ai]
        if sc:
            colors[ai] = Color.from_str(sc)

    # Pre-extract structure_opacity per atom — avoids two dict lookups per atom inside the main loop.
    struct_opacities: list[float | None] = [graph.nodes[nid].get("structure_opacity") for nid in node_ids]

    # Per-atom opacity override (cfg.atom_opacity is 0-indexed).  Affects the
    # atom's own fill-opacity in the render loop; bonds are NOT read from this
    # list so an atom can be faded without dimming its connectivity.
    _atom_only_op: list[float | None] = (
        [cfg.atom_opacity.get(ai) for ai in range(n)] if cfg.atom_opacity else [None] * n
    )

    # Molecule color: override atom + bond colors with a single color.
    # Skip atoms that carry structure_color (overlay mol2 / ensemble extras)
    # so those structures keep their own colour rather than being over-painted.
    mol_bond_color: str | None = None
    if cfg.mol_color is not None:
        flat = Color.from_str(cfg.mol_color)
        for ai in range(n):
            if struct_colors[ai] is None:
                colors[ai] = flat
        mol_bond_color = bond_color_from_atom(flat)

    # Highlight: override colors for user-specified atom groups
    hl_atom_group: dict[int, int] = {}  # atom_idx → group_id
    hl_group_bond_color: list[str] = []  # group_id → darkened bond hex
    if cfg.highlight_groups:
        for gid, group in enumerate(cfg.highlight_groups):
            gc = Color.from_str(group.color)
            hl_group_bond_color.append(bond_color_from_atom(gc))
            for ai in group._index_set:
                colors[ai] = gc
                hl_atom_group[ai] = gid

    # Pre-cache hex strings so .hex property isn't recomputed in the render loop
    _color_hex = [c.hex for c in colors]

    # Bond lookup: per-edge attrs needed by the render loop.
    bonds: dict[tuple[int, int], _BondAttrs] = {}
    if not cfg.hide_bonds:
        for i, j, d in graph.edges(data=True):
            if d.get("TS", False):
                style = BondStyle.DASHED
            elif d.get("NCI", False):
                style = BondStyle.DOTTED
            else:
                style = BondStyle.SOLID
            attrs = _BondAttrs(
                order=d.get("bond_order", 1.0),  # always store raw; bond_orders flag applied per-bond at render
                style=style,
                color=d.get("bond_color_override"),
                width=d.get("bond_width_override"),
                outline_width=d.get("bond_outline_width_override"),
                outline_color=d.get("bond_outline_color_override"),
            )
            bonds[(i, j)] = bonds[(j, i)] = attrs
        # Manual overrides (add or restyle)
        _default = _BondAttrs(order=1.0, style=BondStyle.SOLID)
        for i, j in cfg.ts_bonds:
            bonds[(i, j)] = bonds[(j, i)] = bonds.get((i, j), _default)._replace(style=BondStyle.DASHED)
        for i, j in cfg.nci_bonds:
            bonds[(i, j)] = bonds[(j, i)] = bonds.get((i, j), _default)._replace(style=BondStyle.DOTTED)
        # Molecule color: paint all SOLID bonds with darkened mol_color
        if mol_bond_color is not None:
            for (i, j), attrs in list(bonds.items()):
                if attrs.color is None and attrs.style == BondStyle.SOLID:
                    bonds[(i, j)] = bonds[(j, i)] = attrs._replace(color=mol_bond_color)
        # Highlight: color bonds between two atoms in the SAME highlight group.
        # Only SOLID covalent bonds — TS/NCI are structural overlays.
        # Overrides mol_color bond coloring (but not explicit per-edge overrides).
        if hl_atom_group:
            for (i, j), attrs in list(bonds.items()):
                gi, gj = hl_atom_group.get(i), hl_atom_group.get(j)
                if (
                    gi is not None
                    and gi == gj
                    and attrs.style == BondStyle.SOLID
                    and (attrs.color is None or attrs.color == mol_bond_color)
                ):
                    bonds[(i, j)] = bonds[(j, i)] = attrs._replace(color=hl_group_bond_color[gi])

    # Pre-build adjacency list for O(degree) bond lookup in render loop
    bond_adj: dict[int, list[int]] = {}
    if bonds:
        for i, j in bonds:
            if i < j:  # bonds has both (i,j) and (j,i); add once
                bond_adj.setdefault(i, []).append(j)
                bond_adj.setdefault(j, []).append(i)

    # Only hide C-H hydrogens (not O-H, N-H, free H, etc.)
    hidden = set()
    if cfg.hide_h:
        show = set(cfg.show_h_indices)
        # Auto-show H atoms involved in manual NCI/TS pairs — these aren't in
        # graph.edges (cfg.{nci,ts}_bonds is renderer-only), so the C-only
        # neighbour check below would otherwise hide them and orphan the bond.
        for i, j in (*cfg.nci_bonds, *cfg.ts_bonds):
            if 0 <= i < n and symbols[i] == "H":
                show.add(i)
            if 0 <= j < n and symbols[j] == "H":
                show.add(j)
        for ai in range(n):
            if symbols[ai] == "H" and ai not in show:
                neighbours = list(graph.neighbors(ai))
                if neighbours and all(symbols[nb] == "C" for nb in neighbours):
                    hidden.add(ai)

    aromatic_rings = [] if cfg.hide_bonds else _compute_aromatic_rings(graph, bonds)

    # Fog factors — normalized across depth range, with a dead-zone near the front
    fog_f = np.zeros(n)
    fog_rgb = np.array([255, 255, 255])
    if cfg.fog:
        zr = max(pos[:, 2].max() - pos[:, 2].min(), 1e-6)
        depth = pos[:, 2].max() - pos[:, 2]  # distance from front atom
        fog_f = cfg.fog_strength * np.clip((depth - _FOG_NEAR) / zr, 0.0, 1.0)

    # Depth-of-field: per-atom blur bucket (0 = sharp front, N-1 = max blur back)
    n_dof_levels = 20
    dof_buckets: list[int] = []
    if cfg.dof:
        if cfg.fog:
            dof_depth = fog_f / max(cfg.fog_strength, 1e-6)  # normalize back to [0, 1]
        else:
            zr = max(pos[:, 2].max() - pos[:, 2].min(), 1e-6)
            dof_depth = np.clip((pos[:, 2].max() - pos[:, 2] - _FOG_NEAR) / zr, 0.0, 1.0)
        dof_buckets = [int(d * (n_dof_levels - 1) + 0.5) for d in dof_depth]
    glow_indices = set(cfg.glow_indices) if cfg.glow_indices else set()
    glow_strength = cfg.glow_strength

    # --- Build SVG ---
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {_cb_svg_w} {canvas_h}" width="{_cb_svg_w}" height="{canvas_h}"'
        + (' style="background:transparent"' if cfg.transparent else "")
        + ">"
    ]
    if not cfg.transparent:
        svg.append(f'  <rect width="100%" height="100%" fill="{cfg.background}"/>')

    # DoF filter definitions
    if cfg.dof:
        svg.append("  <defs>")
        for lvl in range(n_dof_levels):
            blur = lvl / max(n_dof_levels - 1, 1) * cfg.dof_strength
            svg.append(
                f'    <filter id="dof{lvl}" x="-50%" y="-50%" width="200%" height="200%">'
                f'<feGaussianBlur stdDeviation="{blur:.2f}"/></filter>'
            )
        svg.append("  </defs>")
    if glow_indices:
        svg.append("  <defs>")
        svg.append(
            f'    <filter id="glow" x="-100%" y="-100%" width="300%" height="300%">'
            f'<feGaussianBlur stdDeviation="{glow_strength:.2f}"/></filter>'
        )
        svg.append("  </defs>")

    # Per-atom gradient and skeletal flags (style-region aware)
    _atom_use_grad: list[bool] | None = None
    if _acfg is not None:
        _atom_use_grad = [_acfg[ai].gradient and not _acfg[ai].skeletal_style for ai in range(n)]
        use_grad = any(_atom_use_grad)
        any_skeletal = any(_acfg[ai].skeletal_style for ai in range(n))
    else:
        use_grad = cfg.gradient and not cfg.skeletal_style
        any_skeletal = cfg.skeletal_style
    if any_skeletal:
        from xyzrender.skeletal import skeletal_atom_svg, skeletal_bond_svg
    # Fog requires per-atom gradient defs: each atom has a unique depth-blended fill
    # AND stroke colour.  Everything else (overlay, ensemble, cmap) just sets colors[ai]
    # to a per-atom value that can be shared by (element, colour_hex) — same def count
    # as default CPK for normal molecules, O(elements x colours) for overlay/ensemble.
    use_per_atom_grad = cfg.fog
    # Per-atom fog stroke colours (only populated when fog is on).
    atom_fog_stroke: list[str] = []
    if use_grad:
        svg.append("  <defs>")
        if use_per_atom_grad:
            # Fog: per-atom radialGradient + per-atom blended stroke colour.
            # Inline <circle> in the render loop references these by id.
            atom_fog_stroke = [cfg.atom_stroke_color] * n
            for ai in range(n):
                if ai in hidden:
                    continue
                if _atom_use_grad is not None and not _atom_use_grad[ai]:
                    continue
                acfg = _acfg[ai] if _acfg is not None else cfg
                hi, me, lo = get_gradient_colors(colors[ai], acfg, strength=acfg.atom_gradient_strength)
                t = min(fog_f[ai] ** 2 * 0.7, 0.70)
                hi, me, lo = hi.blend(WHITE, t), me.blend(WHITE, t), lo.blend(WHITE, t)
                _stroke_src = struct_stroke_colors[ai] or acfg.atom_stroke_color
                _base_stroke = colors[ai].hex if _stroke_src == "atom" else _stroke_src
                atom_fog_stroke[ai] = blend_fog(_base_stroke, fog_rgb, fog_f[ai])
                svg.append(
                    f'    <radialGradient id="g{ai}" cx=".5" cy=".5" fx=".33" fy=".33" r=".66">'
                    f'<stop offset="0%" stop-color="{hi.hex}"/>'
                    f'<stop offset="40%" stop-color="{me.hex}"/>'
                    f'<stop offset="100%" stop-color="{lo.hex}"/>'
                    f"</radialGradient>"
                )
        else:
            # Shared gradient defs keyed by (atomic_number, colour_hex, shift_factors).
            # Default CPK: one def per element. Overlay/ensemble/cmap: one per
            # (element, colour) pair — O(elements x colours), not O(atoms).
            # Inline <circle fill="url(#g...)"> in the render loop: avoids the
            # O(N²) cairosvg <use href> ID-lookup cost for large/ensemble molecules.
            seen: dict[tuple, str] = {}
            for ai in range(n):
                if _atom_use_grad is not None and not _atom_use_grad[ai]:
                    continue
                an = a_nums[ai]
                chex = colors[ai].hex
                acfg = _acfg[ai] if _acfg is not None else cfg
                key = (
                    an,
                    chex,
                    acfg.hue_shift_factor,
                    acfg.light_shift_factor,
                    acfg.saturation_shift_factor,
                    acfg.atom_gradient_strength,
                )
                if key in seen or ai in hidden:
                    continue
                gid = f"{an}_{chex[1:]}"
                if _acfg is not None:
                    gid += f"_{id(acfg) & 0xFFFF:04x}"
                seen[key] = gid
                hi, me, lo = get_gradient_colors(colors[ai], acfg, strength=acfg.atom_gradient_strength)
                svg.append(
                    f'    <radialGradient id="g{gid}" cx=".5" cy=".5" fx=".33" fy=".33" r=".66">'
                    f'<stop offset="0%" stop-color="{hi.hex}"/>'
                    f'<stop offset="40%" stop-color="{me.hex}"/>'
                    f'<stop offset="100%" stop-color="{lo.hex}"/>'
                    f"</radialGradient>"
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
                lo = colors[ai].darken(
                    strength=cfg.vdw_gradient_strength,
                    hue_shift_factor=cfg.hue_shift_factor,
                    light_shift_factor=cfg.light_shift_factor,
                    saturation_shift_factor=cfg.saturation_shift_factor,
                )
                svg.append(
                    f'    <radialGradient id="vg{an}" cx=".5" cy=".5" fx=".33" fy=".33" r=".66">'
                    f'<stop offset="0%" stop-color="{hi.hex}"/><stop offset="100%" stop-color="{lo.hex}"/>'
                    f"</radialGradient>"
                )
        svg.append("  </defs>")

    # MO lobe front/back classification
    mo_is_front = None
    if cfg.mo_contours is not None:
        mo = cfg.mo_contours
        if cfg.flat_mo:
            mo_is_front = [True] * len(mo.lobes)
        else:
            mo_is_front = classify_mo_lobes(mo.lobes, float(pos[:, 2].mean()))

    # --- Back MO orbital lobes (behind molecule) — flat faded fill ---
    if cfg.mo_contours is not None:
        assert mo_is_front is not None
        svg.extend(
            mo_back_lobes_svg(
                cfg.mo_contours,
                mo_is_front,
                cfg.surface_opacity,
                scale,
                cx,
                cy,
                canvas_w,
                canvas_h,
                surface_style=cfg.surface_style,
                stroke_width=_mesh_sw,
                mesh_inner_width=_mesh_inner_sw,
            )
        )

    # --- Convex hull facets (low-alpha plane behind molecule) ---
    if cfg.show_convex_hull:
        palette = [resolve_color(c) for c in cfg.hull_colors]
        hull_color_hex = palette[0]
        per_color: list[str] | None = None

        _raw_idx = cfg.hull_atom_indices
        subsets = normalize_hull_subsets(_raw_idx) if _raw_idx is not None else None

        if subsets:
            all_facets: list[tuple[np.ndarray, float]] = []
            subset_indices: list[int] = []
            for idx, subset in enumerate(subsets):
                if cfg.hull_ordered:
                    valid = [i for i in subset if 0 <= i < n]
                    sub_facets = get_ring_facets(pos, valid)
                else:
                    include_mask = np.zeros(n, dtype=bool)
                    for i in subset:
                        if 0 <= i < n:
                            include_mask[i] = True
                    sub_facets = get_silhouette_polygon(pos, include_mask)
                all_facets.extend(sub_facets)
                subset_indices.extend([idx] * len(sub_facets))
            facets = all_facets
            # Per-subset colors (cycling palette)
            if subset_indices:
                with_idx = list(zip(all_facets, subset_indices, strict=True))
                with_idx.sort(key=lambda x: x[0][1])
                sorted_facets = [f for f, _ in with_idx]
                indices_sorted = [si for _, si in with_idx]
                per_color = [palette[i % len(palette)] for i in indices_sorted]
                facets = sorted_facets
        elif subsets is None:
            # No indices specified — use all heavy (non-H, non-dummy) atoms
            include_mask = np.array([s not in ("*", "H") for s in symbols]) if n > 0 else None
            facets = get_silhouette_polygon(pos, include_mask)
        else:
            # Empty indices list — no hull
            facets = []

        if facets:
            svg.extend(
                hull_facets_svg(
                    facets,
                    hull_color_hex,
                    cfg.hull_opacity,
                    scale,
                    cx,
                    cy,
                    canvas_w,
                    canvas_h,
                    per_facet_color_hex=per_color,
                )
            )

        # Non-bond hull edges (1-skeleton) — per-subset color matches the fill
        if cfg.show_hull_edges:
            bond_pairs = {(min(i, j), max(i, j)) for (i, j) in bonds}
            # Each entry: ((ni, nj), mid_z, edge_color)
            hull_edges_with_z: list[tuple[tuple[int, int], float, str]] = []
            if subsets:
                for sidx, subset in enumerate(subsets):
                    sub_color = palette[sidx % len(palette)]
                    if cfg.hull_ordered:
                        valid = [i for i in subset if 0 <= i < n]
                        edges = get_ring_edges(valid)
                    else:
                        include_mask = np.zeros(n, dtype=bool)
                        for i in subset:
                            if 0 <= i < n:
                                include_mask[i] = True
                        edges = get_convex_hull_edges_silhouette(pos, include_mask)
                    for ni, nj in edges:
                        if (ni, nj) not in bond_pairs:
                            mid_z = (pos[ni][2] + pos[nj][2]) / 2.0
                            hull_edges_with_z.append(((ni, nj), mid_z, sub_color))
            elif subsets is None:
                include_mask = np.array([s not in ("*", "H") for s in symbols]) if n > 0 else None
                for ni, nj in get_convex_hull_edges_silhouette(pos, include_mask):
                    if (ni, nj) not in bond_pairs:
                        mid_z = (pos[ni][2] + pos[nj][2]) / 2.0
                        hull_edges_with_z.append(((ni, nj), mid_z, hull_color_hex))
            hull_edges_with_z.sort(key=lambda x: x[1])
            hull_lw = max(bw * cfg.hull_edge_width_ratio, 1.0)
            for (ni, nj), _, edge_color in hull_edges_with_z:
                x1, y1 = _proj(pos[ni], scale, cx, cy, canvas_w, canvas_h)
                x2, y2 = _proj(pos[nj], scale, cx, cy, canvas_w, canvas_h)
                svg.append(
                    f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="{edge_color}" stroke-width="{hull_lw:.1f}" stroke-linecap="round" '
                    f'stroke-opacity="{cfg.hull_opacity:.3f}"/>'
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
            _global = 1.0 if va.is_axis else cfg.vector_scale
            scaled_vec = _vec_dirs[vi] * va.scale * _global
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
        nci_lobes_flat = nci_lobe_svg_items(
            cfg.nci_contours,
            cfg.surface_opacity,
            scale,
            cx,
            cy,
            canvas_w,
            canvas_h,
            surface_style=cfg.surface_style,
            stroke_width=_mesh_sw,
            mesh_inner_width=_mesh_inner_sw,
        )

    def _drain_nci(next_z: float) -> None:
        nonlocal nci_lobe_idx
        while nci_lobe_idx < len(nci_lobes_flat) and nci_lobes_flat[nci_lobe_idx][0] < next_z:
            svg.extend(nci_lobes_flat[nci_lobe_idx][1])
            nci_lobe_idx += 1

    # --- Pore volume spheres (z-interleaved) ---
    # Uses cfg.pore_node_ids: list of node-ID lists per pore.
    # Centroid + radius computed from oriented positions (post-PCA).
    pore_spheres_flat: list[tuple[float, list[str]]] = []
    pore_sphere_idx = 0
    if cfg.pore_spheres and cfg.pore_node_ids:
        from xyzrender.hull import pore_size_colors

        fp_colors = pore_size_colors(cfg.pore_node_ids, graph)
        # Use fingerprint colours only when multiple distinct pore types exist.
        # Otherwise use the configured pore sphere colour.
        if len(set(fp_colors)) > 1:
            pore_colors = fp_colors
        else:
            pore_colors = [cfg.pore_sphere_color] * len(cfg.pore_node_ids)
        _pore_grad_cache: dict[str, str] = {}
        svg.append("  <defs>")
        for hex_color in set(pore_colors):
            base = Color.from_hex(hex_color)
            dark = base.darken(
                strength=cfg.vdw_gradient_strength,
                hue_shift_factor=cfg.hue_shift_factor,
                light_shift_factor=cfg.light_shift_factor,
                saturation_shift_factor=cfg.saturation_shift_factor,
            )
            gid = f"pore_{hex_color[1:]}"
            _pore_grad_cache[hex_color] = gid
            svg.append(
                f'    <radialGradient id="{gid}" cx=".5" cy=".5" fx=".33" fy=".33" r=".66">'
                f'<stop offset="0%" stop-color="{base.hex}"/>'
                f'<stop offset="100%" stop-color="{dark.hex}"/>'
                f"</radialGradient>"
            )
        svg.append("  </defs>")
        for pidx, node_ids in enumerate(cfg.pore_node_ids):
            # Use true centroids/radii when available (accurate coarse-grain positions).
            # Fall back to node-mean for backwards compatibility.
            if cfg.pore_centroids and cfg.pore_radii and pidx < len(cfg.pore_centroids):
                centroid = np.array(cfg.pore_centroids[pidx])
                radius = cfg.pore_radii[pidx]
            else:
                valid = [i for i in node_ids if 0 <= i < n]
                if len(valid) < 3:
                    continue
                ring_pos = pos[valid]
                centroid = ring_pos.mean(axis=0)
                dists = np.linalg.norm(ring_pos - centroid, axis=1)
                radius = float(dists.min()) * 0.7
            sx, sy = _proj(centroid, scale, cx, cy, canvas_w, canvas_h)
            sr = radius * scale
            z_depth = float(centroid[2])
            gid = _pore_grad_cache[pore_colors[pidx]]
            sphere_svg = [
                f'  <circle cx="{sx:.1f}" cy="{sy:.1f}" r="{sr:.1f}" '
                f'fill="url(#{gid})" fill-opacity="{cfg.pore_sphere_opacity:.2f}" stroke="none"/>'
            ]
            pore_spheres_flat.append((z_depth, sphere_svg))
        pore_spheres_flat.sort(key=lambda x: x[0])

    def _drain_pore_spheres(next_z: float) -> None:
        nonlocal pore_sphere_idx
        while pore_sphere_idx < len(pore_spheres_flat) and pore_spheres_flat[pore_sphere_idx][0] < next_z:
            svg.extend(pore_spheres_flat[pore_sphere_idx][1])
            pore_sphere_idx += 1

    # Interleaved z-order: for each atom, render it then its bonds to deeper atoms

    # Bond config resolution for style regions
    def _bond_cfg(ai: int, aj: int) -> RenderConfig:
        if _acfg is None:
            return cfg
        ca, cb = _acfg[ai], _acfg[aj]
        return ca if (ca is cb and ca is not cfg) else cfg

    # Cylinder shading: cache gradient colours and counter for unique IDs
    _bs_counter = itertools.count()
    _shade_color_cache: dict[str, tuple[str, str]] = {}
    # Deferred atom layers: draw all edges first, then place nodes on top for a
    # clean diagram-like aesthetic (enabled by atoms_above_bonds).
    _deferred_atom_layers: list[str] = []
    # Bond edge stroke: a wider shadow line behind each bond, collected into a
    # deferred layer inserted at the base of the molecule group.
    _bond_outline_layer: list[str] = []
    _atoms_above = cfg.atoms_above_bonds if _acfg is None else any(c.atoms_above_bonds for c in _acfg)

    def _shaded_stroke(color_hex, lx1, ly1, lx2, ly2, w, lpx, lpy, shade_cfg):
        """Return an SVG stroke value — flat colour or perpendicular gradient.

        When *shade_cfg* is not None, creates a cylinder-shading gradient using
        ``get_gradient_colors`` (same system as atom radial gradients) and
        returns ``url(#id)``.  Otherwise returns the plain hex colour.
        """
        if shade_cfg is None:
            return color_hex
        chex = color_hex
        if chex not in _shade_color_cache:
            hi, _me, lo = get_gradient_colors(
                Color.from_str(chex), shade_cfg, strength=shade_cfg.bond_gradient_strength
            )
            _shade_color_cache[chex] = (hi.hex, lo.hex)
        hi_hex, lo_hex = _shade_color_cache[chex]
        sid = f"bs{next(_bs_counter)}"
        half = w * 0.5
        mx, my = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        gx1, gy1 = mx - lpx * half, my - lpy * half
        gx2, gy2 = mx + lpx * half, my + lpy * half
        # 3-stop gradient: lo → hi → lo  (symmetric cylinder shading —
        # specular highlight at centre, dark edges)
        svg.append(
            f'  <defs><linearGradient id="{sid}" x1="{gx1:.1f}" y1="{gy1:.1f}" '
            f'x2="{gx2:.1f}" y2="{gy2:.1f}" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0%" stop-color="{lo_hex}"/>'
            f'<stop offset="50%" stop-color="{hi_hex}"/>'
            f'<stop offset="100%" stop-color="{lo_hex}"/>'
            f"</linearGradient></defs>"
        )
        return f"url(#{sid})"

    def _bond_line(lx1, ly1, lx2, ly2, w, color_hex, lpx, lpy, shade_cfg, op_attr, dash=""):
        """Emit a single bond line — flat or cylinder-shaded."""
        stroke = _shaded_stroke(color_hex, lx1, ly1, lx2, ly2, w, lpx, lpy, shade_cfg)
        svg.append(
            f'  <line x1="{lx1:.1f}" y1="{ly1:.1f}" x2="{lx2:.1f}" y2="{ly2:.1f}" '
            f'stroke="{stroke}" stroke-width="{w:.1f}" stroke-linecap="round"{dash}{op_attr}/>'
        )

    def _element_line(
        lx1,
        ly1,
        lx2,
        ly2,
        w,
        ci_hex,
        cj_hex,
        ri,
        rj,
        lpx,
        lpy,
        *,
        fog_enabled,
        fi,
        fj,
        shade_cfg,
        op_attr,
        dash="",
    ):
        """Emit a half-bond split line with element colouring.

        *ri*, *rj* are raw VdW radii for radius-weighted midpoint.
        Each half is individually cylinder-shaded when *shade_cfg* is set.
        """
        avg_fog = (fi + fj) / 2 * 0.75
        c1 = blend_fog(ci_hex, fog_rgb, avg_fog) if fog_enabled else ci_hex
        c2 = blend_fog(cj_hex, fog_rgb, avg_fog) if fog_enabled else cj_hex
        # Skip split when both endpoints are the same colour (e.g. C-C bonds)
        if c1 == c2:
            _bond_line(lx1, ly1, lx2, ly2, w, c1, lpx, lpy, shade_cfg, op_attr, dash)
        else:
            t = ri / (ri + rj) if (ri + rj) > 0 else 0.5
            # Keep dashed stroke continuity (e.g. aromatic dashed side) by using
            # one line with a hard-stop gradient at the endpoint split ratio.
            if dash:
                sid = f"be{next(_bs_counter)}"
                off = max(0.0, min(100.0, 100.0 * t))
                svg.append(
                    f'  <defs><linearGradient id="{sid}" x1="{lx1:.1f}" y1="{ly1:.1f}" '
                    f'x2="{lx2:.1f}" y2="{ly2:.1f}" gradientUnits="userSpaceOnUse">'
                    f'<stop offset="0%" stop-color="{c1}"/>'
                    f'<stop offset="{off:.4f}%" stop-color="{c1}"/>'
                    f'<stop offset="{off:.4f}%" stop-color="{c2}"/>'
                    f'<stop offset="100%" stop-color="{c2}"/>'
                    f"</linearGradient></defs>"
                )
                svg.append(
                    f'  <line x1="{lx1:.1f}" y1="{ly1:.1f}" x2="{lx2:.1f}" y2="{ly2:.1f}" '
                    f'stroke="url(#{sid})" stroke-width="{w:.1f}" stroke-linecap="round"{dash}{op_attr}/>'
                )
            else:
                xm = lx1 + (lx2 - lx1) * t
                ym = ly1 + (ly2 - ly1) * t
                _bond_line(lx1, ly1, xm, ym, w, c1, lpx, lpy, shade_cfg, op_attr, dash)
                _bond_line(xm, ym, lx2, ly2, w, c2, lpx, lpy, shade_cfg, op_attr, dash)

    # Pre-resolve bond config for the common case (no style regions)
    _base_bcfg = cfg
    _base_bw = cfg.bond_width * scale_ratio
    _base_gap = cfg.bond_gap * _base_bw
    _base_bond_color = cfg.bond_color
    _base_outline_color = cfg.bond_outline_color
    _base_outline_width = cfg.bond_outline_width * scale_ratio
    _base_by_element = cfg.bond_color_by_element
    _base_scfg = cfg if cfg.bond_gradient else None

    def _emit_line(
        lx1,
        ly1,
        lx2,
        ly2,
        w,
        color_hex,
        lpx,
        lpy,
        *,
        shade,
        op_attr,
        dash,
        by_element,
        ci_hex,
        cj_hex,
        ri_vdw,
        rj_vdw,
        fi,
        fj,
        stroke_i,
        stroke_j,
        stroke_w,
    ):
        """Dispatch a single bond line — element-coloured or uniform."""
        if stroke_i and stroke_w > 0:
            stroke = stroke_i
            if stroke_j and stroke_j != stroke_i:
                sid = f"bo{next(_bs_counter)}"
                svg.append(
                    f'  <defs><linearGradient id="{sid}" x1="{lx1:.1f}" y1="{ly1:.1f}" '
                    f'x2="{lx2:.1f}" y2="{ly2:.1f}" gradientUnits="userSpaceOnUse">'
                    f'<stop offset="0%" stop-color="{stroke_i}"/>'
                    f'<stop offset="100%" stop-color="{stroke_j}"/>'
                    f"</linearGradient></defs>"
                )
                stroke = f"url(#{sid})"
            ow = w + 2 * stroke_w
            _bond_outline_layer.append(
                f'  <line x1="{lx1:.1f}" y1="{ly1:.1f}" x2="{lx2:.1f}" y2="{ly2:.1f}" '
                f'stroke="{stroke}" stroke-width="{ow:.1f}" stroke-linecap="round"{dash}{op_attr}/>'
            )
        if by_element:
            _element_line(
                lx1,
                ly1,
                lx2,
                ly2,
                w,
                ci_hex,
                cj_hex,
                ri_vdw,
                rj_vdw,
                lpx,
                lpy,
                fog_enabled=cfg.fog,
                fi=fi,
                fj=fj,
                shade_cfg=shade,
                op_attr=op_attr,
                dash=dash,
            )
        else:
            _bond_line(lx1, ly1, lx2, ly2, w, color_hex, lpx, lpy, shade, op_attr, dash)

    def add_bond(
        ai,
        aj,
        bo,
        style,
        opacity: float = 1.0,
        color_override: str | None = None,
        width_override: float | None = None,
        outline_width_override: float | None = None,
        outline_color_override: str | None = None,
    ):
        """Render bond — closure captures shared rendering state."""
        # Config: use base config unless style regions exist and bond is solid
        if _acfg is not None and style == BondStyle.SOLID:
            ca, cb = _acfg[ai], _acfg[aj]
            bcfg = ca if (ca is cb and ca is not cfg) else cfg
        else:
            bcfg = cfg
        bo = bo if bcfg.bond_orders else 1.0

        # Use pre-resolved values when config is the base (common path)
        if bcfg is cfg:
            _bw = _base_bw
            _gap = _base_gap
            _bond_color = _base_bond_color
            _stroke_color = _base_outline_color
            _stroke_width = _base_outline_width
            _scfg = _base_scfg
        else:
            _bw = bcfg.bond_width * scale_ratio
            _gap = bcfg.bond_gap * _bw
            _bond_color = bcfg.bond_color
            _stroke_color = bcfg.bond_outline_color
            _stroke_width = bcfg.bond_outline_width * scale_ratio
            _scfg = bcfg if bcfg.bond_gradient else None

        # Per-edge overlay / structure overrides beat base and style-region values.
        if width_override is not None:
            _bw = width_override * scale_ratio
            _gap = bcfg.bond_gap * _bw
        if outline_width_override is not None:
            _stroke_width = outline_width_override * scale_ratio
        if outline_color_override is not None:
            _stroke_color = outline_color_override

        if style != BondStyle.SOLID:
            _bw = min(_bw, 20.0 * scale_ratio)
            _gap = bcfg.bond_gap * _bw
        if style == BondStyle.DASHED and bcfg.ts_color is not None:
            _bond_color = bcfg.ts_color
        if style == BondStyle.DOTTED and bcfg.nci_color is not None:
            _bond_color = bcfg.nci_color

        if bcfg.skeletal_style:
            skeletal_bond_svg(
                svg,
                ai,
                aj,
                bo,
                style,
                opacity,
                pos=pos,
                symbols=symbols,
                radii=radii,
                bw=_bw,
                gap=_gap,
                fs_label=fs_label,
                scale=scale,
                cx=cx,
                cy=cy,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                fog_f=fog_f,
                fog_rgb=fog_rgb,
                fog_enabled=cfg.fog,
                bond_color=_bond_color,
                color_override=color_override,
                aromatic_rings=aromatic_rings,
            )
            return

        _bg = bond_geom.get((ai, aj))
        if _bg is None:
            return
        x1, y1, x2, y2, px, py = _bg

        by_element = bcfg.bond_color_by_element and color_override is None and style == BondStyle.SOLID
        ci_hex = cj_hex = color = ""
        if by_element:
            ci_hex = _color_hex[ai]
            cj_hex = _color_hex[aj]
        else:
            color = color_override if color_override is not None else _bond_color
            if cfg.fog:
                color = blend_fog(color, fog_rgb, (fog_f[ai] + fog_f[aj]) / 2 * 0.75)

        op_attr = f' opacity="{opacity:.2f}"' if opacity < 1.0 else ""

        # Per-bond args shared by every _emit_line call
        _fi = fog_f[ai]
        _fj = fog_f[aj]
        _ri_vdw = raw_vdw[ai]
        _rj_vdw = raw_vdw[aj]
        _si = _stroke_color
        _sj = _stroke_color
        if _stroke_color and cfg.fog:
            _si = blend_fog(_stroke_color, fog_rgb, _fi)
            _sj = blend_fog(_stroke_color, fog_rgb, _fj)

        if style == BondStyle.DASHED:
            dd, gg = _bw * 1.2, _bw * 2.2
            dash = f' stroke-dasharray="{dd:.1f},{gg:.1f}"'
            _emit_line(
                x1,
                y1,
                x2,
                y2,
                _bw * 1.2,
                color,
                px,
                py,
                shade=None,
                op_attr=op_attr,
                dash=dash,
                by_element=by_element,
                ci_hex=ci_hex,
                cj_hex=cj_hex,
                ri_vdw=_ri_vdw,
                rj_vdw=_rj_vdw,
                fi=_fi,
                fj=_fj,
                stroke_i=_si,
                stroke_j=_sj,
                stroke_w=_stroke_width,
            )
            return
        if style == BondStyle.DOTTED:
            dd, gg = _bw * 0.08, _bw * 2
            dash = f' stroke-dasharray="{dd:.1f},{gg:.1f}"'
            _emit_line(
                x1,
                y1,
                x2,
                y2,
                _bw,
                color,
                px,
                py,
                shade=None,
                op_attr=op_attr,
                dash=dash,
                by_element=by_element,
                ci_hex=ci_hex,
                cj_hex=cj_hex,
                ri_vdw=_ri_vdw,
                rj_vdw=_rj_vdw,
                fi=_fi,
                fj=_fj,
                stroke_i=_si,
                stroke_j=_sj,
                stroke_w=_stroke_width,
            )
            return

        is_aromatic = 1.3 < bo < 1.7
        if is_aromatic:
            side = _ring_side(pos, ai, aj, aromatic_rings, x1, y1, x2, y2, px, py, scale, cx, cy, canvas_w, canvas_h)
            w = _bw * 0.7
            for ib in [-1, 1]:
                ox, oy = px * ib * _gap, py * ib * _gap
                dash = f' stroke-dasharray="{w * 1.0:.1f},{w * 2.0:.1f}"' if ib == side else ""
                _emit_line(
                    x1 + ox,
                    y1 + oy,
                    x2 + ox,
                    y2 + oy,
                    w,
                    color,
                    px,
                    py,
                    shade=_scfg if not dash else None,
                    op_attr=op_attr,
                    dash=dash,
                    by_element=by_element,
                    ci_hex=ci_hex,
                    cj_hex=cj_hex,
                    ri_vdw=_ri_vdw,
                    rj_vdw=_rj_vdw,
                    fi=_fi,
                    fj=_fj,
                    stroke_i=_si,
                    stroke_j=_sj,
                    stroke_w=_stroke_width,
                )
        else:
            nb = max(1, round(bo))
            w = _bw if nb == 1 else _bw * 0.7
            for ib in range(-nb + 1, nb, 2):
                ox, oy = px * ib * _gap, py * ib * _gap
                _emit_line(
                    x1 + ox,
                    y1 + oy,
                    x2 + ox,
                    y2 + oy,
                    w,
                    color,
                    px,
                    py,
                    shade=_scfg,
                    op_attr=op_attr,
                    dash="",
                    by_element=by_element,
                    ci_hex=ci_hex,
                    cj_hex=cj_hex,
                    ri_vdw=_ri_vdw,
                    rj_vdw=_rj_vdw,
                    fi=_fi,
                    fj=_fj,
                    stroke_i=_si,
                    stroke_j=_sj,
                    stroke_w=_stroke_width,
                )

    # --- Vectorized bond geometry precomputation ---
    # Precompute projected start/end/perpendicular for all bonds in one pass.
    # bond_geom[(ai, aj)] = (x1, y1, x2, y2, px, py) or None if degenerate.
    bond_geom: dict[tuple[int, int], tuple[float, float, float, float, float, float] | None] = {}
    if bonds and not cfg.hide_bonds and bw > 0:
        # Collect unique undirected bond pairs
        _bpairs = [(i, j) for i, j in bonds if i < j]
        if _bpairs:
            _bi = np.array([p[0] for p in _bpairs])
            _bj = np.array([p[1] for p in _bpairs])
            # Vectorized 3D geometry
            _rij = pos[_bj] - pos[_bi]  # (nb, 3)
            _dist = np.sqrt((_rij * _rij).sum(axis=1))  # (nb,)
            _valid = _dist > 1e-6
            _d = np.zeros_like(_rij)
            _d[_valid] = _rij[_valid] / _dist[_valid, None]
            _ri = radii[_bi]
            _rj = radii[_bj]
            _start = pos[_bi] + _d * (_ri * 0.9)[:, None]
            _end = pos[_bj] - _d * (_rj * 0.9)[:, None]
            _dot_check = ((_end - _start) * _d).sum(axis=1)
            _valid &= _dot_check > 0
            # Vectorized 2D projection
            _sx = canvas_w / 2 + scale * (_start[:, 0] - cx)
            _sy = canvas_h / 2 - scale * (_start[:, 1] - cy)
            _ex = canvas_w / 2 + scale * (_end[:, 0] - cx)
            _ey = canvas_h / 2 - scale * (_end[:, 1] - cy)
            _ddx = _ex - _sx
            _ddy = _ey - _sy
            _ln = np.sqrt(_ddx * _ddx + _ddy * _ddy)
            _valid &= _ln >= 1
            _ppx = np.zeros_like(_ln)
            _ppy = np.zeros_like(_ln)
            _ppx[_valid] = -_ddy[_valid] / _ln[_valid]
            _ppy[_valid] = _ddx[_valid] / _ln[_valid]
            # Store results
            for k in range(len(_bpairs)):
                ai_k, aj_k = _bpairs[k]
                if _valid[k]:
                    g = (float(_sx[k]), float(_sy[k]), float(_ex[k]), float(_ey[k]), float(_ppx[k]), float(_ppy[k]))
                    bond_geom[(ai_k, aj_k)] = g
                    bond_geom[(aj_k, ai_k)] = (
                        float(_ex[k]),
                        float(_ey[k]),
                        float(_sx[k]),
                        float(_sy[k]),
                        float(-_ppx[k]),
                        float(-_ppy[k]),
                    )
                else:
                    bond_geom[(ai_k, aj_k)] = bond_geom[(aj_k, ai_k)] = None

    _molecule_insert_idx = len(svg)
    for idx, ai in enumerate(z_order):
        # Flush all vectors whose origin depth <= this atom's depth.  The hidden
        # check is intentionally after the flush so hidden atoms still act as
        # depth markers, keeping vector z-ordering correct.
        while _pv_pos < len(_pending_vecs) and _vec_origins[_pending_vecs[_pv_pos]][2] <= pos[ai][2]:
            vi = _pending_vecs[_pv_pos]
            va = cfg.vectors[vi]
            if not va.draw_on_top:
                # Interleaved: draw shaft now, head later if front-protruding
                _fs = (va.font_size * scale_ratio) if va.font_size is not None else _fs_vec
                _lw = (va.width * scale_ratio) if va.width is not None else _vec_lw
                _draw_arrow_svg(
                    svg,
                    _vec_tail3d[vi],
                    _vec_tip3d[vi],
                    va.color,
                    va.label,
                    _lw,
                    _fs,
                    scale,
                    cx,
                    cy,
                    canvas_w,
                    canvas_h,
                    draw_head=not _vec_head_front[vi],
                )
            _pv_pos += 1

        if ai in hidden:
            continue

        # Drain NCI patches and pore spheres that belong behind this atom
        if nci_lobes_flat:
            _drain_nci(float(pos[ai][2]))
        if pore_spheres_flat:
            _drain_pore_spheres(float(pos[ai][2]))

        xi, yi = _px[ai], _py[ai]
        is_image = _is_image[ai]
        if is_image:
            atom_op = cfg.periodic_image_opacity
        elif struct_opacities[ai] is not None:
            atom_op = struct_opacities[ai]
        else:
            atom_op = 1.0
        # Per-atom fill override: composes multiplicatively with whatever is above
        # so fading a specific atom on a half-opaque overlay still fades it further.
        if _atom_only_op[ai] is not None:
            atom_op = min(atom_op, _atom_only_op[ai])
        op_attr_atom = f' opacity="{atom_op:.2f}"' if atom_op < 1.0 else ""

        # Atom graphics / labels — per-atom config for style regions.
        # NCI centroid nodes ("*") are structural overlays — always use the
        # base config so they stay visible regardless of region styling.
        acfg = cfg if symbols[ai] == "*" else (_acfg[ai] if _acfg is not None else cfg)
        _atom_layer_start = len(svg)
        if acfg.skeletal_style:
            if not is_image:
                skeletal_atom_svg(
                    svg,
                    ai,
                    xi,
                    yi,
                    symbols=symbols,
                    colors=colors,
                    fs_label=fs_label,
                    fog_enabled=cfg.fog,
                    fog_rgb=fog_rgb,
                    fog_f=fog_f,
                    label_color_override=acfg.skeletal_label_color,
                )
        else:
            # Atom circle (gradient or flat fill)
            _sw_ai = _atom_sw[ai] if _atom_sw is not None else sw
            _grad_ai = _atom_use_grad[ai] if _atom_use_grad is not None else use_grad
            _stroke_src = struct_stroke_colors[ai] or acfg.atom_stroke_color
            _stroke_atom = _color_hex[ai] if _stroke_src == "atom" else _stroke_src
            dof_attr = f' filter="url(#dof{dof_buckets[ai]})"' if cfg.dof else ""
            if ai in glow_indices:
                _glow_fill = colors[ai].blend(WHITE, acfg.atom_wash).hex if acfg.atom_wash > 0 else _color_hex[ai]
                if cfg.fog:
                    _glow_fill = blend_fog(_glow_fill, fog_rgb, fog_f[ai])
                svg.append(
                    f'  <circle cx="{xi:.1f}" cy="{yi:.1f}" r="{radii[ai] * scale:.1f}" '
                    f'fill="{_glow_fill}" filter="url(#glow)"{op_attr_atom}/>'
                )
            if _grad_ai:
                if use_per_atom_grad:
                    grad_id = f"g{ai}"
                    fs_atom = atom_fog_stroke[ai]
                else:
                    gid_suffix = f"{a_nums[ai]}_{_color_hex[ai][1:]}"
                    if _acfg is not None:
                        gid_suffix += f"_{id(acfg) & 0xFFFF:04x}"
                    grad_id = f"g{gid_suffix}"
                    fs_atom = _stroke_atom
                svg.append(
                    f'  <circle cx="{xi:.1f}" cy="{yi:.1f}" r="{radii[ai] * scale:.1f}" '
                    f'fill="url(#{grad_id})" stroke="{fs_atom}" stroke-width="{_sw_ai:.1f}"{op_attr_atom}{dof_attr}/>'
                )
            else:
                fill = colors[ai].blend(WHITE, acfg.atom_wash).hex if acfg.atom_wash > 0 else _color_hex[ai]
                stroke = _stroke_atom
                if cfg.fog:
                    fill = blend_fog(fill, fog_rgb, fog_f[ai])
                    stroke = blend_fog(stroke, fog_rgb, fog_f[ai])
                svg.append(
                    f'  <circle cx="{xi:.1f}" cy="{yi:.1f}" r="{radii[ai] * scale:.1f}" '
                    f'fill="{fill}" stroke="{stroke}" stroke-width="{_sw_ai:.1f}"{op_attr_atom}{dof_attr}/>'
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
            # Defer this atom's layers when its config has atoms_above_bonds
            if acfg.atoms_above_bonds and len(svg) > _atom_layer_start:
                _deferred_atom_layers.extend(svg[_atom_layer_start:])
                del svg[_atom_layer_start:]

        # Bonds to deeper atoms (adjacency list → O(degree) instead of O(n))
        if not cfg.hide_bonds and bw > 0:
            for aj_int in bond_adj.get(ai, ()):
                if aj_int in hidden or _z_rank[aj_int] <= idx:
                    continue
                battrs = bonds[(ai, aj_int)]
                # Use periodic_image_opacity if either endpoint is an image atom
                _aj_image = _is_image[aj_int]
                _aj_struct_op = struct_opacities[aj_int] if not _aj_image else None
                _ai_struct_op = struct_opacities[ai]
                if is_image or _aj_image:
                    bond_op = cfg.periodic_image_opacity
                elif _ai_struct_op is not None or _aj_struct_op is not None:
                    bond_op = min(v for v in (_ai_struct_op, _aj_struct_op) if v is not None)
                else:
                    bond_op = 1.0
                # Diffuse GIF: fade stretched bonds
                _diff_op = _diffuse_op.get((ai, aj_int))
                if _diff_op is not None:
                    bond_op = min(bond_op, _diff_op)
                if bond_op < 0.01:
                    continue  # skip invisible bonds
                add_bond(
                    ai,
                    aj_int,
                    battrs.order,
                    battrs.style,
                    opacity=bond_op,
                    color_override=battrs.color,
                    width_override=battrs.width,
                    outline_width_override=battrs.outline_width,
                    outline_color_override=battrs.outline_color,
                )

    # Insert edge stroke shadow layer at the base of the molecule group
    if _bond_outline_layer:
        svg[_molecule_insert_idx:_molecule_insert_idx] = _bond_outline_layer

    # NCI patches in front of all atoms (z_depth > frontmost atom)
    while nci_lobe_idx < len(nci_lobes_flat):
        svg.extend(nci_lobes_flat[nci_lobe_idx][1])
        nci_lobe_idx += 1

    # Flush any vectors whose origin is in front of all atoms
    while _pv_pos < len(_pending_vecs):
        vi = _pending_vecs[_pv_pos]
        va = cfg.vectors[vi]
        if not va.draw_on_top:
            _fs = (va.font_size * scale_ratio) if va.font_size is not None else _fs_vec
            _lw = (va.width * scale_ratio) if va.width is not None else _vec_lw
            _draw_arrow_svg(
                svg, _vec_tail3d[vi], _vec_tip3d[vi], va.color, va.label, _lw, _fs, scale, cx, cy, canvas_w, canvas_h
            )
        _pv_pos += 1
    if _deferred_atom_layers:
        svg.extend(_deferred_atom_layers)

    # --- Second pass: redraw arrowheads that protrude in front of their host atom ---
    # These were skipped in the first pass (_draw_vector_arrow) so that the shaft
    # is still painter-sorted correctly, but the head must appear on top of the atom.
    if cfg.vectors:
        for vi in range(len(cfg.vectors)):
            va = cfg.vectors[vi]
            if not va.draw_on_top and _vec_head_front[vi]:
                _fs = (va.font_size * scale_ratio) if va.font_size is not None else _fs_vec
                _lw = (va.width * scale_ratio) if va.width is not None else _vec_lw
                _draw_arrow_svg(
                    svg,
                    _vec_tail3d[vi],
                    _vec_tip3d[vi],
                    va.color,
                    va.label,
                    _lw,
                    _fs,
                    scale,
                    cx,
                    cy,
                    canvas_w,
                    canvas_h,
                    draw_shaft=False,
                )

    # Drain remaining pore spheres (in front of all atoms)
    if pore_spheres_flat:
        _drain_pore_spheres(float("inf"))

    # --- Front MO orbital lobes (on top of molecule) ---
    if cfg.mo_contours is not None:
        assert mo_is_front is not None
        svg.extend(
            mo_front_lobes_svg(
                cfg.mo_contours,
                mo_is_front,
                cfg.surface_opacity,
                scale,
                cx,
                cy,
                canvas_w,
                canvas_h,
                surface_style=cfg.surface_style,
                stroke_width=_mesh_sw,
                mesh_inner_width=_mesh_inner_sw,
            )
        )

    # --- Density surface (stacked z-layers on top of molecule) ---
    if cfg.dens_contours is not None:
        svg.extend(
            dens_layers_svg(
                cfg.dens_contours,
                cfg.surface_opacity,
                scale,
                cx,
                cy,
                canvas_w,
                canvas_h,
                surface_style=cfg.surface_style,
                stroke_width=_mesh_sw,
                mesh_inner_width=_mesh_inner_sw,
            )
        )

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

    # --- Final pass: Vectors with draw_on_top=True ---
    if cfg.vectors:
        for vi, va in enumerate(cfg.vectors):
            if va.draw_on_top:
                _fs = (va.font_size * scale_ratio) if va.font_size is not None else _fs_vec
                _lw = (va.width * scale_ratio) if va.width is not None else _vec_lw
                _draw_arrow_svg(
                    svg,
                    _vec_tail3d[vi],
                    _vec_tip3d[vi],
                    va.color,
                    va.label,
                    _lw,
                    _fs,
                    scale,
                    cx,
                    cy,
                    canvas_w,
                    canvas_h,
                )

    # --- Colorbar (right side) ---
    if cfg.cbar and cbar_vmin is not None and cbar_vmax is not None and cbar_palette is not None:
        svg.extend(colorbar_svg(cbar_vmin, cbar_vmax, cbar_palette, canvas_w, canvas_h, fs_label, cfg.label_color))

    svg.append("</svg>")
    raw = "\n".join(svg)
    # SVG id= values are global in an HTML document — multiple renders in the same
    # Jupyter notebook page collide, causing atoms/gradients from the first render to
    # appear in all subsequent ones.  Prefix every id, href, and url() reference with
    # a unique token so each SVG is self-contained regardless of embedding context.
    # Skip when _unique_ids=False (GIF frames: converted to PNG immediately, never shown as SVG).
    if _unique_ids:
        p = f"x{next(_render_counter)}"
        raw = raw.replace('id="', f'id="{p}')
        raw = raw.replace('href="#', f'href="#{p}')
        raw = raw.replace("url(#", f"url(#{p}")
    return raw


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_aromatic_rings(
    graph,
    bonds: dict[tuple[int, int], _BondAttrs],
) -> list[set[int]]:
    """Return list of aromatic ring atom index sets, with fallback when graph has no ring data.

    If the graph has ``aromatic_rings``, use it. If any bond with order in (1.3, 1.7)
    is not covered by those rings, build an aromatic subgraph and use minimum_cycle_basis.
    """
    aromatic_rings = [set(r) for r in graph.graph.get("aromatic_rings", [])]
    aromatic_ring_edges = set()
    for ring in aromatic_rings:
        rl = list(ring)
        for ii in range(len(rl)):
            for jj in range(ii + 1, len(rl)):
                if (rl[ii], rl[jj]) in bonds or (rl[jj], rl[ii]) in bonds:
                    aromatic_ring_edges.add((min(rl[ii], rl[jj]), max(rl[ii], rl[jj])))
    missing = False
    for (i, j), attrs in bonds.items():
        if i < j and 1.3 < attrs.order < 1.7 and (i, j) not in aromatic_ring_edges:
            missing = True
            break
    if missing:
        arom_g = nx.Graph()
        for (i, j), attrs in bonds.items():
            if i < j and 1.3 < attrs.order < 1.7:
                arom_g.add_edge(i, j)
        if arom_g.number_of_edges() > 0:
            aromatic_rings = [set(c) for c in nx.minimum_cycle_basis(arom_g)]
    return aromatic_rings


def _draw_arrow_svg(
    svg: list[str],
    tail3d: np.ndarray,
    tip3d: np.ndarray,
    color: str,
    label: str,
    lw: float,
    fs: float,
    scale: float,
    cx: float,
    cy: float,
    cw: float,
    ch: float,
    draw_shaft: bool = True,
    draw_head: bool = True,
) -> None:
    """SVG arrow rendering.

    When the 2D projected length is shorter than the arrowhead size, a dot
    (tip facing viewer) or 'x' (tip facing away) is drawn instead and the
    label is suppressed.  The label reappears automatically once the arrow
    is long enough to draw a proper arrowhead, where it is centred on the
    arrowhead tip.
    """
    ox, oy = _proj(tail3d, scale, cx, cy, cw, ch)
    tx, ty = _proj(tip3d, scale, cx, cy, cw, ch)
    dx, dy = tx - ox, ty - oy
    px_len = (dx * dx + dy * dy) ** 0.5
    arr = max(lw * 3.5, 7.0)

    # If the projected length is shorter than the arrowhead itself, draw a dot or
    # 'x' and suppress the label.
    if px_len < arr:
        if tip3d[2] > tail3d[2]:
            # Facing viewer: draw a dot
            r = max(lw * 0.8, 2.0)
            svg.append(f'  <circle cx="{ox:.1f}" cy="{oy:.1f}" r="{r:.1f}" fill="{color}"/>')
        else:
            # Facing away: draw an 'x'
            r = max(lw * 0.8, 2.0)
            svg.append(
                f'  <line x1="{ox - r:.1f}" y1="{oy - r:.1f}" x2="{ox + r:.1f}" y2="{oy + r:.1f}" '
                f'stroke="{color}" stroke-width="{lw:.1f}" stroke-linecap="round"/>'
            )
            svg.append(
                f'  <line x1="{ox - r:.1f}" y1="{oy + r:.1f}" x2="{ox + r:.1f}" y2="{oy - r:.1f}" '
                f'stroke="{color}" stroke-width="{lw:.1f}" stroke-linecap="round"/>'
            )
        return

    if draw_shaft:
        # Stop shaft at arrowhead base so the round linecap doesn't poke through the head
        frac = max(0.0, 1.0 - arr / px_len)
        sx, sy = ox + dx * frac, oy + dy * frac
        svg.append(
            f'  <line x1="{ox:.1f}" y1="{oy:.1f}" x2="{sx:.1f}" y2="{sy:.1f}" '
            f'stroke="{color}" stroke-width="{lw:.1f}" stroke-linecap="round"/>'
        )

    if draw_head:
        nvx, nvy = dx / px_len, dy / px_len
        pvx, pvy = -nvy, nvx
        p1x = tx - nvx * arr + pvx * arr * 0.38
        p1y = ty - nvy * arr + pvy * arr * 0.38
        p2x = tx - nvx * arr - pvx * arr * 0.38
        p2y = ty - nvy * arr - pvy * arr * 0.38
        svg.append(f'  <polygon points="{tx:.1f},{ty:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" fill="{color}"/>')
        if label:
            sep = fs * 0.65
            lx = tx + nvx * sep
            ly = ty + nvy * sep + fs * 0.35
            svg.append(
                f'  <text x="{lx:.1f}" y="{ly:.1f}" font-size="{fs:.1f}" fill="{color}" '
                f'font-family="Arial,sans-serif" text-anchor="middle" font-weight="bold">{label}</text>'
            )


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
    from xyzrender.annotations import AngleLabel, AtomValueLabel, BondLabel, CentroidLabel, DihedralLabel

    svg: list[str] = []
    col = cfg.label_color

    # Separate passes for each annotation type
    dihedral_idx = 0
    for ann in cfg.annotations:
        if isinstance(ann, AtomValueLabel):
            xi, yi = _proj(pos[ann.index], scale, cx, cy, canvas_w, canvas_h)
            if ann.on_atom:
                # NB: overlaps with --idx labels which also render at (xi, yi);
                # use on_atom=False (--stereo label) when combining with --idx.
                svg.append(_text_svg(xi, yi, ann.text, fs, col))
            else:
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

        elif isinstance(ann, CentroidLabel):
            centroid = pos[list(ann.atoms)].mean(axis=0)
            cx2, cy2 = _proj(centroid, scale, cx, cy, canvas_w, canvas_h)
            svg.append(_text_svg(cx2, cy2, ann.text, fs, col))

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
