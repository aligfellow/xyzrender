"""Density isosurface extraction and SVG rendering."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from xyzrender.mo import (
    _MIN_LOOP_PERIMETER,
    _UPSAMPLE_FACTOR,
    LobeContour2D,
    MOContours,
    _gaussian_blur_2d,
    _loop_perimeter,
    _mo_combined_path_d,
    _resample_loop,
    chain_segments,
    compute_grid_positions,
    cube_corners_ang,
    marching_squares,
)

if TYPE_CHECKING:
    import networkx as nx

    from xyzrender.cube import CubeData
    from xyzrender.types import RenderConfig

logger = logging.getLogger(__name__)

_N_LAYERS = 6  # number of threshold levels for depth-graded rendering
_PROJ_MULT = 2  # projection grid multiplier to reduce Moiré when tilted
_DENS_BLUR = 1.8  # larger blur (vs MO 0.8) to fill gaps in tilted projection


# ---------------------------------------------------------------------------
# Multi-threshold projection: one 2D projection, many contour rings
# ---------------------------------------------------------------------------


def build_density_contours(
    cube: CubeData,
    isovalue: float,
    color: str,
    *,
    rot: np.ndarray | None = None,
    atom_centroid: np.ndarray | None = None,
    target_centroid: np.ndarray | None = None,
    pos_flat_ang: np.ndarray | None = None,
    flat_indices: np.ndarray | None = None,
    fixed_bounds: tuple[float, float, float, float] | None = None,
    n_layers: int = _N_LAYERS,
) -> MOContours:
    """Build density contours from a parsed cube file.

    Projects all above-isovalue voxels to a 2D grid (max-intensity), then
    extracts contours at *n_layers* threshold levels.  Outer rings (at the
    base isovalue) cover the full surface extent; inner rings (at higher
    thresholds) are smaller concentric shapes.  Stacking them with per-layer
    opacity produces a depth-graded appearance.
    """
    n1, n2, n3 = cube.grid_shape
    base_res = max(n1, n2, n3)

    if pos_flat_ang is None:
        pos_flat_ang = compute_grid_positions(cube)

    if flat_indices is None:
        mask = cube.grid_data >= isovalue
        flat_indices = np.flatnonzero(mask)

    values_flat = cube.grid_data.ravel()
    lobe_pos = pos_flat_ang[flat_indices].copy()
    lobe_vals = values_flat[flat_indices]

    # Rotate positions
    if rot is not None:
        if atom_centroid is not None:
            lobe_pos -= atom_centroid
        lobe_pos = lobe_pos @ rot.T
        if target_centroid is not None:
            lobe_pos += target_centroid

    z_depth = float(lobe_pos[:, 2].mean())

    # Tight bounds from actual voxel positions (for canvas fitting)
    vx_xmin, vx_xmax = float(lobe_pos[:, 0].min()), float(lobe_pos[:, 0].max())
    vx_ymin, vx_ymax = float(lobe_pos[:, 1].min()), float(lobe_pos[:, 1].max())
    vx_xpad = (vx_xmax - vx_xmin) * 0.02 + 1e-9
    vx_ypad = (vx_ymax - vx_ymin) * 0.02 + 1e-9

    # 2D projection bounds (use tight voxel bounds, not full cube grid)
    if fixed_bounds is not None:
        x_min, x_max, y_min, y_max = fixed_bounds
    else:
        x_min = vx_xmin - vx_xpad
        x_max = vx_xmax + vx_xpad
        y_min = vx_ymin - vx_ypad
        y_max = vx_ymax + vx_ypad

    # Use a finer projection grid to avoid artifacts when the
    # molecule is tilted relative to the cube grid axes.
    proj_res = base_res * _PROJ_MULT

    # Max-intensity projection to 2D (single pass over all voxels)
    grid_2d = np.zeros((proj_res, proj_res))
    lx, ly = lobe_pos[:, 0], lobe_pos[:, 1]
    xi = np.clip(((lx - x_min) / (x_max - x_min) * (proj_res - 1)).astype(int), 0, proj_res - 1)
    yi = np.clip(((ly - y_min) / (y_max - y_min) * (proj_res - 1)).astype(int), 0, proj_res - 1)
    np.maximum.at(grid_2d, (yi, xi), lobe_vals)

    # Crop to non-zero bounding box + blur padding
    nz_rows, nz_cols = np.nonzero(grid_2d)
    if len(nz_rows) == 0:
        return MOContours(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, pos_color=color, neg_color=color)

    pad = max(3, int(_DENS_BLUR * 4) + 1)
    r0 = max(0, int(nz_rows.min()) - pad)
    r1 = min(proj_res, int(nz_rows.max()) + pad + 1)
    c0 = max(0, int(nz_cols.min()) - pad)
    c1 = min(proj_res, int(nz_cols.max()) + pad + 1)
    cropped = grid_2d[r0:r1, c0:c1]

    # Blur (on cropped grid — much smaller than full projection)
    blurred = np.maximum(_gaussian_blur_2d(cropped, _DENS_BLUR), 0.0)

    # Multi-level contour extraction on the blurred grid (before upsample).
    # Running marching squares on the smaller grid is much faster;
    # coordinates are scaled by _up to match the final resolution.
    _up = max(1, _UPSAMPLE_FACTOR // _PROJ_MULT + 1)
    above = blurred[blurred > isovalue]
    if above.size == 0:
        return MOContours(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, pos_color=color, neg_color=color)

    # Use 85th percentile as upper bound to avoid nuclear-peak outliers
    upper = float(np.percentile(above, 85))
    upper = max(upper, isovalue * 1.5)  # ensure at least some spread
    thresholds = np.geomspace(isovalue, upper, n_layers)

    scale_offset = np.array([r0 * _up, c0 * _up])
    res = proj_res * _up

    layers: list[LobeContour2D] = []
    for threshold in thresholds:
        raw_loops = chain_segments(marching_squares(blurred, float(threshold)))
        # Scale from blurred-grid coords to upsampled-grid coords + crop offset
        offset_loops = [loop * _up + scale_offset for loop in raw_loops]
        loops = [_resample_loop(lp) for lp in offset_loops if _loop_perimeter(lp) >= _MIN_LOOP_PERIMETER]
        if loops:
            layers.append(LobeContour2D(loops=loops, phase="pos", z_depth=z_depth))

    total_loops = sum(len(lc.loops) for lc in layers)
    if total_loops == 0:
        logger.warning("No density contours at isovalue %.4g — try a smaller value with --iso", isovalue)
    else:
        logger.debug("Density contours: %d layers (%d loops total, isovalue=%.4g)", len(layers), total_loops, isovalue)

    return MOContours(
        lobes=layers,
        resolution=res,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        pos_color=color,
        neg_color=color,
    )


# ---------------------------------------------------------------------------
# Per-frame density recomputation for gif-rot
# ---------------------------------------------------------------------------


def recompute_dens(graph: nx.Graph, config: RenderConfig, dens_data: dict) -> None:
    """Recompute density contours for the current graph orientation.

    Caches flat indices, grid positions, and bounding sphere on first call;
    only the rotation changes per frame.
    """
    from xyzrender.utils import kabsch_rotation

    cube_data = dens_data["cube_data"]

    # Cache invariants on first call
    if "flat_indices" not in dens_data:
        mask = cube_data.grid_data >= dens_data["isovalue"]
        dens_data["flat_indices"] = np.flatnonzero(mask)
        dens_data["pos_flat_ang"] = compute_grid_positions(cube_data)
    if "_orig_atoms" not in dens_data:
        orig = np.array([p for _, p in cube_data.atoms], dtype=float)
        dens_data["_orig_atoms"] = orig
        dens_data["_atom_centroid"] = orig.mean(axis=0)
    if "_bounding_radius" not in dens_data:
        corners = cube_corners_ang(cube_data)
        r_max = float(np.linalg.norm(corners - dens_data["_atom_centroid"], axis=1).max())
        dens_data["_bounding_radius"] = r_max + r_max * 0.01 + 1e-9

    orig = dens_data["_orig_atoms"]
    atom_centroid = dens_data["_atom_centroid"]
    curr = np.array([graph.nodes[i]["position"] for i in graph.nodes()], dtype=float)
    target_centroid = curr.mean(axis=0)

    r = dens_data["_bounding_radius"]
    fixed_bounds = (
        float(target_centroid[0] - r),
        float(target_centroid[0] + r),
        float(target_centroid[1] - r),
        float(target_centroid[1] + r),
    )

    rot = kabsch_rotation(orig, curr)

    config.dens_contours = build_density_contours(
        cube_data,
        isovalue=dens_data["isovalue"],
        color=dens_data["color"],
        rot=rot,
        atom_centroid=atom_centroid,
        target_centroid=target_centroid,
        flat_indices=dens_data["flat_indices"],
        pos_flat_ang=dens_data["pos_flat_ang"],
        fixed_bounds=fixed_bounds,
        n_layers=dens_data.get("n_layers", _N_LAYERS),
    )
    config.surface_opacity = dens_data["surface_opacity"]


# ---------------------------------------------------------------------------
# Density SVG rendering
# ---------------------------------------------------------------------------


_DENS_BASE_OPACITY = 0.95  # base opacity for density surface (scaled by surface_opacity)


def dens_layers_svg(
    dens: MOContours,
    surface_opacity: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> list[str]:
    """Render density threshold layers as stacked semi-transparent paths.

    Each layer gets a fraction of the total opacity.  Where more layers
    overlap (inner rings at higher thresholds stack on top of outer rings),
    opacity accumulates, creating a depth-graded appearance from edge to center.
    """
    n = len(dens.lobes)
    if n == 0:
        return []
    per_layer = _DENS_BASE_OPACITY * surface_opacity / n
    color = dens.pos_color
    lines: list[str] = []
    for lobe in dens.lobes:
        d_all = _mo_combined_path_d(lobe.loops, dens, scale, cx, cy, canvas_w, canvas_h)
        if d_all:
            lines.append(f'  <g opacity="{per_layer:.3f}">')
            lines.append(f'    <path d="{d_all}" fill="{color}" fill-rule="evenodd" stroke="none"/>')
            lines.append("  </g>")
    return lines
