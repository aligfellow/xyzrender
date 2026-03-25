"""MO (molecular orbital) contour extraction, classification, and SVG rendering."""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, cast

import numpy as np

from xyzrender.contours import (
    BLUR_SIGMA,
    MIN_LOBE_VOLUME_BOHR3,
    MIN_LOOP_PERIMETER,
    UPSAMPLE_FACTOR,
    Lobe3D,
    LobeContour2D,
    MOContours,
    SurfaceContours,
    chain_segments,
    combined_path_d,
    compute_grid_positions,
    cube_corners_ang,
    gaussian_blur_2d,
    loop_perimeter,
    marching_squares,
    resample_loop,
    upsample_2d,
)

if TYPE_CHECKING:
    import networkx as nx

    from xyzrender.cube import CubeData
    from xyzrender.types import MOParams, RenderConfig

logger = logging.getLogger(__name__)

# Lobe classification (physical units)
_COPLANARITY_THRESHOLD_ANG = 0.3  # Å — below this z-depth difference, lobes are coplanar


# ---------------------------------------------------------------------------
# 3D connected component labeling (BFS flood-fill)
# ---------------------------------------------------------------------------


def find_3d_lobes(grid_3d: np.ndarray, isovalue: float, steps: np.ndarray | None = None) -> list[Lobe3D]:
    """Find connected 3D orbital lobes at ±isovalue via BFS flood-fill."""
    shape = grid_3d.shape
    s1, s2 = shape[1] * shape[2], shape[2]
    lobes: list[Lobe3D] = []

    # Derive cell count threshold from physical volume and voxel size
    if steps is not None:
        voxel_vol = abs(float(np.linalg.det(steps)))
        min_cells = max(2, int(MIN_LOBE_VOLUME_BOHR3 / voxel_vol + 0.5))
    else:
        min_cells = 5  # fallback for callers without cube metadata
    logger.debug("Voxel volume: %.4g Bohr³, min lobe cells: %d", voxel_vol if steps is not None else 0.0, min_cells)

    for phase in ("pos", "neg"):
        mask = grid_3d >= isovalue if phase == "pos" else grid_3d <= -isovalue
        visited = np.zeros(shape, dtype=bool)
        visited[~mask] = True  # non-mask cells don't need visiting

        candidates = np.argwhere(mask)
        for idx in range(len(candidates)):
            i, j, k = int(candidates[idx, 0]), int(candidates[idx, 1]), int(candidates[idx, 2])
            if visited[i, j, k]:
                continue

            component: list[int] = []
            queue = deque([(i, j, k)])
            visited[i, j, k] = True
            while queue:
                ci, cj, ck = queue.popleft()
                component.append(ci * s1 + cj * s2 + ck)
                for di, dj, dk in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
                    ni, nj, nk = ci + di, cj + dj, ck + dk
                    if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
                        if not visited[ni, nj, nk]:
                            visited[ni, nj, nk] = True
                            queue.append((ni, nj, nk))

            if len(component) >= min_cells:
                lobes.append(Lobe3D(flat_indices=np.array(component, dtype=np.intp), phase=phase))
            else:
                logger.debug(
                    "Discarded %s component with %d voxels (< %d minimum)",
                    phase,
                    len(component),
                    min_cells,
                )

    logger.debug("Found %d 3D lobes at isovalue %.4g", len(lobes), isovalue)
    return lobes


# ---------------------------------------------------------------------------
# Per-lobe 2D projection + contouring
# ---------------------------------------------------------------------------


def _project_lobe_2d(
    lobe: Lobe3D,
    pos_flat_ang: np.ndarray,
    values_flat: np.ndarray,
    resolution: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    isovalue: float,
    *,
    rot: np.ndarray | None = None,
    atom_centroid: np.ndarray | None = None,
    target_centroid: np.ndarray | None = None,
    blur_sigma: float = BLUR_SIGMA,
    upsample_factor: int = UPSAMPLE_FACTOR,
) -> LobeContour2D | None:
    """Project one 3D lobe to 2D, blur, upsample, and extract contours."""
    lobe_pos = pos_flat_ang[lobe.flat_indices].copy()
    lobe_vals = values_flat[lobe.flat_indices]

    # Rotate only this lobe's positions
    if rot is not None:
        if atom_centroid is not None:
            lobe_pos -= atom_centroid
        lobe_pos = lobe_pos @ rot.T
        if target_centroid is not None:
            lobe_pos += target_centroid

    z_depth = float(lobe_pos[:, 2].mean())

    # Bin lobe values into a 2D grid (max-intensity for pos, min for neg)
    grid_2d = np.zeros((resolution, resolution))
    lx = lobe_pos[:, 0]
    ly = lobe_pos[:, 1]
    xi = np.clip(((lx - x_min) / (x_max - x_min) * (resolution - 1)).astype(int), 0, resolution - 1)
    yi = np.clip(((ly - y_min) / (y_max - y_min) * (resolution - 1)).astype(int), 0, resolution - 1)

    if lobe.phase == "pos":
        np.maximum.at(grid_2d, (yi, xi), lobe_vals)
    else:
        np.minimum.at(grid_2d, (yi, xi), lobe_vals)

    # Crop to lobe's bounding box + blur kernel padding
    nz_rows, nz_cols = np.nonzero(grid_2d)
    if len(nz_rows) == 0:
        return None
    pad = max(3, int(blur_sigma * 4) + 1)
    r0 = max(0, int(nz_rows.min()) - pad)
    r1 = min(resolution, int(nz_rows.max()) + pad + 1)
    c0 = max(0, int(nz_cols.min()) - pad)
    c1 = min(resolution, int(nz_cols.max()) + pad + 1)
    cropped = grid_2d[r0:r1, c0:c1]

    # Blur + upsample the cropped region only
    blurred = gaussian_blur_2d(cropped, blur_sigma)
    if lobe.phase == "pos":
        blurred = np.maximum(blurred, 0.0)
    else:
        blurred = np.minimum(blurred, 0.0)

    upsampled = upsample_2d(blurred, upsample_factor)

    # Extract contours on cropped grid
    if lobe.phase == "pos":
        raw_loops = chain_segments(marching_squares(upsampled, isovalue))
    else:
        raw_loops = chain_segments(marching_squares(-upsampled, isovalue))

    # Offset contour coords back to full-grid space
    offset = np.array([r0 * upsample_factor, c0 * upsample_factor])
    offset_loops = [loop + offset for loop in raw_loops]

    loops = [resample_loop(lp) for lp in offset_loops if loop_perimeter(lp) >= MIN_LOOP_PERIMETER]

    if not loops:
        return None
    cent_3d = (float(lobe_pos[:, 0].mean()), float(lobe_pos[:, 1].mean()), z_depth)
    return LobeContour2D(loops=loops, phase=lobe.phase, z_depth=z_depth, centroid_3d=cent_3d)


# ---------------------------------------------------------------------------
# Integration: build MO contours from cube data
# ---------------------------------------------------------------------------


def build_mo_contours(
    cube: CubeData,
    params: MOParams,
    *,
    rot: np.ndarray | None = None,
    atom_centroid: np.ndarray | None = None,
    target_centroid: np.ndarray | None = None,
    resolution: int | None = None,
    lobes_3d: list[Lobe3D] | None = None,
    pos_flat_ang: np.ndarray | None = None,
    fixed_bounds: tuple[float, float, float, float] | None = None,
) -> SurfaceContours:
    """Build MO contour data from a parsed cube file.

    Each 3D lobe is projected and contoured independently.  Surface
    appearance (isovalue, colors, blur, upsampling) is driven by *params*.

    Pre-computed *lobes_3d*, *pos_flat_ang*, and *fixed_bounds* may be
    passed to avoid redundant computation across GIF frames.

    Parameters
    ----------
    cube:
        Parsed Gaussian cube file containing the orbital data.
    params:
        MO surface parameters (isovalue, colors, blur, upsampling).
    rot:
        Optional 3x3 rotation matrix to align the cube grid with the
        current atom orientation (output of :func:`~xyzrender.utils.kabsch_rotation`).
    atom_centroid:
        Centroid of the original cube atom positions (Å).
    target_centroid:
        Centroid of the current (possibly rotated) atom positions (Å).
    resolution:
        Override the projection grid resolution (default: largest grid dimension).
    lobes_3d:
        Pre-computed 3D lobes (cached between GIF frames).
    pos_flat_ang:
        Pre-computed flattened grid positions in Å (cached between GIF frames).
    fixed_bounds:
        Fixed ``(x_min, x_max, y_min, y_max)`` in Å (cached between GIF frames).

    Returns
    -------
    SurfaceContours
        Projection data and contour loops ready for SVG rendering.
    """
    from xyzrender.colors import resolve_color

    isovalue = params.isovalue
    pos_color = resolve_color(params.pos_color)
    neg_color = resolve_color(params.neg_color)
    blur_sigma = params.blur_sigma
    upsample_factor = params.upsample_factor
    n1, n2, n3 = cube.grid_shape
    base_res = resolution or max(n1, n2, n3)

    # Pre-compute grid positions in Angstrom (reuse if cached)
    if pos_flat_ang is None:
        pos_flat_ang = compute_grid_positions(cube)

    values_flat = cube.grid_data.ravel()

    # 2D bounds: use fixed bounds (gif-rot) or compute from cube corners
    if fixed_bounds is not None:
        x_min, x_max, y_min, y_max = fixed_bounds
    else:
        corners = cube_corners_ang(cube)
        if rot is not None:
            if atom_centroid is not None:
                corners = corners - atom_centroid
            corners = corners @ rot.T
            if target_centroid is not None:
                corners = corners + target_centroid
        x_min, x_max = float(corners[:, 0].min()), float(corners[:, 0].max())
        y_min, y_max = float(corners[:, 1].min()), float(corners[:, 1].max())
        x_pad = (x_max - x_min) * 0.01 + 1e-9
        y_pad = (y_max - y_min) * 0.01 + 1e-9
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

    # Find 3D lobes (reuse if cached)
    if lobes_3d is None:
        lobes_3d = find_3d_lobes(cube.grid_data, isovalue, steps=cube.steps)

    # Project and contour each lobe independently (rotation per-lobe)
    lobe_contours: list[LobeContour2D] = []
    for lobe in lobes_3d:
        lc = _project_lobe_2d(
            lobe,
            pos_flat_ang,
            values_flat,
            base_res,
            x_min,
            x_max,
            y_min,
            y_max,
            isovalue,
            rot=rot,
            atom_centroid=atom_centroid,
            target_centroid=target_centroid,
            blur_sigma=blur_sigma,
            upsample_factor=upsample_factor,
        )
        if lc is not None:
            lobe_contours.append(lc)

    # Sort back-to-front by z-depth
    lobe_contours.sort(key=lambda lc: lc.z_depth)

    res = base_res * upsample_factor
    total_loops = sum(len(lc.loops) for lc in lobe_contours)
    if total_loops == 0:
        logger.warning(
            "No MO contours at isovalue %.4g — try a smaller value with --isovalue",
            isovalue,
        )

    logger.debug(
        "MO contours: %d lobes (%d loops total, isovalue=%.4g)",
        len(lobe_contours),
        total_loops,
        isovalue,
    )
    # Compute tight Angstrom extent from actual contour loops
    lobe_x_min = lobe_x_max = lobe_y_min = lobe_y_max = None
    all_loops = [loop for lc in lobe_contours for loop in lc.loops]
    if all_loops:
        pts = np.concatenate(all_loops, axis=0)
        res_m1 = max(res - 1, 1)
        lobe_x_min = float(x_min + (pts[:, 1].min() / res_m1) * (x_max - x_min))
        lobe_x_max = float(x_min + (pts[:, 1].max() / res_m1) * (x_max - x_min))
        lobe_y_min = float(y_min + (pts[:, 0].min() / res_m1) * (y_max - y_min))
        lobe_y_max = float(y_min + (pts[:, 0].max() / res_m1) * (y_max - y_min))

    return MOContours(
        lobes=lobe_contours,
        resolution=res,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        pos_color=pos_color,
        neg_color=neg_color,
        lobe_x_min=lobe_x_min,
        lobe_x_max=lobe_x_max,
        lobe_y_min=lobe_y_min,
        lobe_y_max=lobe_y_max,
    )


# ---------------------------------------------------------------------------
# MO lobe classification (front/back)
# ---------------------------------------------------------------------------


def classify_mo_lobes(lobes: list[LobeContour2D], mol_z: float) -> list[bool]:
    """Classify each lobe as front (True) or back (False).

    Pairs opposite-phase lobes by 3D centroid proximity; within each pair
    the higher-z lobe is front.  Unpaired lobes use the molecule z-centroid.
    """
    n = len(lobes)
    if n == 0:
        return []
    is_front: list[bool | None] = [None] * n

    # Build all candidate opposite-phase pairs sorted by 3D distance
    pos_idx = [i for i in range(n) if lobes[i].phase == "pos"]
    neg_idx = [i for i in range(n) if lobes[i].phase == "neg"]
    candidates = []
    for pi in pos_idx:
        pc = lobes[pi].centroid_3d
        for ni in neg_idx:
            nc = lobes[ni].centroid_3d
            d2 = (pc[0] - nc[0]) ** 2 + (pc[1] - nc[1]) ** 2 + (pc[2] - nc[2]) ** 2
            candidates.append((d2, pi, ni))
    candidates.sort()  # closest pairs first

    # Greedy matching — closest pair wins
    used_pos: set[int] = set()
    used_neg: set[int] = set()
    for _, pi, ni in candidates:
        if pi in used_pos or ni in used_neg:
            continue
        used_pos.add(pi)
        used_neg.add(ni)
        # Within pair: higher z = front — but if z-depths are nearly equal
        # (in-plane orbital) both lobes are visible, so render both as front
        dz = abs(lobes[pi].z_depth - lobes[ni].z_depth)
        if dz < _COPLANARITY_THRESHOLD_ANG:
            is_front[pi] = True
            is_front[ni] = True
        elif lobes[pi].z_depth >= lobes[ni].z_depth:
            is_front[pi] = True
            is_front[ni] = False
        else:
            is_front[pi] = False
            is_front[ni] = True

    # Unpaired lobes: fallback to molecule z-centroid
    for i in range(n):
        if is_front[i] is None:
            is_front[i] = lobes[i].z_depth >= mol_z

    return cast("list[bool]", is_front)


# ---------------------------------------------------------------------------
# MO SVG rendering
# ---------------------------------------------------------------------------


_MO_BASE_OPACITY = 0.6  # base opacity for MO lobes (scaled by surface_opacity)
_MO_BACK_FADE = 0.9  # back lobes rendered at this fraction of front opacity


def mo_back_lobes_svg(
    mo: SurfaceContours,
    mo_is_front: list[bool],
    surface_opacity: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> list[str]:
    """Return SVG lines for back MO lobes (faded flat fill, behind molecule)."""
    opacity = _MO_BASE_OPACITY * _MO_BACK_FADE * surface_opacity
    lines: list[str] = []
    for idx_l, lobe in enumerate(mo.lobes):
        if mo_is_front[idx_l]:
            continue
        color_hex = mo.pos_color if lobe.phase == "pos" else mo.neg_color
        d_all = combined_path_d(lobe.loops, mo, scale, cx, cy, canvas_w, canvas_h)
        if d_all:
            lines.append(f'  <g opacity="{opacity:.2f}">')
            lines.append(f'    <path d="{d_all}" fill="{color_hex}" fill-rule="evenodd" stroke="none"/>')
            lines.append("  </g>")
    return lines


def mo_front_lobes_svg(
    mo: SurfaceContours,
    mo_is_front: list[bool],
    surface_opacity: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> list[str]:
    """Return SVG lines for front MO lobes (flat fill, on top of molecule)."""
    opacity = _MO_BASE_OPACITY * surface_opacity
    lines: list[str] = []
    for idx_l, lobe in enumerate(mo.lobes):
        if not mo_is_front[idx_l]:
            continue
        color_hex = mo.pos_color if lobe.phase == "pos" else mo.neg_color
        d_all = combined_path_d(lobe.loops, mo, scale, cx, cy, canvas_w, canvas_h)
        if d_all:
            lines.append(f'  <g opacity="{opacity:.2f}">')
            lines.append(f'    <path d="{d_all}" fill="{color_hex}" fill-rule="evenodd" stroke="none"/>')
            lines.append("  </g>")
    return lines


# ---------------------------------------------------------------------------
# Per-frame MO recomputation for gif-rot
# ---------------------------------------------------------------------------


def recompute_mo(
    graph: nx.Graph,
    config: RenderConfig,
    params: MOParams,
    cube: CubeData,
    surface_opacity: float,
    _cache: dict,
) -> None:
    """Recompute MO contours for the current graph orientation (GIF frames).

    *_cache* is a mutable dict managed by the caller across frames.  On the
    first call it is populated with pre-computed 3D lobes, grid positions,
    and a bounding sphere radius.  Subsequent calls reuse these cached values
    and only update the Kabsch rotation.

    Parameters
    ----------
    graph:
        Molecular graph at the current GIF frame orientation.
    config:
        Render configuration; ``mo_contours`` and ``surface_opacity`` are
        updated in-place.
    params:
        MO surface parameters (isovalue, colors, blur, upsampling).
    cube:
        Gaussian cube file data (read-only; cached values stored in ``_cache``).
    surface_opacity:
        Opacity to apply to the MO surface.
    _cache:
        Mutable dict for inter-frame caching.  Populated on first call.
    """
    from xyzrender.utils import kabsch_rotation

    # Cache lobes and positions on first call
    if "lobes_3d" not in _cache:
        _cache["lobes_3d"] = find_3d_lobes(cube.grid_data, params.isovalue, steps=cube.steps)
        _cache["pos_flat_ang"] = compute_grid_positions(cube)

    orig = np.array([p for _, p in cube.atoms], dtype=float)
    curr = np.array([graph.nodes[i]["position"] for i in graph.nodes()], dtype=float)
    atom_centroid = orig.mean(axis=0)
    target_centroid = curr.mean(axis=0)

    # Cache bounding sphere: rotation-invariant bounds from cube corners.
    if "_bounding_radius" not in _cache:
        corners = cube_corners_ang(cube)
        r_max = float(np.linalg.norm(corners - atom_centroid, axis=1).max())
        _cache["_bounding_radius"] = r_max + r_max * 0.01 + 1e-9

    r = _cache["_bounding_radius"]
    fixed_bounds = (
        float(target_centroid[0] - r),
        float(target_centroid[0] + r),
        float(target_centroid[1] - r),
        float(target_centroid[1] + r),
    )

    rot = kabsch_rotation(orig, curr)

    config.mo_contours = build_mo_contours(
        cube,
        params,
        rot=rot,
        atom_centroid=atom_centroid,
        target_centroid=target_centroid,
        lobes_3d=_cache["lobes_3d"],
        pos_flat_ang=_cache["pos_flat_ang"],
        fixed_bounds=fixed_bounds,
    )
    config.surface_opacity = surface_opacity
