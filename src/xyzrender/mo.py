"""MO (molecular orbital) contour extraction, classification, and SVG rendering."""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from xyzrender.colors import _FOG_NEAR, blend_fog
from xyzrender.contours import (
    BLUR_SIGMA,
    MIN_LOBE_VOLUME_BOHR3,
    UPSAMPLE_FACTOR,
    Lobe3D,
    LobeContour2D,
    MOContours,
    SurfaceContours,
    compute_grid_positions,
    cube_corners_ang,
    project_region_to_contours,
    render_lobe_svg,
)

if TYPE_CHECKING:
    import networkx as nx

    from xyzrender.cube import CubeData
    from xyzrender.types import MOParams, RenderConfig

logger = logging.getLogger(__name__)


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
    surface_style: str = "solid",
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

    return project_region_to_contours(
        grid_2d,
        resolution,
        lobe_pos,
        lobe.phase,
        isovalue,
        blur_sigma=blur_sigma,
        upsample_factor=upsample_factor,
        surface_style=surface_style,
    )


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
    surface_style: str = "solid",
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
            surface_style=surface_style,
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
# MO SVG rendering
# ---------------------------------------------------------------------------


# Default surface_opacity for MO renders when --opacity is unset; MO lobes are
# conceptually translucent isosurfaces so the out-of-the-box look is partial.
MO_DEFAULT_OPACITY = 0.6

# Fog strength multiplier for MO lobes — lobes are large and diffuse, so full
# atom-strength fog washes them out.
_MO_FOG_FACTOR = 0.7


def _lobe_effective_z(
    lobe: LobeContour2D,
    atom_pos: np.ndarray,
    atom_radii: np.ndarray,
) -> float:
    """Compute a single effective z for a lobe via per-atom occlusion constraints.

    For each atom whose 2D circle (in the same Å frame as the lobe's voxel
    positions) overlaps any of the lobe's voxels, derive a constraint:

    - If the lobe's local average z at the atom's screen location is in front
      of the atom centre, the lobe must drain *after* the atom
      (``effective_z >= atom_z``).
    - If the local average z is behind, the lobe must drain *before* the atom
      (``effective_z <= atom_z``).

    Local average z (rather than max) is used so a thin tail of voxels near the
    atom doesn't over-promote the whole lobe to "in front" — ordering flips
    only when the lobe's local *bulk* is genuinely on the opposite side.

    The returned ``effective_z`` lies in the feasible interval when one exists,
    otherwise the conservative fallback (``lower_bound + epsilon``) biases the
    lobe toward "in front of locally-front atoms" — the locally-behind atom
    that lost the constraint gets the centroid-style artifact, but isolated to
    that single atom rather than affecting the whole lobe.

    Returns the lobe's ``z_depth`` if no atom overlaps in 2D (no visual conflict
    is possible) or if ``voxel_pos`` is unavailable.
    """
    if lobe.voxel_pos is None or len(atom_pos) == 0:
        return lobe.z_depth

    vox = lobe.voxel_pos
    lower = -np.inf  # effective_z must be >= lower (lobe drains AFTER these atoms)
    upper = np.inf  # effective_z must be <= upper (lobe drains BEFORE these atoms)
    n_overlap = 0

    # Quick lobe 2D bbox cull to skip atoms far from the lobe
    lx_min, ly_min = vox[:, 0].min(), vox[:, 1].min()
    lx_max, ly_max = vox[:, 0].max(), vox[:, 1].max()

    for ai in range(len(atom_pos)):
        ax, ay, az = atom_pos[ai]
        ar = atom_radii[ai]
        if ax + ar < lx_min or ax - ar > lx_max:
            continue
        if ay + ar < ly_min or ay - ar > ly_max:
            continue
        dx = vox[:, 0] - ax
        dy = vox[:, 1] - ay
        overlap = (dx * dx + dy * dy) < (ar * ar)
        if not overlap.any():
            continue
        local_z = float(vox[overlap, 2].mean())
        n_overlap += 1
        if local_z > az:
            lower = max(lower, az)
        elif az < upper:
            upper = az

    if n_overlap == 0:
        return lobe.z_depth
    if upper > lower:
        # Feasible: any value in (lower, upper) satisfies all per-pair
        # constraints.  Pick a value that anchors to a finite bound so the
        # lobe's queue position stays meaningfully scaled to the molecule.
        if np.isfinite(lower) and np.isfinite(upper):
            return float(0.5 * (lower + upper))
        if np.isfinite(lower):
            # All overlapping atoms are behind the lobe → drain just after the
            # frontmost such atom.
            return float(lower + 1e-3)
        if np.isfinite(upper):
            # All overlapping atoms are in front of the lobe → drain just before
            # the backmost such atom.
            return float(upper - 1e-3)
        return lobe.z_depth
    # Conflict: lobe wraps two atoms in opposite directions. Conservative
    # fallback — favour "lobe in front of locally-front atoms".
    logger.debug("Lobe occlusion conflict (lower=%.3f upper=%.3f), falling back", lower, upper)
    return float(lower + 1e-6)


def mo_lobe_svg_items(
    mo: SurfaceContours,
    surface_opacity: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
    *,
    surface_style: str = "solid",
    stroke_width: float = 1.5,
    mesh_inner_width: float = 0.8,
    flat: bool = False,
    outline_width: float = 0.0,
    outline_color: str = "#000000",
    atom_pos: np.ndarray | None = None,
    atom_radii: np.ndarray | None = None,
    fog_enabled: bool = False,
    fog_strength: float = 0.0,
    fog_rgb: np.ndarray | None = None,
    fog_z_front: float = 0.0,
    fog_z_range: float = 1.0,
) -> list[tuple[float, list[str]]]:
    """Return per-lobe ``(z_depth, svg_lines)`` items for z-interleaved rendering.

    Each lobe is rendered independently — the painter's-algorithm drain in the
    renderer places each lobe at its own z among the atoms.

    When ``atom_pos`` and ``atom_radii`` are supplied, the lobe's queue z is
    resolved via per-atom occlusion constraints (see :func:`_lobe_effective_z`);
    otherwise it falls back to the lobe's centroid z.

    Depth cueing comes from the z-interleaved drawing order plus the shared fog
    colour blend, attenuated by :data:`_MO_FOG_FACTOR` (full atom-strength fog
    washes diffuse lobes out).  ``flat=True`` disables fog on MO lobes.

    Items are returned sorted ascending by their queue z; the caller pools them
    with other surface overlays and re-sorts.
    """
    items: list[tuple[float, list[str]]] = []
    use_constraints = atom_pos is not None and atom_radii is not None
    use_fog = fog_enabled and fog_rgb is not None and fog_strength > 0 and fog_z_range > 0 and not flat
    for lobe in mo.lobes:
        color_hex = mo.pos_color if lobe.phase == "pos" else mo.neg_color
        if use_fog:
            assert fog_rgb is not None
            # Same depth normalization as atoms (renderer.py): distance from
            # frontmost atom, scaled by molecule depth range, with a near
            # dead-zone before fog kicks in.
            depth = max(fog_z_front - lobe.z_depth, 0.0)
            fog_f = fog_strength * _MO_FOG_FACTOR * float(np.clip((depth - _FOG_NEAR) / fog_z_range, 0.0, 1.0))
            color_hex = blend_fog(color_hex, fog_rgb, fog_f)
        if use_constraints:
            assert atom_pos is not None
            assert atom_radii is not None
            queue_z = _lobe_effective_z(lobe, atom_pos, atom_radii)
        else:
            queue_z = lobe.z_depth
        svg_lines = render_lobe_svg(
            lobe,
            mo,
            color_hex,
            surface_opacity,
            scale,
            cx,
            cy,
            canvas_w,
            canvas_h,
            surface_style=surface_style,
            stroke_width=stroke_width,
            mesh_inner_width=mesh_inner_width,
            outline_width=outline_width,
            outline_color=outline_color,
        )
        if svg_lines:
            items.append((queue_z, svg_lines))

    items.sort(key=lambda x: x[0])
    return items


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
        surface_style=config.surface_style,
    )
    config.surface_opacity = surface_opacity
