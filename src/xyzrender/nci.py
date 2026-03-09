"""NCI (Non-Covalent Interaction) surface extraction and SVG rendering.

Finds regions of low reduced density gradient (RDG, s(r)) in interstitial
space — H-bonds, pi-stacking, vdW contacts, steric repulsion — and renders
each as an individual flat-filled loop (MO-style, not concentric rings).

Usage::

    xyzrender mol-dens.cube --nci mol-grad.cube -o nci.svg

The grad cube defines the surface (low-RDG regions); the dens cube provides
atom geometry and (in future phases) per-voxel coloring data.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from xyzrender.esp import _build_lut
from xyzrender.mo import (
    _MIN_LOOP_PERIMETER,
    _UPSAMPLE_FACTOR,
    Lobe3D,
    LobeContour2D,
    _gaussian_blur_2d,
    _loop_perimeter,
    _mo_combined_path_d,
    _resample_loop,
    _upsample_2d,
    chain_segments,
    compute_grid_positions,
    cube_corners_ang,
    marching_squares,
)

# Blur is kept tight so the rendered patch faithfully represents the RDG
# isosurface without over-inflation.
_NCI_BLUR_SIGMA = 1.0
_NCI_MIN_REGION_VOLUME_BOHR3 = 0.1  # discard 3D NCI regions smaller than this (Bohr^3)

# ---------------------------------------------------------------------------
# NCI colormap — CSS4 named colors, same pattern as ESP_COLORMAP in esp.py
# ---------------------------------------------------------------------------
# (position, CSS4 color name) — most negative sign(l2)*rho to most positive
# Blue -> H-bond (attractive), Green -> vdW (near-zero), Red -> steric (repulsive)
NCI_COLORMAP: list[tuple[float, str]] = [
    (0.00, "midnightblue"),  # strong attractive (H-bond)
    (0.50, "limegreen"),  # weak / vdW contact
    (1.00, "maroon"),  # steric repulsion
]

# Standard sign(l2)*rho range (a.u.) -- saturates to blue/red outside this range
_NCI_VMIN: float = -0.5
_NCI_VMAX: float = +0.5
# Minimum PNG resolution for the static colored raster (upsampled if below this)
_NCI_MIN_PNG_RES: int = 300

# 256-entry RGB LUT built once at import (same mechanism as ESP)
_NCI_LUT = _build_lut(NCI_COLORMAP)

# 26-connectivity (face + edge + corner neighbours): diagonal NCI sheets viewed at an
# angle to the grid axes would fragment into isolated voxels under 6-connectivity.
_NCI_NEIGHBOURS = tuple(
    (di, dj, dk) for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1) if (di, dj, dk) != (0, 0, 0)
)


def _nci_colormap(
    value: float,
    vmin: float = _NCI_VMIN,
    vmax: float = _NCI_VMAX,
) -> tuple[int, int, int]:
    """Map a sign(l2)*rho value to an RGB colour via the NCI LUT."""
    idx = int(max(0.0, min(1.0, (value - vmin) / (vmax - vmin))) * 255)
    return int(_NCI_LUT[idx, 0]), int(_NCI_LUT[idx, 1]), int(_NCI_LUT[idx, 2])


def _nci_colormap_hex(
    value: float,
    vmin: float = _NCI_VMIN,
    vmax: float = _NCI_VMAX,
) -> str:
    r, g, b = _nci_colormap(value, vmin, vmax)
    return f"#{r:02x}{g:02x}{b:02x}"


def _dilate_binary_2d(grid: np.ndarray) -> np.ndarray:
    """One step of 8-connected binary dilation.

    Fills single-pixel checkerboard gaps that arise when a thin 3D NCI patch
    (viewed nearly edge-on) projects to a sparse 2D pixel pattern.  Without
    this step the Gaussian blur never reaches the 0.5 membership threshold and
    no contour is found for edge-on patches such as H-bond NCI disks.
    """
    padded = np.pad(grid, 1, mode="constant")
    result = grid.copy()
    for di, dj in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        result = np.maximum(
            result,
            padded[1 + di : 1 + di + grid.shape[0], 1 + dj : 1 + dj + grid.shape[1]],
        )
    return result


if TYPE_CHECKING:
    from xyzrender.cube import CubeData
    from xyzrender.types import NCIParams

logger = logging.getLogger(__name__)


@dataclass
class NCIContours:
    """Pre-computed NCI contour data ready for SVG rendering.

    Structurally compatible with :class:`~xyzrender.mo.ContourGrid` so that
    :func:`~xyzrender.mo._mo_combined_path_d` accepts it directly.
    """

    lobes: list[LobeContour2D] = field(default_factory=list)  # sorted by z_depth
    resolution: int = 0
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    color: str = "#228b22"  # fallback color for uniform mode
    # Tight Angstrom extent of actual lobe contours (for canvas fitting)
    lobe_x_min: float | None = None
    lobe_x_max: float | None = None
    lobe_y_min: float | None = None
    lobe_y_max: float | None = None
    # Per-pixel sign(lambda2)*rho colored raster (data URI, static rendering only)
    raster_png: str | None = None


# Threshold for marching-squares contouring of a binary membership map
_MEMBERSHIP_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Step 1: Find 3D NCI regions
# ---------------------------------------------------------------------------


def find_nci_regions(
    grad_data: np.ndarray,
    steps: np.ndarray,
    isovalue: float = 0.3,
) -> list[Lobe3D]:
    """Find connected 3D NCI patches via BFS flood-fill.

    Regions are where the reduced density gradient (RDG) is below *isovalue*.

    Parameters
    ----------
    grad_data:
        3D array of RDG values (from grad.cube).
    steps:
        (3, 3) step vectors in Bohr (from grad_cube.steps).
    isovalue:
        RDG threshold.  Voxels with s < isovalue are NCI candidates.
    """
    shape = grad_data.shape
    s1, s2 = shape[1] * shape[2], shape[2]

    voxel_vol = abs(float(np.linalg.det(steps)))
    min_cells = max(2, int(_NCI_MIN_REGION_VOLUME_BOHR3 / voxel_vol + 0.5))
    logger.debug("NCI voxel volume: %.4g Bohr³, min region cells: %d", voxel_vol, min_cells)

    mask = grad_data < isovalue

    visited = np.zeros(shape, dtype=bool)
    visited[~mask] = True  # non-mask cells don't need visiting

    regions: list[Lobe3D] = []
    n_discarded = 0
    candidates = np.argwhere(mask)
    for start in candidates:
        i, j, k = int(start[0]), int(start[1]), int(start[2])
        if visited[i, j, k]:
            continue

        component: list[int] = []
        queue: deque[tuple[int, int, int]] = deque([(i, j, k)])
        visited[i, j, k] = True
        while queue:
            ci, cj, ck = queue.popleft()
            component.append(ci * s1 + cj * s2 + ck)
            for di, dj, dk in _NCI_NEIGHBOURS:
                ni, nj, nk = ci + di, cj + dj, ck + dk
                if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
                    if not visited[ni, nj, nk]:
                        visited[ni, nj, nk] = True
                        queue.append((ni, nj, nk))

        if len(component) >= min_cells:
            regions.append(Lobe3D(flat_indices=np.array(component, dtype=np.intp), phase="pos"))
        else:
            n_discarded += 1

    if n_discarded:
        logger.debug("Discarded %d NCI regions smaller than %d voxels", n_discarded, min_cells)

    logger.debug("Found %d NCI regions at RDG isovalue %.4g", len(regions), isovalue)
    return regions


# ---------------------------------------------------------------------------
# Step 2: Project each region to a 2D contour loop
# ---------------------------------------------------------------------------


def _project_nci_region_2d(
    region: Lobe3D,
    pos_flat_ang: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution: int,
    rot: np.ndarray | None,
    atom_centroid: np.ndarray | None,
    target_centroid: np.ndarray | None,
) -> LobeContour2D | None:
    """Project a 3D NCI region to a 2D contour loop.

    Uses binary membership projection (1.0 for member voxels) then
    Gaussian blur + upsampling + marching squares at 0.5.
    """
    lobe_pos = pos_flat_ang[region.flat_indices].copy()

    if rot is not None:
        if atom_centroid is not None:
            lobe_pos -= atom_centroid
        lobe_pos = lobe_pos @ rot.T
        if target_centroid is not None:
            lobe_pos += target_centroid

    z_depth = float(lobe_pos[:, 2].mean())

    lx, ly = lobe_pos[:, 0], lobe_pos[:, 1]
    xi = np.clip(((lx - x_min) / (x_max - x_min) * (resolution - 1)).astype(int), 0, resolution - 1)
    yi = np.clip(((ly - y_min) / (y_max - y_min) * (resolution - 1)).astype(int), 0, resolution - 1)

    # Binary membership: 1.0 where NCI voxels project, 0 elsewhere.
    grid_2d = np.zeros((resolution, resolution))
    grid_2d[yi, xi] = 1.0

    nz_rows, nz_cols = np.nonzero(grid_2d)
    if len(nz_rows) == 0:
        return None

    pad = max(3, int(_NCI_BLUR_SIGMA * 4) + 1)
    r0 = max(0, int(nz_rows.min()) - pad)
    r1 = min(resolution, int(nz_rows.max()) + pad + 1)
    c0 = max(0, int(nz_cols.min()) - pad)
    c1 = min(resolution, int(nz_cols.max()) + pad + 1)
    cropped = grid_2d[r0:r1, c0:c1]

    # Dilation before blur: fills single-pixel gaps so the Gaussian blur reaches
    # the 0.5 threshold.  Two conditions trigger it:
    #   - very few projected pixels: small or near-invisible patch needs dilation
    #     to produce any closed contour at all
    #   - low fill fraction (n_unique / bbox area): catches thin patches of any
    #     orientation — including diagonal pi-pi stacking viewed at an angle,
    #     which has large row_span AND col_span but few pixels relative to bbox.
    #     A solid face-on patch fills ~π/4 ≈ 0.78 of its bbox; 0.4 is a safe
    #     threshold below that.
    n_unique = len(nz_rows)
    row_span = int(nz_rows.max()) - int(nz_rows.min()) + 1
    col_span = int(nz_cols.max()) - int(nz_cols.min()) + 1
    fill_fraction = n_unique / max(1, row_span * col_span)
    to_blur = _dilate_binary_2d(cropped) if (n_unique < 20 or fill_fraction < 0.4) else cropped
    blurred = np.maximum(_gaussian_blur_2d(to_blur, _NCI_BLUR_SIGMA), 0.0)
    upsampled = _upsample_2d(blurred, _UPSAMPLE_FACTOR)

    raw_loops = chain_segments(marching_squares(upsampled, _MEMBERSHIP_THRESHOLD))
    offset = np.array([r0 * _UPSAMPLE_FACTOR, c0 * _UPSAMPLE_FACTOR])
    offset_loops = [loop + offset for loop in raw_loops]
    loops = [_resample_loop(lp) for lp in offset_loops if _loop_perimeter(lp) >= _MIN_LOOP_PERIMETER]

    if not loops:
        return None

    cent_3d = (float(lobe_pos[:, 0].mean()), float(lobe_pos[:, 1].mean()), z_depth)
    return LobeContour2D(loops=loops, phase="pos", z_depth=z_depth, centroid_3d=cent_3d)


# ---------------------------------------------------------------------------
# Step 2a: Build colored raster for static per-pixel NCI surface
# ---------------------------------------------------------------------------


def _build_nci_color_raster(
    regions_3d: list[Lobe3D],
    pos_flat_ang: np.ndarray,
    dens_cube: "CubeData",
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution: int,
    rot: np.ndarray | None,
    atom_centroid: np.ndarray | None,
    target_centroid: np.ndarray | None,
    *,
    vmin: float = _NCI_VMIN,
    vmax: float = _NCI_VMAX,
) -> str | None:
    """Build a colored RGBA raster of NCI patches for static SVG embedding.

    Each pixel is colored by the average sign(l2)*rho of the voxels that
    project to it: blue=H-bond, green=vdW, red=steric repulsion.
    Returns a ``data:image/png;base64,...`` URI, or ``None`` if no voxels
    project at all.
    """
    import base64
    import io

    from PIL import Image

    dens_sum = np.zeros((resolution, resolution))
    count = np.zeros((resolution, resolution), dtype=np.float64)
    dens_flat = dens_cube.grid_data.ravel()

    for region in regions_3d:
        lobe_pos = pos_flat_ang[region.flat_indices].copy()
        if rot is not None:
            if atom_centroid is not None:
                lobe_pos -= atom_centroid
            lobe_pos = lobe_pos @ rot.T
            if target_centroid is not None:
                lobe_pos += target_centroid

        lx, ly = lobe_pos[:, 0], lobe_pos[:, 1]
        xi = np.clip(((lx - x_min) / (x_max - x_min) * (resolution - 1)).astype(int), 0, resolution - 1)
        yi = np.clip(((ly - y_min) / (y_max - y_min) * (resolution - 1)).astype(int), 0, resolution - 1)

        np.add.at(dens_sum, (yi, xi), dens_flat[region.flat_indices])
        np.add.at(count, (yi, xi), 1.0)

    if count.max() == 0:
        return None

    raster_blur = 1.5
    dens_blurred = _gaussian_blur_2d(dens_sum, raster_blur)
    count_blurred = _gaussian_blur_2d(count, raster_blur)

    # Per-pixel average density; smooth alpha with solid interior
    dens_avg = np.where(count_blurred > 1e-4, dens_blurred / np.maximum(count_blurred, 1e-10), 0.0)
    alpha_raw = _gaussian_blur_2d((count > 0).astype(float), raster_blur)
    alpha_peak = float(alpha_raw.max())
    alpha_norm = np.clip(alpha_raw / (alpha_peak + 1e-10), 0.0, 1.0)
    # Steepen: interior (≥50% of peak) → fully opaque; edge pixels keep smooth falloff
    alpha_f = np.minimum(alpha_norm * 2.0, 1.0)

    # Apply NCI colormap via LUT (same mechanism as ESP)
    lut_idx = np.clip(((dens_avg - vmin) / (vmax - vmin) * 255), 0, 255).astype(np.uint8)
    r_u8 = _NCI_LUT[lut_idx, 0]
    g_u8 = _NCI_LUT[lut_idx, 1]
    b_u8 = _NCI_LUT[lut_idx, 2]
    a_u8 = np.clip(alpha_f * 255, 0, 255).astype(np.uint8)

    # Flip: numpy row 0 = y_min (bottom); PNG row 0 = top = y_max
    rgba = np.flipud(np.stack([r_u8, g_u8, b_u8, a_u8], axis=-1))

    img = Image.fromarray(rgba, "RGBA")
    if img.width < _NCI_MIN_PNG_RES or img.height < _NCI_MIN_PNG_RES:
        target = max(_NCI_MIN_PNG_RES, img.width * 3, img.height * 3)
        img = img.resize((target, target), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{png_b64}"


# ---------------------------------------------------------------------------
# Step 2b: Build all NCI contours from cube data
# ---------------------------------------------------------------------------


def build_nci_contours(
    grad_cube: CubeData,
    dens_cube: CubeData,
    params: NCIParams,
    *,
    rot: np.ndarray | None = None,
    atom_centroid: np.ndarray | None = None,
    target_centroid: np.ndarray | None = None,
    pos_flat_ang: np.ndarray | None = None,
    fixed_bounds: tuple[float, float, float, float] | None = None,
    regions_3d: list[Lobe3D] | None = None,
) -> NCIContours:
    """Build NCI contour data from a grad cube file.

    Each connected low-RDG region is projected and contoured independently
    (MO-style), giving individual flat-filled patches for each interaction.

    Parameters
    ----------
    grad_cube:
        Gaussian cube file containing the reduced density gradient (RDG) values.
    dens_cube:
        Gaussian cube file containing the electron density (sign(lambda2)*rho values).
    params:
        NCI surface parameters (isovalue, color, color_mode, dens_cutoff).
    rot:
        Optional rotation matrix for grid alignment.
    atom_centroid:
        Centroid of the original cube atom positions (Å).
    target_centroid:
        Centroid of the current (possibly rotated) atom positions (Å).
    pos_flat_ang:
        Pre-computed flattened grid positions (cached between GIF frames).
    fixed_bounds:
        Fixed ``(x_min, x_max, y_min, y_max)`` in Å (cached between GIF frames).
    regions_3d:
        Pre-computed 3D NCI regions (cached between GIF frames).

    Returns
    -------
    NCIContours
        Contour loops and coloring data ready for SVG rendering.
    """
    from xyzrender.types import resolve_color

    isovalue = params.isovalue
    color = resolve_color(params.color)
    color_mode = params.color_mode

    n1, n2, n3 = grad_cube.grid_shape
    base_res = max(n1, n2, n3)

    if pos_flat_ang is None:
        pos_flat_ang = compute_grid_positions(grad_cube)

    # 2D bounds from cube corners (consistent with atom positions)
    if fixed_bounds is not None:
        x_min, x_max, y_min, y_max = fixed_bounds
    else:
        corners = cube_corners_ang(grad_cube)
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

    # Find 3D NCI regions (BFS flood-fill on low-RDG mask)
    if regions_3d is None:
        regions_3d = find_nci_regions(
            grad_cube.grid_data,
            grad_cube.steps,
            isovalue=isovalue,
        )

    # Project each region to 2D
    use_dens_color = color_mode in ("avg", "pixel")
    dens_flat = dens_cube.grid_data.ravel() if use_dens_color else None

    # Auto-scale colormap range from actual NCI-region density values
    if use_dens_color and regions_3d:
        assert dens_flat is not None
        all_dens_vals = np.concatenate([dens_flat[r.flat_indices] for r in regions_3d])
        p2 = abs(float(np.percentile(all_dens_vals, 2)))
        p98 = abs(float(np.percentile(all_dens_vals, 98)))
        p = max(p2, p98, 0.01)
        vmin, vmax = -p, +p
    else:
        vmin, vmax = _NCI_VMIN, _NCI_VMAX

    paired: list[tuple[Lobe3D, LobeContour2D]] = []
    for region in regions_3d:
        lc = _project_nci_region_2d(
            region,
            pos_flat_ang,
            x_min,
            x_max,
            y_min,
            y_max,
            base_res,
            rot,
            atom_centroid,
            target_centroid,
        )
        if lc is not None:
            if use_dens_color:
                # Average sign(l2)*rho over all voxels in this region -> lobe color
                assert dens_flat is not None
                mean_dens = float(dens_flat[region.flat_indices].mean())
                lc.lobe_color = _nci_colormap_hex(mean_dens, vmin, vmax)
            paired.append((region, lc))

    lobe_contours = [lc for _, lc in paired]

    # Sort back-to-front by z_depth for proper SVG layering
    lobe_contours.sort(key=lambda lc: lc.z_depth)

    total_loops = sum(len(lc.loops) for lc in lobe_contours)
    if total_loops == 0:
        logger.warning(
            "No NCI patches found at RDG isovalue %.4g — try adjusting --iso",
            isovalue,
        )
    else:
        logger.debug(
            "NCI contours: %d regions (%d loops total, RDG isovalue=%.4g, mode=%s)",
            len(lobe_contours),
            total_loops,
            isovalue,
            color_mode,
        )

    # Per-pixel raster only for pixel mode (PIL encode is expensive)
    nci_raster_png: str | None = None
    if color_mode == "pixel":
        nci_raster_png = _build_nci_color_raster(
            regions_3d,
            pos_flat_ang,
            dens_cube,
            x_min,
            x_max,
            y_min,
            y_max,
            base_res,
            rot,
            atom_centroid,
            target_centroid,
            vmin=vmin,
            vmax=vmax,
        )

    # Tight Angstrom extent for canvas fitting
    lobe_x_min: float | None = None
    lobe_x_max: float | None = None
    lobe_y_min: float | None = None
    lobe_y_max: float | None = None
    if lobe_contours:
        lobe_x_min = x_min
        lobe_x_max = x_max
        lobe_y_min = y_min
        lobe_y_max = y_max

    res = base_res * _UPSAMPLE_FACTOR
    return NCIContours(
        lobes=lobe_contours,
        resolution=res,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        color=color,
        lobe_x_min=lobe_x_min,
        lobe_x_max=lobe_x_max,
        lobe_y_min=lobe_y_min,
        lobe_y_max=lobe_y_max,
        raster_png=nci_raster_png,
    )


# ---------------------------------------------------------------------------
# Step 3: SVG rendering — individual flat-filled NCI patches
# ---------------------------------------------------------------------------


def nci_static_svg_defs(
    nci: NCIContours,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> list[str]:
    """Emit the ``<defs>`` block for pixel-mode NCI: shared raster image + per-lobe clip paths.

    The ``<use>`` elements that actually paint the raster are emitted separately
    (z-sorted into the atom/bond render loop by the renderer).
    """
    if not nci.raster_png or not nci.lobes:
        return []

    img_x = canvas_w / 2 + scale * (nci.x_min - cx)
    img_y = canvas_h / 2 - scale * (nci.y_max - cy)
    img_w = scale * (nci.x_max - nci.x_min)
    img_h = scale * (nci.y_max - nci.y_min)

    lines: list[str] = ["  <defs>"]
    lines.append(
        f'    <image id="nci_raster" x="{img_x:.1f}" y="{img_y:.1f}" '
        f'width="{img_w:.1f}" height="{img_h:.1f}" '
        f'href="{nci.raster_png}" '
        f'preserveAspectRatio="none" image-rendering="optimizeQuality"/>'
    )
    for i, lobe in enumerate(nci.lobes):
        d = _mo_combined_path_d(lobe.loops, nci, scale, cx, cy, canvas_w, canvas_h)
        if d:
            lines.append(f'    <clipPath id="nci_clip_{i}">')
            lines.append(f'      <path d="{d}" fill-rule="evenodd"/>')
            lines.append("    </clipPath>")
    lines.append("  </defs>")
    return lines


def nci_lobe_svg_items(
    nci: NCIContours,
    surface_opacity: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> list[tuple[float, list[str]]]:
    """Return per-lobe ``(z_depth, svg_lines)`` pairs for z-sorted rendering.

    Sorted ascending by z_depth (back-to-front) to match the painter's-algorithm
    z_order traversal in the renderer.  The renderer interleaves these between
    atoms at the correct depth rather than painting all patches on top.

    For pixel mode the returned lines are ``<use>`` references to the shared
    ``#nci_raster`` image (clip path defs must be emitted first via
    ``nci_static_svg_defs``).  For avg/uniform mode they are flat ``<path>``
    elements.
    """
    if not nci.lobes:
        return []

    items: list[tuple[float, list[str]]] = []
    opacity = surface_opacity

    if nci.raster_png:
        for i, lobe in enumerate(nci.lobes):
            use_str = f'  <use href="#nci_raster" clip-path="url(#nci_clip_{i})" opacity="{opacity:.3f}"/>'
            items.append((lobe.z_depth, [use_str]))
    else:
        for lobe in nci.lobes:
            color = lobe.lobe_color if lobe.lobe_color is not None else nci.color
            d = _mo_combined_path_d(lobe.loops, nci, scale, cx, cy, canvas_w, canvas_h)
            if d:
                path_str = f'  <path d="{d}" fill="{color}" fill-rule="evenodd" stroke="none" opacity="{opacity:.3f}"/>'
                items.append((lobe.z_depth, [path_str]))

    # nci.lobes is already sorted ascending z_depth; items preserves that order
    return items


def nci_loops_svg(
    nci: NCIContours,
    surface_opacity: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> list[str]:
    """Render NCI patches as individual flat-filled closed loops.

    Each patch uses its per-lobe average sign(l2)*rho color when available
    (blue=H-bond, green=vdW, red=steric), otherwise falls back to the
    uniform ``nci.color``.  Drawn back-to-front by z_depth.
    """
    if not nci.lobes:
        return []

    opacity = surface_opacity
    lines: list[str] = []

    for lobe in nci.lobes:
        color = lobe.lobe_color if lobe.lobe_color is not None else nci.color
        d = _mo_combined_path_d(lobe.loops, nci, scale, cx, cy, canvas_w, canvas_h)
        if d:
            lines.append(f'  <path d="{d}" fill="{color}" fill-rule="evenodd" stroke="none" opacity="{opacity:.3f}"/>')

    return lines


def nci_static_svg(
    nci: NCIContours,
    surface_opacity: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> list[str]:
    """Render the per-pixel colored NCI raster clipped to each lobe's loop shape.

    The sign(l2)*rho heatmap is defined once as a ``<image>``; each lobe's
    marching-squares outline is used as a ``<clipPath>`` so the raster is
    only visible inside the actual NCI isosurface patches.  This mirrors the
    ESP surface rendering approach.
    """
    if not nci.raster_png or not nci.lobes:
        return []

    img_x = canvas_w / 2 + scale * (nci.x_min - cx)
    img_y = canvas_h / 2 - scale * (nci.y_max - cy)
    img_w = scale * (nci.x_max - nci.x_min)
    img_h = scale * (nci.y_max - nci.y_min)
    opacity = surface_opacity

    lines: list[str] = ["  <defs>"]
    lines.append(
        f'    <image id="nci_raster" x="{img_x:.1f}" y="{img_y:.1f}" '
        f'width="{img_w:.1f}" height="{img_h:.1f}" '
        f'href="{nci.raster_png}" '
        f'preserveAspectRatio="none" image-rendering="optimizeQuality"/>'
    )

    clip_ids: list[str] = []
    for i, lobe in enumerate(nci.lobes):
        d = _mo_combined_path_d(lobe.loops, nci, scale, cx, cy, canvas_w, canvas_h)
        if d:
            clip_id = f"nci_clip_{i}"
            clip_ids.append(clip_id)
            lines.append(f'    <clipPath id="{clip_id}">')
            lines.append(f'      <path d="{d}" fill-rule="evenodd"/>')
            lines.append("    </clipPath>")

    lines.append("  </defs>")

    for clip_id in clip_ids:
        lines.append(f'  <use href="#nci_raster" clip-path="url(#{clip_id})" opacity="{opacity:.3f}"/>')

    return lines
