"""ESP (electrostatic potential) surface rendering.

Hybrid approach: projects density and ESP cube data to 2D, builds a
smooth-gradient heatmap PNG from ESP values with 3D lighting, then
extracts density contour rings (like dens.py) for depth-graded
clip-path rendering in SVG.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from xyzrender.mo import (
    _MIN_LOOP_PERIMETER,
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

if TYPE_CHECKING:
    import networkx as nx

    from xyzrender.cube import CubeData
    from xyzrender.types import RenderConfig

logger = logging.getLogger(__name__)

_N_LAYERS = 6  # contour threshold levels for depth-graded clip-path rings
_PROJ_MULT = 3  # projection grid multiplier to avoid Moiré when tilted
_PROJ_BLUR = 3.5  # Gaussian blur sigma (fills gaps + smooths grid structure)
_SHELL_UPPER = 6.0  # max density/isovalue ratio for shell voxels
_FRONT_DECAY = 2.0  # angstroms — exponential decay from frontmost surface
_LIGHT_DIR = np.array([-0.45, 0.5, 0.75])
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)
_SAT_BOOST = 2.0  # hyper-saturation for SVG layered opacity rendering
_MIN_PNG_RES = 500  # upscale threshold for low-res heatmap PNGs


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ESPSurface:
    """Pre-computed ESP surface: heatmap PNG + density contour layers."""

    png_data_uri: str = ""
    resolution: int = 0
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    # Density contour layers for depth-graded clip-path rendering
    layers: list[LobeContour2D] = field(default_factory=list)
    # ESP value range used for color mapping (for caching across GIF frames)
    esp_vmin: float = 0.0
    esp_vmax: float = 0.0


# ---------------------------------------------------------------------------
# ESP colormap — CSS4 named colors, user-changeable via presets
# ---------------------------------------------------------------------------

# (position, CSS4 color name) — most positive ESP to most negative ESP
ESP_COLORMAP: list[tuple[float, str]] = [
    (0.00, "midnightblue"),  # most positive ESP (electron-poor)
    (0.25, "steelblue"),
    (0.50, "darkseagreen"),  # neutral
    (0.75, "peru"),
    (1.00, "maroon"),  # most negative ESP (electron-rich)
]


def _build_lut(cmap: list[tuple[float, str]]) -> np.ndarray:
    """Build a 256-entry RGB LUT from a named-color colormap."""
    from xyzrender.types import Color

    stops = [(t, Color.from_str(name)) for t, name in cmap]

    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for j in range(len(stops) - 1):
            t0, c0 = stops[j]
            t1, c1 = stops[j + 1]
            if t <= t1:
                s = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                lut[i] = (
                    int(c0.r + s * (c1.r - c0.r)),
                    int(c0.g + s * (c1.g - c0.g)),
                    int(c0.b + s * (c1.b - c0.b)),
                )
                break
        else:
            c = stops[-1][1]
            lut[i] = (c.r, c.g, c.b)
    return lut


_LUT = _build_lut(ESP_COLORMAP)


# ---------------------------------------------------------------------------
# ESP surface building
# ---------------------------------------------------------------------------


def build_esp_surface(
    dens_cube: CubeData,
    esp_cube: CubeData,
    isovalue: float,
    *,
    rot: np.ndarray | None = None,
    atom_centroid: np.ndarray | None = None,
    target_centroid: np.ndarray | None = None,
    pos_flat_ang: np.ndarray | None = None,
    flat_indices: np.ndarray | None = None,
    fixed_bounds: tuple[float, float, float, float] | None = None,
    n_layers: int = _N_LAYERS,
    upsample: int = 6,
    esp_range: tuple[float, float] | None = None,
    normals_phys: np.ndarray | None = None,
) -> ESPSurface:
    """Build an ESP surface: heatmap PNG + density contour layers.

    Projects density and ESP to 2D, builds an RGB heatmap from ESP values
    with 3D lighting, then extracts density contour rings for depth-graded
    clip-path rendering.
    """
    n1, n2, n3 = dens_cube.grid_shape
    base_res = max(n1, n2, n3)

    if pos_flat_ang is None:
        pos_flat_ang = compute_grid_positions(dens_cube)

    if flat_indices is None:
        mask = dens_cube.grid_data >= isovalue
        flat_indices = np.flatnonzero(mask)

    dens_values_flat = dens_cube.grid_data.ravel()
    esp_values_flat = esp_cube.grid_data.ravel()
    lobe_pos = pos_flat_ang[flat_indices].copy()
    lobe_dens = dens_values_flat[flat_indices]
    lobe_esp = esp_values_flat[flat_indices]

    # Shell mask: only voxels near the isosurface (not deep interior).
    # ESP colors, lighting, and depth all sample from the surface shell only.
    shell = lobe_dens <= isovalue * _SHELL_UPPER

    # Rotate positions
    if rot is not None:
        if atom_centroid is not None:
            lobe_pos -= atom_centroid
        lobe_pos = lobe_pos @ rot.T
        if target_centroid is not None:
            lobe_pos += target_centroid

    # Tight bounds
    vx_xmin = float(lobe_pos[:, 0].min())
    vx_xmax = float(lobe_pos[:, 0].max())
    vx_ymin = float(lobe_pos[:, 1].min())
    vx_ymax = float(lobe_pos[:, 1].max())
    vx_xpad = (vx_xmax - vx_xmin) * 0.02 + 1e-9
    vx_ypad = (vx_ymax - vx_ymin) * 0.02 + 1e-9

    if fixed_bounds is not None:
        x_min, x_max, y_min, y_max = fixed_bounds
    else:
        x_min = vx_xmin - vx_xpad
        x_max = vx_xmax + vx_xpad
        y_min = vx_ymin - vx_ypad
        y_max = vx_ymax + vx_ypad

    # Use a finer projection grid to avoid Moiré artifacts when tilted.
    proj_res = base_res * _PROJ_MULT

    # Pixel coordinates for all above-isovalue voxels
    lx, ly = lobe_pos[:, 0], lobe_pos[:, 1]
    xi = np.clip(
        ((lx - x_min) / (x_max - x_min) * (proj_res - 1)).astype(int),
        0,
        proj_res - 1,
    )
    yi = np.clip(
        ((ly - y_min) / (y_max - y_min) * (proj_res - 1)).astype(int),
        0,
        proj_res - 1,
    )

    # Density max-intensity projection (all voxels — needed for contours)
    grid_2d = np.zeros((proj_res, proj_res))
    np.maximum.at(grid_2d, (yi, xi), lobe_dens)

    # Shell-only pixel coords for ESP/lighting/depth
    s_xi, s_yi = xi[shell], yi[shell]
    s_esp = lobe_esp[shell]
    s_z = lobe_pos[shell, 2]

    # Front-depth weight: exponential decay from the frontmost surface
    s_z_max = float(s_z.max()) if s_z.size > 0 else 0.0
    s_front_wt = np.exp((s_z - s_z_max) / _FRONT_DECAY)

    # Shared front-weight accumulation (ESP, lighting, Z all use the same
    # scatter coordinates and weights — accumulate once, reuse for all three)
    wt_sum = np.zeros((proj_res, proj_res))
    np.add.at(wt_sum, (s_yi, s_xi), s_front_wt)
    has_wt = wt_sum > 0

    # ESP weighted projection (shell only)
    esp_sum = np.zeros((proj_res, proj_res))
    np.add.at(esp_sum, (s_yi, s_xi), s_front_wt * s_esp)
    grid_2d_esp = np.divide(esp_sum, wt_sum, out=np.zeros_like(esp_sum), where=has_wt)

    # --- 3D surface normal lighting (shell only) ---
    if normals_phys is not None:
        s_norms = normals_phys[flat_indices[shell]].copy()
        if rot is not None:
            s_norms = s_norms @ rot.T
        s_norms = -s_norms  # outward = -gradient
        lengths = np.sqrt((s_norms * s_norms).sum(axis=1))
        lengths = np.maximum(lengths, 1e-12)
        s_norms /= lengths[:, np.newaxis]
        s_lambert = np.clip(s_norms @ _LIGHT_DIR, 0.0, 1.0)
    else:
        s_lambert = np.full(shell.sum(), 0.65)

    light_sum = np.zeros((proj_res, proj_res))
    np.add.at(light_sum, (s_yi, s_xi), s_front_wt * s_lambert)
    grid_2d_light = np.divide(light_sum, wt_sum, out=np.full_like(light_sum, 0.65), where=has_wt)

    # --- Z-depth map for depth fading (shell only) ---
    z_sum = np.zeros((proj_res, proj_res))
    np.add.at(z_sum, (s_yi, s_xi), s_front_wt * s_z)
    grid_2d_z = np.divide(z_sum, wt_sum, out=np.zeros_like(z_sum), where=has_wt)

    # Crop to non-zero bounding box + blur padding before blur/upsample.
    # Avoids processing large empty regions of the projection grid.
    nz_rows, nz_cols = np.nonzero(grid_2d)
    if len(nz_rows) == 0:
        return ESPSurface(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    blur_pad = max(3, int(_PROJ_BLUR * 2 * 1.5) + 1)  # accommodate largest blur sigma
    r0 = max(0, int(nz_rows.min()) - blur_pad)
    r1 = min(proj_res, int(nz_rows.max()) + blur_pad + 1)
    c0 = max(0, int(nz_cols.min()) - blur_pad)
    c1 = min(proj_res, int(nz_cols.max()) + blur_pad + 1)

    _up = max(1, upsample // _PROJ_MULT)
    blurred_dens = np.maximum(_gaussian_blur_2d(grid_2d[r0:r1, c0:c1], _PROJ_BLUR), 0.0)
    blurred_esp = _gaussian_blur_2d(grid_2d_esp[r0:r1, c0:c1], _PROJ_BLUR * 1.5)
    blurred_light = _gaussian_blur_2d(grid_2d_light[r0:r1, c0:c1], _PROJ_BLUR * 0.5)
    blurred_z = _gaussian_blur_2d(grid_2d_z[r0:r1, c0:c1], _PROJ_BLUR * 0.8)
    up_dens = _upsample_2d(blurred_dens, _up)
    up_esp = _upsample_2d(blurred_esp, _up)
    up_light = _upsample_2d(blurred_light, _up)
    up_z = _upsample_2d(blurred_z, _up)

    h, w = up_dens.shape
    # Resolution and offsets for coordinate mapping (contour loops → screen)
    res = proj_res * _up
    crop_off_r = r0 * _up
    crop_off_c = c0 * _up

    # --- Build ESP heatmap PNG ---
    esp_above = up_esp[up_dens > isovalue]
    if esp_above.size == 0:
        return ESPSurface(
            resolution=res,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    # Use fixed ESP range if provided (stabilizes colors across GIF frames),
    # otherwise compute from this frame's projected data.
    if esp_range is not None:
        esp_vmin, esp_vmax = esp_range
    else:
        esp_vmin = float(np.percentile(esp_above, 5))
        esp_vmax = float(np.percentile(esp_above, 95))

    # Zero-centered normalization: ESP=0 → t=0.5 (green/neutral).
    # Positive ESP (electron-poor) → t<0.5 → blue.
    # Negative ESP (electron-rich) → t>0.5 → red.
    half_range = max(abs(esp_vmin), abs(esp_vmax), 1e-10)
    esp_norm = np.clip(0.5 - up_esp / (2 * half_range), 0.0, 1.0)

    surface_mask = up_dens > isovalue * 0.5
    lut_idx = np.where(surface_mask, (esp_norm * 255).astype(np.uint8), 0)

    # --- Lighting: directional + depth fog ---
    # 3D surface normals → Lambertian shading
    diffuse = np.clip(up_light, 0.0, 1.0)

    # Z-depth fog: front vivid, back washed toward white
    z_surf = up_z[surface_mask]
    if z_surf.size > 0:
        z_lo, z_hi = float(z_surf.min()), float(z_surf.max())
        z_range = max(z_hi - z_lo, 1e-10)
        z_norm = np.clip((up_z - z_lo) / z_range, 0.0, 1.0)
    else:
        z_norm = np.ones((h, w))
    depth_fade = 0.40 + 0.60 * z_norm  # 1.0 at front, 0.40 at back

    # Step 1: Get base LUT colors at full brightness and boost saturation
    # FIRST, before any darkening.  Each layer is displayed at ~0.11
    # opacity over white, so the source PNG must be hyper-saturated.
    rgb_f = np.full((h, w, 3), 255.0)
    for ch in range(3):
        rgb_f[:, :, ch] = np.where(
            surface_mask,
            _LUT[lut_idx, ch].astype(float),
            255.0,
        )
    gray = np.where(
        surface_mask,
        (rgb_f[:, :, 0] + rgb_f[:, :, 1] + rgb_f[:, :, 2]) / 3.0,
        255.0,
    )
    for ch in range(3):
        rgb_f[:, :, ch] = np.where(
            surface_mask,
            gray + _SAT_BOOST * (rgb_f[:, :, ch] - gray),
            255.0,
        )

    # Step 2: Apply lighting and depth as brightness modulation.
    # The sat-boosted colors are vivid enough to survive the darkening.
    brightness = 0.20 + 0.95 * diffuse * depth_fade

    # Fog: blend back parts toward white for subtle depth cue.
    fog = (1.0 - depth_fade) * 0.12

    for ch in range(3):
        rgb_f[:, :, ch] = np.where(
            surface_mask,
            rgb_f[:, :, ch] * brightness * (1.0 - fog) + 255.0 * fog,
            255.0,
        )
    # Post-upsample smooth: remove residual grid structure (checkerboard)
    for ch in range(3):
        rgb_f[:, :, ch] = _gaussian_blur_2d(rgb_f[:, :, ch], 0.8)
    rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)

    # --- Alpha: fully opaque within surface, soft edge at boundary ---
    # Clip paths handle the depth-graded ring shape; PNG just needs
    # clean color data with a smooth boundary.
    edge = np.clip(
        (up_dens - isovalue * 0.3) / (isovalue * 1.2),
        0.0,
        1.0,
    )
    alpha = (edge * 255).astype(np.uint8)

    # --- Extract contours for depth-graded clipping ---
    # Additive Z-shift: front-facing regions get a density boost so
    # contour thresholds are reached further out → wider ring spacing.
    # Back regions are unshifted → rings pack tighter → depth cueing.
    # (Additive avoids the edge-clustering of multiplicative Z-modulation.)
    z_surf_vals = up_z[surface_mask]
    if z_surf_vals.size > 0:
        z_s_lo = float(z_surf_vals.min())
        z_s_range = max(float(z_surf_vals.max()) - z_s_lo, 1e-10)
        z_01 = np.clip((up_z - z_s_lo) / z_s_range, 0.0, 1.0)
    else:
        z_01 = np.ones((h, w))
    contour_field = up_dens + isovalue * 0.5 * z_01
    # Extra smoothing prevents checkerboard artifacts from grid structure
    contour_field = _gaussian_blur_2d(contour_field, _PROJ_BLUR * 0.25)

    above = contour_field[surface_mask]
    layers: list[LobeContour2D] = []
    if above.size > 0:
        upper = float(np.percentile(above, 85))
        upper = max(upper, isovalue * 1.5)
        thresholds = np.geomspace(isovalue, upper, n_layers)
        crop_offset = np.array([crop_off_r, crop_off_c])
        for threshold in thresholds:
            raw_loops = chain_segments(marching_squares(contour_field, float(threshold)))
            offset_loops = [loop + crop_offset for loop in raw_loops]
            loops = [_resample_loop(lp) for lp in offset_loops if _loop_perimeter(lp) >= _MIN_LOOP_PERIMETER]
            if loops:
                layers.append(LobeContour2D(loops=loops, phase="pos", z_depth=0.0))

    # Flip vertically (numpy row 0 = y_min, PNG row 0 = top = y_max)
    rgba = np.dstack([rgb, alpha])
    rgba = np.flipud(rgba)

    from PIL import Image

    img = Image.fromarray(rgba, "RGBA")
    # Smooth upscale low-res images (e.g. GIF frames at 240x240) so that
    # individual voxel pixels aren't visible in the final output.
    if img.width < _MIN_PNG_RES or img.height < _MIN_PNG_RES:
        target = max(_MIN_PNG_RES, img.width * 3, img.height * 3)
        img = img.resize((target, target), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    data_uri = f"data:image/png;base64,{png_b64}"

    total_loops = sum(len(lc.loops) for lc in layers)
    logger.debug(
        "ESP surface: %dx%d heatmap, %d contour layers (%d loops), ESP [%.4g, %.4g]",
        w,
        h,
        len(layers),
        total_loops,
        esp_vmin,
        esp_vmax,
    )

    return ESPSurface(
        png_data_uri=data_uri,
        resolution=res,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        layers=layers,
        esp_vmin=esp_vmin,
        esp_vmax=esp_vmax,
    )


# ---------------------------------------------------------------------------
# Per-frame ESP recomputation for gif-rot
# ---------------------------------------------------------------------------


def _compute_normals_phys(cube: CubeData) -> np.ndarray:
    """Compute 3D density gradient in physical space for all voxels.

    Returns a flat (N_total, 3) array of gradient vectors (one per voxel).
    The gradient points toward increasing density; negate for outward normals.
    """
    n1, n2, n3 = cube.grid_shape
    vol = cube.grid_data.reshape(n1, n2, n3)
    g0, g1, g2 = np.gradient(vol)

    # Convert grid-index-space gradient to physical space:
    # ∇_phys V = steps^{-1} @ [g0, g1, g2]  (chain rule)
    m_inv = np.linalg.inv(cube.steps)  # steps are in Bohr, but direction is all we need

    flat_g = np.column_stack([g0.ravel(), g1.ravel(), g2.ravel()])
    return flat_g @ m_inv.T


def recompute_esp(graph: nx.Graph, config: RenderConfig, esp_data: dict) -> None:
    """Recompute ESP surface for the current graph orientation."""
    from xyzrender.utils import kabsch_rotation

    dens_cube = esp_data["dens_cube"]
    esp_cube = esp_data["esp_cube"]

    if "flat_indices" not in esp_data:
        mask = dens_cube.grid_data >= esp_data["isovalue"]
        esp_data["flat_indices"] = np.flatnonzero(mask)
        esp_data["pos_flat_ang"] = compute_grid_positions(dens_cube)
    if "_orig_atoms" not in esp_data:
        orig = np.array([p for _, p in dens_cube.atoms], dtype=float)
        esp_data["_orig_atoms"] = orig
        esp_data["_atom_centroid"] = orig.mean(axis=0)
    if "_bounding_radius" not in esp_data:
        corners = cube_corners_ang(dens_cube)
        r_max = float(
            np.linalg.norm(corners - esp_data["_atom_centroid"], axis=1).max(),
        )
        esp_data["_bounding_radius"] = r_max + r_max * 0.01 + 1e-9
    # Precompute 3D normals (once — the gradient field doesn't change)
    if "_normals_phys" not in esp_data:
        esp_data["_normals_phys"] = _compute_normals_phys(dens_cube)
    # Stable ESP color range from 3D shell data (view-independent).
    if "_esp_range" not in esp_data:
        iso = esp_data["isovalue"]
        dens_flat = dens_cube.grid_data.ravel()
        shell_3d = (dens_flat >= iso) & (dens_flat <= iso * _SHELL_UPPER)
        esp_shell = esp_cube.grid_data.ravel()[shell_3d]
        if esp_shell.size > 0:
            esp_data["_esp_range"] = (
                float(np.percentile(esp_shell, 5)),
                float(np.percentile(esp_shell, 95)),
            )

    orig = esp_data["_orig_atoms"]
    atom_centroid = esp_data["_atom_centroid"]
    curr = np.array(
        [graph.nodes[i]["position"] for i in graph.nodes()],
        dtype=float,
    )
    target_centroid = curr.mean(axis=0)

    r = esp_data["_bounding_radius"]
    fixed_bounds = (
        float(target_centroid[0] - r),
        float(target_centroid[0] + r),
        float(target_centroid[1] - r),
        float(target_centroid[1] + r),
    )

    rot = kabsch_rotation(orig, curr)

    config.esp_surface = build_esp_surface(
        dens_cube,
        esp_cube,
        isovalue=esp_data["isovalue"],
        rot=rot,
        atom_centroid=atom_centroid,
        target_centroid=target_centroid,
        flat_indices=esp_data["flat_indices"],
        pos_flat_ang=esp_data["pos_flat_ang"],
        fixed_bounds=fixed_bounds,
        n_layers=esp_data.get("n_layers", _N_LAYERS),
        upsample=3,  # lower resolution for GIF speed
        esp_range=esp_data.get("_esp_range"),
        normals_phys=esp_data["_normals_phys"],
    )


# ---------------------------------------------------------------------------
# ESP SVG rendering — PNG heatmap clipped by density contour rings
# ---------------------------------------------------------------------------


_ESP_BASE_OPACITY = 1.0  # base opacity for ESP surface (scaled by surface_opacity)


def esp_surface_svg(
    esp: ESPSurface,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
    surface_opacity: float = 1.0,
) -> list[str]:
    """Render ESP surface as contour-clipped heatmap layers.

    The PNG provides smooth ESP color gradients.  Each density contour
    layer clips the PNG and is rendered at ``base_opacity * surface_opacity / n_layers``
    opacity.  Where more layers overlap (inner rings), opacity accumulates
    — producing the same depth-graded ring aesthetic as ``dens_layers_svg``.
    """
    if not esp.png_data_uri or not esp.layers:
        return []

    img_x = canvas_w / 2 + scale * (esp.x_min - cx)
    img_y = canvas_h / 2 - scale * (esp.y_max - cy)
    img_w = scale * (esp.x_max - esp.x_min)
    img_h = scale * (esp.y_max - esp.y_min)

    n = len(esp.layers)
    per_layer = _ESP_BASE_OPACITY * surface_opacity / n

    lines: list[str] = ["  <defs>"]
    lines.append(
        f'    <image id="esp_heatmap" x="{img_x:.1f}" y="{img_y:.1f}" '
        f'width="{img_w:.1f}" height="{img_h:.1f}" '
        f'href="{esp.png_data_uri}" '
        f'preserveAspectRatio="none" image-rendering="optimizeQuality"/>'
    )

    # Build clip paths from contour layers (duck-type: ESPSurface has
    # the same .resolution/.x_min/.x_max/.y_min/.y_max as MOContours)
    clip_ids: list[str] = []
    for i, lobe in enumerate(esp.layers):
        d = _mo_combined_path_d(lobe.loops, esp, scale, cx, cy, canvas_w, canvas_h)  # type: ignore[arg-type]
        if d:
            clip_id = f"esp_clip_{i}"
            clip_ids.append(clip_id)
            lines.append(f'    <clipPath id="{clip_id}">')
            lines.append(f'      <path d="{d}" fill-rule="evenodd"/>')
            lines.append("    </clipPath>")

    lines.append("  </defs>")

    for clip_id in clip_ids:
        lines.append(f'  <use href="#esp_heatmap" clip-path="url(#{clip_id})" opacity="{per_layer:.3f}"/>')

    return lines
