"""Shared contour extraction infrastructure for surface modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from xyzrender.cube import BOHR_TO_ANG, CubeData

# --- Contour processing ---
# 3D lobe filtering (physical units — scales with cube grid spacing)
MIN_LOBE_VOLUME_BOHR3 = 0.1  # discard 3D orbital components smaller than this (Bohr^3)

# 2D projected-grid properties (grid-cell units, not related to cube spacing)
UPSAMPLE_FACTOR = 3  # 80x80 -> 400x400 -- smooth enough for publication
BLUR_SIGMA = 0.8  # Gaussian sigma in 2D grid cells before upsampling
MIN_LOOP_PERIMETER = 15.0  # upsampled grid units — discard tiny contour fragments


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Lobe3D:
    """A spatially connected 3D orbital lobe (connected component)."""

    flat_indices: np.ndarray  # indices into flattened grid/position arrays
    phase: str  # "pos" or "neg"


@dataclass
class LobeContour2D:
    """Contour loops for one 3D lobe projected to 2D."""

    loops: list[np.ndarray]  # each (M, 2) array of [row, col] points
    phase: str  # "pos" or "neg"
    z_depth: float  # average z-coordinate (for front/back ordering)
    centroid_3d: tuple[float, float, float] = (0.0, 0.0, 0.0)  # for pairing
    lobe_color: str | None = None  # per-lobe color override (NCI avg coloring)
    # Rotated 3D voxel positions (Å) — lets the renderer resolve queue-z
    # against the atoms each lobe overlaps.  None when not needed.
    voxel_pos: np.ndarray | None = None
    # Mesh geometry (populated only when surface_style == "mesh")
    mesh_iso_loops: list[np.ndarray] = field(default_factory=list)  # closed inner contour rings
    mesh_grid_lines: list[np.ndarray] = field(default_factory=list)  # open grid scan lines (H + V clipped to surface)


@runtime_checkable
class ContourGrid(Protocol):
    """Protocol for 2D projection grid metadata used by SVG path converters.

    Both :class:`SurfaceContours` and :class:`~xyzrender.nci.NCIContours`
    satisfy this protocol, allowing :func:`combined_path_d` to accept
    either type without inheritance coupling.
    """

    resolution: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass
class SurfaceContours:
    """Pre-computed MO or density contour data ready for SVG rendering."""

    lobes: list[LobeContour2D] = field(default_factory=list)  # sorted by z_depth
    resolution: int = 0
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    pos_color: str = "#2554A5"
    neg_color: str = "#851639"
    # Tight Angstrom extent of actual lobe contours (for canvas fitting)
    lobe_x_min: float | None = None
    lobe_x_max: float | None = None
    lobe_y_min: float | None = None
    lobe_y_max: float | None = None


# Backward-compatible alias (used by renderer and other callers)
MOContours = SurfaceContours


# ---------------------------------------------------------------------------
# Grid/projection functions
# ---------------------------------------------------------------------------


def cube_corners_ang(cube: CubeData) -> np.ndarray:
    """Compute the 8 corner positions of the cube grid in Angstrom."""
    n1, n2, n3 = cube.grid_shape
    ii, jj, kk = np.mgrid[0:2, 0:2, 0:2]
    ii = ii * (n1 - 1)
    jj = jj * (n2 - 1)
    kk = kk * (n3 - 1)
    corners = (
        cube.origin + ii[..., None] * cube.steps[0] + jj[..., None] * cube.steps[1] + kk[..., None] * cube.steps[2]
    ).reshape(-1, 3)
    return corners * BOHR_TO_ANG


def compute_grid_positions(cube: CubeData) -> np.ndarray:
    """Compute all grid positions in Angstrom (flattened). Cached for reuse."""
    n1, n2, n3 = cube.grid_shape
    ii, jj, kk = np.mgrid[0:n1, 0:n2, 0:n3]
    positions = (
        cube.origin + ii[..., None] * cube.steps[0] + jj[..., None] * cube.steps[1] + kk[..., None] * cube.steps[2]
    )
    return positions.reshape(-1, 3) * BOHR_TO_ANG


# ---------------------------------------------------------------------------
# Marching squares
# ---------------------------------------------------------------------------

# Lookup table: for each 4-bit case index, list of (edge_a, edge_b) pairs
# Corners: 0=top-left(i,j), 1=top-right(i,j+1), 2=bottom-right(i+1,j+1), 3=bottom-left(i+1,j)
# Edges: 0=top, 1=right, 2=bottom, 3=left
_MS_TABLE: dict[int, list[tuple[int, int]]] = {
    0: [],
    1: [(3, 0)],
    2: [(0, 1)],
    3: [(3, 1)],
    4: [(1, 2)],
    5: [(3, 0), (1, 2)],  # saddle — resolved below
    6: [(0, 2)],
    7: [(3, 2)],
    8: [(2, 3)],
    9: [(2, 0)],
    10: [(0, 3), (2, 1)],  # saddle — resolved below
    11: [(2, 1)],
    12: [(1, 3)],
    13: [(1, 0)],
    14: [(0, 3)],
    15: [],
}


def marching_squares(
    grid: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Extract contour line segments from a 2D scalar field.

    Returns (N, 4) array where each row is [row1, col1, row2, col2].
    """
    ny, nx = grid.shape
    _empty = np.empty((0, 4))
    if ny < 2 or nx < 2:
        return _empty

    # Corner values for all (ny-1) x (nx-1) cells
    v0 = grid[:-1, :-1]  # top-left
    v1 = grid[:-1, 1:]  # top-right
    v2 = grid[1:, 1:]  # bottom-right
    v3 = grid[1:, :-1]  # bottom-left

    # 4-bit case index per cell
    case = (
        (v0 >= threshold).view(np.uint8)
        | ((v1 >= threshold).view(np.uint8) << 1)
        | ((v2 >= threshold).view(np.uint8) << 2)
        | ((v3 >= threshold).view(np.uint8) << 3)
    )

    # Early exit: no contour crossings
    if not np.any(case & (case != 15)):
        return _empty

    # Cell row/col index grids
    ri, ci = np.indices((ny - 1, nx - 1), dtype=float)

    # Interpolation parameter t on each edge, clamped to [0, 1]
    def _t(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
        dv = vb - va
        safe_dv = np.where(np.abs(dv) > 1e-12, dv, 1.0)
        t = np.where(np.abs(dv) > 1e-12, (threshold - va) / safe_dv, 0.5)
        return np.clip(t, 0.0, 1.0)

    t01, t12, t23, t30 = _t(v0, v1), _t(v1, v2), _t(v2, v3), _t(v3, v0)

    # Edge crossing positions (row, col) for each of the 4 edges:
    er = [ri, ri + t12, ri + 1, ri + 1 - t30]
    ec = [ci + t01, ci + 1, ci + 1 - t23, ci]

    # Saddle-point centre value (only used for cases 5 and 10)
    center = (v0 + v1 + v2 + v3) * 0.25

    # Gather segments per case (14 iterations, not ny*nx)
    seg_r1, seg_c1, seg_r2, seg_c2 = [], [], [], []

    def _gather(mask: np.ndarray, ea: int, eb: int) -> None:
        seg_r1.append(er[ea][mask])
        seg_c1.append(ec[ea][mask])
        seg_r2.append(er[eb][mask])
        seg_c2.append(ec[eb][mask])

    for cv in range(1, 15):
        mask = case == cv
        if not mask.any():
            continue

        if cv == 5:
            alt = mask & (center >= threshold)
            std = mask & ~alt
            if std.any():
                _gather(std, 3, 0)
                _gather(std, 1, 2)
            if alt.any():
                _gather(alt, 3, 2)
                _gather(alt, 1, 0)
        elif cv == 10:
            alt = mask & (center >= threshold)
            std = mask & ~alt
            if std.any():
                _gather(std, 0, 3)
                _gather(std, 2, 1)
            if alt.any():
                _gather(alt, 0, 1)
                _gather(alt, 2, 3)
        else:
            for ea, eb in _MS_TABLE[cv]:
                _gather(mask, ea, eb)

    if not seg_r1:
        return _empty

    return np.column_stack(
        [
            np.concatenate(seg_r1),
            np.concatenate(seg_c1),
            np.concatenate(seg_r2),
            np.concatenate(seg_c2),
        ]
    )


# ---------------------------------------------------------------------------
# Segment chaining into closed loops
# ---------------------------------------------------------------------------


def chain_segments(
    segments: np.ndarray,
    decimals: int = 4,
) -> list[np.ndarray]:
    """Connect line segments into closed contour loops."""
    n_seg = len(segments)
    if n_seg == 0:
        return []

    # 2*n_seg endpoints: endpoint 2i = start of segment i, 2i+1 = end
    endpoints = np.empty((2 * n_seg, 2))
    endpoints[0::2] = segments[:, :2]
    endpoints[1::2] = segments[:, 2:]

    # Integer keys for fast pair matching via sort
    kscale = 10.0**decimals
    ikeys = np.rint(endpoints * kscale).astype(np.int64)
    ikeys[:, 0] -= ikeys[:, 0].min()
    ikeys[:, 1] -= ikeys[:, 1].min()
    max_col = int(ikeys[:, 1].max()) + 1
    combined = ikeys[:, 0] * max_col + ikeys[:, 1]

    # Sort and pair-match consecutive equal keys
    order = np.argsort(combined, kind="mergesort")
    sorted_keys = combined[order]

    match = np.full(2 * n_seg, -1, dtype=np.intp)
    i = 0
    while i < 2 * n_seg - 1:
        if sorted_keys[i] == sorted_keys[i + 1]:
            a, b = int(order[i]), int(order[i + 1])
            match[a] = b
            match[b] = a
            i += 2
        else:
            i += 1

    # Walk chains using array indexing
    used = np.zeros(n_seg, dtype=bool)
    loops: list[np.ndarray] = []

    for start_seg in range(n_seg):
        if used[start_seg]:
            continue
        used[start_seg] = True
        chain_pts = [endpoints[2 * start_seg], endpoints[2 * start_seg + 1]]
        cur = 2 * start_seg + 1

        while True:
            partner = match[cur]
            if partner < 0:
                break
            seg = partner >> 1
            if used[seg]:
                break
            used[seg] = True
            exit_ep = partner ^ 1
            chain_pts.append(endpoints[exit_ep])
            cur = exit_ep

        if len(chain_pts) >= 3:
            loops.append(np.array(chain_pts))

    return loops


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------


def resample_loop(
    loop: np.ndarray,
    target_spacing: float = 1.5,
) -> np.ndarray:
    """Resample a closed contour loop at uniform arc-length intervals."""
    n = len(loop)
    if n < 3:
        return loop

    closed = np.vstack([loop, loop[:1]])
    diffs = np.diff(closed, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    total_len = float(dists.sum())
    if total_len < 1e-6:
        return loop

    n_pts = max(int(total_len / target_spacing + 0.5), 8)

    cum = np.empty(n + 1)
    cum[0] = 0.0
    np.cumsum(dists, out=cum[1:])

    targets = np.linspace(0, total_len, n_pts, endpoint=False)
    seg_idx = np.searchsorted(cum[1:], targets, side="right")
    seg_idx = np.clip(seg_idx, 0, n - 1)

    seg_len = dists[seg_idx]
    safe_len = np.where(seg_len > 1e-12, seg_len, 1.0)
    t = np.where(seg_len > 1e-12, (targets - cum[seg_idx]) / safe_len, 0.0)

    p0 = closed[seg_idx]
    p1 = closed[seg_idx + 1]
    return p0 + t[:, np.newaxis] * (p1 - p0)


# ---------------------------------------------------------------------------
# Gaussian smoothing + bilinear upsampling
# ---------------------------------------------------------------------------


def loop_perimeter(loop: np.ndarray) -> float:
    """Sum of segment lengths around a contour loop."""
    diffs = np.diff(np.vstack([loop, loop[:1]]), axis=0)
    return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())


def gaussian_blur_2d(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Apply separable Gaussian blur to 2D grid (vectorized, pure numpy)."""
    size = int(4 * sigma + 0.5) * 2 + 1
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    pad = size // 2
    ny, nx = grid.shape

    # Horizontal pass: convolve each row via matrix multiply
    padded = np.pad(grid, ((0, 0), (pad, pad)), mode="edge")
    idx = np.arange(nx)[:, None] + np.arange(size)[None, :]
    temp = padded[:, idx] @ kernel  # (ny, nx)

    # Vertical pass: convolve each column via matrix multiply
    padded = np.pad(temp, ((pad, pad), (0, 0)), mode="edge")
    idx = np.arange(ny)[:, None] + np.arange(size)[None, :]
    return padded[idx, :].transpose(0, 2, 1) @ kernel  # (ny, nx)


def upsample_2d(grid: np.ndarray, factor: int) -> np.ndarray:
    """Upsample 2D array by integer factor using separable bilinear interpolation."""
    ny, nx = grid.shape
    if ny < 2 or nx < 2:
        return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)

    # Horizontal pass: vectorised across all rows simultaneously
    x_new = np.linspace(0, nx - 1, nx * factor)
    x0 = np.clip(np.searchsorted(np.arange(nx), x_new, side="right") - 1, 0, nx - 2)
    dx = x_new - x0
    temp = grid[:, x0] + dx * (grid[:, x0 + 1] - grid[:, x0])

    # Vertical pass: vectorised across all columns simultaneously
    y_new = np.linspace(0, ny - 1, ny * factor)
    y0 = np.clip(np.searchsorted(np.arange(ny), y_new, side="right") - 1, 0, ny - 2)
    dy = (y_new - y0)[:, np.newaxis]
    return temp[y0] + dy * (temp[y0 + 1] - temp[y0])


# ---------------------------------------------------------------------------
# SVG path conversion
# ---------------------------------------------------------------------------


def loop_to_path_d(
    loop: np.ndarray,
    grid: ContourGrid,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> str | None:
    """Convert a contour loop to a smooth SVG path (Catmull-Rom to cubic Bezier)."""
    if len(loop) < 3:
        return None
    res = max(grid.resolution - 1, 1)

    # Vectorized grid → SVG coordinate transform
    x_ang = grid.x_min + (loop[:, 1] / res) * (grid.x_max - grid.x_min)
    y_ang = grid.y_min + (loop[:, 0] / res) * (grid.y_max - grid.y_min)
    sx = canvas_w / 2 + scale * (x_ang - cx)
    sy = canvas_h / 2 - scale * (y_ang - cy)

    # Catmull-Rom control points via rolled arrays
    p0x, p0y = np.roll(sx, 1), np.roll(sy, 1)
    p2x, p2y = np.roll(sx, -1), np.roll(sy, -1)
    p3x, p3y = np.roll(sx, -2), np.roll(sy, -2)

    cp1x = sx + (p2x - p0x) / 6
    cp1y = sy + (p2y - p0y) / 6
    cp2x = p2x - (p3x - sx) / 6
    cp2y = p2y - (p3y - sy) / 6

    # Build SVG path string
    coords = np.column_stack([cp1x, cp1y, cp2x, cp2y, p2x, p2y])
    cmds = [f"C {a:.1f} {b:.1f} {c:.1f} {d:.1f} {e:.1f} {f:.1f}" for a, b, c, d, e, f in coords.tolist()]
    return f"M {sx[0]:.1f} {sy[0]:.1f} " + " ".join(cmds) + " Z"


def combined_path_d(
    loops: list[np.ndarray],
    grid: ContourGrid,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> str | None:
    """Combine all contour loops of one phase into a single SVG path d-string.

    Uses fill-rule="evenodd" so inner loops become holes automatically.
    """
    parts = []
    for loop in loops:
        d = loop_to_path_d(loop, grid, scale, cx, cy, canvas_w, canvas_h)
        if d:
            parts.append(d)
    return " ".join(parts) if parts else None


def open_path_to_d(
    points: np.ndarray,
    grid: ContourGrid,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> str | None:
    """Convert an open polyline to a smooth SVG path (Catmull-Rom, no Z closure).

    Uses reflected phantom points at the two endpoints so the curve
    starts and ends tangent to the first/last segment.
    """
    n = len(points)
    if n < 2:
        return None
    res = max(grid.resolution - 1, 1)

    # Grid → SVG coordinate transform (same as loop_to_path_d)
    x_ang = grid.x_min + (points[:, 1] / res) * (grid.x_max - grid.x_min)
    y_ang = grid.y_min + (points[:, 0] / res) * (grid.y_max - grid.y_min)
    sx = canvas_w / 2 + scale * (x_ang - cx)
    sy = canvas_h / 2 - scale * (y_ang - cy)

    if n == 2:
        return f"M {sx[0]:.1f} {sy[0]:.1f} L {sx[1]:.1f} {sy[1]:.1f}"

    # Build extended arrays with reflected phantom endpoints
    ex = np.empty(n + 2)
    ey = np.empty(n + 2)
    ex[1:-1] = sx
    ey[1:-1] = sy
    ex[0] = 2 * sx[0] - sx[1]  # phantom start
    ey[0] = 2 * sy[0] - sy[1]
    ex[-1] = 2 * sx[-1] - sx[-2]  # phantom end
    ey[-1] = 2 * sy[-1] - sy[-2]

    # Catmull-Rom → cubic Bezier for segments 1..n-1 in the extended array
    parts = [f"M {sx[0]:.1f} {sy[0]:.1f}"]
    for i in range(1, n):
        p0x, p0y = ex[i - 1], ey[i - 1]
        p1x, p1y = ex[i], ey[i]
        p2x, p2y = ex[i + 1], ey[i + 1]
        p3x, p3y = ex[i + 2] if i + 2 < n + 2 else ex[i + 1], ey[i + 2] if i + 2 < n + 2 else ey[i + 1]
        cp1x = p1x + (p2x - p0x) / 6
        cp1y = p1y + (p2y - p0y) / 6
        cp2x = p2x - (p3x - p1x) / 6
        cp2y = p2y - (p3y - p1y) / 6
        parts.append(f"C {cp1x:.1f} {cp1y:.1f} {cp2x:.1f} {cp2y:.1f} {p2x:.1f} {p2y:.1f}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Mesh geometry extraction (shared by MO, density, NCI)
# ---------------------------------------------------------------------------

_MESH_N_ISO_LEVELS = 5  # number of intermediate iso-contour rings (contour style)
_MESH_N_LINES = 20  # number of lines per direction (mesh style)
_MESH_MIN_SEGMENT = 5  # minimum segment span (grid cells)
_MESH_N_PTS = 24  # points sampled per mesh line for SVG rendering
_MESH_WARP_STRENGTH = 0.25  # perpendicular warp amplitude
_MESH_SMOOTH_SIGMA = 3.0  # 1D Gaussian sigma for warp smoothing


def _smooth_1d(values: np.ndarray, sigma: float = _MESH_SMOOTH_SIGMA) -> np.ndarray:
    """Apply 1-D Gaussian smoothing to an array."""
    if len(values) < 3 or sigma < 0.5:
        return values
    size = int(4 * sigma + 0.5) * 2 + 1
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, size // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _find_iso_crossing(field_1d: np.ndarray, isovalue: float, from_start: bool) -> float:
    """Find fractional index where *field_1d* crosses *isovalue*.

    Scans from the start or end and linearly interpolates the exact crossing.
    Returns a fractional index into *field_1d*.
    """
    n = len(field_1d)

    # Define traversal parameters based on direction
    range_args = (0, n - 1, 1) if from_start else (n - 1, 0, -1)
    step = range_args[2]

    for i in range(*range_args):
        v_curr, v_next = field_1d[i], field_1d[i + step]

        if v_curr < isovalue <= v_next:
            t = (isovalue - v_curr) / max(v_next - v_curr, 1e-12)
            return i + t * step

    return float(range_args[0])


def _bilinear_sample(field: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Sample *field* at fractional (row, col) positions with bilinear interpolation."""
    nr, nc = field.shape
    r0 = np.clip(np.floor(rows).astype(int), 0, nr - 2)
    c0 = np.clip(np.floor(cols).astype(int), 0, nc - 2)
    dr = rows - r0
    dc = cols - c0
    return (
        field[r0, c0] * (1 - dr) * (1 - dc)
        + field[r0, c0 + 1] * (1 - dr) * dc
        + field[r0 + 1, c0] * dr * (1 - dc)
        + field[r0 + 1, c0 + 1] * dr * dc
    )


def _warped_scan_lines(
    work: np.ndarray,
    isovalue: float,
    n_lines: int,
    axis: int,
    *,
    warp: float = _MESH_WARP_STRENGTH,
    n_pts: int = _MESH_N_PTS,
    min_segment: int = _MESH_MIN_SEGMENT,
) -> list[np.ndarray]:
    """Extract scan-line segments warped to emulate 3D surface curvature.

    Each scan line starts as a straight horizontal (*axis* = 0) or vertical
    (*axis* = 1) line.  Endpoints are interpolated to the exact isovalue
    crossing so the line starts and ends precisely on the outer contour.

    The warp basis is a **heavily-blurred binary mask** (not the raw scalar
    field).  This is smooth everywhere — no nuclear cusps or field spikes —
    and naturally peaks in the geometric center of the surface, creating a
    uniform 3D-curvature impression for any surface type (MO, density, NCI).

    Returns open polylines as ``(M, 2)`` arrays in ``[row, col]`` coords.
    """
    mask = work >= isovalue
    if not mask.any():
        return []

    peak = float(work.max())
    if peak <= isovalue:
        return []

    # Generalize axes:
    collapse_axis = 1 - axis
    active_indices = np.nonzero(mask.any(axis=collapse_axis))[0]

    if len(active_indices) < 2:
        return []

    p_min, p_max = int(active_indices[0]), int(active_indices[-1])
    span = p_max - p_min
    positions = np.linspace(p_min, p_max, n_lines + 2, dtype=int)[1:-1]
    line_length = work.shape[collapse_axis]

    lines: list[np.ndarray] = []

    for pi in positions:
        # Extract 1D field along the current line dynamically
        slices = [slice(None), slice(None)]
        slices[axis] = pi
        line_field = work[tuple(slices)]

        above = line_field >= isovalue
        changes = np.diff(above.astype(np.int8))
        entries = np.where(changes == 1)[0]
        exits = np.where(changes == -1)[0] + 1

        if above[0]:
            entries = np.concatenate([[0], entries])
        if above[-1]:
            exits = np.concatenate([exits, [line_length - 1]])

        for ei, xi in zip(entries, exits, strict=False):
            if xi - ei < min_segment:
                continue

            segment = line_field[ei : xi + 1]
            v_start = _find_iso_crossing(segment, isovalue, from_start=True) + ei
            v_end = _find_iso_crossing(segment, isovalue, from_start=False) + ei

            n_dense = max(xi - ei, 40)
            dense_sweep = np.linspace(v_start, v_end, n_dense)
            dense_fixed = np.full(n_dense, float(pi))

            # Assign row/col depending on orientation
            if axis == 0:
                rows_dense, cols_dense = dense_fixed, dense_sweep
            else:
                rows_dense, cols_dense = dense_sweep, dense_fixed

            # Bilinear field sample + smooth for warp
            f_vals = _bilinear_sample(work, rows_dense, cols_dense)
            f_norm = np.clip((f_vals - isovalue) / (peak - isovalue), 0.0, 1.0)
            f_smooth = _smooth_1d(f_norm)
            f_smooth[0] = 0.0
            f_smooth[-1] = 0.0

            displacement = f_smooth * warp * span
            warped_fixed = pi + displacement

            # Apply warp to the correct dimension
            if axis == 0:
                warped_rows, warped_cols = warped_fixed, cols_dense
            else:
                warped_rows, warped_cols = rows_dense, warped_fixed

            idx = np.linspace(0, n_dense - 1, n_pts, dtype=int)
            pts = np.column_stack([warped_rows[idx], warped_cols[idx]])
            lines.append(pts)

    return lines


def extract_mesh_geometry(
    field_2d: np.ndarray,
    isovalue: float,
    crop_offset: np.ndarray,
    *,
    is_negative: bool = False,
    n_iso_levels: int = _MESH_N_ISO_LEVELS,
    n_lines: int = _MESH_N_LINES,
    warp: float = _MESH_WARP_STRENGTH,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract contour rings and warped grid lines from a 2D projected scalar field.

    Two types of geometry are extracted:

    * **Iso-contour rings** (``contour`` style): concentric curved loops at
      intermediate field thresholds showing surface depth.
    * **Warped grid lines** (``mesh`` style): horizontal and vertical scan
      lines clipped to the exact outer contour and displaced perpendicular
      by the smoothed field value, emulating 3D surface curvature.

    Parameters
    ----------
    field_2d:
        The upsampled/blurred 2D scalar field (cropped region).
    isovalue:
        Outer contour threshold.
    crop_offset:
        ``[r0 * upsample, c0 * upsample]`` to map back to full-grid coords.
    is_negative:
        If True, negate the field (for neg-phase MO lobes).
    n_iso_levels:
        Number of intermediate iso-contour rings.
    n_lines:
        Number of grid lines per direction (horizontal and vertical).
    warp:
        Perpendicular warp amplitude (higher = more curvature).

    Returns
    -------
    (iso_loops, grid_lines):
        *iso_loops* — closed contour arrays in full-grid coords.
        *grid_lines* — open warped scan-line arrays in full-grid coords.
    """
    work = -field_2d if is_negative else field_2d

    # --- Iso-contour rings ---
    above = work[work > isovalue]
    iso_loops: list[np.ndarray] = []
    if above.size > 0:
        peak = float(np.percentile(above, 95))
        if peak > isovalue * 1.05:
            thresholds = np.linspace(isovalue * 1.05, peak * 0.9, n_iso_levels)
            for thr in thresholds:
                raw = chain_segments(marching_squares(work, float(thr)))
                for lp in raw:
                    offset_lp = lp + crop_offset
                    if loop_perimeter(offset_lp) >= MIN_LOOP_PERIMETER:
                        iso_loops.append(resample_loop(offset_lp))

    # --- Warped grid lines clipped to surface ---
    h_lines = _warped_scan_lines(work, isovalue, n_lines, axis=0, warp=warp)
    v_lines = _warped_scan_lines(work, isovalue, n_lines, axis=1, warp=warp)
    grid_lines = [pts + crop_offset for pts in h_lines + v_lines]

    return iso_loops, grid_lines


# ---------------------------------------------------------------------------
# Shared 2D projection pipeline (used by MO and NCI)
# ---------------------------------------------------------------------------


def project_region_to_contours(
    grid_2d: np.ndarray,
    resolution: int,
    lobe_pos: np.ndarray,
    phase: str,
    threshold: float,
    *,
    blur_sigma: float = BLUR_SIGMA,
    upsample_factor: int = UPSAMPLE_FACTOR,
    surface_style: str = "solid",
    n_mesh_iso: int | None = None,
    n_mesh_lines: int = _MESH_N_LINES,
) -> LobeContour2D | None:
    """Crop, blur, upsample, and contour a 2D projected scalar field.

    Shared pipeline for MO and NCI surface modules.  The caller populates
    *grid_2d* (value-binned for MO, binary membership for NCI) and handles
    any pre-processing such as dilation.

    Parameters
    ----------
    grid_2d:
        ``(resolution, resolution)`` 2D projected scalar field.
    resolution:
        Grid side length.
    lobe_pos:
        ``(N, 3)`` transformed voxel positions in Angstrom (for z-depth
        and centroid).
    phase:
        ``"pos"`` or ``"neg"`` — controls blur clamping and marching-
        squares field negation.
    threshold:
        Marching-squares iso threshold.
    blur_sigma:
        Gaussian blur sigma in grid cells.
    upsample_factor:
        Integer upsampling factor before contouring.
    surface_style:
        ``"solid"``, ``"mesh"``, ``"contour"``, or ``"dot"``.
    n_mesh_iso:
        Intermediate iso-contour rings for mesh geometry (default:
        10 for ``"dot"``, :data:`_MESH_N_ISO_LEVELS` otherwise).
    n_mesh_lines:
        Grid lines per direction for mesh geometry.

    Returns
    -------
    LobeContour2D | None
        Contour data, or ``None`` if no contours survive filtering.
    """
    # Crop to bounding box + blur kernel padding
    nz_rows, nz_cols = np.nonzero(grid_2d)
    if len(nz_rows) == 0:
        return None
    pad = max(3, int(blur_sigma * 4) + 1)
    r0 = max(0, int(nz_rows.min()) - pad)
    r1 = min(resolution, int(nz_rows.max()) + pad + 1)
    c0 = max(0, int(nz_cols.min()) - pad)
    c1 = min(resolution, int(nz_cols.max()) + pad + 1)
    cropped = grid_2d[r0:r1, c0:c1]

    # Blur + phase-dependent clamp
    blurred = gaussian_blur_2d(cropped, blur_sigma)
    if phase == "pos":
        blurred = np.maximum(blurred, 0.0)
    else:
        blurred = np.minimum(blurred, 0.0)

    upsampled = upsample_2d(blurred, upsample_factor)

    # Marching squares (negate field for neg-phase lobes)
    if phase == "pos":
        raw_loops = chain_segments(marching_squares(upsampled, threshold))
    else:
        raw_loops = chain_segments(marching_squares(-upsampled, threshold))

    # Offset contour coords back to full-grid space
    offset = np.array([r0 * upsample_factor, c0 * upsample_factor])
    offset_loops = [loop + offset for loop in raw_loops]

    loops = [resample_loop(lp) for lp in offset_loops if loop_perimeter(lp) >= MIN_LOOP_PERIMETER]

    if not loops:
        return None

    z_depth = float(lobe_pos[:, 2].mean())
    cent_3d = (float(lobe_pos[:, 0].mean()), float(lobe_pos[:, 1].mean()), z_depth)
    lc = LobeContour2D(
        loops=loops,
        phase=phase,
        z_depth=z_depth,
        centroid_3d=cent_3d,
        voxel_pos=lobe_pos.copy(),
    )

    # Mesh geometry
    if surface_style in ("mesh", "contour", "dot"):
        if n_mesh_iso is None:
            n_mesh_iso = 10 if surface_style == "dot" else _MESH_N_ISO_LEVELS
        iso_loops, grid_lines = extract_mesh_geometry(
            upsampled,
            threshold,
            offset,
            is_negative=(phase == "neg"),
            n_iso_levels=n_mesh_iso,
            n_lines=n_mesh_lines,
        )
        lc.mesh_iso_loops = iso_loops
        lc.mesh_grid_lines = grid_lines

    return lc


# ---------------------------------------------------------------------------
# Shared mesh/wire SVG rendering
# ---------------------------------------------------------------------------


def render_lobe_svg(
    lobe: LobeContour2D,
    grid: ContourGrid,
    fill_color: str,
    opacity: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
    surface_style: str = "solid",
    stroke_width: float = 1.5,
    mesh_inner_width: float = 0.8,
    outline_width: float = 0.0,
    outline_color: str = "#000000",
) -> list[str]:
    """Render one lobe in solid/mesh/wire style. Returns SVG element strings.

    ``outline_width`` and ``outline_color`` apply only in solid mode; mesh /
    contour / dot styles already render strokes around the boundary.
    """
    d_all = combined_path_d(lobe.loops, grid, scale, cx, cy, canvas_w, canvas_h)
    if not d_all:
        return []

    if surface_style == "solid":
        if outline_width > 0:
            stroke_attr = f' stroke="{outline_color}" stroke-width="{outline_width:.2f}"'
        else:
            stroke_attr = ' stroke="none"'
        return [
            f'  <g opacity="{opacity:.2f}">',
            f'    <path d="{d_all}" fill="{fill_color}" fill-rule="evenodd"{stroke_attr}/>',
            "  </g>",
        ]

    # Derive stroke color (darkened fill)
    from xyzrender.colors import Color

    stroke_hex = Color.from_str(fill_color).darken(strength=0.4).hex

    # --- dot: outer boundary + iso-contour rings, all as round dots ---
    if surface_style == "dot":
        dot_sw = stroke_width * 1.2
        dot_inner_sw = mesh_inner_width * 1.4
        dot_gap = max(2.5, dot_sw * 1.8)
        dot_inner_gap = max(2.0, dot_inner_sw * 1.8)
        lines: list[str] = [f'  <g opacity="{opacity:.2f}">']
        lines.append(
            f'    <path d="{d_all}" fill="none" fill-rule="evenodd"'
            f' stroke="{stroke_hex}" stroke-width="{dot_sw:.1f}"'
            f' stroke-dasharray="0 {dot_gap:.1f}" stroke-linecap="round"/>'
        )
        for iso_loop in lobe.mesh_iso_loops:
            d = loop_to_path_d(iso_loop, grid, scale, cx, cy, canvas_w, canvas_h)
            if d:
                lines.append(
                    f'    <path d="{d}" fill="none"'
                    f' stroke="{stroke_hex}" stroke-width="{dot_inner_sw:.1f}"'
                    f' stroke-dasharray="0 {dot_inner_gap:.1f}" stroke-linecap="round"/>'
                )
        lines.append("  </g>")
        return lines

    # --- contour / mesh: solid strokes ---
    lines = [f'  <g opacity="{opacity:.2f}">']
    lines.append(
        f'    <path d="{d_all}" fill="none" fill-rule="evenodd"'
        f' stroke="{stroke_hex}" stroke-width="{stroke_width:.1f}"/>'
    )
    if surface_style == "contour":
        for iso_loop in lobe.mesh_iso_loops:
            d = loop_to_path_d(iso_loop, grid, scale, cx, cy, canvas_w, canvas_h)
            if d:
                lines.append(
                    f'    <path d="{d}" fill="none" stroke="{stroke_hex}" stroke-width="{mesh_inner_width:.1f}"/>'
                )
    elif surface_style == "mesh":
        for grid_pts in lobe.mesh_grid_lines:
            d = open_path_to_d(grid_pts, grid, scale, cx, cy, canvas_w, canvas_h)
            if d:
                lines.append(
                    f'    <path d="{d}" fill="none" stroke="{stroke_hex}" stroke-width="{mesh_inner_width:.1f}"/>'
                )
    lines.append("  </g>")
    return lines
