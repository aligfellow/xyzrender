"""MO (molecular orbital) contour extraction, classification, and SVG rendering."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from xyzrender.cube import BOHR_TO_ANG, CubeData

if TYPE_CHECKING:
    import networkx as nx

    from xyzrender.types import RenderConfig

logger = logging.getLogger(__name__)

# --- Contour processing ---
_UPSAMPLE_FACTOR = 3  # 80x80 -> 240x240 -- smooth enough for publication
_BLUR_SIGMA = 0.8  # Gaussian sigma (in grid cells) before upsampling
_MIN_LOBE_CELLS = 20  # discard 3D components smaller than this
_MIN_LOOP_PERIMETER = 15.0  # grid units — discard tiny contour fragments


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

    loops: list[list[tuple[float, float]]]
    phase: str  # "pos" or "neg"
    z_depth: float  # average z-coordinate (for front/back ordering)
    centroid_3d: tuple[float, float, float] = (0.0, 0.0, 0.0)  # for pairing


@dataclass
class MOContours:
    """Pre-computed MO contour data ready for SVG rendering."""

    lobes: list[LobeContour2D] = field(default_factory=list)  # sorted by z_depth
    resolution: int = 0
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    pos_color: str = "#2554A5"
    neg_color: str = "#851639"


# ---------------------------------------------------------------------------
# 3D connected component labeling (BFS flood-fill)
# ---------------------------------------------------------------------------


def find_3d_lobes(grid_3d: np.ndarray, isovalue: float) -> list[Lobe3D]:
    """Find connected 3D orbital lobes at ±isovalue via BFS flood-fill."""
    shape = grid_3d.shape
    s1, s2 = shape[1] * shape[2], shape[2]
    lobes: list[Lobe3D] = []

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

            if len(component) >= _MIN_LOBE_CELLS:
                lobes.append(Lobe3D(flat_indices=np.array(component, dtype=np.intp), phase=phase))

    logger.debug("Found %d 3D lobes at isovalue %.4g", len(lobes), isovalue)
    return lobes


def cube_corners_ang(cube: CubeData) -> np.ndarray:
    """Compute the 8 corner positions of the cube grid in Angstrom."""
    n1, n2, n3 = cube.grid_shape
    corners = np.empty((8, 3))
    idx = 0
    for i in (0, n1 - 1):
        for j in (0, n2 - 1):
            for k in (0, n3 - 1):
                corners[idx] = cube.origin + i * cube.steps[0] + j * cube.steps[1] + k * cube.steps[2]
                idx += 1
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
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Extract contour line segments from a 2D scalar field.

    Returns list of ((row1, col1), (row2, col2)) segments in grid coordinates.
    """
    ny, nx = grid.shape
    if ny < 2 or nx < 2:
        return []

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
        return []

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
    #   Edge 0 (top):    corners 0->1  ->  (i,       j + t)
    #   Edge 1 (right):  corners 1->2  ->  (i + t,   j + 1)
    #   Edge 2 (bottom): corners 2->3  ->  (i + 1,   j + 1 - t)
    #   Edge 3 (left):   corners 3->0  ->  (i + 1-t, j)
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
        return []

    # Stack into (N, 4) array and convert to list of tuple pairs
    pts = np.column_stack(
        [
            np.concatenate(seg_r1),
            np.concatenate(seg_c1),
            np.concatenate(seg_r2),
            np.concatenate(seg_c2),
        ]
    )
    return [((r[0], r[1]), (r[2], r[3])) for r in pts.tolist()]


# ---------------------------------------------------------------------------
# Segment chaining into closed loops
# ---------------------------------------------------------------------------


def chain_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    decimals: int = 4,
) -> list[list[tuple[float, float]]]:
    """Connect line segments into closed contour loops."""
    if not segments:
        return []

    # Build adjacency: rounded endpoint -> list of (segment_index, other_endpoint)
    adj: dict[tuple[float, float], list[tuple[int, tuple[float, float]]]] = {}
    for idx, (p1, p2) in enumerate(segments):
        k1 = (round(p1[0], decimals), round(p1[1], decimals))
        k2 = (round(p2[0], decimals), round(p2[1], decimals))
        adj.setdefault(k1, []).append((idx, p2))
        adj.setdefault(k2, []).append((idx, p1))

    used = set()
    loops: list[list[tuple[float, float]]] = []

    for seg_idx, (start, end) in enumerate(segments):
        if seg_idx in used:
            continue
        used.add(seg_idx)
        loop = [start, end]
        current_key = (round(end[0], decimals), round(end[1], decimals))
        start_key = (round(start[0], decimals), round(start[1], decimals))

        while current_key != start_key:
            found = False
            for next_idx, next_pt in adj.get(current_key, []):
                if next_idx not in used:
                    used.add(next_idx)
                    loop.append(next_pt)
                    current_key = (round(next_pt[0], decimals), round(next_pt[1], decimals))
                    found = True
                    break
            if not found:
                break

        if len(loop) >= 3:
            loops.append(loop)

    return loops


def _resample_loop(
    loop: list[tuple[float, float]],
    target_spacing: float = 1.5,
) -> list[tuple[float, float]]:
    """Resample a closed contour loop at uniform arc-length intervals."""
    if len(loop) < 3:
        return loop

    # Compute cumulative arc length (closed: include segment back to start)
    dists = []
    for i in range(len(loop)):
        j = (i + 1) % len(loop)
        dx = loop[j][0] - loop[i][0]
        dy = loop[j][1] - loop[i][1]
        dists.append((dx * dx + dy * dy) ** 0.5)
    total_len = sum(dists)
    if total_len < 1e-6:
        return loop

    n_pts = max(int(total_len / target_spacing + 0.5), 8)

    cum = [0.0]
    for d in dists:
        cum.append(cum[-1] + d)

    step = total_len / n_pts
    resampled = []
    seg = 0
    for i in range(n_pts):
        target = i * step
        while seg < len(dists) - 1 and cum[seg + 1] < target:
            seg += 1
        seg_len = dists[seg]
        if seg_len < 1e-12:
            t = 0.0
        else:
            t = (target - cum[seg]) / seg_len
        p0 = loop[seg % len(loop)]
        p1 = loop[(seg + 1) % len(loop)]
        resampled.append((p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])))
    return resampled


# ---------------------------------------------------------------------------
# Gaussian smoothing + bilinear upsampling
# ---------------------------------------------------------------------------


def _loop_perimeter(loop: list[tuple[float, float]]) -> float:
    """Sum of segment lengths around a contour loop."""
    total = 0.0
    n = len(loop)
    for i in range(n):
        j = (i + 1) % n
        dx = loop[j][0] - loop[i][0]
        dy = loop[j][1] - loop[i][1]
        total += (dx * dx + dy * dy) ** 0.5
    return total


def _gaussian_blur_2d(grid: np.ndarray, sigma: float = _BLUR_SIGMA) -> np.ndarray:
    """Apply separable Gaussian blur to 2D grid (vectorized, pure numpy)."""
    size = int(4 * sigma + 0.5) * 2 + 1
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    pad = size // 2
    ny, nx = grid.shape

    # Horizontal pass: convolve each row via matrix multiply
    padded = np.pad(grid, ((0, 0), (pad, pad)), mode="edge")
    # Build column indices for sliding window: (nx, size)
    idx = np.arange(nx)[:, None] + np.arange(size)[None, :]
    temp = padded[:, idx] @ kernel  # (ny, nx)

    # Vertical pass: convolve each column via matrix multiply
    padded = np.pad(temp, ((pad, pad), (0, 0)), mode="edge")
    idx = np.arange(ny)[:, None] + np.arange(size)[None, :]
    return padded[idx, :].transpose(0, 2, 1) @ kernel  # (ny, nx)


def _upsample_2d(grid: np.ndarray, factor: int) -> np.ndarray:
    """Upsample 2D array by integer factor using bilinear interpolation."""
    ny, nx = grid.shape
    x_old = np.arange(nx)
    x_new = np.linspace(0, nx - 1, nx * factor)
    # Interpolate along columns first
    temp = np.array([np.interp(x_new, x_old, grid[i]) for i in range(ny)])
    # Then along rows
    y_old = np.arange(ny)
    y_new = np.linspace(0, ny - 1, ny * factor)
    return np.array([np.interp(y_new, y_old, temp[:, j]) for j in range(nx * factor)]).T


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
    pad = max(3, int(_BLUR_SIGMA * 4) + 1)
    r0 = max(0, int(nz_rows.min()) - pad)
    r1 = min(resolution, int(nz_rows.max()) + pad + 1)
    c0 = max(0, int(nz_cols.min()) - pad)
    c1 = min(resolution, int(nz_cols.max()) + pad + 1)
    cropped = grid_2d[r0:r1, c0:c1]

    # Blur + upsample the cropped region only
    blurred = _gaussian_blur_2d(cropped)
    if lobe.phase == "pos":
        blurred = np.maximum(blurred, 0.0)
    else:
        blurred = np.minimum(blurred, 0.0)

    upsampled = _upsample_2d(blurred, _UPSAMPLE_FACTOR)

    # Extract contours on cropped grid
    if lobe.phase == "pos":
        raw_loops = chain_segments(marching_squares(upsampled, isovalue))
    else:
        raw_loops = chain_segments(marching_squares(-upsampled, isovalue))

    # Offset contour coords back to full-grid space
    off_r = r0 * _UPSAMPLE_FACTOR
    off_c = c0 * _UPSAMPLE_FACTOR
    offset_loops = [[(r + off_r, c + off_c) for r, c in loop] for loop in raw_loops]

    loops = [_resample_loop(lp) for lp in offset_loops if _loop_perimeter(lp) >= _MIN_LOOP_PERIMETER]

    if not loops:
        return None
    cent_3d = (float(lobe_pos[:, 0].mean()), float(lobe_pos[:, 1].mean()), z_depth)
    return LobeContour2D(loops=loops, phase=lobe.phase, z_depth=z_depth, centroid_3d=cent_3d)


# ---------------------------------------------------------------------------
# Integration: build MO contours from cube data
# ---------------------------------------------------------------------------


def build_mo_contours(
    cube: CubeData,
    rot: np.ndarray | None = None,
    isovalue: float = 0.05,
    pos_color: str = "#2554A5",
    neg_color: str = "#851639",
    resolution: int | None = None,
    atom_centroid: np.ndarray | None = None,
    target_centroid: np.ndarray | None = None,
    *,
    lobes_3d: list[Lobe3D] | None = None,
    pos_flat_ang: np.ndarray | None = None,
    fixed_bounds: tuple[float, float, float, float] | None = None,
) -> MOContours:
    """Build MO contour data from a parsed cube file.

    Each 3D lobe is projected and contoured independently.
    Pre-computed *lobes_3d*, *pos_flat_ang*, and *fixed_bounds* can be
    passed to avoid recomputation across frames.
    """
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
        lobes_3d = find_3d_lobes(cube.grid_data, isovalue)

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
        )
        if lc is not None:
            lobe_contours.append(lc)

    # Sort back-to-front by z-depth
    lobe_contours.sort(key=lambda lc: lc.z_depth)

    res = base_res * _UPSAMPLE_FACTOR
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
    return MOContours(
        lobes=lobe_contours,
        resolution=res,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        pos_color=pos_color,
        neg_color=neg_color,
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
        # Within pair: higher z = front
        if lobes[pi].z_depth >= lobes[ni].z_depth:
            is_front[pi] = True
            is_front[ni] = False
        else:
            is_front[pi] = False
            is_front[ni] = True

    # Unpaired lobes: fallback to molecule z-centroid
    for i in range(n):
        if is_front[i] is None:
            is_front[i] = lobes[i].z_depth >= mol_z

    return is_front  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# MO SVG rendering
# ---------------------------------------------------------------------------


def _mo_loop_to_path_d(
    loop: list[tuple[float, float]],
    mo: MOContours,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> str | None:
    """Convert a contour loop to a smooth SVG path (Catmull-Rom to cubic Bezier)."""
    if len(loop) < 3:
        return None
    res = max(mo.resolution - 1, 1)
    # Convert grid coords to SVG pixel coords
    pts = []
    for row, col in loop:
        x_ang = mo.x_min + (col / res) * (mo.x_max - mo.x_min)
        y_ang = mo.y_min + (row / res) * (mo.y_max - mo.y_min)
        sx = canvas_w / 2 + scale * (x_ang - cx)
        sy = canvas_h / 2 - scale * (y_ang - cy)
        pts.append((sx, sy))

    n = len(pts)
    # Build Catmull-Rom cubic Bezier path (closed loop)
    parts = [f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"]
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        # Control points
        cp1x = p1[0] + (p2[0] - p0[0]) / 6
        cp1y = p1[1] + (p2[1] - p0[1]) / 6
        cp2x = p2[0] - (p3[0] - p1[0]) / 6
        cp2y = p2[1] - (p3[1] - p1[1]) / 6
        parts.append(f"C {cp1x:.1f} {cp1y:.1f} {cp2x:.1f} {cp2y:.1f} {p2[0]:.1f} {p2[1]:.1f}")
    parts.append("Z")
    return " ".join(parts)


def _mo_combined_path_d(
    loops: list[list[tuple[float, float]]],
    mo: MOContours,
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
        d = _mo_loop_to_path_d(loop, mo, scale, cx, cy, canvas_w, canvas_h)
        if d:
            parts.append(d)
    return " ".join(parts) if parts else None


def mo_gradient_defs_svg(mo: MOContours) -> list[str]:
    """Return SVG radialGradient defs for MO lobe front fills."""
    from xyzrender.types import Color

    lines: list[str] = []
    for phase, color_hex in [("pos", mo.pos_color), ("neg", mo.neg_color)]:
        c = Color.from_hex(color_hex)
        hi_f = c.lighten(0.25)
        lo_f = c.darken(0.40)
        lines.append(
            f'    <radialGradient id="mo_{phase}_front" cx=".5" cy=".5" fx=".33" fy=".33" r=".66">'
            f'<stop offset="0%" stop-color="{hi_f.hex}"/>'
            f'<stop offset="100%" stop-color="{lo_f.hex}"/>'
            f"</radialGradient>"
        )
    return lines


def mo_back_lobes_svg(
    mo: MOContours,
    mo_is_front: list[bool],
    mo_opacity: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> list[str]:
    """Return SVG lines for back MO lobes (flat faded fill, behind molecule)."""
    from xyzrender.types import Color

    lines: list[str] = []
    for idx_l, lobe in enumerate(mo.lobes):
        if mo_is_front[idx_l]:
            continue
        color_hex = mo.pos_color if lobe.phase == "pos" else mo.neg_color
        flat_color = Color.from_hex(color_hex).lighten(0.30).hex
        d_all = _mo_combined_path_d(lobe.loops, mo, scale, cx, cy, canvas_w, canvas_h)
        if d_all:
            lines.append(f'  <g opacity="{mo_opacity:.2f}">')
            lines.append(f'    <path d="{d_all}" fill="{flat_color}" fill-rule="evenodd" stroke="none"/>')
            lines.append("  </g>")
    return lines


def mo_front_lobes_svg(
    mo: MOContours,
    mo_is_front: list[bool],
    mo_opacity: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: int,
    canvas_h: int,
) -> list[str]:
    """Return SVG lines for front MO lobes (gradient fill, on top of molecule)."""
    lines: list[str] = []
    for idx_l, lobe in enumerate(mo.lobes):
        if not mo_is_front[idx_l]:
            continue
        grad_id = f"mo_{lobe.phase}_front"
        d_all = _mo_combined_path_d(lobe.loops, mo, scale, cx, cy, canvas_w, canvas_h)
        if d_all:
            lines.append(f'  <g opacity="{mo_opacity:.2f}">')
            lines.append(f'    <path d="{d_all}" fill="url(#{grad_id})" fill-rule="evenodd" stroke="none"/>')
            lines.append("  </g>")
    return lines


# ---------------------------------------------------------------------------
# Per-frame MO recomputation for gif-rot
# ---------------------------------------------------------------------------


def recompute_mo(graph: nx.Graph, config: RenderConfig, mo_data: dict) -> None:
    """Recompute MO contours for the current graph orientation.

    Caches 3D lobes, grid positions, and bounding sphere on first call;
    only the rotation changes per frame.
    """
    from xyzrender.utils import kabsch_rotation

    cube_data = mo_data["cube_data"]

    # Cache lobes and positions on first call
    if "lobes_3d" not in mo_data:
        mo_data["lobes_3d"] = find_3d_lobes(cube_data.grid_data, mo_data["isovalue"])
        mo_data["pos_flat_ang"] = compute_grid_positions(cube_data)

    orig = np.array([p for _, p in cube_data.atoms], dtype=float)
    curr = np.array([graph.nodes[i]["position"] for i in graph.nodes()], dtype=float)
    atom_centroid = orig.mean(axis=0)
    target_centroid = curr.mean(axis=0)

    # Cache bounding sphere: rotation-invariant bounds from cube corners.
    if "fixed_bounds" not in mo_data:
        corners = cube_corners_ang(cube_data)
        r_max = float(np.linalg.norm(corners - atom_centroid, axis=1).max())
        pad = r_max * 0.01 + 1e-9
        mo_data["_bounding_radius"] = r_max + pad

    r = mo_data["_bounding_radius"]
    fixed_bounds = (
        float(target_centroid[0] - r),
        float(target_centroid[0] + r),
        float(target_centroid[1] - r),
        float(target_centroid[1] + r),
    )
    mo_data["fixed_bounds"] = fixed_bounds

    rot = kabsch_rotation(orig, curr)

    config.mo_contours = build_mo_contours(
        cube_data,
        rot=rot,
        isovalue=mo_data["isovalue"],
        pos_color=mo_data["pos_color"],
        neg_color=mo_data["neg_color"],
        atom_centroid=atom_centroid,
        target_centroid=target_centroid,
        lobes_3d=mo_data["lobes_3d"],
        pos_flat_ang=mo_data["pos_flat_ang"],
        fixed_bounds=fixed_bounds,
    )
    config.mo_opacity = mo_data["mo_opacity"]
