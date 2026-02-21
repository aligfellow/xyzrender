"""GIF animation generation for xyzrender."""

from __future__ import annotations

import logging
import sys
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np

from xyzrender.renderer import render_svg

logger = logging.getLogger(__name__)


def _progress(current: int, total: int) -> None:
    """Overwrite the current line with a progress indicator."""
    if not logger.isEnabledFor(logging.INFO):
        return
    sys.stderr.write(f"\r  frame {current}/{total}")
    if current == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


ROTATION_AXES = [
    "x",
    "y",
    "z",
    "-x",
    "-y",
    "-z",
    "xy",
    "xz",
    "yz",
    "yx",
    "zx",
    "zy",
    "-xy",
    "-xz",
    "-yz",
    "-yx",
    "-zx",
    "-zy",
]


def _rotation_axis(axis_str: str) -> tuple[np.ndarray, float]:
    """Convert axis string to (unit_axis_vector, sign).

    Returns a unit vector defining the rotation axis and a sign (+1/-1)
    for direction. Uses true axis-angle rotation (Rodrigues), not Euler angles.

    Single axes: ``x`` → (1,0,0), ``y`` → (0,1,0), ``z`` → (0,0,1).
    Diagonal axes: ``xy`` → (1,1,0)/sqrt(2), a 45-degree axis between x and y.
    Reversed diagonals: ``yx`` → (1,-1,0)/sqrt(2), the other 45-degree diagonal.
    The ``-`` prefix reverses the rotation direction.
    """
    sign = -1.0 if axis_str.startswith("-") else 1.0
    ax = axis_str.lstrip("-")

    _unit = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}

    if len(ax) == 1:
        return _unit[ax], sign

    # Two-axis diagonal: order determines which diagonal
    # xy → (1,1,0), yx → (1,-1,0) — two perpendicular 45° diagonals
    _idx = {"x": 0, "y": 1, "z": 2}
    first, second = _idx[ax[0]], _idx[ax[1]]
    v = np.zeros(3)
    v[first] = 1.0
    v[second] = 1.0 if first < second else -1.0
    return v / np.linalg.norm(v), sign


def _pca_matrix(positions: np.ndarray) -> np.ndarray:
    from xyzrender.utils import pca_matrix

    return pca_matrix(positions)


def _orient_frames(frames: list[dict], vt: np.ndarray) -> list[dict]:
    """Apply a fixed PCA rotation to all frames (center each frame independently)."""
    oriented = []
    for frame in frames:
        pos = np.array(frame["positions"])
        centered = pos - pos.mean(axis=0)
        oriented.append({"symbols": frame["symbols"], "positions": (centered @ vt.T).tolist()})
    return oriented


def _orient_graph(graph, vt: np.ndarray) -> None:
    """Apply PCA rotation to graph node positions in-place."""
    n = graph.number_of_nodes()
    pos = np.array([graph.nodes[i]["position"] for i in range(n)])
    centered = pos - pos.mean(axis=0)
    rotated = centered @ vt.T
    for i in range(n):
        graph.nodes[i]["position"] = tuple(rotated[i].tolist())


if TYPE_CHECKING:
    import networkx as nx

    from xyzrender.types import RenderConfig


def render_vibration_gif(
    path: str,
    config: RenderConfig,
    output: str,
    *,
    charge: int = 0,
    multiplicity: int | None = None,
    mode: int = 0,
    ts_frame: int = 0,
    fps: int = 10,
    reference_graph: nx.Graph | None = None,
) -> None:
    """Render a TS vibrational mode as an animated GIF.

    Uses graphRC to generate the trajectory, renders each frame as SVG
    with TS bonds dashed, converts to PNG via cairosvg, and stitches into a GIF.

    If ``reference_graph`` is provided (e.g. from ``-V`` viewer rotation),
    all frames are rotated to match that orientation.
    """
    try:
        from graphrc import run_vib_analysis
    except ImportError:
        msg = "Vibration GIF requires graphrc"
        raise ImportError(msg) from None

    results = run_vib_analysis(
        input_file=path,
        mode=mode,
        ts_frame=ts_frame,
        enable_graph=True,
        charge=charge,
        multiplicity=multiplicity,
        print_output=False,
    )

    ts_graph = results["graph"]["ts_graph"]
    frames = results["trajectory"]["frames"]

    # Apply viewer rotation to all frames if a reference orientation was given
    if reference_graph is not None:
        logger.debug("Applying Kabsch rotation from viewer orientation")
        rot = _compute_rotation(ts_graph, reference_graph)
        frames = _rotate_frames(frames, rot)

    # PCA: compute once from first frame, apply consistently to all
    if config.auto_orient:
        import copy

        vt = _pca_matrix(np.array(frames[0]["positions"]))
        frames = _orient_frames(frames, vt)
        _orient_graph(ts_graph, vt)
        config = copy.copy(config)
        config.auto_orient = False

    # graphRC's frames are already a full oscillation cycle — just loop
    logger.info("Rendering vibration GIF (%d frames)", len(frames))
    pngs = _render_frames(ts_graph, frames, config)
    _stitch_gif(pngs, output, fps)
    logger.info("Wrote %s", output)


def render_vibration_rotation_gif(
    path: str,
    config: RenderConfig,
    output: str,
    *,
    charge: int = 0,
    multiplicity: int | None = None,
    mode: int = 0,
    ts_frame: int = 0,
    fps: int = 10,
    axis: str = "y",
    n_frames: int | None = None,
    reference_graph: nx.Graph | None = None,
) -> None:
    """Render a combined vibration + rotation GIF.

    Total frames = n_vib_frames * rotations, so the vibration cycles exactly
    ``rotations`` times during one full 360° spin.  If ``n_frames`` is given,
    rotations is computed as ``round(n_frames / n_vib)`` (minimum 1).
    """
    try:
        from graphrc import run_vib_analysis
    except ImportError:
        msg = "Vibration+rotation GIF requires graphrc"
        raise ImportError(msg) from None

    from xyzrender.io import apply_axis_angle_rotation

    results = run_vib_analysis(
        input_file=path,
        mode=mode,
        ts_frame=ts_frame,
        enable_graph=True,
        charge=charge,
        multiplicity=multiplicity,
        print_output=False,
    )

    ts_graph = results["graph"]["ts_graph"]
    vib_frames = results["trajectory"]["frames"]

    if reference_graph is not None:
        logger.debug("Applying Kabsch rotation from viewer orientation")
        rot = _compute_rotation(ts_graph, reference_graph)
        vib_frames = _rotate_frames(vib_frames, rot)

    # PCA: compute once from first frame, apply consistently to all
    if config.auto_orient:
        vt = _pca_matrix(np.array(vib_frames[0]["positions"]))
        vib_frames = _orient_frames(vib_frames, vt)
        _orient_graph(ts_graph, vt)

    n_vib = len(vib_frames)
    rotations = max(1, round(n_frames / n_vib)) if n_frames is not None else 3
    total = n_vib * rotations

    # Use first frame positions for bounding sphere
    pos0 = np.array(vib_frames[0]["positions"])
    rot_cfg = _rotation_config(pos0, config)

    axis_vec, axis_sign = _rotation_axis(axis)
    step = 360.0 / total
    logger.info("Rendering vibration+rotation GIF (%d vib x %d = %d frames, axis=%s)", n_vib, rotations, total, axis)
    pngs = []
    for frame_idx in range(total):
        # Set vibration positions (cycling)
        vib_pos = vib_frames[frame_idx % n_vib]["positions"]
        for i, (x, y, z) in enumerate(vib_pos):
            ts_graph.nodes[i]["position"] = (float(x), float(y), float(z))

        # Apply axis-angle rotation (clean single-axis, not Euler screw)
        apply_axis_angle_rotation(ts_graph, axis_vec, axis_sign * step * frame_idx)

        svg = render_svg(ts_graph, rot_cfg, _log=False)
        pngs.append(_svg_to_png(svg, config.canvas_size))
        _progress(frame_idx + 1, total)

    _stitch_gif(pngs, output, fps)
    logger.info("Wrote %s", output)


def render_rotation_gif(
    graph: nx.Graph,
    config: RenderConfig,
    output: str,
    *,
    n_frames: int = 60,
    fps: int = 10,
    axis: str = "y",
) -> None:
    """Render a rotation animation as a GIF.

    Rotates the molecule around the given axis over a full 360 degrees.
    Uses a fixed viewport (bounding sphere) so the molecule doesn't
    appear to zoom or shift during rotation.
    """
    from xyzrender.io import apply_axis_angle_rotation

    n = graph.number_of_nodes()

    # PCA: apply once to initial positions so GIF matches static SVG orientation
    if config.auto_orient:
        _orient_graph(graph, _pca_matrix(np.array([graph.nodes[i]["position"] for i in range(n)])))

    original_positions = [graph.nodes[i]["position"] for i in range(n)]
    rot_cfg = _rotation_config(np.array(original_positions), config)

    axis_vec, axis_sign = _rotation_axis(axis)
    step = 360.0 / n_frames
    logger.info("Rendering rotation GIF (%d frames, axis=%s)", n_frames, axis)
    pngs = []
    for frame_idx in range(n_frames):
        for i in range(n):
            graph.nodes[i]["position"] = original_positions[i]

        apply_axis_angle_rotation(graph, axis_vec, axis_sign * step * frame_idx)

        svg = render_svg(graph, rot_cfg, _log=False)
        pngs.append(_svg_to_png(svg, config.canvas_size))
        _progress(frame_idx + 1, n_frames)

    for i in range(n):
        graph.nodes[i]["position"] = original_positions[i]

    _stitch_gif(pngs, output, fps)
    logger.info("Wrote %s", output)


def render_trajectory_gif(
    frames: list[dict],
    config: RenderConfig,
    output: str,
    *,
    charge: int = 0,
    multiplicity: int | None = None,
    fps: int = 10,
    reference_graph: nx.Graph | None = None,
) -> None:
    """Render optimization/trajectory path as an animated GIF.

    Builds the molecular graph once from the last frame (optimized geometry)
    to get correct bond orders, then updates positions per frame.
    If ``reference_graph`` is provided, all frames are rotated to match.
    """
    from xyzgraph import build_graph

    # Build graph from last frame (optimized geometry → correct bond orders)
    last = frames[-1]
    last_atoms = list(zip(last["symbols"], [tuple(p) for p in last["positions"]], strict=True))
    graph = build_graph(last_atoms, charge=charge, multiplicity=multiplicity)

    # Copy TS/NCI edge attributes from reference graph
    if reference_graph is not None:
        for i, j, d in reference_graph.edges(data=True):
            if graph.has_edge(i, j):
                for attr in ("TS", "NCI", "bond_type"):
                    if attr in d:
                        graph[i][j][attr] = d[attr]

    # Apply viewer rotation if reference orientation was given
    if reference_graph is not None:
        logger.debug("Applying Kabsch rotation from viewer orientation")
        rot = _compute_rotation(graph, reference_graph)
        frames = _rotate_frames(frames, rot)

    # PCA: compute once from first frame, apply consistently to all
    if config.auto_orient:
        import copy

        vt = _pca_matrix(np.array(frames[0]["positions"]))
        frames = _orient_frames(frames, vt)
        config = copy.copy(config)
        config.auto_orient = False

    logger.info("Rendering trajectory GIF (%d frames)", len(frames))
    pngs = _render_frames(graph, frames, config)
    _stitch_gif(pngs, output, fps)
    logger.info("Wrote %s", output)


def _rotation_config(positions: np.ndarray, config: RenderConfig) -> RenderConfig:
    """Create a config with fixed viewport sized to the bounding sphere."""
    import copy

    # Conservative padding: largest VdW ~2.0 Å
    atom_pad = config.atom_scale * 0.075 * 2.0
    centroid = positions.mean(axis=0)
    max_r = np.linalg.norm(positions - centroid, axis=1).max()
    fixed_span = 2 * (max_r + atom_pad)

    rot_cfg = copy.copy(config)
    rot_cfg.fixed_span = fixed_span
    rot_cfg.fixed_center = (float(centroid[0]), float(centroid[1]))
    rot_cfg.auto_orient = False  # PCA per-frame would fight rotation
    logger.debug("Bounding sphere: r=%.2f, fixed_span=%.2f", max_r, fixed_span)
    return rot_cfg


def _compute_rotation(original_graph: nx.Graph, rotated_graph: nx.Graph) -> np.ndarray:
    """Compute 3x3 rotation matrix from original to rotated positions via SVD."""
    n = original_graph.number_of_nodes()
    orig = np.array([original_graph.nodes[i]["position"] for i in range(n)])
    rot = np.array([rotated_graph.nodes[i]["position"] for i in range(n)])

    # Center both
    orig_c = orig - orig.mean(axis=0)
    rot_c = rot - rot.mean(axis=0)

    # Kabsch: R = V @ U^T  where  H = orig^T @ rot = U S V^T
    h = orig_c.T @ rot_c
    u, _, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    sign = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])  # correct for reflection
    return vt.T @ sign @ u.T


def _rotate_frames(frames: list[dict], rot: np.ndarray) -> list[dict]:
    """Apply rotation matrix to all frame positions."""
    rotated = []
    for frame in frames:
        pos = np.array(frame["positions"])
        centroid = pos.mean(axis=0)
        pos_rotated = (rot @ (pos - centroid).T).T + centroid
        rotated.append({"symbols": frame["symbols"], "positions": pos_rotated})
    return rotated


def _render_frames(graph: nx.Graph, frames: list[dict], config: RenderConfig) -> list[bytes]:
    """Render each trajectory frame to PNG, keeping graph topology fixed."""
    pngs = []
    for idx, frame in enumerate(frames):
        positions = frame["positions"]
        for i, (x, y, z) in enumerate(positions):
            graph.nodes[i]["position"] = (float(x), float(y), float(z))
        svg = render_svg(graph, config, _log=False)
        pngs.append(_svg_to_png(svg, config.canvas_size))
        _progress(idx + 1, len(frames))
    return pngs


def _svg_to_png(svg: str, size: int) -> bytes:
    """Convert SVG string to PNG bytes."""
    try:
        import cairosvg
    except ImportError:
        msg = "GIF generation requires cairosvg"
        raise ImportError(msg) from None

    return cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)


def _stitch_gif(pngs: list[bytes], output: str, fps: int) -> None:
    """Stitch PNG frames into an animated GIF."""
    try:
        from PIL import Image
    except ImportError:
        msg = "GIF generation requires Pillow"
        raise ImportError(msg) from None

    images = []
    for png_data in pngs:
        img = Image.open(BytesIO(png_data)).convert("RGBA")
        images.append(img)

    duration = int(1000 / fps)
    logger.debug("Stitching %d frames at %d fps (%d ms/frame)", len(images), fps, duration)
    images[0].save(output, save_all=True, append_images=images[1:], duration=duration, loop=0)
