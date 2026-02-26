"""GIF animation generation for xyzrender."""

from __future__ import annotations

import logging
import sys
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np

from xyzrender.renderer import render_svg
from xyzrender.utils import kabsch_rotation, pca_matrix

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
    nodes = list(graph.nodes())
    pos = np.array([graph.nodes[n]["position"] for n in nodes])
    centered = pos - pos.mean(axis=0)
    rotated = centered @ vt.T
    for i, nid in enumerate(nodes):
        graph.nodes[nid]["position"] = tuple(rotated[i].tolist())


if TYPE_CHECKING:
    import networkx as nx
    from xyzgraph.nci import NCIAnalyzer

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
    detect_nci: bool = False,
) -> None:
    """Render a TS vibrational mode as an animated GIF.

    Uses graphRC to generate the trajectory, renders each frame as SVG
    with TS bonds dashed, converts to PNG via cairosvg, and stitches into a GIF.

    If ``reference_graph`` is provided (e.g. from ``-V`` viewer rotation),
    all frames are rotated to match that orientation.
    If ``detect_nci`` is True, NCI interactions are detected once on the
    TS geometry and applied to every frame (centroids recomputed per frame).
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

    # NCI: detect once on TS geometry, apply to all frames
    fixed_ncis = None
    if detect_nci:
        from xyzgraph import detect_ncis

        detect_ncis(ts_graph)
        fixed_ncis = ts_graph.graph.get("ncis", [])

    # Apply viewer rotation to all frames if a reference orientation was given
    if reference_graph is not None:
        logger.debug("Applying Kabsch rotation from viewer orientation")
        rot = _compute_rotation(ts_graph, reference_graph)
        frames = _rotate_frames(frames, rot)

    # PCA: compute once from first frame, apply consistently to all
    if config.auto_orient:
        import copy

        vt = pca_matrix(np.array(frames[0]["positions"]))
        frames = _orient_frames(frames, vt)
        _orient_graph(ts_graph, vt)
        config = copy.copy(config)
        config.auto_orient = False

    # Fixed viewport across all frames so every PNG has identical dimensions
    config = _fixed_viewport(frames, config)

    # graphRC's frames are already a full oscillation cycle — just loop
    logger.info("Rendering vibration GIF (%d frames)", len(frames))
    pngs = _render_frames(ts_graph, frames, config, fixed_ncis=fixed_ncis)
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
    detect_nci: bool = False,
) -> None:
    """Render a combined vibration + rotation GIF.

    Total frames = n_vib_frames * rotations, so the vibration cycles exactly
    ``rotations`` times during one full 360° spin.  If ``n_frames`` is given,
    rotations is computed as ``round(n_frames / n_vib)`` (minimum 1).
    If ``detect_nci`` is True, NCI interactions are detected once on the
    TS geometry and applied to every frame.
    """
    try:
        from graphrc import run_vib_analysis
    except ImportError:
        msg = "Vibration+rotation GIF requires graphrc"
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
    vib_frames = results["trajectory"]["frames"]

    # NCI: detect once on TS geometry, apply to all frames
    fixed_ncis = None
    if detect_nci:
        from xyzgraph import detect_ncis

        detect_ncis(ts_graph)
        fixed_ncis = ts_graph.graph.get("ncis", [])

    if reference_graph is not None:
        logger.debug("Applying Kabsch rotation from viewer orientation")
        rot = _compute_rotation(ts_graph, reference_graph)
        vib_frames = _rotate_frames(vib_frames, rot)

    # PCA: compute once from first frame, apply consistently to all
    if config.auto_orient:
        vt = pca_matrix(np.array(vib_frames[0]["positions"]))
        vib_frames = _orient_frames(vib_frames, vt)
        _orient_graph(ts_graph, vt)

    n_vib = len(vib_frames)
    rotations = max(1, round(n_frames / n_vib)) if n_frames is not None else 3
    total = n_vib * rotations
    if n_frames is not None and total != n_frames:
        logger.info("Rounded --rot-frames %d -> %d (%d vib x %d rotations)", n_frames, total, n_vib, rotations)

    # Cycle vibration frames for the full rotation
    all_frames = [vib_frames[i % n_vib] for i in range(total)]

    axis_vec, axis_sign = _rotation_axis(axis)

    # Fixed viewport across all frames so every PNG has identical dimensions
    rot_cfg = _fixed_viewport(vib_frames, config, rotation_axis=axis_vec)
    logger.info(
        "Rendering vibration+rotation GIF (%d vib x %d rot = %d frames, axis=%s)", n_vib, rotations, total, axis
    )
    pngs = _render_frames(
        ts_graph, all_frames, rot_cfg, fixed_ncis=fixed_ncis, rotation_axis=axis_vec, rotation_sign=axis_sign
    )
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
    mo_data: dict | None = None,
    dens_data: dict | None = None,
) -> None:
    """Render a rotation animation as a GIF.

    Rotates the molecule around the given axis over a full 360 degrees.
    Uses a fixed viewport (bounding sphere) so the molecule doesn't
    appear to zoom or shift during rotation.

    If *mo_data* is provided (dict with cube_data, isovalue, colors, opacity),
    MO contours are recomputed for each frame to match the rotation.
    If *dens_data* is provided (dict with cube_data, isovalue, color, opacity),
    density contours are recomputed for each frame to match the rotation.
    """
    from xyzrender.io import apply_axis_angle_rotation

    nodes = list(graph.nodes())

    # PCA: apply once to initial positions so GIF matches static SVG orientation
    if config.auto_orient:
        _orient_graph(graph, pca_matrix(np.array([graph.nodes[n]["position"] for n in nodes])))

    original_positions = {n: graph.nodes[n]["position"] for n in nodes}

    axis_vec, axis_sign = _rotation_axis(axis)
    frame = {"positions": list(original_positions.values()), "symbols": [graph.nodes[n]["symbol"] for n in nodes]}
    rot_cfg = _fixed_viewport([frame], config, rotation_axis=axis_vec)

    # Expand viewport if density surface extends beyond atoms
    if dens_data is not None:
        from xyzrender.mo import compute_grid_positions

        cube = dens_data["cube_data"]
        pos_flat = compute_grid_positions(cube)
        mask = cube.grid_data >= dens_data["isovalue"]
        flat_indices = np.flatnonzero(mask)
        # Pre-populate cache for recompute_dens
        dens_data["pos_flat_ang"] = pos_flat
        dens_data["flat_indices"] = flat_indices

        dens_pos = pos_flat[flat_indices]
        orig_atoms = np.array([p for _, p in cube.atoms], dtype=float)
        atom_cent = orig_atoms.mean(axis=0)
        # Pre-populate cache for recompute_dens
        dens_data["_orig_atoms"] = orig_atoms
        dens_data["_atom_centroid"] = atom_cent

        dens_r = float(np.linalg.norm(dens_pos - atom_cent, axis=1).max())
        needed = dens_r * 1.05  # 5% padding for blur/smoothing
        assert rot_cfg.fixed_span is not None  # set by _fixed_viewport
        current_half = rot_cfg.fixed_span / 2
        if needed > current_half:
            rot_cfg.fixed_span = 2 * needed
            logger.debug("Expanded GIF viewport for density: r=%.2f -> span=%.2f", dens_r, rot_cfg.fixed_span)

    axis_vec, axis_sign = _rotation_axis(axis)
    step = 360.0 / n_frames
    logger.info("Rendering rotation GIF (%d frames, axis=%s)", n_frames, axis)
    pngs = []
    for frame_idx in range(n_frames):
        for n in nodes:
            graph.nodes[n]["position"] = original_positions[n]

        apply_axis_angle_rotation(graph, axis_vec, axis_sign * step * frame_idx)

        if mo_data is not None:
            from xyzrender.mo import recompute_mo

            recompute_mo(graph, rot_cfg, mo_data)

        if dens_data is not None:
            from xyzrender.dens import recompute_dens

            recompute_dens(graph, rot_cfg, dens_data)

        svg = render_svg(graph, rot_cfg, _log=False)
        pngs.append(_svg_to_png(svg, config.canvas_size))
        _progress(frame_idx + 1, n_frames)

    for n in nodes:
        graph.nodes[n]["position"] = original_positions[n]

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
    detect_nci: bool = False,
    axis: str | None = None,
    kekule: bool = False,
) -> None:
    """Render optimization/trajectory path as an animated GIF.

    Builds the molecular graph once from the last frame (optimized geometry)
    to get correct bond orders, then updates positions per frame.
    If ``reference_graph`` is provided, all frames are rotated to match.
    If ``detect_nci`` is True, NCI interactions are re-detected per frame
    using xyzgraph's NCIAnalyzer (topology built once, geometry per frame).
    If ``axis`` is provided, the molecule rotates 360° around that axis
    over the course of the trajectory.
    """
    from xyzgraph import build_graph

    # Build graph from last frame (optimized geometry → correct bond orders)
    last = frames[-1]
    last_atoms = list(zip(last["symbols"], [tuple(p) for p in last["positions"]], strict=True))
    graph = build_graph(last_atoms, charge=charge, multiplicity=multiplicity, kekule=kekule)

    # Copy TS/NCI edge attributes from reference graph
    if reference_graph is not None:
        for i, j, d in reference_graph.edges(data=True):
            if graph.has_edge(i, j):
                for attr in ("TS", "NCI", "bond_type"):
                    if attr in d:
                        graph[i][j][attr] = d[attr]

    # Build NCI analyzer once from topology, detect per frame later
    nci_analyzer = None
    if detect_nci:
        from xyzgraph.nci import NCIAnalyzer

        nci_analyzer = NCIAnalyzer(graph)

    # Apply viewer rotation if reference orientation was given
    if reference_graph is not None:
        logger.debug("Applying Kabsch rotation from viewer orientation")
        rot = _compute_rotation(graph, reference_graph)
        frames = _rotate_frames(frames, rot)

    # PCA: compute once from first frame, apply consistently to all
    if config.auto_orient:
        import copy

        vt = pca_matrix(np.array(frames[0]["positions"]))
        frames = _orient_frames(frames, vt)
        config = copy.copy(config)
        config.auto_orient = False

    axis_vec = None
    axis_sign = 1.0
    if axis:
        axis_vec, axis_sign = _rotation_axis(axis)

    # Fixed viewport across all frames so every PNG has identical dimensions
    config = _fixed_viewport(frames, config, rotation_axis=axis_vec)

    logger.info("Rendering trajectory GIF (%d frames%s)", len(frames), f", axis={axis}" if axis else "")
    pngs = _render_frames(
        graph, frames, config, nci_analyzer=nci_analyzer, rotation_axis=axis_vec, rotation_sign=axis_sign
    )
    _stitch_gif(pngs, output, fps)
    logger.info("Wrote %s", output)


def _fixed_viewport(frames: list[dict], config: RenderConfig, rotation_axis: np.ndarray | None = None) -> RenderConfig:
    """Create a config with fixed viewport so every frame has identical canvas size.

    When *rotation_axis* is given, computes the exact projection envelope for
    rotation around that axis: each atom's max screen-X extent is its distance
    from the axis, and screen-Y extent is its component along the axis.
    Otherwise uses the tighter 2D (XY) bounding box.

    """
    import copy

    from xyzgraph import DATA

    max_vdw = max(DATA.vdw.get(s, 1.5) for s in set(frames[0]["symbols"]))
    if config.vdw_indices is not None:
        atom_pad = max_vdw * config.vdw_scale
    else:
        atom_pad = max_vdw * config.atom_scale * 0.075

    all_pos = np.vstack([np.array(f["positions"]) for f in frames])

    if rotation_axis is not None:
        centroid = all_pos.mean(axis=0)
        rel = all_pos - centroid
        # Component along the rotation axis (stays fixed as screen-Y)
        along = rel @ rotation_axis
        # Distance from the rotation axis (sweeps into screen-X)
        perp = np.linalg.norm(rel - np.outer(along, rotation_axis), axis=1)
        half_x = perp.max() + atom_pad
        half_y = max(along.max() - along.min(), 0) / 2 + atom_pad
        fixed_span = 2 * max(half_x, half_y)
        center_xy = (float(centroid[0]), float(centroid[1]))
    else:
        xy = all_pos[:, :2]
        lo = xy.min(axis=0) - atom_pad
        hi = xy.max(axis=0) + atom_pad
        center_xy = (float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2))
        fixed_span = float(max((hi - lo).max(), 1e-6))

    cfg = copy.copy(config)
    cfg.fixed_span = fixed_span
    cfg.fixed_center = center_xy
    cfg.auto_orient = False
    logger.debug("Fixed viewport: span=%.2f, center=(%.2f, %.2f)", fixed_span, *center_xy)
    return cfg


def _compute_rotation(original_graph: nx.Graph, rotated_graph: nx.Graph) -> np.ndarray:
    """Compute 3x3 rotation matrix from original to rotated graph positions."""
    n = original_graph.number_of_nodes()
    orig = np.array([original_graph.nodes[i]["position"] for i in range(n)])
    rot = np.array([rotated_graph.nodes[i]["position"] for i in range(n)])
    return kabsch_rotation(orig, rot)


def _rotate_frames(frames: list[dict], rot: np.ndarray) -> list[dict]:
    """Apply rotation matrix to all frame positions."""
    rotated = []
    for frame in frames:
        pos = np.array(frame["positions"])
        centroid = pos.mean(axis=0)
        pos_rotated = (rot @ (pos - centroid).T).T + centroid
        rotated.append({"symbols": frame["symbols"], "positions": pos_rotated})
    return rotated


def _render_frames(
    graph: nx.Graph,
    frames: list[dict],
    config: RenderConfig,
    *,
    nci_analyzer: NCIAnalyzer | None = None,
    fixed_ncis: list | None = None,
    rotation_axis: np.ndarray | None = None,
    rotation_sign: float = 1.0,
) -> list[bytes]:
    """Render each trajectory frame to PNG, keeping graph topology fixed.

    If *nci_analyzer* is provided, NCI interactions are re-detected per
    frame and the graph is decorated with the frame-specific NCI edges.
    If *fixed_ncis* is provided, the same NCI set is applied to every frame
    (centroids recomputed from current atom positions each frame).
    If *rotation_axis* is provided, each frame is incrementally rotated
    around that axis for a full 360° over all frames.
    """
    if nci_analyzer is not None or fixed_ncis is not None:
        from xyzgraph.nci import build_nci_graph
    if rotation_axis is not None:
        from xyzrender.io import apply_axis_angle_rotation

    total = len(frames)
    step = 360.0 / total if rotation_axis is not None else 0
    pngs = []
    for idx, frame in enumerate(frames):
        positions = frame["positions"]
        for i, (x, y, z) in enumerate(positions):
            graph.nodes[i]["position"] = (float(x), float(y), float(z))

        if nci_analyzer is not None:
            ncis = nci_analyzer.detect(np.array(positions))
            render_graph = build_nci_graph(graph, ncis)
        elif fixed_ncis is not None:
            render_graph = build_nci_graph(graph, fixed_ncis)
        else:
            render_graph = graph

        if rotation_axis is not None:
            apply_axis_angle_rotation(render_graph, rotation_axis, rotation_sign * step * idx)

        svg = render_svg(render_graph, config, _log=False)
        pngs.append(_svg_to_png(svg, config.canvas_size))
        _progress(idx + 1, total)
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
