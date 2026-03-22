"""GIF animation generation for xyzrender."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np

from xyzrender.export import svg_to_png_bytes
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


def _rotation_axis(axis_str: str, lattice: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Convert axis string to (unit_axis_vector, sign).

    Returns a unit vector defining the rotation axis and a sign (+1/-1)
    for direction. Uses true axis-angle rotation (Rodrigues), not Euler angles.

    Single axes: ``x`` → (1,0,0), ``y`` → (0,1,0), ``z`` → (0,0,1).
    Diagonal axes: ``xy`` → (1,1,0)/sqrt(2), a 45-degree axis between x and y.
    Reversed diagonals: ``yx`` → (1,-1,0)/sqrt(2), the other 45-degree diagonal.
    The ``-`` prefix reverses the rotation direction.

    Crystallographic hkl: a 3-digit string such as ``111``, ``110``, ``001``
    converts Miller indices to a Cartesian axis using the lattice vectors
    (requires *lattice* to be provided).  Only single-digit Miller indices
    (0-9) per component are supported.
    """
    sign = -1.0 if axis_str.startswith("-") else 1.0
    ax = axis_str.lstrip("-")

    # Crystallographic hkl format: all-digit string of length 3
    if ax.isdigit() and len(ax) == 3:
        if lattice is None:
            msg = (
                f"Crystallographic axis {axis_str!r} requires crystal lattice data (--interface-mode or crystal input)"
            )
            raise ValueError(msg)
        h, k, l_idx = int(ax[0]), int(ax[1]), int(ax[2])
        v = h * lattice[0] + k * lattice[1] + l_idx * lattice[2]
        norm = float(np.linalg.norm(v))
        if norm < 1e-10:
            msg = f"Crystallographic axis [{ax}] has zero length (h={h}, k={k}, l={l_idx})"
            raise ValueError(msg)
        return v / norm, sign

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
    centroid = pos.mean(axis=0)
    rotated = (pos - centroid) @ vt.T
    for i, nid in enumerate(nodes):
        graph.nodes[nid]["position"] = tuple(rotated[i].tolist())
    if "lattice" in graph.graph:
        lat = np.array(graph.graph["lattice"], dtype=float)
        graph.graph["lattice"] = (vt @ lat.T).T
        # Always set up the origin (even if implicit) so the GIF matches
        # the static SVG: cell_origin = PCA(file_origin - pre_centroid).
        origin = np.array(graph.graph.get("lattice_origin", np.zeros(3)), dtype=float)
        graph.graph["lattice_origin"] = vt @ (origin - centroid)  # new centroid = 0


if TYPE_CHECKING:
    import networkx as nx
    from xyzgraph.nci import NCIAnalyzer

    from xyzrender.cube import CubeData
    from xyzrender.types import DensParams, MOParams, RenderConfig


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
    mo_params: MOParams | None = None,
    mo_cube: CubeData | None = None,
    dens_params: DensParams | None = None,
    dens_cube: CubeData | None = None,
) -> None:
    """Render a rotation animation as a GIF.

    Rotates the molecule around the given axis over a full 360 degrees.
    Uses a fixed viewport (bounding sphere) so the molecule doesn't
    appear to zoom or shift during rotation.

    If *mo_params* and *mo_cube* are provided, MO contours are recomputed
    for each frame to match the rotation.
    If *dens_params* and *dens_cube* are provided, density contours are
    recomputed for each frame to match the rotation.
    """
    nodes = list(graph.nodes())

    # PCA: apply once to initial positions so GIF matches static SVG orientation
    if config.auto_orient:
        _pca_pos = np.array([graph.nodes[n]["position"] for n in nodes])
        _pca_centroid = _pca_pos.mean(axis=0)
        _pca_vt = pca_matrix(_pca_pos)
        _orient_graph(graph, _pca_vt)
        if config.vectors:
            import copy as _cp

            config = _cp.copy(config)
            _o = np.array([va.origin for va in config.vectors])
            _d = np.array([va.vector for va in config.vectors])
            new_vecs = []
            for va, o, d in zip(
                config.vectors,
                (_pca_vt @ (_o - _pca_centroid).T).T,
                (_pca_vt @ _d.T).T,
                strict=True,
            ):
                nv = _cp.copy(va)
                nv.origin = o
                nv.vector = d
                new_vecs.append(nv)
            config.vectors = new_vecs
        # MO surfaces use a -30° x-axis tilt (same as resolve_orientation tilt_degrees=-30.0)
        # so the GIF starting frame matches the static render orientation.
        if mo_params is not None:
            theta = np.radians(-30.0)
            c, s = np.cos(theta), np.sin(theta)
            rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
            _pos = np.array([graph.nodes[n]["position"] for n in nodes])
            _tilted = _pos @ rx.T  # centroid ≈ 0 after _orient_graph, so no translation needed
            for i, nid in enumerate(nodes):
                graph.nodes[nid]["position"] = tuple(_tilted[i].tolist())

    # Set up the cell origin (0,0,0) so it rotates it around
    # the molecular centre of mass, not around the origin.
    # _orient_graph already does this for the PCA path; this covers --no-orient.
    if "lattice" in graph.graph and "lattice_origin" not in graph.graph:
        graph.graph["lattice_origin"] = np.zeros(3)

    original_pos_array = np.array([graph.nodes[n]["position"] for n in nodes], dtype=float)
    original_lattice = np.array(graph.graph["lattice"], dtype=float) if "lattice" in graph.graph else None
    original_lattice_origin = (
        np.array(graph.graph["lattice_origin"], dtype=float) if "lattice_origin" in graph.graph else None
    )

    # Pass lattice to _rotation_axis for crystallographic hkl axis strings
    _lattice = config.cell_data.lattice if config.cell_data is not None else None
    axis_vec, axis_sign = _rotation_axis(axis, lattice=_lattice)
    frame = {"positions": original_pos_array.tolist(), "symbols": [graph.nodes[n]["symbol"] for n in nodes]}
    rot_cfg = _fixed_viewport([frame], config, rotation_axis=axis_vec)

    # Save original crystal state so we can recompute the rotated box each frame.
    _orig_lattice: np.ndarray | None = None
    _orig_cell_origin: np.ndarray | None = None
    _atom_centroid: np.ndarray | None = None
    if rot_cfg.cell_data is not None:
        _orig_lattice = rot_cfg.cell_data.lattice.copy()
        _orig_cell_origin = rot_cfg.cell_data.cell_origin.copy()
        _atom_centroid = original_pos_array.mean(axis=0)

    # Expand viewport if density surface extends beyond atoms
    if dens_params is not None and dens_cube is not None:
        from xyzrender.contours import compute_grid_positions

        pos_flat = compute_grid_positions(dens_cube)
        mask = dens_cube.grid_data >= dens_params.isovalue
        flat_indices = np.flatnonzero(mask)
        dens_pos = pos_flat[flat_indices]
        orig_atoms = np.array([p for _, p in dens_cube.atoms], dtype=float)
        atom_cent = orig_atoms.mean(axis=0)
        dens_r = float(np.linalg.norm(dens_pos - atom_cent, axis=1).max())
        needed = dens_r * 1.05  # 5% padding for blur/smoothing
        assert rot_cfg.fixed_span is not None  # set by _fixed_viewport
        current_half = rot_cfg.fixed_span / 2
        if needed > current_half:
            rot_cfg.fixed_span = 2 * needed
            logger.debug("Expanded GIF viewport for density: r=%.2f -> span=%.2f", dens_r, rot_cfg.fixed_span)

    step = 360.0 / n_frames
    logger.info("Rendering rotation GIF (%d frames, axis=%s)", n_frames, axis)
    # Save pre-rotation vector data so each frame applies a fresh rotation (no drift).
    _gif_vec_origins = np.array([va.origin for va in rot_cfg.vectors]) if rot_cfg.vectors else np.full((0, 3), np.nan)
    _gif_vec_dirs = np.array([va.vector for va in rot_cfg.vectors]) if rot_cfg.vectors else np.full((0, 3), np.nan)
    _gif_vec_centroid = original_pos_array.mean(axis=0) if rot_cfg.vectors else np.full(3, np.nan)

    # Pre-populate surface caches with cube-invariant data (lobes, grid positions,
    # bounding radius) so workers only do the per-frame Kabsch rotation.
    import copy as _copy

    _mo_cache: dict = {}
    _dens_cache: dict = {}
    if mo_params is not None and mo_cube is not None:
        from xyzrender.mo import recompute_mo

        recompute_mo(graph, _copy.copy(rot_cfg), mo_params, mo_cube, rot_cfg.surface_opacity, _mo_cache)
    if dens_params is not None and dens_cube is not None:
        from xyzrender.dens import recompute_dens

        recompute_dens(graph, _copy.copy(rot_cfg), dens_params, dens_cube, rot_cfg.surface_opacity, _dens_cache)

    ctx = RotationFrameContext(
        axis_vec=axis_vec,
        axis_sign=axis_sign,
        step=step,
        rot_cfg=rot_cfg,
        vec_origins=_gif_vec_origins,
        vec_dirs=_gif_vec_dirs,
        vec_centroid=_gif_vec_centroid,
        orig_lattice=_orig_lattice,
        orig_cell_origin=_orig_cell_origin,
        atom_centroid=_atom_centroid,
        mo_params=mo_params,
        mo_cube=mo_cube,
        dens_params=dens_params,
        dens_cube=dens_cube,
        mo_cache=_mo_cache,
        dens_cache=_dens_cache,
    )
    worker = partial(
        _render_rot_frame,
        graph=graph,
        config=config,
        nodes=nodes,
        original_pos_array=original_pos_array,
        original_lattice=original_lattice,
        original_lattice_origin=original_lattice_origin,
        ctx=ctx,
    )
    pngs = _parallel_render(worker, range(n_frames), n_frames)

    for i, n in enumerate(nodes):
        graph.nodes[n]["position"] = tuple(original_pos_array[i].tolist())

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


def render_diffuse_gif(
    graph: "nx.Graph",
    config: "RenderConfig",
    output: str,
    *,
    n_frames: int = 60,
    noise: float = 0.3,
    bonds: str = "fade",
    reverse: bool = True,
    fps: int = 10,
    rotation_axis: str | None = None,
    rotation_degrees: float = 360.0,
    anchor: set[int] | None = None,
) -> None:
    """Render a diffuse/assembly GIF.

    *bonds* controls bond visibility: ``"fade"`` (default) fades
    stretched bonds, ``"show"`` keeps all bonds, ``"hide"`` removes them.

    *rotation_degrees* controls how far the molecule rotates during the
    animation (default 180°). Only used when *rotation_axis* is set.
    """
    import copy

    if config.auto_orient:
        vt = pca_matrix(np.array([graph.nodes[n]["position"] for n in graph.nodes()]))
        _orient_graph(graph, vt)
        config = copy.copy(config)
        config.auto_orient = False

    from xyzrender.diffuse import diffuse_frames

    frames = diffuse_frames(
        graph,
        n_frames=n_frames,
        noise=noise,
        bonds=bonds,
        reverse=reverse,
        anchor=anchor,
    )

    axis_vec = None
    axis_sign = 1.0
    if rotation_axis:
        axis_vec, axis_sign = _rotation_axis(rotation_axis)

    config = _fixed_viewport(frames, config, rotation_axis=axis_vec)

    logger.info("Rendering diffuse GIF (%d frames%s)", len(frames), f", axis={rotation_axis}" if rotation_axis else "")
    pngs = _render_frames(
        graph, frames, config, rotation_axis=axis_vec, rotation_sign=axis_sign, rotation_degrees=rotation_degrees
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
    # Deep-copy cell_data so GIF frame updates to lattice/cell_origin don't
    # mutate the caller's config object.
    if cfg.cell_data is not None:
        cfg.cell_data = copy.deepcopy(cfg.cell_data)
    logger.debug("Fixed viewport: span=%.2f, center=(%.2f, %.2f)", fixed_span, *center_xy)
    return cfg


def _axis_angle_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation matrix for *axis* (unit vector) and *angle_deg* (degrees)."""
    theta = np.radians(angle_deg)
    k = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return c * np.eye(3) + s * kx + (1 - c) * np.outer(k, k)


def _rotate_vectors_in_cfg(
    cfg: "RenderConfig",
    rot: np.ndarray,
    centroid: np.ndarray,
    src_origins: np.ndarray,
    src_dirs: np.ndarray,
) -> "RenderConfig":
    """Return a shallow copy of *cfg* with vector arrows rotated by *rot* around *centroid*.

    Uses *src_origins* / *src_dirs* as the reference un-rotated coordinates so
    the function can be called from a loop without accumulating floating-point
    error across frames.
    """
    import copy

    new_cfg = copy.copy(cfg)
    new_origins = centroid + (rot @ (src_origins - centroid).T).T
    new_dirs = (rot @ src_dirs.T).T
    new_vecs = []
    for va, o, d in zip(cfg.vectors, new_origins, new_dirs, strict=True):
        nv = copy.copy(va)
        nv.origin = o
        nv.vector = d
        new_vecs.append(nv)
    new_cfg.vectors = new_vecs
    return new_cfg


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


@dataclass
class RotationFrameContext:
    """Bundled parameters for rotation GIF frame rendering."""

    axis_vec: np.ndarray
    axis_sign: float
    step: float
    rot_cfg: "RenderConfig"
    vec_origins: np.ndarray
    vec_dirs: np.ndarray
    vec_centroid: np.ndarray
    orig_lattice: np.ndarray | None
    orig_cell_origin: np.ndarray | None
    atom_centroid: np.ndarray | None
    mo_params: "MOParams | None"
    mo_cube: "CubeData | None"
    dens_params: "DensParams | None"
    dens_cube: "CubeData | None"
    mo_cache: dict
    dens_cache: dict


def _render_rot_frame(
    frame_idx: int,
    graph: "nx.Graph",
    config: "RenderConfig",
    nodes: list,
    original_pos_array: np.ndarray,
    original_lattice: np.ndarray | None,
    original_lattice_origin: np.ndarray | None,
    ctx: RotationFrameContext,
) -> tuple[int, bytes]:
    """Worker: render one rotation GIF frame to PNG. Surfaces recomputed via Kabsch per frame."""
    import copy

    if original_lattice is not None:
        graph.graph["lattice"] = original_lattice.copy()
    if original_lattice_origin is not None:
        graph.graph["lattice_origin"] = original_lattice_origin.copy()

    total_angle = ctx.axis_sign * ctx.step * frame_idx
    rot_mat = _axis_angle_matrix(ctx.axis_vec, total_angle)

    centroid = original_pos_array.mean(axis=0)
    rotated = (rot_mat @ (original_pos_array - centroid).T).T + centroid
    for i, n in enumerate(nodes):
        graph.nodes[n]["position"] = tuple(rotated[i].tolist())

    frame_cfg = copy.copy(ctx.rot_cfg)

    if ctx.rot_cfg.vectors:
        frame_cfg = _rotate_vectors_in_cfg(ctx.rot_cfg, rot_mat, ctx.vec_centroid, ctx.vec_origins, ctx.vec_dirs)

    if ctx.rot_cfg.cell_data is not None and ctx.orig_lattice is not None:
        assert ctx.orig_cell_origin is not None
        assert ctx.atom_centroid is not None
        frame_cfg.cell_data = copy.deepcopy(ctx.rot_cfg.cell_data)
        frame_cfg.cell_data.lattice = ctx.orig_lattice @ rot_mat.T
        frame_cfg.cell_data.cell_origin = rot_mat @ (ctx.orig_cell_origin - ctx.atom_centroid) + ctx.atom_centroid

    if ctx.mo_params is not None and ctx.mo_cube is not None:
        from xyzrender.mo import recompute_mo

        recompute_mo(graph, frame_cfg, ctx.mo_params, ctx.mo_cube, frame_cfg.surface_opacity, ctx.mo_cache)

    if ctx.dens_params is not None and ctx.dens_cube is not None:
        from xyzrender.dens import recompute_dens

        recompute_dens(graph, frame_cfg, ctx.dens_params, ctx.dens_cube, frame_cfg.surface_opacity, ctx.dens_cache)

    svg = render_svg(graph, frame_cfg, _log=False, _unique_ids=False)
    return frame_idx, svg_to_png_bytes(svg, size=config.canvas_size)


def _parallel_render(worker, items, total: int) -> list[bytes]:
    """Run *worker* over *items* in parallel; return ordered PNG list.

    Workers must return ``(idx, png_bytes)`` so results can be placed in order.
    Callers are responsible for including the ordering index in each item.
    """
    import multiprocessing

    pngs = [b""] * total
    pool = multiprocessing.Pool()
    try:
        for i, (idx, png_data) in enumerate(pool.imap_unordered(worker, items)):
            pngs[idx] = png_data
            _progress(i + 1, total)
    finally:
        pool.close()
        pool.join()
    return pngs


def _render_traj_frame(
    idx_frame: tuple[int, dict],
    graph: "nx.Graph",
    config: "RenderConfig",
    nci_analyzer: "NCIAnalyzer | None",
    fixed_ncis: list | None,
    rotation_axis: np.ndarray | None,
    rotation_sign: float,
    step: float,
    rf_vec_origins: np.ndarray,
    rf_vec_dirs: np.ndarray,
) -> tuple[int, bytes]:
    """Worker: render one trajectory/vibration frame to PNG."""
    if nci_analyzer is not None or fixed_ncis is not None:
        from xyzgraph.nci import build_nci_graph
    if rotation_axis is not None:
        from xyzrender.utils import apply_axis_angle_rotation

    idx, frame = idx_frame
    positions = frame["positions"]
    for i, (x, y, z) in enumerate(positions):
        graph.nodes[i]["position"] = (float(x), float(y), float(z))

    # Apply per-frame bond opacities (from diffuse GIF: fades stretched bonds)
    bond_opacities = frame.get("bond_opacities")
    if bond_opacities:
        for (i, j), op in bond_opacities.items():
            if graph.has_edge(i, j):
                graph[i][j]["diffuse_opacity"] = op

    if nci_analyzer is not None:
        ncis = nci_analyzer.detect(np.array(positions))
        render_graph = build_nci_graph(graph, ncis)
    elif fixed_ncis is not None:
        render_graph = build_nci_graph(graph, fixed_ncis)
    else:
        render_graph = graph

    frame_config = config
    if rotation_axis is not None:
        _rg_nodes = list(render_graph.nodes())
        _rg_centroid = np.mean([render_graph.nodes[n]["position"] for n in _rg_nodes], axis=0)
        rot_mat = _axis_angle_matrix(rotation_axis, rotation_sign * step * idx)
        apply_axis_angle_rotation(render_graph, rotation_axis, rotation_sign * step * idx)
        if config.vectors:
            frame_config = _rotate_vectors_in_cfg(config, rot_mat, _rg_centroid, rf_vec_origins, rf_vec_dirs)

    svg = render_svg(render_graph, frame_config, _log=False)
    return idx, svg_to_png_bytes(svg, size=config.canvas_size)


def _render_frames(
    graph: nx.Graph,
    frames: list[dict],
    config: RenderConfig,
    *,
    nci_analyzer: NCIAnalyzer | None = None,
    fixed_ncis: list | None = None,
    rotation_axis: np.ndarray | None = None,
    rotation_sign: float = 1.0,
    rotation_degrees: float = 360.0,
) -> list[bytes]:
    """Render each trajectory frame to PNG, keeping graph topology fixed.

    If *nci_analyzer* is provided, NCI interactions are re-detected per
    frame and the graph is decorated with the frame-specific NCI edges.
    If *fixed_ncis* is provided, the same NCI set is applied to every frame
    (centroids recomputed from current atom positions each frame).
    If *rotation_axis* is provided, each frame is incrementally rotated
    around that axis over *rotation_degrees* (default 360°).
    """
    total = len(frames)
    step = rotation_degrees / total if rotation_axis is not None else 0
    _rf_vec_origins = (
        np.array([va.origin for va in config.vectors])
        if (config.vectors and rotation_axis is not None)
        else np.full((0, 3), np.nan)
    )
    _rf_vec_dirs = (
        np.array([va.vector for va in config.vectors])
        if (config.vectors and rotation_axis is not None)
        else np.full((0, 3), np.nan)
    )
    worker = partial(
        _render_traj_frame,
        graph=graph,
        config=config,
        nci_analyzer=nci_analyzer,
        fixed_ncis=fixed_ncis,
        rotation_axis=rotation_axis,
        rotation_sign=rotation_sign,
        step=step,
        rf_vec_origins=_rf_vec_origins,
        rf_vec_dirs=_rf_vec_dirs,
    )
    return _parallel_render(worker, enumerate(frames), total)


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
    images[0].save(output, save_all=True, append_images=images[1:], duration=duration, loop=0, disposal=2)
