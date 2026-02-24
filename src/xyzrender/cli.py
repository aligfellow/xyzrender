"""Command-line interface for xyzrender."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from xyzrender.config import build_render_config, load_config

if TYPE_CHECKING:
    from xyzrender.cube import CubeData
    from xyzrender.types import RenderConfig

from xyzrender.io import (
    detect_nci,
    load_molecule,
    load_stdin,
    load_trajectory_frames,
    load_ts_molecule,
    rotate_with_viewer,
)
from xyzrender.renderer import render_svg

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {"svg", "png", "pdf"}


def _basename(input_path: str | None, from_stdin: bool) -> str:
    """Derive output basename from input file or fallback for stdin."""
    if from_stdin or not input_path:
        return "graphic"
    return Path(input_path).stem


def _write_output(svg: str, output: str, cfg: RenderConfig, parser: argparse.ArgumentParser) -> None:
    """Write SVG to file, converting to PNG/PDF based on extension."""
    ext = output.rsplit(".", 1)[-1].lower()
    if ext == "svg":
        with open(output, "w") as f:
            f.write(svg)
    elif ext == "png":
        from xyzrender.export import svg_to_png

        svg_to_png(svg, output, size=cfg.canvas_size, dpi=cfg.dpi)
    elif ext == "pdf":
        from xyzrender.export import svg_to_pdf

        svg_to_pdf(svg, output)
    else:
        supported = ", ".join("." + e for e in sorted(_SUPPORTED_EXTENSIONS))
        parser.error(f"Unsupported output format: .{ext} (use {supported})")


def _parse_pairs(s: str) -> list[tuple[int, int]]:
    """Parse '1-6,3-4' → [(0,5), (2,3)] (1-indexed input → 0-indexed)."""
    if not s.strip():
        return []
    pairs = []
    for part in s.split(","):
        a, b = part.split("-")
        pairs.append((int(a) - 1, int(b) - 1))
    return pairs


def _parse_indices(s: str) -> list[int]:
    """Parse '1-20,25,30' → [0..19, 24, 29] (1-indexed input → 0-indexed)."""
    if not s.strip():
        return []
    indices = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            indices.extend(range(int(a) - 1, int(b)))
        else:
            indices.append(int(part) - 1)
    return indices


def main() -> None:
    """Entry point for the CLI."""
    p = argparse.ArgumentParser(
        prog="xyzrender", description="Publication-quality molecular graphics from the command line."
    )

    # --- Input / Output ---
    io_g = p.add_argument_group("input/output")
    io_g.add_argument("input", nargs="?", help="XYZ file (reads stdin if omitted)")
    io_g.add_argument("-o", "--output", help="Output file (.svg, .png, .pdf)")
    io_g.add_argument("-c", "--charge", type=int, default=0)
    io_g.add_argument("-m", "--multiplicity", type=int, default=None)
    io_g.add_argument("-d", "--debug", action="store_true", help="Debug output")

    # --- Styling ---
    style_g = p.add_argument_group("styling")
    style_g.add_argument("--config", default=None, help="Config preset or JSON path (default, flat, custom)")
    style_g.add_argument("-S", "--canvas-size", type=int, default=None)
    style_g.add_argument("-a", "--atom-scale", type=float, default=None)
    style_g.add_argument("-b", "--bond-width", type=float, default=None)
    style_g.add_argument("-s", "--atom-stroke-width", type=float, default=None)
    style_g.add_argument("--bond-color", default=None, help="Bond color (hex)")
    style_g.add_argument("-B", "--background", default=None)
    style_g.add_argument("-G", "--gradient-strength", type=float, default=None, help="Gradient contrast")
    style_g.add_argument("--grad", action=argparse.BooleanOptionalAction, default=None, help="Radial gradients")
    style_g.add_argument("-F", "--fog-strength", type=float, default=None)
    style_g.add_argument("--vdw-opacity", type=float, default=None, help="VdW sphere opacity")
    style_g.add_argument("--vdw-scale", type=float, default=None, help="VdW sphere radius scale")
    style_g.add_argument("--vdw-gradient", type=float, default=None, help="VdW sphere gradient strength")

    # --- Display ---
    disp_g = p.add_argument_group("display")
    disp_g.add_argument("--hy", nargs="*", type=int, default=None, help="Show H atoms (no args=all, or 1-indexed)")
    disp_g.add_argument("--no-hy", action="store_true", default=False, help="Hide all H atoms")
    disp_g.add_argument("--bo", action=argparse.BooleanOptionalAction, default=None, help="Bond orders")
    disp_g.add_argument(
        "-k", "--kekule", action="store_true", default=False, help="Use Kekule bond orders (no aromatic 1.5)"
    )
    disp_g.add_argument("--fog", action=argparse.BooleanOptionalAction, default=None, help="Depth fog")
    disp_g.add_argument("--vdw", nargs="?", const="", default=None, help='VdW spheres (no args=all, or "1-20,25")')
    disp_g.add_argument("--mo", action="store_true", default=False, help="Render MO lobes from .cube input")
    disp_g.add_argument("--iso", type=float, default=0.05, help="MO isosurface threshold (default: 0.05)")
    disp_g.add_argument("--mo-opacity", type=float, default=None, help="MO lobe opacity (default: 0.6)")
    disp_g.add_argument(
        "--mo-colors",
        nargs=2,
        default=None,
        metavar=("POS", "NEG"),
        help="MO lobe colors as hex (default: #2554A5 #851639)",
    )

    # --- Orientation ---
    orient_g = p.add_argument_group("orientation")
    orient_g.add_argument(
        "--orient", action=argparse.BooleanOptionalAction, default=None, help="Auto-orientation (default: on)"
    )
    orient_g.add_argument("-I", "--interactive", action="store_true", help="Open in v viewer for interactive rotation")

    # --- TS / NCI ---
    ts_g = p.add_argument_group("transition state / NCI")
    ts_g.add_argument("--ts", action="store_true", dest="ts_detect", help="Auto-detect TS bonds via graphRC")
    ts_g.add_argument("--ts-frame", type=int, default=0, help="TS reference frame for graphRC (0-indexed)")
    ts_g.add_argument("--ts-bond", default="", help='Manual TS bond pair(s), 1-indexed: "1-6,3-4"')
    ts_g.add_argument("--nci", action="store_true", help="Auto-detect NCI interactions via xyzgraph")
    ts_g.add_argument("--nci-bond", default="", help='Manual NCI bond pair(s), 1-indexed: "1-5,2-8"')

    # --- GIF animation ---
    gif_g = p.add_argument_group("GIF animation")
    gif_g.add_argument("--gif-ts", action="store_true", help="TS vibration GIF (via graphRC)")
    gif_g.add_argument("--gif-trj", action="store_true", help="Trajectory/optimization GIF (multi-frame input)")
    gif_g.add_argument(
        "--gif-rot",
        nargs="?",
        const="y",
        default=None,
        help="Rotation GIF (default axis: y). Combinable with --gif-ts.",
    )
    gif_g.add_argument("-go", "--gif-output", default=None, help="GIF output path")
    gif_g.add_argument("--gif-fps", type=int, default=10, help="GIF frames per second (default: 10)")
    gif_g.add_argument("--rot-frames", type=int, default=120, help="Rotation frames (default: 120)")

    args = p.parse_args()

    from xyzrender import configure_logging

    configure_logging(verbose=True, debug=args.debug)

    # Build config: preset/JSON base + CLI overrides
    config_data = load_config(args.config or "default")

    cli_overrides: dict = {}
    for attr, key in [
        ("canvas_size", "canvas_size"),
        ("atom_scale", "atom_scale"),
        ("bond_width", "bond_width"),
        ("atom_stroke_width", "atom_stroke_width"),
        ("bond_color", "bond_color"),
        ("gradient_strength", "gradient_strength"),
        ("fog_strength", "fog_strength"),
        ("background", "background"),
        ("vdw_opacity", "vdw_opacity"),
        ("vdw_scale", "vdw_scale"),
    ]:
        val = getattr(args, attr)
        if val is not None:
            cli_overrides[key] = val
    if args.vdw_gradient is not None:
        cli_overrides["vdw_gradient_strength"] = args.vdw_gradient
    if args.grad is not None:
        cli_overrides["gradient"] = args.grad
    if args.fog is not None:
        cli_overrides["fog"] = args.fog
    if args.bo is not None:
        cli_overrides["bond_orders"] = args.bo

    cfg = build_render_config(config_data, cli_overrides)

    # Per-molecule settings (always from CLI)
    if args.no_hy:
        cfg.hide_h = True  # --no-hy: hide all H
    elif args.hy is None:
        cfg.hide_h = True  # default: hide C-H
    elif len(args.hy) == 0:
        cfg.hide_h = False  # --hy with no args: show all
    else:
        cfg.hide_h = True  # --hy 1 3 5: show specific only
        cfg.show_h_indices = [i - 1 for i in args.hy]
    cfg.ts_bonds = _parse_pairs(args.ts_bond)
    cfg.nci_bonds = _parse_pairs(args.nci_bond)
    cfg.vdw_indices = (
        _parse_indices(args.vdw) if args.vdw is not None and args.vdw != "" else ([] if args.vdw == "" else None)
    )
    # Auto-orient: on by default, off for interactive/stdin
    from_stdin = not args.input and not sys.stdin.isatty()
    if args.orient is not None:
        cfg.auto_orient = args.orient
    elif args.interactive or from_stdin:
        cfg.auto_orient = False
    else:
        cfg.auto_orient = True

    # Output path defaults and validation
    base = _basename(args.input, from_stdin)
    if not args.output:
        args.output = f"{base}.svg"

    static_ext = args.output.rsplit(".", 1)[-1].lower()
    if static_ext not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join("." + e for e in sorted(_SUPPORTED_EXTENSIONS))
        p.error(f"Unsupported static output format: .{static_ext} (use {supported})")

    wants_gif = args.gif_ts or args.gif_rot or args.gif_trj
    if args.rot_frames != 120 and not args.gif_rot:
        logger.warning("--rot-frames ignored without --gif-rot")
    if args.gif_ts and args.gif_trj:
        p.error(
            "--gif-ts and --gif-trj are mutually exclusive. "
            "Use --gif-trj with --ts if you want TS bonds shown in the trj gif"
        )

    is_cube = args.input and args.input.endswith(".cube")

    # MO mutual exclusivity
    if args.mo and args.vdw is not None:
        p.error("--mo and --vdw are mutually exclusive")
    if args.mo and args.gif_ts:
        p.error("--mo and --gif-ts are mutually exclusive")
    if args.mo and args.gif_trj:
        p.error("--mo and --gif-trj are mutually exclusive")
    if args.mo and not is_cube:
        p.error("--mo requires a .cube input file")
    if wants_gif:
        gif_path = args.gif_output or f"{base}.gif"
        gif_ext = gif_path.rsplit(".", 1)[-1].lower()
        if gif_ext != "gif":
            p.error(f"GIF output must have .gif extension, got: .{gif_ext}")

    # Load molecule (--gif-ts implies TS detection)
    cube_data = None
    needs_ts = args.ts_detect or args.gif_ts
    if is_cube:
        from xyzrender.io import load_cube

        graph, cube_data = load_cube(args.input, charge=args.charge, multiplicity=args.multiplicity, kekule=args.kekule)
    elif needs_ts and args.input:
        graph, _ts_frames = load_ts_molecule(
            args.input,
            charge=args.charge,
            multiplicity=args.multiplicity,
            ts_frame=args.ts_frame,
            kekule=args.kekule,
        )
    elif args.input:
        graph = load_molecule(args.input, charge=args.charge, multiplicity=args.multiplicity, kekule=args.kekule)
    elif not sys.stdin.isatty():
        graph = load_stdin(charge=args.charge, multiplicity=args.multiplicity, kekule=args.kekule)
    else:
        p.error("No input file and stdin is a terminal")

    # Post-load analysis
    if args.nci:
        graph = detect_nci(graph)

    # Orientation
    if args.interactive:
        rotate_with_viewer(graph)

    # MO: resolve colors once (used for both static render and gif-rot)
    mo_colors = None
    if args.mo:
        mo_colors = args.mo_colors or [
            config_data.get("mo_pos_color", "#2554A5"),
            config_data.get("mo_neg_color", "#851639"),
        ]

    # MO contour computation (PCA must happen here so rot is available for the grid)
    if args.mo and cube_data is not None:
        from typing import cast

        import numpy as np

        from xyzrender.mo import build_mo_contours
        from xyzrender.utils import kabsch_rotation, pca_orient

        assert mo_colors is not None  # set when args.mo is True
        cube = cast("CubeData", cube_data)
        rot = None
        if cfg.auto_orient:
            node_ids = list(graph.nodes())
            pos = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
            oriented, rot = pca_orient(pos, return_matrix=True)

            # Tilt -45° around x-axis so orbital lobes above/below the
            # molecular plane are clearly separated in the projection.
            tilt = np.radians(-45)
            rx = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(tilt), -np.sin(tilt)],
                    [0, np.sin(tilt), np.cos(tilt)],
                ]
            )
            rot = rx @ rot
            oriented = oriented @ rx.T

            for idx, nid in enumerate(node_ids):
                graph.nodes[nid]["position"] = tuple(oriented[idx].tolist())
            cfg.auto_orient = False  # already applied, don't re-apply in render_svg

        # If positions were modified (e.g. by interactive viewer) but PCA was not
        # applied, compute R from the correspondence between original cube atom
        # positions and current graph positions (Kabsch algorithm).
        if rot is None:
            orig = np.array([p for _, p in cube.atoms], dtype=float)
            curr = np.array([graph.nodes[i]["position"] for i in graph.nodes()], dtype=float)
            if not np.allclose(orig, curr, atol=1e-6):
                rot = kabsch_rotation(orig, curr)

        # Centroids for aligning the orbital grid with atom positions
        atom_centroid = np.array([p for _, p in cube.atoms], dtype=float).mean(axis=0)
        node_ids_mo = list(graph.nodes())
        curr_centroid = np.array(
            [graph.nodes[i]["position"] for i in node_ids_mo],
            dtype=float,
        ).mean(axis=0)

        cfg.mo_contours = build_mo_contours(
            cube,
            rot=rot,
            isovalue=args.iso,
            pos_color=mo_colors[0],
            neg_color=mo_colors[1],
            atom_centroid=atom_centroid,
            target_centroid=curr_centroid,
        )
        if args.mo_opacity is not None:
            cfg.mo_opacity = args.mo_opacity

    # Render static output
    svg = render_svg(graph, cfg)
    _write_output(svg, args.output, cfg, p)

    # GIF output
    if wants_gif:
        from xyzrender.gif import (
            ROTATION_AXES,
            render_rotation_gif,
            render_trajectory_gif,
            render_vibration_gif,
            render_vibration_rotation_gif,
        )

        if args.gif_rot and args.gif_rot not in ROTATION_AXES:
            p.error(f"Invalid rotation axis: {args.gif_rot} (valid: {', '.join(ROTATION_AXES)})")

        if args.gif_ts and args.gif_rot:
            if not args.input:
                p.error("--gif-ts requires an input file")
            render_vibration_rotation_gif(
                args.input,
                cfg,
                gif_path,
                charge=args.charge,
                multiplicity=args.multiplicity,
                ts_frame=args.ts_frame,
                fps=args.gif_fps,
                axis=args.gif_rot,
                n_frames=args.rot_frames,
                reference_graph=graph,
                detect_nci=args.nci,
            )
        elif args.gif_ts:
            if not args.input:
                p.error("--gif-ts requires an input file")
            render_vibration_gif(
                args.input,
                cfg,
                gif_path,
                charge=args.charge,
                multiplicity=args.multiplicity,
                ts_frame=args.ts_frame,
                fps=args.gif_fps,
                reference_graph=graph,
                detect_nci=args.nci,
            )
        elif args.gif_trj:
            if not args.input:
                p.error("--gif-trj requires an input file")
            frames = load_trajectory_frames(args.input)
            if len(frames) < 2:
                p.error("--gif-trj requires multi-frame input")
            render_trajectory_gif(
                frames,
                cfg,
                gif_path,
                charge=args.charge,
                multiplicity=args.multiplicity,
                fps=args.gif_fps,
                reference_graph=graph,
                detect_nci=args.nci,
                axis=args.gif_rot or None,
                kekule=args.kekule,
            )
        elif args.gif_rot:
            mo_data = None
            if args.mo and cube_data is not None:
                assert mo_colors is not None  # set when args.mo is True
                mo_data = {
                    "cube_data": cube_data,
                    "isovalue": args.iso,
                    "pos_color": mo_colors[0],
                    "neg_color": mo_colors[1],
                    "mo_opacity": args.mo_opacity if args.mo_opacity is not None else cfg.mo_opacity,
                }
            render_rotation_gif(
                graph,
                cfg,
                gif_path,
                n_frames=args.rot_frames,
                fps=args.gif_fps,
                axis=args.gif_rot,
                mo_data=mo_data,
            )


if __name__ == "__main__":
    main()
