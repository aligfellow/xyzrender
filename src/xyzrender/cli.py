"""Command-line interface for xyzrender."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from xyzrender.config import build_render_config, load_config

if TYPE_CHECKING:
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
    io_g.add_argument("--debug", action="store_true", help="Debug output")

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
    disp_g.add_argument("--fog", action=argparse.BooleanOptionalAction, default=None, help="Depth fog")
    disp_g.add_argument("--vdw", nargs="?", const="", default=None, help='VdW spheres (no args=all, or "1-20,25")')

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
    gif_g.add_argument("--gif-frames", type=int, default=120, help="Rotation GIF frames (default: 120)")

    args = p.parse_args()

    from xyzrender import configure_logging

    configure_logging(verbose=True, debug=args.debug)

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

    # Output path defaults
    base = _basename(args.input, from_stdin)
    if not args.output:
        args.output = f"{base}.svg"

    # Load molecule (--gif-ts implies TS detection)
    needs_ts = args.ts_detect or args.gif_ts
    if needs_ts and args.input:
        graph, _ts_frames = load_ts_molecule(
            args.input, charge=args.charge, multiplicity=args.multiplicity, ts_frame=args.ts_frame
        )
    elif args.input:
        graph = load_molecule(args.input, charge=args.charge, multiplicity=args.multiplicity)
    elif not sys.stdin.isatty():
        graph = load_stdin(charge=args.charge, multiplicity=args.multiplicity)
    else:
        p.error("No input file and stdin is a terminal")

    # Post-load analysis
    if args.nci:
        detect_nci(graph)

    # Orientation
    if args.interactive:
        rotate_with_viewer(graph)

    # Render static output
    svg = render_svg(graph, cfg)
    _write_output(svg, args.output, cfg, p)

    # GIF output
    wants_gif = args.gif_ts or args.gif_rot or args.gif_trj
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

        gif_path = args.gif_output or f"{base}.gif"

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
                n_frames=args.gif_frames,
                reference_graph=graph,
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
            )
        elif args.gif_rot:
            render_rotation_gif(graph, cfg, gif_path, n_frames=args.gif_frames, fps=args.gif_fps, axis=args.gif_rot)
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
            )


if __name__ == "__main__":
    main()
