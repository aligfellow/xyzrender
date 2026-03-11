"""Command-line interface for xyzrender."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from xyzrender.api import Molecule, load, orient, render, render_gif
from xyzrender.config import build_config
from xyzrender.readers import load_stdin

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {"svg", "png", "pdf"}


def _basename(input_path: str | None, from_stdin: bool) -> str:
    """Derive output basename from input file or fallback for stdin."""
    if from_stdin or not input_path:
        return "graphic"
    return Path(input_path).stem


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
    io_g.add_argument("input", nargs="?", help="Input file (.xyz, .mol, .sdf, .mol2, .pdb, .smi, .cif, .cube, …)")
    io_g.add_argument("-o", "--output", help="Output file (.svg, .png, .pdf)")
    io_g.add_argument("-c", "--charge", type=int, default=0)
    io_g.add_argument("-m", "--multiplicity", type=int, default=None)
    io_g.add_argument("-d", "--debug", action="store_true", help="Debug output")
    io_g.add_argument(
        "--smi",
        default=None,
        metavar="SMILES",
        help="Parse a SMILES string directly (requires rdkit: pip install 'xyzrender[smiles]')",
    )
    io_g.add_argument(
        "--mol-frame",
        type=int,
        default=0,
        metavar="N",
        help="Record index to read from a multi-molecule SDF file (default: 0)",
    )
    io_g.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Ignore file connectivity and re-detect bonds with xyzgraph",
    )

    # --- Styling ---
    style_g = p.add_argument_group("styling")
    style_g.add_argument("--config", default=None, help="Config preset or JSON path (default, flat, custom)")
    style_g.add_argument("-S", "--canvas-size", type=int, default=None)
    style_g.add_argument("-a", "--atom-scale", type=float, default=None)
    style_g.add_argument("-b", "--bond-width", type=float, default=None)
    style_g.add_argument("-s", "--atom-stroke-width", type=float, default=None)
    style_g.add_argument("--bond-color", default=None, help="Bond color (hex or named)")
    style_g.add_argument("-B", "--background", default=None)
    style_g.add_argument("-t", "--transparent", action="store_true", help="Transparent background")
    style_g.add_argument("-G", "--gradient-strength", type=float, default=None, help="Gradient contrast")
    style_g.add_argument("--grad", action=argparse.BooleanOptionalAction, default=None, help="Radial gradients")
    style_g.add_argument("-F", "--fog-strength", type=float, default=None, help="Fog strength")
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

    # --- Surfaces (MO / density / ESP) ---
    surf_g = p.add_argument_group("surfaces")
    surf_g.add_argument("--mo", action="store_true", default=False, help="Render MO lobes from .cube input")
    surf_g.add_argument(
        "--mo-colors",
        nargs=2,
        default=None,
        metavar=("POS", "NEG"),
        help="MO lobe colors as hex or named color (default: steelblue maroon)",
    )
    surf_g.add_argument("--dens", action="store_true", default=False, help="Render density isosurface from .cube input")
    surf_g.add_argument("--dens-color", default=None, help="Density surface color (hex or named, default: steelblue)")
    surf_g.add_argument(
        "--esp", default=None, metavar="CUBE", help="ESP cube file for potential coloring (implies --dens)"
    )
    surf_g.add_argument(
        "--nci-surf",
        default=None,
        metavar="CUBE",
        dest="nci_surf",
        help="NCI gradient cube file — find patches where RDG is low (implies density rendering)",
    )
    surf_g.add_argument(
        "--nci-color", default=None, help="NCI patch colour for uniform mode (hex or named, default: forestgreen)"
    )
    surf_g.add_argument(
        "--nci-coloring",
        default=None,
        choices=["avg", "pixel", "uniform"],
        dest="nci_coloring",
        help="NCI surface coloring: avg=per-lobe mean sign(l2)*rho blue/green/red (default), pixel=per-pixel raster, "
        "uniform=flat color (see --nci-color)",
    )
    surf_g.add_argument(
        "--iso",
        type=float,
        default=None,
        help="Isosurface threshold (MO default: 0.05, density/ESP default: 0.001, NCI/RDG default: 0.3)",
    )
    surf_g.add_argument(
        "--flat-mo",
        action="store_true",
        default=False,
        help="Disable MO depth classification (all lobes rendered as front)",
    )
    surf_g.add_argument("--mo-blur", type=float, default=None, help="MO Gaussian blur sigma (default: 0.8)")
    surf_g.add_argument("--mo-upsample", type=int, default=None, help="MO upsample factor (default: 3)")
    surf_g.add_argument("--opacity", type=float, default=None, help="Surface opacity (default: 1.0, >1 boosts)")

    # --- Overlay ---
    ov_g = p.add_argument_group("overlay")
    ov_g.add_argument(
        "--overlay",
        default=None,
        metavar="FILE",
        help="Overlay molecule file — aligned onto the main input and drawn in magenta",
    )
    ov_g.add_argument(
        "--overlay-color",
        default=None,
        dest="overlay_color",
        help="Overlay molecule color (hex or named, default: darkmagenta)",
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
    ts_g.add_argument(
        "--nci",
        action="store_true",
        dest="nci_detect",
        help="Auto-detect NCI interactions via xyzgraph",
    )
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

    # --- Measurements & annotations ---
    annot_g = p.add_argument_group("measurements & annotations")
    annot_g.add_argument(
        "--label-size", type=float, default=None, metavar="PT", help="Label font size (overrides preset)"
    )
    annot_g.add_argument(
        "--measure",
        nargs="*",
        default=None,
        metavar="TYPE",
        help="Print bond measurements to stdout: d (distances), a (angles), t (dihedrals). "
        "Combine: --measure d a. Omit types for all.",
    )
    annot_g.add_argument(
        "--idx",
        nargs="?",
        const="sn",
        default=None,
        metavar="FMT",
        help="Label all atoms with index in SVG: sn (C1, default), s (C only), n (number only)",
    )
    annot_g.add_argument(
        "-l",
        dest="label_specs",
        nargs="+",
        action="append",
        default=None,
        metavar="TOKEN",
        help=(
            "Annotate SVG (repeatable): "
            '"-l 1 2 d" (bond distance), "-l 2 a" (all angles at atom 2), '
            '"-l 1 2 3 4 t" (dihedral polyline), "-l 1 +0.5" (custom atom label), '
            '"-l 1 2 NBO" (custom bond label). Indices 1-based.'
        ),
    )
    annot_g.add_argument(
        "--label",
        default=None,
        metavar="FILE",
        help="Annotation file (same spec syntax as -l, one per line, # comments, comma or space separated)",
    )
    annot_g.add_argument(
        "--cmap",
        default=None,
        metavar="FILE",
        help="Atom property colormap file: two columns (1-indexed atom index, value); header lines are skipped",
    )
    annot_g.add_argument(
        "--cmap-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("VMIN", "VMAX"),
        help="Explicit colormap range (default: auto from file values)",
    )
    annot_g.add_argument(
        "--vectors",
        default=None,
        metavar="FILE",
        help=(
            "JSON file defining vector arrows to overlay on the image.  "
            'Each entry: {"origin": "com"|<atom_index>|[x,y,z], '
            '"vector": [vx,vy,vz], "color": "#rrggbb", "label": "μ", "scale": 1.0}'
        ),
    )
    annot_g.add_argument(
        "--vector-scale",
        type=float,
        default=None,
        metavar="FACTOR",
        help="Global length scale factor applied to all vector arrows (default: 1.0)",
    )

    # --- Crystal / periodic structures ---
    crystal_g = p.add_argument_group("crystal / periodic structures")
    crystal_g.add_argument(
        "--crystal",
        nargs="?",
        const="auto",
        default=None,
        metavar="{vasp,qe}",
        help=(
            "Load as a periodic crystal structure via phonopy (requires xyzrender[crystal]). "
            "Enables unit cell box, crystallographic axes (a/b/c), and image atoms. "
            "Optionally specify the phonopy interface: vasp, qe (auto-detected from filename "
            "if omitted). Examples: --crystal, --crystal vasp, --crystal qe"
        ),
    )
    crystal_g.add_argument(
        "--cell",
        action="store_true",
        default=False,
        help=(
            "Draw the unit cell box from an extXYZ Lattice= header. "
            "No crystallographic axes or image atoms — just the box. "
            "Does not require phonopy."
        ),
    )
    crystal_g.add_argument("--no-cell", action="store_true", default=False, help="Hide the unit cell box (--crystal)")
    crystal_g.add_argument(
        "--ghosts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show/hide periodic image atoms (default: on when --cell/--crystal, off otherwise)",
    )
    crystal_g.add_argument(
        "--axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show/hide crystallographic axis arrows a/b/c (--crystal)",
    )
    crystal_g.add_argument("--cell-color", default=None, help="Unit cell box color (hex or named, default: #333333)")
    crystal_g.add_argument("--cell-width", type=float, default=None, help="Unit cell box line width (default: 1.5)")
    crystal_g.add_argument(
        "--ghost-opacity",
        type=float,
        default=None,
        help="Opacity for ghost (periodic image) atoms/bonds (default: 0.5)",
    )
    crystal_g.add_argument(
        "--axis",
        default=None,
        metavar="HKL",
        help=(
            "Orient crystal looking down a crystallographic direction, e.g. --axis 111. "
            "Each digit is one Miller index (0-9). Requires --crystal or --cell."
        ),
    )

    args = p.parse_args()
    from xyzrender import configure_logging

    configure_logging(verbose=True, debug=args.debug)

    # Normalise argparse --hy (None | [] | [1,3,5]) → shared (None | True | [1,3,5])
    hy_spec: bool | list[int] | None = None
    if args.hy is not None:
        hy_spec = True if len(args.hy) == 0 else args.hy

    # Resolve orient flag before build_config so it can be passed in directly
    from_stdin = not args.input and not sys.stdin.isatty()
    _orient: bool | None = args.orient
    if _orient is None and (args.interactive or from_stdin):
        _orient = False

    cfg = build_config(
        args.config or "default",
        canvas_size=args.canvas_size,
        atom_scale=args.atom_scale,
        bond_width=args.bond_width,
        atom_stroke_width=args.atom_stroke_width,
        bond_color=args.bond_color,
        background=args.background,
        transparent=args.transparent,
        gradient=args.grad,
        gradient_strength=args.gradient_strength,
        fog=args.fog,
        fog_strength=args.fog_strength,
        bo=args.bo,
        hy=hy_spec,
        no_hy=args.no_hy,
        orient=_orient,
        opacity=args.opacity,
        label_font_size=args.label_size,
        vdw_opacity=args.vdw_opacity,
        vdw_scale=args.vdw_scale,
        vdw_gradient_strength=args.vdw_gradient,
        ts_bonds=_parse_pairs(args.ts_bond),
        nci_bonds=_parse_pairs(args.nci_bond),
        vdw_indices=(
            _parse_indices(args.vdw) if args.vdw is not None and args.vdw != "" else ([] if args.vdw == "" else None)
        ),
        show_indices=args.idx is not None,
        idx_format=args.idx or "sn",
        cmap_range=tuple(args.cmap_range) if args.cmap_range else None,
    )

    # Output path defaults and validation
    base = _basename(args.input, from_stdin)
    if not args.output:
        args.output = f"{base}.svg"

    static_ext = args.output.rsplit(".", 1)[-1].lower()
    if static_ext not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join("." + e for e in sorted(_SUPPORTED_EXTENSIONS))
        p.error(f"Unsupported static output format: .{static_ext} (use {supported})")

    wants_gif = args.gif_ts or args.gif_rot or args.gif_trj

    # Warn when annotation flags (static-SVG only) are combined with GIF output
    annotation_flags_used = args.idx is not None or args.label_specs or args.label
    if annotation_flags_used and wants_gif:
        print(
            "Warning: --idx, -l and --label apply to static SVG output only and will not appear in the GIF.",
            file=sys.stderr,
        )

    is_cube = args.input and args.input.endswith(".cube")

    # Validate --smi / --mol-frame / --rebuild usage
    if args.smi and args.input:
        p.error("--smi cannot be combined with a positional input file")
    if args.mol_frame != 0 and not (args.input and args.input.endswith(".sdf")):
        logger.warning("--mol-frame has no effect on non-SDF input")
    if args.rebuild and args.smi:
        logger.warning("--rebuild has no effect on SMILES input (rdkit bonds are always used)")

    # --crystal: interface_mode is non-None iff phonopy crystal loading is requested.
    # Auto-detection (stem/extension) is handled inside load() → _resolve_crystal_interface().
    # --cell is a separate lighter path: extXYZ box only, no phonopy, no image atoms.
    interface_mode: str | None = None
    if args.crystal is not None:
        if args.input is None:
            p.error("--crystal requires an input file")
        # Resolve now so we know interface_mode for axes/ghosts defaulting below.
        # load() will call _resolve_crystal_interface() again; ValueError → p.error().
        from xyzrender.api import _resolve_crystal_interface

        try:
            interface_mode = _resolve_crystal_interface(Path(args.input), args.crystal)
        except ValueError as e:
            p.error(str(e))

    if wants_gif:
        gif_path = args.gif_output or f"{base}.gif"
        gif_ext = gif_path.rsplit(".", 1)[-1].lower()
        if gif_ext != "gif":
            p.error(f"GIF output must have .gif extension, got: .{gif_ext}")

    # --- Load molecule ---
    needs_ts = args.ts_detect or args.gif_ts
    if is_cube and needs_ts:
        print(
            "Warning: --ts/--gif-ts has no effect with cube files (single geometry, no frequency data). "
            "Use --ts-bond to manually specify TS bonds."
        )

    if args.smi:
        mol = load(args.smi, smiles=True, charge=args.charge, multiplicity=args.multiplicity, kekule=args.kekule)
        xyz_path = Path(args.output).with_suffix(".xyz")
        mol.to_xyz(xyz_path, title=args.smi)
        logger.info("3D geometry written to %s", xyz_path)
    elif from_stdin:
        graph = load_stdin(charge=args.charge, multiplicity=args.multiplicity, kekule=args.kekule)
        mol = Molecule(graph=graph, cube_data=None, cell_data=None, oriented=False)
    elif not args.input:
        p.error("No input file and stdin is a terminal")
    else:
        try:
            mol = load(
                args.input,
                charge=args.charge,
                multiplicity=args.multiplicity,
                kekule=args.kekule,
                rebuild=args.rebuild,
                mol_frame=args.mol_frame,
                ts_detect=needs_ts,
                ts_frame=args.ts_frame,
                nci_detect=args.nci_detect,
                crystal=interface_mode or False,
                cell=args.cell,
                quick=args.bo is False,
            )
        except ValueError as e:
            p.error(str(e))

    # Pre-load overlay once so render() + render_gif() don't each load it from disk.
    if args.overlay and isinstance(args.overlay, str):
        _ov_charge = mol.graph.graph.get("total_charge", 0)
        _ov_mult = mol.graph.graph.get("multiplicity")
        args.overlay = load(args.overlay, charge=_ov_charge, multiplicity=_ov_mult)

    # --- Measurements (terminal output only) ---
    if args.measure is not None:
        from xyzrender.measure import print_measurements

        modes = args.measure if args.measure else ["all"]
        valid_modes = {"all", "d", "a", "t", "tor", "dih"}
        for m in modes:
            if m.lower() not in valid_modes:
                p.error(f"--measure: unknown type {m!r} (valid: d, a, t, or omit for all)")
        print_measurements(mol.graph, modes)

    # --- Annotations (store on cfg; render() uses cfg.annotations directly) ---
    if args.label_specs or args.label:
        from xyzrender.annotations import parse_annotations

        try:
            cfg.annotations = parse_annotations(
                inline_specs=args.label_specs or [],
                file_path=args.label,
                graph=mol.graph,
            )
        except (ValueError, FileNotFoundError) as e:
            p.error(str(e))

    # --- Atom property colormap (store on cfg) ---
    if args.cmap:
        from xyzrender.annotations import load_cmap

        try:
            cfg.atom_cmap = load_cmap(args.cmap, mol.graph)
        except (ValueError, FileNotFoundError) as e:
            p.error(str(e))

    # --- Interactive viewer ---
    if args.interactive:
        orient(mol)
        if not mol.oriented:
            sys.exit(1)

    # --- Crystal ghost resolution ---
    # Ghosts default: on whenever the molecule carries cell_data (auto-detected or explicit)
    _show_ghosts = args.ghosts if args.ghosts is not None else mol.cell_data is not None

    # --- Render static SVG ---
    try:
        render(
            mol,
            config=cfg,
            no_cell=args.no_cell,
            axes=args.axes,
            axis=args.axis,
            ghosts=_show_ghosts,
            cell_color=args.cell_color,
            cell_width=args.cell_width,
            ghost_opacity=args.ghost_opacity,
            mo=args.mo,
            dens=args.dens,
            esp=args.esp,
            nci=args.nci_surf,
            iso=args.iso,
            mo_pos_color=args.mo_colors[0] if args.mo_colors else None,
            mo_neg_color=args.mo_colors[1] if args.mo_colors else None,
            mo_blur=args.mo_blur,
            mo_upsample=args.mo_upsample,
            flat_mo=args.flat_mo,
            dens_color=args.dens_color,
            nci_color=args.nci_color,
            nci_coloring=args.nci_coloring,
            overlay=args.overlay,
            overlay_color=args.overlay_color,
            vectors=args.vectors,
            vector_scale=args.vector_scale,
            output=args.output,
        )
    except ValueError as e:
        p.error(str(e))

    # --- GIF output ---
    if wants_gif:
        if args.gif_rot:
            from xyzrender.gif import ROTATION_AXES

            if args.gif_rot not in ROTATION_AXES:
                _test_ax = args.gif_rot.lstrip("-")
                if not (_test_ax.isdigit() and len(_test_ax) >= 3 and mol.cell_data is not None):
                    p.error(
                        f"Invalid rotation axis: {args.gif_rot!r} "
                        f"(valid: {', '.join(ROTATION_AXES)}, or 3-digit hkl for crystal inputs)"
                    )

        mol_or_path: str | Molecule = args.input if (args.gif_ts or args.gif_trj) else mol
        # For gif_ts/gif_trj the trajectory is read from disk (mol_or_path is a path),
        # but pass mol.graph as reference_graph so -I orientation and TS/NCI bonds are respected.
        _ref_graph = mol.graph if (args.gif_ts or args.gif_trj) else None
        try:
            render_gif(
                mol_or_path,
                config=cfg,
                gif_rot=args.gif_rot or None,
                gif_trj=args.gif_trj,
                gif_ts=args.gif_ts,
                output=gif_path,
                gif_fps=args.gif_fps,
                rot_frames=args.rot_frames,
                ts_frame=args.ts_frame,
                overlay=args.overlay,
                overlay_color=args.overlay_color,
                reference_graph=_ref_graph,
                detect_nci=args.nci_detect,
                mo=args.mo,
                dens=args.dens,
                iso=args.iso,
                mo_pos_color=args.mo_colors[0] if args.mo_colors else None,
                mo_neg_color=args.mo_colors[1] if args.mo_colors else None,
                mo_blur=args.mo_blur,
                mo_upsample=args.mo_upsample,
                flat_mo=args.flat_mo,
                dens_color=args.dens_color,
                no_cell=args.no_cell,
                axes=args.axes,
                axis=args.axis,
                ghosts=_show_ghosts,
                cell_color=args.cell_color,
                cell_width=args.cell_width,
                ghost_opacity=args.ghost_opacity,
                vectors=args.vectors,
                vector_scale=args.vector_scale,
            )
        except ValueError as e:
            p.error(str(e))


if __name__ == "__main__":
    main()
