"""Command-line interface for xyzrender."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {"svg", "png", "pdf", "tiff", "tif"}


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
        segment = part.strip()
        if not segment:
            continue
        tokens = segment.split("-")
        if len(tokens) != 2:
            raise ValueError(f"Invalid pair {segment!r}: expected exactly one '-' per segment (e.g. 1-6,3-4)")
        try:
            a, b = int(tokens[0].strip()), int(tokens[1].strip())
        except ValueError as e:
            raise ValueError(f"Invalid pair {segment!r}: atom indices must be integers (e.g. 1-6,3-4)") from e
        pairs.append((a - 1, b - 1))
    return pairs


def _flatten_specs(items: list[str]) -> list[str]:
    """Split each item on commas and flatten: ['M-L,sbm', '1-3'] → ['M-L', 'sbm', '1-3']."""
    out: list[str] = []
    for item in items:
        for part in item.split(","):
            stripped = part.strip()
            if stripped:
                out.append(stripped)
    return out


_COMPACT_HELP = """\
Usage: xyzrender [input] [options]

Input/Output:
  input                   .xyz .mol .sdf .mol2 .pdb .cif .cube .com .nw ...
  -o FILE                 Output file (.svg, .png, .pdf, .tif, .tiff)
  -c N / -m N             Charge / multiplicity (for bond order detection)
  --smi SMILES            Render from SMILES string (requires rdkit)

Styling:
  --config NAME           Preset: default flat paton skeletal bubble tube
                          mtube wire graph pmol (or path to custom .json)
  -a F / -b F             Atom scale / Bond width
  -B COLOR / -t           Background colour / transparent
  --grad / --no-grad      Atom gradients on/off
  --fog / --no-fog        Depth fog on/off
  --dof                   Depth of field - blur "back" atoms (not bonds)

Display:
  --hy [ATOMS] / --no-hy  Show/hide H atoms (all or specific indices)
  --bo / --no-bo          Show/hide bond orders
  --unbond SPEC           Hide bonds: M-L  sbm  Fe-het  1-3  pi
  --bond PAIR             Force-show bonds: 1-3  4-5
  --haptic                Pi-coordination → single centroid bond
  --vdw [ATOMS]           VdW spheres (all or specific atoms/elements: 1,2,6-8,M,H)
  --no-bonds              Hide all bonds

Orientation:
  --orient / --no-orient  PCA auto-orientation on/off
  -I                      Interactive viewer (requires vmol)
  --ref [FILE]            Save/load orientation reference

TS / NCI:
  --ts / --nci            Auto-detect TS bonds / NCI interactions
  --ts-bond / --nci-bond  Manual bond pairs (1-indexed)

GIF Animation:
  --gif-rot [AXIS]        Rotation GIF (x/y/z/xy/xz/.../hkl, default: y)
  --gif-trj               Trajectory - requires mutiframe .xyz or calculation output
  --gif-ts                TS vibration GIF - required vib trj or output with Hessian
  --gif-diffuse           Diffuse assembly GIF
  -go FILE / --gif-fps N  GIF output path / frames per second

Surfaces:
  --mo / --dens           MO lobes / density isosurface (.cube input)
  --esp CUBE              ESP colour mapping (density + ESP cubes)
  --nci-surf CUBE         NCI interaction surface (density + gradient cubes)
  --iso F                 Isovalue (MO: 0.05, dens: 0.001, NCI: 0.3)
  --hull [INDICES]        Convex hull (all / "rings" / atom subsets)
  --surface-style STYLE   solid, mesh, contour, dot

Overlay / Ensemble:
  --overlay FILE          Overlay molecule (aligned, drawn in magenta)
  --ensemble              Multi-frame overlay from trajectory

Annotations:
  --idx [FMT]             Atom index labels: sn (C1), s (C), n (1)
  -l TOKENS               Annotation: "-l 1 2 d" (distance), "-l 2 a" (angles)
  --stereo [CLASSES]      Stereochemistry labels

Highlight / Regions:
  --mol-color COLOR       Flat colour for all atoms
  --hl ATOMS [COLOR]      Highlight atom group (repeatable)
  --region ATOMS CONFIG   Per-atom style region
  --radius-scale ATOMS FACTOR
                          Per-atom radius scale (multiplies on top of -a)

Crystal:
  --cell                  Draw unit cell box
  --ghosts / --no-ghosts  Periodic image atoms on/off
  --axes / --no-axes      Crystallographic axis arrows
  --axis HKL              Orient along Miller index (e.g. 111)
  --supercell M N L       Repeat unit cell

Run 'xyzrender --help' for full details on every option.
"""


def main() -> None:
    """Entry point for the CLI."""
    from xyzrender import __version__

    # Intercept -h before argparse to print compact help
    if "-h" in sys.argv[1:] and "--help" not in sys.argv[1:]:
        print(f"xyzrender v{__version__} — publication-quality molecular graphics\n")
        print(_COMPACT_HELP)
        raise SystemExit(0)

    p = argparse.ArgumentParser(prog="xyzrender", description="Publication-quality molecular graphics.")
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    # --- Input / Output ---
    io_g = p.add_argument_group("input/output")
    io_g.add_argument(
        "input",
        nargs="?",
        help="Input file (.xyz, .mol, .sdf, .mol2, .pdb, .smi, .cif, .cube, "
        ".com, .gjf, .inp, .nw, .vasp, POSCAR, .in, …)",
    )
    io_g.add_argument("-o", "--output", help="Output file (.svg, .png, .pdf)")
    io_g.add_argument("-c", "--charge", type=int, default=0)
    io_g.add_argument("-m", "--multiplicity", type=int, default=None)
    io_g.add_argument("--bohr", action="store_true", default=False, help="Input coordinates are in Bohr")
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
    style_g.add_argument(
        "--config",
        default=None,
        help=("Config preset or JSON path (default, flat, paton, skeletal, bubble, tube, mtube, wire, graph, custom)"),
    )
    style_g.add_argument("-S", "--canvas-size", type=int, default=None)
    style_g.add_argument("-a", "--atom-scale", type=float, default=None)
    style_g.add_argument(
        "--radius-scale",
        nargs=2,
        action="append",
        default=None,
        metavar=("ATOMS", "FACTOR"),
        help='Scale atom radii: --radius-scale "1,2-4,M" 2.0. Repeatable. Multiplies on top of -a.',
    )
    style_g.add_argument("-b", "--bond-width", type=float, default=None)
    style_g.add_argument("-s", "--atom-stroke-width", type=float, default=None)
    style_g.add_argument("--bond-color", default=None, help="Bond color (hex or named)")
    style_g.add_argument("--bond-outline-color", default=None, help="Bond outline color (hex or named)")
    style_g.add_argument("--bond-outline-width", type=float, default=None, help="Bond outline width in px (0 = off)")
    style_g.add_argument("-B", "--background", default=None)
    style_g.add_argument("-t", "--transparent", action="store_true", help="Transparent background")
    style_g.add_argument("-Hls", "--hue-shift-factor", type=float, default=None, help="Hue gradient contrast")
    style_g.add_argument("-hLs", "--light-shift-factor", type=float, default=None, help="Lightness gradient contrast")
    style_g.add_argument(
        "-hlS", "--saturation-shift-factor", type=float, default=None, help="Saturation gradient contrast"
    )
    style_g.add_argument("--grad", action=argparse.BooleanOptionalAction, default=None, help="Radial gradients")
    style_g.add_argument("-F", "--fog-strength", type=float, default=None, help="Fog strength")
    style_g.add_argument("--vdw-opacity", type=float, default=None, help="VdW sphere opacity")
    style_g.add_argument("--vdw-scale", type=float, default=None, help="VdW sphere radius scale")
    style_g.add_argument("--atom-gradient-strength", type=float, default=None, help="Atom gradient strength")
    style_g.add_argument("--bond-gradient-strength", type=float, default=None, help="Bond gradient strength")
    style_g.add_argument("--vdw-gradient-strength", type=float, default=None, help="VdW sphere gradient strength")

    # --- Display ---
    disp_g = p.add_argument_group("display")
    disp_g.add_argument(
        "--hy",
        nargs="?",
        const="",
        default=None,
        metavar="ATOMS",
        help='Show H atoms (no args=all, or indices like "1-5,8")',
    )
    disp_g.add_argument("--no-hy", action="store_true", default=False, help="Hide all H atoms")
    disp_g.add_argument(
        "--no-bonds", action="store_true", default=False, help="Hide all bonds (e.g. space-filling style)"
    )
    disp_g.add_argument(
        "--unbond",
        nargs="+",
        default=[],
        metavar="SPEC",
        help='Hide bonds by rule or index pair: "M-L sbm Fe-het 1-3" (comma or space separated)',
    )
    disp_g.add_argument(
        "--bond",
        nargs="+",
        default=[],
        metavar="PAIR",
        help='Force-show/add bonds: "1-3 4-5" (1-indexed, overrides --unbond)',
    )
    disp_g.add_argument("--bo", action=argparse.BooleanOptionalAction, default=None, help="Bond orders")
    disp_g.add_argument(
        "--haptic",
        action="store_true",
        default=False,
        help="Replace pi-coordination bonds with single metal-to-centroid bond",
    )
    disp_g.add_argument(
        "-k", "--kekule", action="store_true", default=False, help="Use Kekule bond orders (no aromatic 1.5)"
    )
    disp_g.add_argument("--fog", action=argparse.BooleanOptionalAction, default=None, help="Depth fog")
    disp_g.add_argument(
        "--skeletal-label-color", default=None, help="Override all element label colors in skeletal mode (hex or named)"
    )
    disp_g.add_argument(
        "--vdw",
        nargs="?",
        const="",
        default=None,
        help='VdW spheres (no args=all, or selectors like "1-20,25", "M", "Pt")',
    )

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
        "--nci-mode",
        default=None,
        help="NCI surface coloring: avg (default), pixel, uniform, or a colour name/hex for uniform mode",
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
    surf_g.add_argument(
        "--surface-style",
        choices=["solid", "mesh", "contour", "dot"],
        default=None,
        dest="surface_style",
        help="Surface rendering style: solid (default), mesh (wireframe grid), wire (outline only)",
    )
    surf_g.add_argument(
        "--hull",
        nargs="*",
        default=None,
        metavar="INDICES",
        help='Convex hull (no args = all heavy atoms; "rings" = per aromatic ring;'
        ' "faces" = 2D structural faces colored by size (sheets);'
        ' "pores" = 3D pore windows via coarse-grained net topology (MOFs);'
        ' or 1-indexed subsets e.g. "1-6" or "1-6 7-12")',
    )
    surf_g.add_argument(
        "--hull-color", nargs="+", default=None, help="Hull fill color(s) (hex or named, one per subset)"
    )
    surf_g.add_argument("--hull-opacity", type=float, default=None, help="Hull fill opacity (0-1)")
    surf_g.add_argument(
        "--hull-color-type",
        choices=["type", "size", "env"],
        default="type",
        help="Ring colouring: 'type' = atom types + size, 'size' = size only, 'env' = type + ring fusion",
    )
    surf_g.add_argument(
        "--hull-edge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Draw/hide non-bond hull edges (default: on)",
    )
    surf_g.add_argument(
        "--hull-edge-width-ratio",
        type=float,
        default=None,
        help="Hull edge stroke width as fraction of bond width (default: 0.4)",
    )
    surf_g.add_argument(
        "--ring-max-size",
        type=int,
        default=100,
        dest="ring_max_size",
        help="Max ring size for --hull faces/pore detection (default: 100)",
    )
    surf_g.add_argument(
        "--ring-min-size",
        type=int,
        default=3,
        dest="ring_min_size",
        help="Min ring size for --hull faces/pore detection (default: 3)",
    )
    surf_g.add_argument(
        "--face-planarity",
        type=float,
        default=0.25,
        help="Planarity tolerance for 3D face detection (0=strict, 1=permissive, default: 0.25)",
    )
    surf_g.add_argument(
        "--pore",
        action="store_true",
        default=False,
        help="Detect pore cavities and draw inscribed sphere(s)",
    )
    surf_g.add_argument("--pore-color", default=None, help="Pore sphere color (default: yellow)")
    surf_g.add_argument("--pore-opacity", type=float, default=None, help="Pore sphere opacity (default: 0.5)")

    # --- Overlay / ensemble ---
    ov_g = p.add_argument_group("overlay / ensemble")
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
    ov_g.add_argument(
        "--ensemble",
        action="store_true",
        default=False,
        help="Ensemble overlay for multi-frame XYZ trajectories (align all frames onto the first)",
    )
    ov_g.add_argument(
        "--align-atoms",
        default=None,
        dest="align_atoms",
        help='Atom indices (min 3) for alignment subset, e.g. "1,2,3", "1-6"',
    )
    ov_g.add_argument(
        "--ensemble-color",
        default=None,
        dest="ensemble_color",
        help="Palette (viridis, spectral, coolwarm), a single color, or comma-separated colors",
    )

    # --- Orientation ---
    orient_g = p.add_argument_group("orientation")
    orient_g.add_argument(
        "--orient", action=argparse.BooleanOptionalAction, default=None, help="Auto-orientation (default: on)"
    )
    orient_g.add_argument("-I", "--interactive", action="store_true", help="Open in v viewer for interactive rotation")
    orient_g.add_argument(
        "--ref",
        nargs="?",
        const="reference.xyz",
        default=None,
        metavar="FILE",
        help="Save/load orientation reference for consistent rendering",
    )

    # --- TS / NCI ---
    ts_g = p.add_argument_group("transition state / NCI")
    ts_g.add_argument("--ts", action="store_true", dest="ts_detect", help="Auto-detect TS bonds via graphRC")
    ts_g.add_argument("--ts-frame", type=int, default=0, help="TS reference frame for graphRC (0-indexed)")
    ts_g.add_argument("--ts-bond", default="", help='Manual TS bond pair(s), 1-indexed: "1-6,3-4"')
    ts_g.add_argument("--ts-color", default=None, help="Color for dashed TS bonds (hex or named)")
    ts_g.add_argument(
        "--nci",
        action="store_true",
        dest="nci_detect",
        help="Auto-detect NCI interactions via xyzgraph",
    )
    ts_g.add_argument("--nci-bond", default="", help='Manual NCI bond pair(s), 1-indexed: "1-5,2-8"')
    ts_g.add_argument("--nci-color", default=None, help="Color for dotted NCI bonds (hex or named)")

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
    gif_g.add_argument("--gif-diffuse", action="store_true", help="Diffuse/assembly GIF — atoms scatter and reassemble")
    gif_g.add_argument("-go", "--gif-output", default=None, help="GIF output path")
    gif_g.add_argument("--gif-fps", type=int, default=10, help="GIF frames per second (default: 10)")
    gif_g.add_argument("--rot-frames", type=int, default=120, help="Rotation frames (default: 120)")
    gif_g.add_argument("--diffuse-frames", type=int, default=60, help="Number of diffuse frames (default: 60)")
    gif_g.add_argument("--diffuse-noise", type=float, default=0.3, help="Per-frame random walk noise (default: 0.3)")
    gif_g.add_argument(
        "--diffuse-bonds",
        choices=["fade", "show", "hide"],
        default="fade",
        help="Bond visibility during diffuse: fade (default), show, or hide",
    )
    gif_g.add_argument(
        "--diffuse-rot",
        type=int,
        nargs="?",
        const=180,
        default=None,
        metavar="DEG",
        help="Add rotation during diffuse (default: 180°)",
    )
    gif_g.add_argument(
        "--diffuse-forward", action="store_true", help="Play forward (molecule → noise) instead of assembly"
    )
    gif_g.add_argument(
        "--anchor",
        default=None,
        metavar="ATOMS",
        help='Atoms that stay fixed, e.g. "1-5,8"',
    )

    # --- Atom color / Highlight ---
    hl_g = p.add_argument_group("highlight")
    hl_g.add_argument(
        "--mol-color", default=None, metavar="COLOR", help="Flat color for all atoms and bonds (overrides CPK)"
    )
    hl_g.add_argument(
        "--hl",
        nargs="+",
        action="append",
        default=None,
        metavar=("ATOMS", "COLOR"),
        help='Highlight atom group: --hl "1-5,8" [color]. Can be repeated. Auto-colors if no color given.',
    )

    # --- Style regions ---
    region_g = p.add_argument_group("style regions")
    region_g.add_argument(
        "--region",
        nargs=2,
        action="append",
        default=None,
        metavar=("ATOMS", "CONFIG"),
        help="Render atom subset with a different style: "
        '--region "1-5" flat or --region "Pt" flat or --region "M" tube.',
    )

    # --- Bond coloring ---
    bond_color_g = p.add_argument_group("bond coloring")
    bond_color_g.add_argument(
        "--bond-by-element",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Color bonds by endpoint atom colors (on by default in tube preset)",
    )
    bond_color_g.add_argument(
        "--bond-gradient",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cylinder shading on bonds (3D tube look). On by default in tube preset.",
    )

    # --- Depth of field ---
    dof_g = p.add_argument_group("depth of field")
    dof_g.add_argument(
        "--dof", action="store_true", default=False, help="Depth-of-field blur (does not affect bonds/lines)"
    )
    dof_g.add_argument(
        "--dof-strength", type=float, default=None, metavar="FLOAT", help="DoF max blur strength (default: 3.0)"
    )

    # --- Measurements & annotations ---
    annot_g = p.add_argument_group("measurements & annotations")
    annot_g.add_argument(
        "--label-size", type=float, default=None, metavar="PT", help="Label font size (overrides preset)"
    )
    annot_g.add_argument(
        "--stereo",
        nargs="?",
        const="",
        default=None,
        metavar="CLASSES",
        help=(
            "Add stereochemistry labels from 3D geometry. "
            "Optional comma-separated class filter: point, ez, axis, plane, helix. "
            "Omit CLASSES to show all."
        ),
    )
    annot_g.add_argument(
        "--stereo-style",
        default="atom",
        choices=["atom", "label"],
        metavar="STYLE",
        help="R/S label placement: atom (centered) or label (offset). Default: atom.",
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
        "--cmap-palette",
        default="viridis",
        metavar="NAME",
        help="Colormap palette name (default: viridis)",
    )
    annot_g.add_argument(
        "--cbar",
        action="store_true",
        default=False,
        help="Add a vertical colorbar on the right showing the data range",
    )
    annot_g.add_argument(
        "--cmap-symm",
        action="store_true",
        default=False,
        help="Symmetric colormap range about zero: [-max(|v|), +max(|v|)]",
    )
    annot_g.add_argument(
        "--vector",
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
        "--cell",
        action="store_true",
        default=False,
        help=(
            "Draw the unit cell box from an extXYZ Lattice= header. "
            "No crystallographic axes or image atoms — just the box."
        ),
    )
    crystal_g.add_argument("--no-cell", action="store_true", default=False, help="Hide the unit cell box")
    crystal_g.add_argument(
        "--ghosts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show/hide periodic image atoms (default: on when --cell, off otherwise)",
    )
    crystal_g.add_argument(
        "--axes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show/hide crystallographic axis arrows a/b/c (default: on unless --no-cell)",
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
            "Each digit is one Miller index (0-9). Requires --cell."
        ),
    )
    crystal_g.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=(1, 1, 1),
        metavar=("M", "N", "L"),
        help="Repeat the unit cell M N L times along a, b, c. Default: 1 1 1.",
    )

    args = p.parse_args()

    from_stdin = not args.input and not sys.stdin.isatty()
    if not args.input and not args.smi and not from_stdin:
        p.error("No input file provided. Pass a file path, --smi, or pipe via stdin.")
    if args.input and not Path(args.input).is_file():
        p.error(f"No such file or directory: {args.input!r}")

    from xyzrender import configure_logging
    from xyzrender.api import (
        Molecule,
        load,
        orient,
        render,
        render_gif,
    )
    from xyzrender.config import build_config
    from xyzrender.readers import load_stdin

    configure_logging(verbose=True, debug=args.debug)

    from xyzrender.utils import parse_atom_indices

    # Normalise --hy: None | "" | "1-5,8" → None | True | [1,2,3,4,5,8] (1-indexed for API)
    hy_spec: bool | list[int] | None = None
    if args.hy is not None:
        hy_spec = True if args.hy == "" else parse_atom_indices(args.hy, one_indexed=True)

    # Resolve orient flag before build_config so it can be passed in directly
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
        bond_outline_color=args.bond_outline_color,
        bond_outline_width=args.bond_outline_width,
        ts_color=args.ts_color,
        nci_color=args.nci_color,
        background=args.background,
        transparent=args.transparent,
        gradient=args.grad,
        hue_shift_factor=args.hue_shift_factor,
        light_shift_factor=args.light_shift_factor,
        saturation_shift_factor=args.saturation_shift_factor,
        fog=args.fog,
        fog_strength=args.fog_strength,
        bo=args.bo,
        hy=hy_spec,
        hide_bonds=args.no_bonds,
        unbond=_flatten_specs(args.unbond) or None,
        bond=_flatten_specs(args.bond) or None,
        no_hy=args.no_hy,
        orient=_orient,
        opacity=args.opacity,
        label_font_size=args.label_size,
        vdw_opacity=args.vdw_opacity,
        vdw_scale=args.vdw_scale,
        atom_gradient_strength=args.atom_gradient_strength,
        bond_gradient_strength=args.bond_gradient_strength,
        vdw_gradient_strength=args.vdw_gradient_strength,
        ts_bonds=_parse_pairs(args.ts_bond),
        nci_bonds=_parse_pairs(args.nci_bond),
        vdw_indices=None,
        show_indices=args.idx is not None,
        idx_format=args.idx or "sn",
        cmap_range=tuple(args.cmap_range) if args.cmap_range else None,
        cmap_palette=args.cmap_palette,
        cbar=args.cbar,
        cmap_symm=args.cmap_symm,
    )

    if args.haptic:
        cfg.haptic = True
    if args.skeletal_label_color is not None:
        cfg.skeletal_label_color = args.skeletal_label_color

    # Highlight atoms (multi-group) — convert argparse lists to tuples for render()
    _highlight: list[tuple[str, ...]] | None = None
    if args.hl is not None:
        for entry in args.hl:
            if len(entry) > 2:
                p.error(f"--hl takes 1-2 arguments (ATOMS [COLOR]), got {len(entry)}")
        _highlight = [tuple(e) for e in args.hl]

    # Bond coloring
    if args.bond_by_element is not None:
        cfg.bond_color_by_element = args.bond_by_element
    elif args.bond_color is not None:
        # --bond-color implies uniform colour unless --bond-by-element is explicit
        cfg.bond_color_by_element = False
    if args.bond_gradient is not None:
        cfg.bond_gradient = args.bond_gradient

    # Depth of field
    if args.dof:
        cfg.dof = True
    if args.dof_strength is not None:
        cfg.dof_strength = args.dof_strength

    # Per-atom radius scale
    if args.radius_scale is not None:
        cfg.radius_scale = [(sel, float(fac)) for sel, fac in args.radius_scale]

    # Output path defaults and validation
    base = _basename(args.input, from_stdin)
    if not args.output:
        args.output = f"{base}.svg"

    static_ext = args.output.rsplit(".", 1)[-1].lower()
    if static_ext not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join("." + e for e in sorted(_SUPPORTED_EXTENSIONS))
        p.error(f"Unsupported static output format: .{static_ext} (use {supported})")

    wants_gif = args.gif_ts or args.gif_rot or args.gif_trj or args.gif_diffuse

    if args.gif_diffuse and (args.gif_ts or args.gif_trj):
        p.error("--gif-diffuse cannot be combined with --gif-ts or --gif-trj")

    # --diffuse-rot without --gif-rot implies y-axis rotation
    if args.gif_diffuse and args.diffuse_rot is not None and not args.gif_rot:
        args.gif_rot = "y"

    # Warn when SVG-only flags are combined with GIF output
    annotation_flags_used = args.idx is not None or args.label_specs or args.label
    if annotation_flags_used and wants_gif:
        logger.warning("--idx, -l and --label apply to static SVG output only and will not appear in the GIF")

    is_cube = args.input and args.input.endswith(".cube")

    # Ensemble overlay is only defined for multi-frame XYZ / QM trajectories
    if args.ensemble and not args.input:
        p.error("--ensemble requires an input trajectory file")
    if args.ensemble and (args.overlay or args.overlay_color):
        p.error("--ensemble cannot be combined with --overlay / --overlay-color")
    if args.ensemble and (args.gif_ts or args.gif_trj):
        p.error("--ensemble cannot be combined with --gif-ts or --gif-trj (use gif_rot only)")
    if args.ensemble and from_stdin:
        p.error("--ensemble cannot be used with stdin input")

    # Validate --smi / --mol-frame / --rebuild usage
    if args.smi and args.input:
        p.error("--smi cannot be combined with a positional input file")
    if args.mol_frame != 0 and not (args.input and args.input.endswith(".sdf")):
        logger.warning("--mol-frame has no effect on non-SDF input")
    if args.rebuild and args.smi:
        logger.warning("--rebuild has no effect on SMILES input (rdkit bonds are always used)")

    # Supercell repetition counts (validated fully after loading so we can
    # ensure the input actually has a unit cell / lattice).
    _supercell = tuple(args.supercell) if args.supercell is not None else (1, 1, 1)
    if any(v < 1 for v in _supercell):
        p.error(f"--supercell values must be >= 1 (got {_supercell})")

    if wants_gif:
        gif_path = args.gif_output or f"{base}.gif"
        gif_ext = gif_path.rsplit(".", 1)[-1].lower()
        if gif_ext != "gif":
            p.error(f"GIF output must have .gif extension, got: .{gif_ext}")

    # --- Load molecule ---
    needs_ts = args.ts_detect or args.gif_ts
    if is_cube and needs_ts:
        logger.warning(
            "--ts/--gif-ts has no effect with cube files (single geometry, no frequency data). "
            "Use --ts-bond to manually specify TS bonds."
        )

    if args.smi:
        mol = load(
            args.smi,
            smiles=True,
            charge=args.charge,
            multiplicity=args.multiplicity,
            kekule=args.kekule,
        )
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
                cell=args.cell,
                quick=args.bo is False,
                bohr=True if args.bohr else None,
            )
        except ValueError as e:
            p.error(str(e))

    # Resolve atom-spec driven options now that atom symbols are available.
    from xyzrender.selectors import resolve_atom_indices

    if args.vdw is not None:
        if args.vdw == "":
            cfg.vdw_indices = []
        else:
            cfg.vdw_indices = sorted(resolve_atom_indices(args.vdw, mol.graph))

    # Style regions (user-defined + preset-defined from JSON)
    from xyzrender.api import _apply_style_regions

    _apply_style_regions(cfg, mol.graph, regions=args.region)

    # Resolve hull argument string → type for render().
    # Actual detection happens inside render() after crystal images / supercell.
    _hull_arg: bool | str | list[int] | list[list[int]] | None = None
    if args.hull is not None:
        if args.hull in (["rings"], ["ring"]):
            _hull_arg = "rings"
        elif args.hull in (["faces"], ["face"]):
            _hull_arg = "faces"
        elif not args.hull:
            _hull_arg = True
        else:
            # Must be numeric atom indices — validate before parsing.
            _valid_modes = ("rings", "ring", "faces", "face")
            for g in args.hull:
                if g in _valid_modes:
                    p.error(f"--hull {g}: did you mean '--hull {g}'? (no other arguments alongside mode names)")
                if not any(c.isdigit() for c in g):
                    p.error(f"--hull {g!r}: expected 'faces', 'rings', or atom indices (e.g. '1-6 7-12')")
            _hull_arg = [parse_atom_indices(g, one_indexed=True) for g in args.hull]

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

    # --- Stereochemistry labels ---
    if args.stereo is not None:
        from xyzrender.stereo import STEREO_CLASSES, build_stereo_annotations

        stereo_cls: set[str] | None = None
        if args.stereo:  # non-empty string → parse classes
            stereo_cls = {c.strip() for c in args.stereo.split(",")}
            bad = stereo_cls - STEREO_CLASSES
            if bad:
                p.error(
                    f"Unknown stereo class(es): {', '.join(sorted(bad))}. Valid: {', '.join(sorted(STEREO_CLASSES))}"
                )

        try:
            cfg.annotations.extend(build_stereo_annotations(mol.graph, rs_style=args.stereo_style, classes=stereo_cls))
        except ValueError as e:
            p.error(str(e))

    # --- Atom property colormap (store on cfg) ---
    if args.cmap:
        from xyzrender.annotations import load_cmap

        try:
            cfg.atom_cmap = load_cmap(args.cmap, mol.graph)
        except (ValueError, FileNotFoundError) as e:
            p.error(str(e))

    # --- Parse align-atoms (comma-separated 1-indexed, e.g. "1,2,3" or "1-6") ---
    _align_atoms: list[int] | None = None
    if args.align_atoms is not None:
        _align_atoms = parse_atom_indices(args.align_atoms, one_indexed=True)

    _anchor_atoms: list[int] | None = None
    if args.anchor:
        _anchor_atoms = parse_atom_indices(args.anchor, one_indexed=True)

    # --- Parse ensemble color: palette name, single color, or comma-separated list ---
    _ens_color: str | list[str] | None = None
    _ens_palette: str | None = None
    if args.ensemble_color is not None:
        from xyzrender.colors import PALETTE_NAMES

        val = args.ensemble_color.strip()
        if val in PALETTE_NAMES:
            _ens_palette = val
        else:
            parts = [c.strip() for c in val.split(",")]
            _ens_color = parts if len(parts) > 1 else parts[0]

    # --- Interactive viewer (operates on the reference frame only) ---
    if args.interactive:
        if args.ref is not None and Path(args.ref).is_file():
            logger.warning(
                "--ref %s already exists — skipping interactive viewer (reference orientation will be used)",
                args.ref,
            )
        else:
            orient(mol)
            if not mol.oriented:
                sys.exit(1)

    # --- Ensemble: load all frames, align onto (possibly oriented) reference ---
    if args.ensemble:
        mol = load(
            args.input,
            ensemble=True,
            align_atoms=_align_atoms,
            ensemble_color=_ens_color,
            ensemble_palette=_ens_palette,
            ensemble_opacity=args.opacity,
            rebuild=args.rebuild,
            nci_detect=args.nci_detect,
            charge=args.charge,
            multiplicity=args.multiplicity,
            kekule=args.kekule,
            reference_mol=mol,
        )

    # --- Crystal ghost resolution ---
    # Ghosts default: on whenever the molecule carries cell_data (auto-detected or explicit)
    _show_ghosts = args.ghosts if args.ghosts is not None else mol.cell_data is not None

    # Validate supercell usage: allowed for any input that has a valid lattice.
    if _supercell != (1, 1, 1):
        if mol.cell_data is None:
            p.error("--supercell requires an input with a unit cell (lattice).")
        lat = getattr(mol.cell_data, "lattice", None)
        if lat is None:
            p.error("--supercell requires an input with a unit cell (lattice).")
        import numpy as np

        lat = np.array(lat, dtype=float)
        if lat.shape != (3, 3) or np.allclose(lat, 0.0):
            p.error("--supercell requires a non-zero 3x3 lattice matrix.")

    # --- Bond rules: apply once to mol.graph so render() and render_gif() both
    #     receive the resolved graph; their internal guards then become no-ops. ---
    if cfg.unbond or cfg.bond or cfg.haptic:
        from xyzrender.bond_rules import apply_bond_rules

        apply_bond_rules(mol.graph, cfg)
        cfg.unbond = []
        cfg.bond = []
        cfg.haptic = False

    # --- Render static SVG ---
    try:
        render(
            mol,
            config=cfg,
            mol_color=args.mol_color,
            highlight=_highlight,
            no_cell=args.no_cell,
            axes=args.axes,
            axis=args.axis,
            supercell=_supercell,
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
            nci_mode=args.nci_mode,
            surface_style=args.surface_style,
            opacity=args.opacity,
            overlay=args.overlay,
            overlay_color=args.overlay_color,
            align_atoms=_align_atoms,
            vector=args.vector,
            vector_scale=args.vector_scale,
            hull=_hull_arg,
            hull_color=args.hull_color,
            hull_opacity=args.hull_opacity,
            hull_edge=args.hull_edge,
            hull_edge_width_ratio=args.hull_edge_width_ratio,
            hull_color_type=args.hull_color_type,
            pore=args.pore,
            ring_max_size=args.ring_max_size,
            ring_min_size=args.ring_min_size,
            face_planarity=args.face_planarity,
            pore_color=args.pore_color,
            pore_opacity=args.pore_opacity,
            bo=args.bo,
            output=args.output,
            ref=args.ref,
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

        if args.ensemble and args.gif_rot:
            # Ensemble rotation GIF: use the pre-built ensemble Molecule.
            mol_or_path = mol
        else:
            mol_or_path: str | Molecule = args.input if (args.gif_ts or args.gif_trj) else mol
        # For gif_ts/gif_trj the trajectory is read from disk (mol_or_path is a path),
        # but pass mol.graph as reference_graph so -I orientation and TS/NCI bonds are respected.
        _ref_graph = mol.graph if (args.gif_ts or args.gif_trj) else None
        try:
            render_gif(
                mol_or_path,
                config=cfg,
                mol_color=args.mol_color,
                highlight=_highlight,
                gif_rot=args.gif_rot or None,
                gif_trj=args.gif_trj,
                gif_ts=args.gif_ts,
                gif_diffuse=args.gif_diffuse,
                diffuse_frames=args.diffuse_frames,
                diffuse_noise=args.diffuse_noise,
                diffuse_bonds=args.diffuse_bonds,
                diffuse_rot=args.diffuse_rot,
                diffuse_reverse=not args.diffuse_forward,
                anchor=_anchor_atoms,
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
                surface_style=args.surface_style,
                no_cell=args.no_cell,
                axes=args.axes,
                axis=args.axis,
                supercell=_supercell,
                ghosts=_show_ghosts,
                cell_color=args.cell_color,
                cell_width=args.cell_width,
                ghost_opacity=args.ghost_opacity,
                vector=args.vector,
                vector_scale=args.vector_scale,
                ref=args.ref,
                hull=_hull_arg,
                hull_color=args.hull_color,
                hull_opacity=args.hull_opacity,
                hull_edge=args.hull_edge,
                hull_edge_width_ratio=args.hull_edge_width_ratio,
                hull_color_type=args.hull_color_type,
                pore=args.pore,
                ring_max_size=args.ring_max_size,
                ring_min_size=args.ring_min_size,
                face_planarity=args.face_planarity,
                pore_color=args.pore_color,
                pore_opacity=args.pore_opacity,
            )
        except ValueError as e:
            p.error(str(e))


if __name__ == "__main__":
    main()
