# CLI Reference

Full flag reference for `xyzrender`. See also `xyzrender --help`.

## Input / Output

| Flag | Description |
|------|-------------|
| `-o`, `--output` | Static output path (`.svg`, `.png`, `.pdf`) |
| `--smi SMILES` | Embed a SMILES string into 3D (requires rdkit) |
| `--mol-frame N` | Record index in multi-molecule SDF (default: 0) |
| `--rebuild` | Ignore file connectivity; re-detect bonds with xyzgraph |
| `-c`, `--charge` | Molecular charge |
| `-m`, `--multiplicity` | Spin multiplicity |
| `--config` | Config preset (`default`, `flat`, `paton`, `skeletal`) or path to JSON file |
| `-d`, `--debug` | Debug logging |

## Styling

| Flag | Description |
|------|-------------|
| `-S`, `--canvas-size` | Canvas size in px (default: 800) |
| `-a`, `--atom-scale` | Atom radius scale factor |
| `-b`, `--bond-width` | Bond stroke width |
| `-s`, `--atom-stroke-width` | Atom outline stroke width |
| `--bond-color` | Bond color (hex or named) |
| `-B`, `--background` | Background color |
| `-t`, `--transparent` | Transparent background |
| `-G`, `--gradient-strength` | Gradient contrast multiplier |
| `--grad` / `--no-grad` | Radial gradient toggle |
| `-F`, `--fog-strength` | Depth fog strength |
| `--fog` / `--no-fog` | Depth fog toggle |
| `--bo` / `--no-bo` | Bond order rendering toggle |

## Display

| Flag | Description |
|------|-------------|
| `--hy` | Show H atoms (no args = all, or specify 1-indexed atom numbers) |
| `--no-hy` | Hide all H atoms |
| `-k`, `--kekule` | Use Kekulé bond orders (no aromatic 1.5) |
| `--vdw` | vdW spheres (no args = all, or index ranges e.g. `1-6`) |
| `--vdw-opacity` | vdW sphere opacity (default: 0.25) |
| `--vdw-scale` | vdW sphere radius scale |
| `--vdw-gradient` | vdW sphere gradient strength |

## Structural overlay

| Flag | Description |
|------|-------------|
| `--overlay FILE` | Second structure to overlay (RMSD-aligned onto the primary) |
| `--overlay-color COLOR` | Color for the overlay structure (hex or named) |

## Orientation

| Flag | Description |
|------|-------------|
| `-I`, `--interactive` | Interactive rotation via `v` viewer |
| `--orient` / `--no-orient` | Auto-orientation toggle |

## TS / NCI

| Flag | Description |
|------|-------------|
| `--ts` | Auto-detect TS bonds via graphRC |
| `--ts-frame` | TS reference frame (0-indexed) |
| `--ts-bond` | Manual TS bond pair(s) (1-indexed, e.g. `1-2`) |
| `--nci` | Auto-detect NCI interactions |
| `--nci-bond` | Manual NCI bond pair(s) (1-indexed) |

## Surfaces

| Flag | Description |
|------|-------------|
| `--mo` | Render MO lobes from `.cube` input |
| `--mo-colors POS NEG` | MO lobe colors (hex or named) |
| `--mo-blur SIGMA` | MO Gaussian blur sigma (default: 0.8, ADVANCED) |
| `--mo-upsample N` | MO contour upsample factor (default: 3, ADVANCED) |
| `--flat-mo` | Render all MO lobes as front-facing (no depth classification) |
| `--dens` | Render density isosurface from `.cube` input |
| `--dens-color` | Density surface color (default: `steelblue`) |
| `--esp CUBE` | ESP cube file for potential coloring (implies `--dens`) |
| `--nci-surf CUBE` | NCI gradient (RDG) cube — render NCI surface lobes |
| `--nci-coloring MODE` | NCI coloring: `avg` (default), `pixel`, `uniform` |
| `--nci-color COLOR` | NCI lobe color for `uniform` mode (default: `forestgreen`) |
| `--iso` | Isosurface threshold (MO default: 0.05, density/ESP: 0.001, NCI: 0.3) |
| `--opacity` | Surface opacity multiplier (default: 1.0) |

## Annotations

| Flag | Description |
|------|-------------|
| `--measure [TYPE...]` | Print bond measurements to stdout (`d`, `a`, `t`; combine or omit for all) |
| `--idx [FMT]` | Atom index labels in SVG (`sn` = C1, `s` = C, `n` = 1) |
| `-l TOKEN...` | Inline SVG annotation (repeatable); 1-based indices |
| `--label FILE` | Bulk annotation file (same syntax as `-l`) |
| `--label-size PT` | Label font size (overrides preset) |
| `--cmap FILE` | Per-atom property colormap (Viridis, 1-indexed) |
| `--cmap-range VMIN VMAX` | Explicit colormap range (default: auto from file) |

## Vector arrows

| Flag | Description |
|------|-------------|
| `--vector FILE` | Path to a JSON file defining 3D vector arrows for overlay |
| `--vector-scale` | Global length multiplier for all vector arrows |

## GIF animations

| Flag | Description |
|------|-------------|
| `--gif-rot [AXIS]` | Rotation GIF (default axis: `y`). Combinable with `--gif-ts` |
| `--gif-ts` | TS vibration GIF (via graphRC) |
| `--gif-trj` | Trajectory / optimisation GIF (multi-frame input) |
| `-go`, `--gif-output` | GIF output path (default: `{basename}.gif`) |
| `--gif-fps` | Frames per second (default: 10) |
| `--rot-frames` | Rotation frame count (default: 120) |

Available rotation axes: `x`, `y`, `z`, `xy`, `xz`, `yz`, `yx`, `zx`, `zy`. Prefix `-` to reverse (e.g. `-xy`). For crystal inputs, a 3-digit Miller index string is also accepted (e.g. `111`, `001`).

## Convex hull

| Flag | Description |
|------|-------------|
| `--hull [INDICES ...]` | Draw convex hull (no args = all heavy atoms; `rings` = auto-detect aromatic rings; or 1-indexed subsets e.g. `1-6` or `1-6 7-12`) |
| `--hull-color COLOR [...]` | Hull fill color(s) (hex or named, one per subset) |
| `--hull-opacity FLOAT` | Hull fill opacity (0-1) |
| `--hull-edge` / `--no-hull-edge` | Draw/hide non-bond hull edges (default: on) |
| `--hull-edge-width-ratio FLOAT` | Hull edge stroke width as fraction of bond width (default: 0.4) |

## Crystal / unit cell

| Flag | Description |
|------|-------------|
| `--cell` | Draw unit cell box from `Lattice=` in extXYZ (usually auto-detected) |
| `--cell-color` | Cell edge color (hex or named, default: `gray`) |
| `--cell-width` | Unit cell box line width (default: 2.0) |
| `--crystal [{vasp,qe}]` | Load as crystal via `phonopy`; format auto-detected or explicit |
| `--no-cell` | Hide the unit cell box |
| `--ghosts` / `--no-ghosts` | Show/hide ghost (periodic image) atoms outside the cell |
| `--ghost-opacity` | Opacity of ghost atoms/bonds (default: 0.5) |
| `--axes` / `--no-axes` | Show/hide the a/b/c axis arrows |
| `--axis HKL` | Orient looking down a crystallographic direction (e.g. `111`, `001`) |
