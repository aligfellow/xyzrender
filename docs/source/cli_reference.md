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
| `--config` | Config preset (`default`, `flat`, `paton`, `pmol`, `skeletal`, `tube`, `mtube`, `wire`, `graph`) or path to JSON file |
| `-d`, `--debug` | Debug logging |

## Styling

| Flag | Description |
|------|-------------|
| `-S`, `--canvas-size` | Canvas size in px (default: 800) |
| `-a`, `--atom-scale` | Atom radius scale factor |
| `-b`, `--bond-width` | Bond stroke width |
| `-s`, `--atom-stroke-width` | Atom outline stroke width |
| `--bond-color` | Bond color (hex or named) |
| `--bond-outline-color` | Bond edge stroke color (default: black) |
| `--bond-outline-width` | Bond edge stroke width in px (0 = off) |
| `--no-bonds` | Hide all bonds (e.g. space-filling style) |
| `--unbond SPEC [...]` | Hide bonds by rule or index*. `all` / `*` hides every covalent bond. |
| `--bond PAIR [...]` | Force-show/add bonds: 1-indexed pairs (`1-3 4-5`). Overrides `--unbond` |
| `--haptic` | Replace pi-coordination bond fans with single centroid bonds (dotted). Inherits overlay / ensemble colour from the metal atom. |
| `--atom-opacity ATOMS VALUE` | Per-atom fill opacity (repeatable), e.g. `--atom-opacity "1-5" 0.3`. Affects the atom circle only; adjacent bonds stay fully opaque. |
| `-B`, `--background` | Background color |
| `-t`, `--transparent` | Transparent background |
| `--grad` / `--no-grad` | Radial gradient toggle |
| `--atom-gradient-strength` | Atom gradient strength (default: 1.0) |
| `--bond-gradient` / `--no-bond-gradient` | Cylinder shading on bonds (3D tube look) |
| `--bond-gradient-strength` | Bond cylinder gradient strength (default: 0.3) |
| `-F`, `--fog-strength` | Depth fog strength |
| `--fog` / `--no-fog` | Depth fog toggle |
| `--bo` / `--no-bo` | Bond order rendering toggle |
| `--bond-by-element` / `--no-bond-by-element` | Color bonds by endpoint atom colors |
| `--radius-scale ATOMS FACTOR` | Scale selected atoms (repeatable). Multiplies on top of `-a` |
| `--region ATOMS CONFIG` | Render atom subset with a different style (repeatable). Selectors: `"1-5"`, `"Pt"`, `"M"` (metals), `"sbm"` (s-block), `"het"` (heteroatoms) |

*- categories (`M-L`, `sbm`, `Fe-het`), pi-coordination (`M-pi`, `pi`), element (`Li`), atom index (`2`), or pair (`1-3`). Comma or space separated

## Display

| Flag | Description |
|------|-------------|
| `--hy [ATOMS]` | Show H atoms (no args = all, or indices like `"1-5,8"`) |
| `--no-hy` | Hide all H atoms |
| `-k`, `--kekule` | Use Kekulé bond orders (no aromatic 1.5) |
| `--vdw` | vdW spheres (no args = all, or selectors like `"1-6"`, `"M"`, `"Pt"`) |
| `--vdw-opacity` | vdW sphere opacity (default: 0.25) |
| `--vdw-scale` | vdW sphere radius scale |
| `--vdw-gradient-strength` | vdW sphere gradient strength (default: 1.6) |
| `--mol-color COLOR` | Flat color for all atoms and bonds (overrides CPK). Highlight paints on top |
| `--hl ATOMS [COLOR]` | Highlight atom group: `--hl "1-5,8" [color]`. Can be repeated for multiple groups. Auto-colors from palette if no color given |
| `--dof` | Depth-of-field blur (does not affect bonds/lines) |
| `--dof-strength FLOAT` | DoF max blur strength (default: 3.0) |

## Structural overlay / ensemble

| Flag | Description |
|------|-------------|
| `--overlay FILE` | Second structure to overlay (RMSD-aligned onto the primary). Different atom counts are handled automatically via shared-scaffold alignment |
| `--overlay-color COLOR` | Overlay atom colour (bonds rendered 30 % darker automatically) |
| `--opacity FLOAT` | Transparency 0–1. Applied to the overlay when `--overlay` is set, to the ensemble when `--ensemble` is set, else to surfaces |
| `--overlay-atom-scale FLOAT` | Absolute atom radius scale for the overlay only (mirrors `--atom-scale`) |
| `--overlay-bond-width FLOAT` | Absolute bond width for the overlay only |
| *(preset JSON / Python only)* | `atom_stroke_width`, `atom_stroke_color`, `bond_color`, `bond_outline_width`, `bond_outline_color` — set inside the `overlay` block of a preset JSON or on `OverlayConfig` to fine-tune styling without CLI flag bloat |
| `--overlay-unbond SPEC [...]` | Hide bonds on the overlay only (same grammar as `--unbond`; applied pre-merge so indices are overlay-local) |
| `--overlay-bond PAIR [...]` | Force-show / add bonds on the overlay only (1-indexed, overlay-local) |
| `--align` / `--no-align` | Force / skip Kabsch/MCS alignment for `--overlay` and `--ensemble`. Default: on. `--align` is useful to override a preset with `auto_align: false`; `--no-align` keeps each structure's raw coordinates (interactive `-I` rotation of the base still propagates to the overlay) |
| `--ensemble` | Ensemble overlay for multi-frame XYZ trajectories; conformers default to CPK atom colours |
| `--ensemble-color VALUE` | Palette name (`viridis`, `plasma`, `spectral`, `coolwarm`, `RdBu`, `rainbow`), a single colour, or comma-separated colours |
| `--align-atoms INDICES` | Atom subset for Kabsch alignment (min 3), e.g. `1,2,3`, `1-6`. Works with `--overlay` and `--ensemble` |

## Orientation

| Flag | Description |
|------|-------------|
| `-I`, `--interactive` | Interactive rotation (see `--viewer`) |
| `--viewer {vmol,ase}` | Viewer backend for `-I`: `vmol` (default, requires vmol) or `ase` (requires ase). Close the ASE window to confirm; press `z` then `q` in vmol. |
| `--orient` / `--no-orient` | Auto-orientation toggle |
| `--ref [FILE]` | Save/load orientation reference (`reference.xyz` by default) |

## TS / NCI

| Flag | Description |
|------|-------------|
| `--ts` | Auto-detect TS bonds via graphRC |
| `--ts-frame` | TS reference frame (0-indexed) |
| `--ts-bond` | Manual TS bond pair(s) (1-indexed, e.g. `1-2`) |
| `--ts-color` | Color for dashed TS bonds (hex or named) |
| `--nci` | Auto-detect NCI interactions |
| `--nci-bond` | Manual NCI bond pair(s) (1-indexed) |
| `--nci-color` | Color for dotted NCI bonds (hex or named) |

## Surfaces

| Flag | Description |
|------|-------------|
| `--mo` | Render MO lobes from `.cube` or `.cub` input |
| `--mo-colors POS NEG` | MO lobe colors (hex or named) |
| `--mo-blur SIGMA` | MO Gaussian blur sigma (default: 0.8, ADVANCED) |
| `--mo-upsample N` | MO contour upsample factor (default: 3, ADVANCED) |
| `--flat-mo` | Render all MO lobes as front-facing (no depth classification) |
| `--dens` | Render density isosurface from `.cube` or `.cub` input |
| `--dens-color` | Density surface color (default: `steelblue`) |
| `--esp CUBE` | ESP cube file for potential coloring (implies `--dens`) |
| `--nci-surf CUBE` | NCI gradient (RDG) cube — render NCI surface lobes |
| `--nci-mode MODE` | NCI surface coloring: `avg` (default), `pixel`, `uniform`, or a colour name/hex |
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
| `--stereo [CLASSES]` | Stereochemistry labels from 3D geometry. Optional comma-separated class filter: `point`, `ez`, `axis`, `plane`, `helix`. Omit to show all |
| `--stereo-style STYLE` | R/S label placement: `atom` (centered, default) or `label` (offset) |
| `--cmap FILE` | Per-atom property colormap (1-indexed atom index, value) |
| `--cmap-range VMIN VMAX` | Explicit colormap range (default: auto from file) |
| `--cmap-symm` | Symmetric colormap range about zero: `[-max(|v|), +max(|v|)]` |
| `--cmap-palette NAME` | Shared scalar palette override (`viridis` for `--cmap`, `rainbow` for `--esp`) |
| `--cbar` | Add a vertical colorbar on the right for `--cmap` or `--esp` |

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
| `--no-cell` | Hide the unit cell box |
| `--ghosts` / `--no-ghosts` | Show/hide ghost (periodic image) atoms outside the cell |
| `--ghost-opacity` | Opacity of ghost atoms/bonds (default: 0.5) |
| `--axes` / `--no-axes` | Show/hide the a/b/c axis arrows |
| `--axis HKL` | Orient looking down a crystallographic direction (e.g. `111`, `001`) |
| `--supercell M N L` | Repeat the unit cell `M×N×L` times along a/b/c (requires lattice/unit-cell data; default: `1 1 1`) |
