# CLI Reference

Full flag reference for `xyzrender`. See also `xyzrender --help`.

## Input / Output

| Flag | Description |
|------|-------------|
| `INPUT` | Input file (`.xyz`, `.mol`, `.sdf`, `.mol2`, `.pdb`, `.smi`, `.cif`, `.cube`/`.cub`, `.com`, `.gjf`, `.inp`, `.nw`, `.vasp`, `POSCAR`, `.in`, `.fdf`, `.abi`, `.coord`, QM `.out` / `.log` …) — omit to read from stdin |
| `-o`, `--output` | Static output path (`.svg`, `.png`, `.pdf`); defaults to `{basename}.svg` |
| `--smi SMILES` | Embed a SMILES string into 3D (requires rdkit: `pip install 'xyzrender[smi]'`) |
| `--mol-frame N` | Record index in multi-molecule SDF (default: 0) |
| `--rebuild` | Ignore file connectivity; re-detect bonds with xyzgraph |
| `-c`, `--charge` | Molecular charge |
| `-m`, `--multiplicity` | Spin multiplicity |
| `--bohr` | Input coordinates are in Bohr (force conversion to Angstrom) |
| `--config` | Config preset (`default`, `flat`, `paton`, `pmol`, `skeletal`, `bubble`, `vdw`, `tube`, `mtube`, `btube`, `wire`, `graph`) or path to JSON file |
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
| `--unbond SPEC [...]` | Hide bonds by rule or index*. `all` / `*` hides every covalent bond |
| `--bond PAIR [...]` | Force-show/add bonds: 1-indexed pairs (`1-3 4-5`). Overrides `--unbond` |
| `--haptic` | Replace pi-coordination bond fans with single centroid bonds (dotted). Inherits overlay / ensemble colour from the metal atom |
| `--atom-opacity ATOMS VALUE` | Per-atom fill opacity (repeatable), e.g. `--atom-opacity "1-5" 0.3`. Affects the atom circle only; adjacent bonds stay fully opaque |
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
| `-Hls`, `--hue-shift-factor` | Hue gradient contrast (advanced; default per preset) |
| `-hLs`, `--light-shift-factor` | Lightness gradient contrast (advanced) |
| `-hlS`, `--saturation-shift-factor` | Saturation gradient contrast (advanced) |
| `--skeletal-label-color` | Override all element label colours in `--config skeletal` |

*— categories (`M-L`, `sbm`, `Fe-het`), pi-coordination (`M-pi`, `pi`), element (`Li`), atom index (`2`), or pair (`1-3`). Comma or space separated.

## Display

| Flag | Description |
|------|-------------|
| `--hy [ATOMS]` | Show H atoms (no args = all, or indices like `"1-5,8"`) |
| `--no-hy` | Hide all H atoms |
| `--only ATOMS` | Render only selected atoms before orientation/canvas fitting. Repeatable; same selector grammar as `--hl`, e.g. `"1-24"`, `"C,N,O"`, `"M"`. Cube/surface fields are not cropped |
| `--exclude ATOMS` | Remove selected atoms before orientation/canvas fitting. Repeatable; same selector grammar as `--hl`, e.g. `"25-40"`, `"Na,Cl"`. Cube/surface fields are not cropped |
| `-k`, `--kekule` | Use Kekulé bond orders (no aromatic 1.5) |
| `--vdw [ATOMS]` | vdW spheres (no args = all, or selectors like `"1-6"`, `"M"`, `"Pt"`) |
| `--vdw-opacity` | vdW sphere opacity (default: 0.25) |
| `--vdw-scale` | vdW sphere radius scale |
| `--vdw-gradient-strength` | vdW sphere gradient strength (default: 1.6) |
| `--vdw-interlocking` / `--no-vdw-interlocking` | Render the `--vdw` overlay as interlocked silhouettes (default: on) |
| `--atom-interlocking` / `--no-atom-interlocking` | Render primary atom spheres as interlocked silhouettes; the `vdw` preset turns this on for space-filling renders |
| `--vdw-outline-width FLOAT` | vdW overlay outline width (default: 0 = no outline) |
| `--vdw-outline-color COLOR` | vdW overlay outline colour |
| `--h-scale FLOAT` | H-atom radius scale (primary atoms; default: 0.6) |
| `--vdw-h-scale FLOAT` | H-atom radius scale on the `--vdw` overlay (default: 0.7) |
| `--mol-color COLOR` | Flat color for all atoms and bonds (overrides CPK). Highlight paints on top |
| `--hl ATOMS [COLOR]` | Highlight atom group. Accepts the same selectors as `--only`/`--region` — indices/ranges (`"1-5,8"`), elements (`"C,N"`), or categories (`"M"`, `"het"`). Repeatable; auto-colours from palette if no colour given |
| `--dof` | Depth-of-field blur (does not affect bonds/lines) |
| `--dof-strength FLOAT` | DoF max blur strength (default: 3.0) |
| `--glow ATOMS` | Add blurred glow under selected atoms (same selector grammar as `--hl` / `--vdw`) |
| `--glow-strength FLOAT` | Glow blur strength (default: 5.0) |

## Orientation

| Flag | Description |
|------|-------------|
| `-I`, `--interactive` | Interactive rotation (see `--viewer`) |
| `--viewer {vmol,ase}` | Viewer backend for `-I`: `vmol` (default, requires vmol) or `ase` (requires ase). Close the ASE window to confirm; press `z` then `q` in vmol |
| `--orient` / `--no-orient` | Auto-orientation toggle |
| `--ref [FILE]` | Save/load orientation reference (`reference.xyz` by default) |

## Transition states / NCI graph overlay

These flags draw graph-detected bond overlays (dashes / dots) on top of the molecular skeleton. The volumetric NCI surface (`--nci-surf`) is documented in the Surfaces section below.

| Flag | Description |
|------|-------------|
| `--ts` | Auto-detect TS bonds via graphRC |
| `--ts-frame` | TS reference frame (0-indexed) |
| `--ts-bond PAIR` | Manual TS bond pair(s) (1-indexed, e.g. `"1-2"`) |
| `--ts-color` | Colour for dashed TS bonds (hex or named); overrides `--ts-element` |
| `--ts-element` / `--no-ts-element` | Atom-coloured halves on TS dashes |
| `--ts-dash LEN,GAP` | TS dash length,gap as bond-width multiples (default `1.2,2.2`) |
| `--ts-width MULT` | TS line width as a bond-width multiple (default `1.2`) |
| `--nci` | Auto-detect NCI interactions (graph topology, not the volumetric surface — see `--nci-surf`) |
| `--nci-bond PAIR` | Manual NCI bond pair(s) (1-indexed) |
| `--nci-color` | Colour for dotted NCI/haptic bonds (hex or named); overrides `--nci-element` |
| `--nci-element` / `--no-nci-element` | Atom-coloured halves on NCI/haptic dots (on in pmol/btube/tube/mtube) |
| `--nci-dash LEN,GAP` | NCI/haptic dot length,gap as bond-width multiples (default `0.08,2.0`) |
| `--nci-width MULT` | NCI/haptic line width as a bond-width multiple (default `1.0`) |

## Surfaces

| Flag | Description |
|------|-------------|
| `--mo` | Render MO lobes from `.cube` or `.cub` input |
| `--mo-colors POS NEG` | MO lobe colors (hex or named) |
| `--mo-outline-width [PX]` | Outline stroke per lobe (solid style only; pair with `--opacity 1.0` for crisp edges). Bare flag = 5.0; 0 = off |
| `--mo-outline-color COLOR` | Outline color (default: black) |
| `--mo-blur SIGMA` | MO Gaussian blur sigma (default: 0.8, advanced) |
| `--mo-upsample N` | MO contour upsample factor (default: 3, advanced) |
| `--flat-mo` | Disable depth-fog colour blend on MO lobes (textbook flat colours) |
| `--dens` | Render density isosurface from `.cube` or `.cub` input |
| `--dens-color` | Density surface color (default: `steelblue`) |
| `--esp CUBE` | ESP cube file for potential coloring (implies `--dens`) |
| `--nci-surf CUBE` | Interaction surface cube — auto-classified as `low_field` (NCIPLOT RDG) or `high_field` (Multiwfn IGMH δg) |
| `--nci-mode MODE` | NCI surface coloring: `avg` (default), `pixel`, `uniform`, or a colour name/hex |
| `--iso` | Isosurface threshold. Defaults: MO `0.05`, density/ESP `0.001`. Interaction surfaces are *starting points* — tune per cube: low-field RDG `0.3`; IGMH `dg_inter` `0.005`, `dg_intra` `0.05`–`0.3` |
| `--opacity` | Surface opacity multiplier (default: 1.0; >1 boosts) |
| `--surface-style STYLE` | Surface rendering style: `solid` (default), `mesh`, `contour`, `dot`. Density falls back to `contour` for `mesh`; ESP raster ignores style |

## Convex hull / faces / pore

| Flag | Description |
|------|-------------|
| `--hull [INDICES ...]` | Draw convex hull (no args = all heavy atoms; `rings` = auto aromatic rings; `faces` = 2D structural faces; or 1-indexed subsets e.g. `1-6` or `1-6 7-12`) |
| `--hull-color COLOR [...]` | Hull fill color(s) (hex or named, one per subset) |
| `--hull-opacity FLOAT` | Hull fill opacity (0–1) |
| `--hull-color-type {type,size,env}` | Ring colouring: `type` (atom types + size, default), `size` (size only), `env` (type + ring fusion) |
| `--hull-edge` / `--no-hull-edge` | Draw/hide non-bond hull edges (default: on) |
| `--hull-edge-width-ratio FLOAT` | Hull edge stroke width as fraction of bond width (default: 0.4) |
| `--ring-min-size N` | Minimum ring size for `--hull faces` / `--pore` (default: 3) |
| `--ring-max-size N` | Maximum ring size for `--hull faces` / `--pore` (default: 100) |
| `--face-planarity FLOAT` | Planarity tolerance for 3D face detection: 0 = strict, 1 = permissive (default: 0.25) |
| `--pore` | Detect pore cavities and draw inscribed sphere(s) |
| `--pore-color` | Pore sphere colour (default: warm yellow) |
| `--pore-opacity FLOAT` | Pore sphere opacity (default: 0.5) |

## Structural overlay / ensemble

| Flag | Description |
|------|-------------|
| `--overlay FILE` | Second structure to overlay (RMSD-aligned onto the primary). Different atom counts handled via shared-scaffold alignment |
| `--overlay-color COLOR` | Overlay atom colour (bonds auto-darkened 30 %) |
| `--opacity FLOAT` | Transparency 0–1. Applied to the overlay when `--overlay` is set, to the ensemble when `--ensemble` is set, else to surfaces |
| `--overlay-atom-scale FLOAT` | Absolute atom radius scale for the overlay only (mirrors `--atom-scale`) |
| `--overlay-bond-width FLOAT` | Absolute bond width for the overlay only |
| `--overlay-unbond SPEC [...]` | Hide bonds on the overlay only (same grammar as `--unbond`; indices are overlay-local) |
| `--overlay-bond PAIR [...]` | Force-show / add bonds on the overlay only (1-indexed, overlay-local) |
| `--overlay-show SPEC [...]` | Render only these overlay atoms (same selector grammar as `--hl`). Applied after alignment so the fit still uses the full scaffold |
| `--overlay-ts` | Run graphRC TS detection on the overlay (mirrors `--ts`; soft-fails if overlay has no freq data) |
| `--overlay-ts-bond PAIR` | Manual TS bond pair(s) on the overlay, 1-indexed in the overlay's atom list |
| `--align` / `--no-align` | Force / skip Kabsch/MCS alignment for `--overlay` and `--ensemble`. Default: on |
| `--ensemble` | Ensemble overlay for multi-frame XYZ trajectories; conformers default to CPK atom colours |
| `--ensemble-color VALUE` | Palette name (`viridis`, `plasma`, `spectral`, `coolwarm`, `RdBu`, `rainbow`, `batlow`, `roma`, `vik`, `bam`, `managua`), a single colour, or comma-separated colours |
| `--align-atoms SELECTOR` | Alignment candidates for `--overlay` and `--ensemble` (min 3 atoms). 1-indexed IDs (`1,2,3`, `1-6`), element symbols (`Fe,P,Cl`), categories (`M`, `L`, `het`, `sbm`), or a mix (`1-5,Fe`). Metal-containing specs pivot on the metal so paired metals coincide; otherwise MCS + K-subset Kabsch picks the lowest-RMSD candidate |

Fine-grained overlay style (`atom_stroke_width`, `atom_stroke_color`, `bond_color`, `bond_outline_*`) lives in preset JSON or `OverlayConfig` only — not exposed as CLI flags.

## Annotations

| Flag | Description |
|------|-------------|
| `--measure [TYPE...]` | Print bond measurements to stdout (`d`, `a`, `t`; combine or omit for all) |
| `--idx [FMT]` | Atom index labels in SVG (`sn` = C1, default; `s` = C; `n` = 1) |
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
| `--vector-scale FACTOR` | Global length multiplier for all vector arrows |

## GIF animations

| Flag | Description |
|------|-------------|
| `--gif-rot [AXIS]` | Rotation GIF (default axis: `y`). Combinable with `--gif-ts` and `--gif-bounce` |
| `--gif-bounce DEG[,AXIS]` | Bounce rotation: starts at the original orientation, then rotates to `+DEG`, back through `0`, and to `-DEG` on the rotation axis (`y` by default). Append `,AXIS` to override |
| `--gif-ts` | TS vibration GIF (via graphRC) |
| `--gif-trj` | Trajectory / optimisation GIF (multi-frame input) |
| `--trj-bonds` | Re-detect bonds for every frame (NEB-TS MEPs and other trajectories with changing connectivity) |
| `--gif-diffuse` | Diffuse / assembly GIF — atoms scatter and reassemble |
| `--diffuse-frames N` | Number of diffuse frames (default: 60) |
| `--diffuse-noise FLOAT` | Per-frame random walk noise (default: 0.3) |
| `--diffuse-bonds {fade,show,hide}` | Bond visibility during diffuse (default: `fade`) |
| `--diffuse-rot [DEG]` | Add partial rotation during diffuse (default: 180°) |
| `--diffuse-forward` | Play forward (molecule → noise) instead of assembly |
| `--anchor ATOMS` | Atoms that stay fixed during `--gif-diffuse`, e.g. `"1-5,8"` |
| `-go`, `--gif-output` | GIF output path (default: `{basename}.gif`) |
| `--gif-fps` | Frames per second (default: 10) |
| `--rot-frames` | Rotation frame count (default: 120) |
| `--vib-frames` | Vibration frames for `--gif-ts` (default: 20) |

Available rotation axes: `x`, `y`, `z`, `xy`, `xz`, `yz`, `yx`, `zx`, `zy`. Prefix `-` to reverse (e.g. `-xy`). For crystal inputs, a 3-digit Miller index string is also accepted (e.g. `111`, `001`).

## Crystal / unit cell

| Flag | Description |
|------|-------------|
| `--cell` | Draw unit cell box from `Lattice=` in extXYZ (usually auto-detected) |
| `--cell-color` | Cell edge color (hex or named, default: `#333333`) |
| `--cell-width` | Unit cell box line width (default: 1.5) |
| `--no-cell` | Hide the unit cell box |
| `--ghosts` / `--no-ghosts` | Show/hide ghost (periodic image) atoms outside the cell |
| `--ghost-opacity` | Opacity of ghost atoms/bonds (default: 0.5) |
| `--axes` / `--no-axes` | Show/hide the a/b/c axis arrows |
| `--axis HKL` | Orient looking down a crystallographic direction (e.g. `111`, `001`) |
| `--supercell M N L` | Repeat the unit cell `M×N×L` times along a/b/c (requires lattice/unit-cell data; default: `1 1 1`) |
