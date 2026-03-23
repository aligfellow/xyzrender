# xyzrender: Publication-quality molecular graphics.

Render molecular structures as publication-quality SVG, PNG, PDF, and animated GIF from XYZ, mol/SDF, MOL2, PDB, SMILES, CIF, cube files, or quantum chemistry output — from the command line or from Python/Jupyter.

[![PyPI Downloads](https://static.pepy.tech/badge/xyzrender)](https://pepy.tech/projects/xyzrender)
[![License](https://img.shields.io/github/license/aligfellow/xyzrender)](https://github.com/aligfellow/xyzrender/blob/main/LICENSE)
[![Powered by: uv](https://img.shields.io/badge/-uv-purple)](https://docs.astral.sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/aligfellow/xyzrender/ci.yml?branch=main&logo=github-actions)](https://github.com/aligfellow/xyzrender/actions)
[![Codecov](https://img.shields.io/codecov/c/github/aligfellow/xyzrender)](https://codecov.io/gh/aligfellow/xyzrender)
[![Documentation](https://readthedocs.org/projects/xyzrender/badge/?version=latest)](https://xyzrender.readthedocs.io/en/latest/)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue?logo=readthedocs)](https://xyzrender.readthedocs.io)

xyzrender turns XYZ files and quantum chemistry input/output (MOl, MOL2, SDF, PDB, ORCA, Gaussian, Q-Chem, etc.) into clean SVG, PNG, PDF, and animated GIF graphics — ready for papers, presentations, and supporting information. The SVG rendering approach is built on and inspired by [**xyz2svg**](https://github.com/briling/xyz2svg) by [Ksenia Briling **@briling**](https://github.com/briling).

Most molecular visualisation tools require manual setup: loading files into a GUI, tweaking camera angles, exporting at the right resolution and adding specific TS or NCI bonds. `xyzrender` skips this. One command gives you a (mostly) oriented, depth-cued structure with correct bond orders, aromatic ring rendering, automatic bond connectivity, with TS bonds and NCI bonds. Orientation control is available through an interface to [**v**](https://github.com/briling/v) by [Ksenia Briling **@briling**](https://github.com/briling).

![TS bimp full nci](examples/images/bimp_nci_ts.gif)

**What it handles out of the box:**

- **Bond orders and aromaticity** — double bonds, triple bonds, and aromatic ring notation detected automatically from geometry via [`xyzgraph`](https://github.com/aligfellow/xyzgraph)
- **Transition state bonds** — forming/breaking bonds rendered as dashed lines, detected automatically from imaginary frequency vibrations via [`graphRC`](https://github.com/aligfellow/graphRC)
- **Stereochemistry labels** — R/S, E/Z, axial, planar (metallocene and CIP), and helical chirality labels detected and annotated automatically via [`xyzgraph`](https://github.com/aligfellow/xyzgraph)
- **Non-covalent interactions** — hydrogen bonds and other weak interactions shown as dotted lines, detected automatically via [`xyzgraph`](https://github.com/aligfellow/xyzgraph)
- **GIF animations** — rotation, TS vibration, and trajectory animations for presentations
- **Molecular orbitals** — render MO lobes from cube files with front/back depth cueing
- **Electron density surfaces** — depth-graded translucent isosurfaces from density cube files
- **Electrostatic potential (ESP)** — ESP colormapped onto the density surface from paired cube files
- **vdW surface overlays** — van der Waals spheres on all or selected atoms
- **Structural overlay** — RMSD-align two conformers and render in contrasting colours
- **Conformer ensemble** — overlay all frames from a multi-frame XYZ trajectory, with palette colouring and opacity control
- **Convex hull** — semi-transparent facets over selected atoms (e.g. aromatic ring carbons, coordination spheres); optional hull-edge lines
- **Depth fog and gradients** — 3D depth cues without needing a 3D viewer
- **Cheminformatics formats** — mol, SDF, MOL2, PDB (with CRYST1 unit cell), SMILES (3D embedding via rdkit), and CIF (via ase) — bond connectivity read directly from file
- **Crystal / periodic structures** — render periodic structures with unit cell box, ghost atoms, and crystallographic axis arrows (a/b/c); extXYZ `Lattice=` auto-detected; VASP/QE via [`phonopy`](https://github.com/phonopy/phonopy)
- **Multiple output formats** — SVG (default), PNG, PDF, and GIF from the same command

**Preconfigured but extensible.** Built-in presets (`default`, `flat`, `paton`, `skeletal`, `bubble`, `tube`, `wire`) cover common use cases. Every setting — colors, radii, bond widths, gradients, fog — can be overridden via CLI flags or a custom JSON config file.

```bash
xyzrender caffeine.xyz                          # SVG with sensible defaults
xyzrender ts.out --ts -o figure.png             # TS with dashed bonds as PNG
xyzrender caffeine.xyz --gif-rot -go movie.gif  # rotation GIF for slides
```

See web app by [@BNNLab](https://github.com/bnnlab) [**xyzrender-web.streamlit.app**](https://xyzrender-web.streamlit.app/).

## Installation

```bash
pip install xyzrender
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install xyzrender
```

To test without installing, you can use [uvx](https://docs.astral.sh/uv/guides/tools/#running-tools)

```bash
uvx xyzrender 
```

### From Source:

Using pip: 

```bash
git clone https://github.com/aligfellow/xyzrender.git
cd xyzrender
pip install .
# install in editable mode
pip install -e .
# or straight from git
pip install git+https://github.com/aligfellow/xyzrender.git
```
For more information on installation and optional dependencies (crystal, SMILES, CIF, GIF), see the [installation docs](https://xyzrender.readthedocs.io/en/latest/installation.html)

## Quick start

```bash
xyzrender caffeine.xyz                                    # render XYZ → SVG
xyzrender calc.out                                        # QM output (ORCA, Gaussian, etc.)
xyzrender caffeine.xyz -o render.png                      # explicit output path/format
xyzrender caffeine.xyz --config paton --hy -o styled.svg  # preset + show hydrogens
xyzrender sn2.out --ts --hy -o ts.svg                     # auto-detect TS bonds
xyzrender caffeine.xyz --gif-rot -go caffeine.gif         # rotation GIF
```

### Python API

```python
from xyzrender import load, render, render_gif

mol = load("caffeine.xyz")
render(mol)                          # displays inline in Jupyter
render(mol, output="caffeine.svg")   # save as SVG/PNG/PDF

render(mol, config="paton", hy=True) # all CLI flags as kwargs
render_gif(mol, gif_rot="y")         # rotation GIF
```

For the full Python API (render options, `build_config()`, `measure()`, `load()` kwargs, return types), see the [Python API guide](https://xyzrender.readthedocs.io/en/latest/python_api.html) or the runnable [`examples/examples.ipynb`](examples/examples.ipynb) notebook.

## Feature gallery

### Presets

| Default | Flat | Paton (pymol-like) | Skeletal | 
|---------|------|--------------------|----------|
| ![default](examples/images/caffeine_default.svg) | ![flat](examples/images/caffeine_flat.svg) | ![paton](examples/images/caffeine_paton.svg) | ![skeletal](examples/images/caffeine_skeletal.svg) | 

| Bubble | Tube | Wire |
|--------|------|------|
| ![bubble](examples/images/caffeine_bubble.svg) | ![tube](examples/images/caffeine_tube.svg) | ![wire](examples/images/caffeine_wire.svg) |

### Style regions

| Tube + ball-stick region | Tube + ball-stick, NCI, vdW |
|--------------------------|------------------------|
| ![region](examples/images/caffeine_region.svg) | ![bimp regions](examples/images/bimp_regions.svg) |

### Display options

| All H | Some H | No H | Aromatic | Kekule |
|-------|--------|------|----------|--------|
| ![all H](examples/images/ethanol_all_h.svg) | ![some H](examples/images/ethanol_some_h.svg) | ![no H](examples/images/ethanol_no_h.svg) | ![benzene](examples/images/benzene.svg) | ![kekule](examples/images/caffeine_kekule.svg) |

### vdW spheres

| All atoms | Partial | Paton-style |
|-----------|---------|-------------|
| ![vdw](examples/images/asparagine_vdw.svg) | ![vdw partial](examples/images/asparagine_vdw_partial.svg) | ![vdw paton](examples/images/asparagine_vdw_paton.svg) |

### Convex hull

| Benzene ring | Anthracene rings | Auto rings | Rotation |
|--------------|------------------|------------|----------|
| ![benzene hull](examples/images/benzene_ring_hull.svg) | ![anthracene hull](examples/images/anthracene_hull.svg) | ![mnh hull](examples/images/mnh_hull_rings.svg) | ![anthracene rot](examples/images/anthracene_hull.gif) |

### Highlight & molecule color

| Default (orchid) | Custom colour | Multi-group | Mol color + highlight |
|------------------|---------------|-------------|-----------------------|
| ![hl](examples/images/caffeine_hl.svg) | ![hl custom](examples/images/caffeine_hl_custom.svg) | ![multi hl](examples/images/caffeine_multi_hl.svg) | ![mol color hl](examples/images/caffeine_mol_color_hl_idx.svg) |

### Depth of field

| DoF | Rotation |
|-----|----------| 
| ![dof](examples/images/caffeine_dof.svg) | ![dof](examples/images/caffeine_dof.gif) |

### Structural overlay & ensemble

| Overlay | Custom colour | Ensemble (CPK) | Ensemble (spectral) |
|---------|---------------|----------------|---------------------|
| ![overlay](examples/images/isothio_overlay.svg) | ![overlay custom](examples/images/isothio_overlay_custom.svg) | ![ensemble](examples/images/triphenylbenzol_ensemble.svg) | ![ensemble custom](examples/images/triphenylbenzol_ensemble_custom.svg) |

### Transition states & NCI

| Auto TS | Manual TS | Auto NCI | TS + NCI custom colours | QM output |
|---------|-----------|----------|-------------------------|-----------|
| ![ts](examples/images/sn2_ts.svg) | ![ts man](examples/images/sn2_ts_man.svg) | ![nci](examples/images/nci.svg) | ![ts nci custom](examples/images/bimp_ts_nci_custom.svg) | ![bimp](examples/images/bimp_qm.svg) |

### Annotations & labels

| Distances + angles + dihedrals | Custom labels | TS with labels |
|--------------------|--------|----------------|
| ![dihedral](examples/images/caffeine_dihedral.svg) | ![labels](examples/images/caffeine_labels.svg) | ![sn2 labels](examples/images/sn2_ts_label.svg) |

### Stereochemistry labels

| Isothiocyanate (R/S, E/Z, planar)                      | TS with stereo (Mn-H₂, `--ts --stereo`)                    |
| ------------------------------------------------------- | ----------------------------------------------------------- |
| ![isothio stereo](examples/images/isothio_stereo.svg)   | ![mn-h2 ts stereo](examples/images/mn-h2_ts_stereo.svg)    |

### Atom property colormap

| Mulliken charges (rotation) | Symmetric range | With colorbar |
|----------------------------|----------------|---------------|
| ![cmap gif](examples/images/caffeine_cmap.gif) | ![cmap](examples/images/caffeine_cmap.svg) | ![cbar](examples/images/caffeine_cmap_colorbar.svg) |

### Surfaces (cube files)

| MO (HOMO) | MO (LUMO) | Density | ESP |
|-----------|-----------|---------|-----|
| ![homo](examples/images/caffeine_homo.svg) | ![lumo](examples/images/caffeine_lumo.svg) | ![dens](examples/images/caffeine_dens.svg) | ![esp](examples/images/caffeine_esp.svg) |

| NCI surface (H-bond) | NCI surface (pi-stack) | Vector arrows | 
|-----------------------|------------------------|---------------|
| ![nci surf](examples/images/base-pair-nci_surf.svg) | ![nci pi](examples/images/phenol_di-nci_surf.svg) | ![vectors](examples/images/ethanol_dip.svg) | 
### File formats

| PDB | SMILES |
|-----|--------|
| ![PDB](examples/images/ala_phe_ala.svg) | ![smiles](examples/images/cyclohexane_smi.svg) |

### Crystal / periodic structures

| Unit cell | Rotation | VASP | Supercell 2×2×1 | Viewing direction |
|-----------|----------|------|-----------------|-------------------|
| ![cell](examples/images/caffeine_cell.svg) | ![cell rot](examples/images/caffeine_cell.gif) | ![vasp](examples/images/NV63_vasp.svg) | ![supercell](examples/images/NV63_cell_supercell_221.svg) | ![111](examples/images/NV63_111.gif) |

### GIF animations

| Rotation | TS + NCI + vdW + rotation | Trajectory | TS |
|----------|---------------------------|------------|------------------|
| ![rotate](examples/images/caffeine.gif) | ![ts rot](examples/images/bimp_nci_ts.gif) | ![trj](examples/images/bimp_trj.gif) | ![ts](examples/images/mn-h2.gif) |

| Overlay rotation | MO | Density | 
|----------|---------------------------|------------|
| ![overlay gif](examples/images/isothio_overlay.gif) | ![homo](examples/images/caffeine_homo.gif) | ![dens](examples/images/caffeine_dens.gif) | 

| Vectors | Diffuse / assembly |
|-----|--------------------|
| ![vectors](examples/images/ethanol_forces_efield.gif) | ![diffuse](examples/images/caffeine_diffuse.gif) |

For usage details and CLI commands, see the [examples](https://xyzrender.readthedocs.io/en/latest/examples.html) and [CLI reference](https://xyzrender.readthedocs.io/en/latest/cli_reference.html) in the docs.

## Documentation

Full documentation at [**xyzrender.readthedocs.io**](https://xyzrender.readthedocs.io):

- [Installation](https://xyzrender.readthedocs.io/en/latest/installation.html) — PyPI, uv, source, optional dependencies
- [CLI Quickstart](https://xyzrender.readthedocs.io/en/latest/quickstart_cli.html) — getting started from the command line
- [Python API Guide](https://xyzrender.readthedocs.io/en/latest/python_api.html) — render options, `build_config()`, `measure()`, return types
- [Examples](https://xyzrender.readthedocs.io/en/latest/examples.html) — presets, overlays, surfaces, crystal, annotations, and more
- [Configuration](https://xyzrender.readthedocs.io/en/latest/configuration.html) — presets, custom JSON, styling flags
- [CLI Reference](https://xyzrender.readthedocs.io/en/latest/cli_reference.html) — all flags
- [Input Formats](https://xyzrender.readthedocs.io/en/latest/formats.html) — XYZ, QM output, SDF, PDB, SMILES, CIF, cube files
- [API Reference](https://xyzrender.readthedocs.io/en/latest/reference.html) — auto-generated from docstrings

## License

[MIT](LICENSE)

## Acknowledgements

The SVG rendering in xyzrender is built on and heavily inspired by [**xyz2svg**](https://github.com/briling/xyz2svg). The CPK colour scheme, core SVG atom/bond rendering logic, fog, and overall approach originate from that project.  
- [Ksenia Briling (@briling)](https://github.com/briling) — [**xyz2svg**](https://github.com/briling/xyz2svg) and [**v**](https://github.com/briling/v)
- [Iñigo Iribarren Aguirre (@iribirii)](https://github.com/iribirii) — radial gradient (pseudo-3D) rendering from [**xyz2svg**](https://github.com/briling/xyz2svg).

Key dependencies:

- [**xyzgraph**](https://github.com/aligfellow/xyzgraph) — bond connectivity, bond orders, aromaticity detection and non-covalent interactions from molecular geometry
- [**graphRC**](https://github.com/aligfellow/graphRC) — reaction coordinate analysis and TS bond detection from imaginary frequency vibrations
- [**cclib**](https://github.com/cclib/cclib) — parsing quantum chemistry output files (ORCA, Gaussian, Q-Chem, etc.)
- [**CairoSVG**](https://github.com/Kozea/CairoSVG) — SVG to PNG/PDF conversion
- [**Pillow**](https://github.com/python-pillow/Pillow) — GIF frame assembly
- [**resvg-py**](https://github.com/nicmr/resvg-py) — SVG to PNG conversion preserving SVG effects

Falls back to CairoSVG automatically (filters silently ignored). SVG output always contains the filters regardless.

Optional dependencies:

- [**phonopy**](https://github.com/phonopy/phonopy) — crystal structure loading (`pip install 'xyzrender[crystal]'`)
- [**rdkit**](https://www.rdkit.org/) — SMILES 3D embedding (`pip install 'xyzrender[smiles]'`)
- [**ase**](https://wiki.fysik.dtu.dk/ase/) — CIF parsing (`pip install 'xyzrender[cif]'`)
- [**v**](https://github.com/briling/v) — interactive molecule orientation (`pip install xyzrender[v]`, Linux only, not included into `[all]`)

Contributors:

- [Ksenia Briling (@briling)](https://github.com/briling) — `vmol` integration and the [xyz2svg](https://github.com/briling/xyz2svg) foundation
- [Sander Cohen-Janes (@scohenjanes5)](https://github.com/scohenjanes5) — crystal/periodic structure support (VASP, Quantum ESPRESSO, ghost atoms, crystallographic axes), vector annotations and gif parallelisation
- [Rubén Laplaza (@rlaplaza)](https://github.com/rlaplaza) — convex hull facets
- [Iñigo Iribarren Aguirre (@iribirii)](https://github.com/iribirii) — radial gradients respecting colour space (pseudo-3D), skeletal rendering, ensemble display, supercell projection
- [James O'Brien (@JamesOBrien2)](https://github.com/JamesOBrien2) — stereochemistry detection and integration, nci and ts colour control
- [Vinicius Port (@caprilesport)](https://github.com/caprilesport) — `v` binary path discovery
- [Lucas Attia (@lucasattia)](https://github.com/lucasattia) — `--transparent` background flag

## Citation

xyzrender uses [xyzgraph](https://github.com/aligfellow/xyzgraph) and [graphRC](https://github.com/aligfellow/graphRC) for all molecular graph construction — bond orders, aromaticity detection, NCI interactions, and TS bond detection. If you use xyzrender in published work, please cite:

> A.S. Goodfellow* and B.N. Nguyen, *J. Chem. Theory Comput.*, 2026, DOI: [10.1021/acs.jctc.5c02073](https://doi.org/10.1021/acs.jctc.5c02073). Preprint [here](https://doi.org/10.26434/chemrxiv-2025-k69gt).

### BibTeX

```bibtex
@article{goodfellow2026xyzgraph,
  author  = {Goodfellow, A.S. and Nguyen, B.N.},
  title   = {Graph-Based Internal Coordinate Analysis for Transition State Characterization},
  journal = {J. Chem. Theory Comput.},
  year    = {2026},
  doi     = {10.1021/acs.jctc.5c02073},
}
```

## Development

<details>
<summary>Information on dev setup and CI</summary>

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).


```bash
git clone https://github.com/aligfellow/xyzrender.git
cd xyzrender
just setup   # install dev dependencies
just check   # lint + type-check + tests
```

| Command | Description |
|---|---|
| `just check` | Run lint + type-check + tests |
| `just lint` | Format and lint with ruff |
| `just type` | Type-check with ty |
| `just test` | Run pytest with coverage |
| `just fix` | Auto-fix lint issues |
| `just build` | Build distribution |
| `just setup` | Install all dev dependencies |

### CI

GitHub Actions runs lint, type-check, and tests on every push to `main` and every PR targeting `main`. Coverage is uploaded to [Codecov](https://codecov.io).

</details>

## Template
Generated from [aligfellow/python-template](https://github.com/aligfellow/python-template).

<details>
<summary>Updating from the template</summary>

If this project was created with [copier](https://copier.readthedocs.io/), you can pull in upstream template improvements:

```bash
# Run from the project root
copier update --trust
```

This will:

1. Fetch the latest version of the template
2. Re-ask any questions whose defaults have changed
3. Re-render the templated files with your existing answers
4. Apply the changes as a diff — your project-specific edits are preserved via a three-way merge

If there are conflicts (e.g. you modified the `justfile` and so did the template), copier will leave standard merge conflict markers (`<<<<<<<` / `>>>>>>>`) for you to resolve manually.

The `--trust` flag is required because the template defines tasks (used for `git init` on first copy). The tasks don't run during update, but copier requires trust for any template that declares them.

Requires that the project was originally created with `copier copy`, not the plain GitHub "Use this template" button.

</details>
