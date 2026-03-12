# Examples

Worked examples covering the full range of xyzrender's features, with rendered outputs and the commands that produced them. Sample structures are in `examples/structures/`. To regenerate all outputs:

```bash
uv run bash examples/generate.sh
```

- [Basics](examples/basics.md) — hydrogen display, bond orders, aromatic notation, vdW spheres, and presets
- [Structural Overlay](examples/overlay.md) — RMSD-aligned conformer comparison
- [Animations](examples/animations.md) — rotation GIFs, TS vibration, trajectory animations, and combined options
- [Transition States and NCI](examples/ts_nci.md) — transition state bonds and non-covalent interactions, auto-detected or manual
- [Molecular Orbitals](examples/mo.md) — molecular orbital lobes from cube files
- [Electron Density and ESP](examples/dens_esp.md) — electron density isosurfaces and ESP colormapping
- [NCI Surface](examples/nci_surf.md) — NCI surface patches from NCIPLOT cube files
- [Convex hull](examples/hull.md) — semi-transparent hull facets over selected atoms (rings, coordination spheres)
- [Crystal Structures](examples/crystal.md) — unit cell rendering, VASP/QE periodic structures, and crystallographic axes
- [Annotations](examples/annotations.md) — atom indices, SVG labels, measurements, and vector arrows
- [Atom Colormap](examples/cmap.md) — per-atom scalar colormaps (charges, shifts, Fukui indices)

```{toctree}
:maxdepth: 1
:hidden:

examples/basics
examples/overlay
examples/animations
examples/ts_nci
examples/mo
examples/dens_esp
examples/nci_surf
examples/hull
examples/crystal
examples/annotations
examples/cmap
```
