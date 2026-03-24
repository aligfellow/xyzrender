# NCI Surface

```{note}
Surface plots are schematic 2D representations suitable for figures. For quantitative isosurface analysis, use a dedicated 3D viewer (VMD, PyMOL).
```

Visualise non-covalent interaction (NCI) regions from two [NCIPlot](https://github.com/juliacontrerasgarcia/NCIPLOT-4.2) cube files: `sign(λ₂)·ρ` density (main input) and reduced density gradient (`--nci-surf`).

The surface is rendered as individual flat-filled patches per interaction region. Coloring is based on the sign of `λ₂` weighted by density: **blue** = strong attractive (H-bond), **green** = weak/vdW, **red** = repulsive (steric).

| H-bond (base pair) | π-stacking (phenol dimer) |
|-------------------|--------------------------|
| ![H-bond (base pair)](../../../examples/images/base-pair-nci_surf.svg) | ![π-stacking (phenol dimer)](../../../examples/images/phenol_di-nci_surf.svg) |

```bash
# avg coloring (default): blue=H-bond, green=vdW, red=steric
xyzrender base-pair-dens.cube --nci-surf base-pair-grad.cube -o base-pair-nci_surf.svg
xyzrender phenol_di-dens.cube --nci-surf phenol_di-grad.cube -o phenol_di-nci_surf.svg

# per-pixel (more detail)
xyzrender base-pair-dens.cube --nci-surf base-pair-grad.cube --nci-mode pixel

# flat color (default: forestgreen)
xyzrender base-pair-dens.cube --nci-surf base-pair-grad.cube --nci-mode uniform

# flat color with custom colour
xyzrender base-pair-dens.cube --nci-surf base-pair-grad.cube --nci-mode teal
```

Coloring modes (`--nci-mode`):

| Mode | Description |
|------|-------------|
| `avg` (default) | Each NCI lobe filled with its mean `sign(λ₂)·ρ`: **blue** = H-bond, **green** = vdW, **red** = steric |
| `pixel` | Per-pixel `sign(λ₂)·ρ` raster — shows intra-lobe variation (not a very nice render styling at the moment) |
| `uniform` | Flat single color for all NCI regions (default: `forestgreen`) |
| *colour* | Any colour name or hex — shorthand for uniform mode with that colour |

All NCI surface flags:

| Flag | Description |
|------|-------------|
| `--nci-surf GRAD_CUBE` | Reduced density gradient cube file (enables NCI surface rendering) |
| `--nci-mode MODE` | Coloring: `avg` (default), `pixel`, `uniform`, or a colour name/hex |
| `--iso` | RDG isovalue threshold (default: 0.5 — larger value = more surface) |
| `--opacity` | Surface opacity multiplier (default: 1.0) |
| `--nci-cutoff CUTOFF` | Density magnitude cutoff (advanced — not needed for standard NCIPLOT output) |

Sample structures from [NCIPlot](https://github.com/juliacontrerasgarcia/NCIPLOT-4.2/tree/master/tests).
