```{image} _static/branding/logo_big.svg
:alt: xyzrender
:width: 720px
:align: center
```

# xyzrender documentation

Publication-quality molecular graphics from XYZ, cube, QM output, and more — as SVG, PNG, PDF, or animated GIF.

```{figure} ../../examples/images/bimp_nci_ts.gif
:width: 600
:alt: xyzrender NCI transition state example

Transition state animation with NCI surface, vdW spheres, and TS bonds — rendered with a single command.
```

Simple CLI input:

```bash
xyzrender bimp.out --gif-ts --gif-rot --nci --vdw 84-169
```

```{toctree}
:maxdepth: 1
:caption: Getting Started
:hidden:

installation
quickstart_cli
quickstart_python
```

```{toctree}
:maxdepth: 1
:caption: User Guide
:hidden:

formats
configuration
orientation
python_api
```

```{toctree}
:maxdepth: 1
:caption: Examples — Styling
:hidden:

examples/basics
examples/highlight
examples/style_regions
examples/cmap
```

```{toctree}
:maxdepth: 1
:caption: Examples — Surfaces & analysis
:hidden:

examples/mo
examples/dens_esp
examples/nci_surf
examples/ts_nci
```

```{toctree}
:maxdepth: 1
:caption: Examples — Geometry tools
:hidden:

examples/overlay
examples/ensemble
examples/hull
examples/crystal
```

```{toctree}
:maxdepth: 1
:caption: Examples — Annotations
:hidden:

examples/annotations
```

```{toctree}
:maxdepth: 1
:caption: Examples — Output
:hidden:

examples/animations
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

cli_reference
api/core
api/types
api/config
```

```{toctree}
:maxdepth: 1
:caption: About
:hidden:

citation
acknowledgements
```
