# Acknowledgements

The SVG rendering in xyzrender is built on and heavily inspired by [**xyz2svg**](https://github.com/briling/xyz2svg). The CPK colour scheme, core SVG atom/bond rendering logic, fog, and overall approach originate from that project.

- [Ksenia Briling (@briling)](https://github.com/briling) — [**xyz2svg**](https://github.com/briling/xyz2svg) and [**v**](https://github.com/briling/v)
- [Iñigo Iribarren Aguirre (@iribirii)](https://github.com/iribirii) — radial gradient (pseudo-3D) rendering from [**xyz2svg**](https://github.com/briling/xyz2svg)

The `paton` colour preset is inspired by the clean styling used by [Rob Paton](https://github.com/patonlab) through PyMOL ([gist](https://gist.github.com/bobbypaton/1cdc4784f3fc8374467bae5eb410edef)).

The interlocked-spheres rendering used by `--config vdw` and the `--vdw` overlay is adapted from [**CineMol**](https://github.com/moltools/CineMol) by David Meijer.

- D. Meijer, M.H. Medema and J.J.J. van der Hooft, *J. Cheminform.*, 2024, **16**, 58 ([DOI](https://doi.org/10.1186/s13321-024-00851-y)).

NCI surface example structures from [NCIPlot](https://github.com/juliacontrerasgarcia/NCIPLOT-4.2/tree/master/tests).

## Key dependencies

- [**xyzgraph**](https://github.com/aligfellow/xyzgraph) — bond connectivity, bond orders, aromaticity detection and non-covalent interactions from molecular geometry
- [**graphRC**](https://github.com/aligfellow/graphRC) — reaction coordinate analysis and TS bond detection from imaginary frequency vibrations
- [**cclib**](https://github.com/cclib/cclib) — parsing quantum chemistry output files (ORCA, Gaussian, Q-Chem, etc.)
- [**CairoSVG**](https://github.com/Kozea/CairoSVG) — SVG to PNG/PDF conversion
- [**Pillow**](https://github.com/python-pillow/Pillow) — GIF frame assembly
- [**resvg-py**](https://github.com/nicmr/resvg-py) — SVG to PNG conversion preserving SVG effects

Falls back to CairoSVG automatically (filters silently ignored). SVG output always contains the filters regardless.

## Optional dependencies

- [**rdkit**](https://www.rdkit.org/) — SMILES 3D embedding (`pip install 'xyzrender[smi]'`)
- [**ase**](https://wiki.fysik.dtu.dk/ase/) — CIF parsing and ASE GUI viewer (`pip install 'xyzrender[cif]'`)
- [**v**](https://github.com/briling/v) — interactive molecule orientation (`-I` flag, `pip install xyzrender[v]`, Linux only, not included in `[all]`)

## Contributors

- [Ksenia Briling (@briling)](https://github.com/briling) — `vmol` integration and the [xyz2svg](https://github.com/briling/xyz2svg) foundation
- [Sander Cohen-Janes (@scohenjanes5)](https://github.com/scohenjanes5) — crystal/periodic structure support (VASP, Quantum ESPRESSO, ghost atoms, crystallographic axes), vector annotations and gif parallelisation, gaussian input parsing
- [Rubén Laplaza (@rlaplaza)](https://github.com/rlaplaza) — convex hull facets
- [Iñigo Iribarren Aguirre (@iribirii)](https://github.com/iribirii) — logo design, radial gradients respecting colour space (pseudo-3D), skeletal rendering, ensemble display, supercell projection, metal tube preset
- [James O'Brien (@JamesOBrien2)](https://github.com/JamesOBrien2) — stereochemistry detection and integration, nci/ts colour control, graph styling, pmol styling, colour palette extension, ase viewer integration, igmh cubes
- [Vinicius Port (@caprilesport)](https://github.com/caprilesport) — `v` binary path discovery
- [Lucas Attia (@lucasattia)](https://github.com/lucasattia) — transparent background

## License

[MIT](https://github.com/aligfellow/xyzrender/blob/main/LICENSE)
