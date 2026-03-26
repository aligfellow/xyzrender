# Input Formats

xyzrender reads bond connectivity directly from file where available (mol, SDF, MOL2, PDB, SMILES, CIF). The parser is selected by file extension.

## XYZ

Standard XYZ files:

```bash
xyzrender molecule.xyz
```

extXYZ (with `Lattice=` header) is handled automatically — the unit cell box, ghost atoms, and axis arrows are enabled without any extra flags. See [Crystal Structures](examples/crystal.md).

## QM Input Files

Render directly from computational chemistry input files. Coordinates and charge/multiplicity are extracted automatically.

```bash
xyzrender calc.com              # Gaussian
xyzrender calc.gjf              # Gaussian
xyzrender calc.inp              # ORCA / CP2K / GAMESS
xyzrender calc.nw               # NWChem
xyzrender calc.in               # Q-Chem (or QE / ABINIT — auto-detected)
xyzrender calc.fdf              # SIESTA
xyzrender calc.abi              # ABINIT
xyzrender calc.coord            # Turbomole (Bohr auto-detected)
```

For Turbomole and other codes that use Bohr units, conversion is auto-detected. Use `--bohr` to force conversion:

```bash
xyzrender calc.coord --bohr
```

Charge and multiplicity are extracted from the input file where possible (ORCA `* xyz C M`, Gaussian charge/mult line, NWChem `charge` directive, Q-Chem `$molecule`, Psi4 `molecule {}`). Override with `-c` / `-m`.

If coordinates are in an external file (e.g. ORCA `* xyzfile 0 1 mol.xyz`), the referenced file is read automatically.

## QM Output

ORCA (`.out`), Gaussian (`.log`), Q-Chem (`.out`) — format is auto-detected from file content via [cclib](https://cclib.github.io/):

```bash
xyzrender calc.out
xyzrender calc.log
```

Use `--charge` and `--multiplicity` if needed for bond detection:

```bash
xyzrender calc.out -c -1 -m 2
```

See [Transition States and NCI](examples/ts_nci.md) for transition state rendering from QM output.

## Cheminformatics formats

```bash
xyzrender molecule.sdf       # SDF — bonds from file
xyzrender molecule.mol       # mol — bonds from file
xyzrender molecule.mol2      # MOL2 — Tripos aromatic bonds
xyzrender structure.pdb      # PDB — ATOM/HETATM + CONECT records
```

**PDB with CRYST1:** if the PDB contains a `CRYST1` record, the unit cell is parsed and crystal rendering is used automatically.

**Multi-record SDF:** use `--mol-frame N` to select a record (default: 0):

```bash
xyzrender multi.sdf --mol-frame 1
```

## SMILES

Requires `pip install 'xyzrender[smi]'` (rdkit). Embeds a SMILES string into 3D using ETKDGv3 + MMFF94.

```bash
xyzrender --smi "C1CCCCC1" --hy -o cyclohexane.svg
```

An XYZ file of the optimised 3D geometry is automatically saved alongside the rendered image (e.g. `cyclohexane.xyz`).

## CIF

Requires `pip install 'xyzrender[cif]'` (ase):

```bash
xyzrender structure.cif
```

## Cube files

Cube files contain both molecular geometry and a 3D volumetric grid. Used for molecular orbitals ([Molecular Orbitals](examples/mo.md)), electron density and ESP ([Electron Density and ESP](examples/dens_esp.md)), and NCI surfaces ([NCI Surface](examples/nci_surf.md)).

```bash
xyzrender homo.cube --mo
xyzrender dens.cube --dens
xyzrender dens.cube --esp esp.cube
xyzrender dens.cube --nci-surf grad.cube
```

## Periodic structures

VASP, Quantum ESPRESSO, SIESTA, ABINIT, and CP2K periodic input files are auto-detected — the unit cell, ghost atoms, and axis arrows are enabled automatically:

```bash
xyzrender NV63.vasp             # VASP POSCAR/CONTCAR
xyzrender NV63.in               # Quantum ESPRESSO pw.in
xyzrender calc.fdf              # SIESTA FDF
xyzrender calc.abi              # ABINIT
xyzrender calc.inp              # CP2K (cell from &CELL block)
```

No extra dependencies or flags required. See [Crystal Structures](examples/crystal.md).

## Re-detecting bonds

`--rebuild` discards file connectivity and re-runs xyzgraph distance-based detection:

```bash
xyzrender molecule.sdf --rebuild
```

## Format-specific flags

| Flag | Description |
|------|-------------|
| `--smi SMILES` | Embed a SMILES string into 3D (requires rdkit) |
| `--mol-frame N` | Record index in multi-molecule SDF (default: 0) |
| `--rebuild` | Ignore file connectivity; re-detect bonds with xyzgraph |
| `-c`, `--charge` | Molecular charge |
| `-m`, `--multiplicity` | Spin multiplicity |
| `--bohr` | Input coordinates are in Bohr (force conversion to Angstrom) |
