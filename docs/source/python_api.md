# Python API Guide

xyzrender has a full Python API. All CLI flags are available as keyword arguments. Results display inline in Jupyter automatically.

See also the [auto-generated API reference](api/core.rst) for docstrings and type signatures, and the runnable [`examples/examples.ipynb`](https://github.com/aligfellow/xyzrender/blob/main/examples/examples.ipynb) notebook.

## Loading molecules

`load()` parses a file and returns a `Molecule` object. Pass it to `render()` to avoid re-reading the file on every call. You can also pass a path string directly to `render()` as a shorthand.

```python
from xyzrender import load, render

mol = load("caffeine.xyz")
render(mol)                          # displays inline in Jupyter
render(mol, output="caffeine.svg")   # save as SVG
render(mol, output="caffeine.png")   # save as PNG

# Short-form: pass a path directly (re-parses each time)
render("caffeine.xyz")
```

### Loading options

Use `load()` keyword arguments for non-default loading behaviour:

```python
mol = load("ts.out", ts_detect=True)            # detect TS bonds via graphRC
mol = load("mol.xyz", nci_detect=True)          # detect NCI interactions
mol = load("mol.sdf", mol_frame=2, kekule=True) # SDF frame + Kekule bonds
mol = load("CC(=O)O", smiles=True)              # SMILES → 3D (requires rdkit)
mol = load("POSCAR")                            # VASP/QE/SIESTA/ABINIT auto-detected
mol = load("caffeine_cell.xyz", cell=True)      # extXYZ Lattice= header
mol = load("mol.xyz", quick=True)               # skip BO detection (faster, use with bo=False)
```

## Render options

All CLI flags are available as keyword arguments to `render()`:

### Styling

```python
render(mol, config="flat")                                     # built-in preset
render(mol, config="paton", transparent=True)                  # preset + transparent bg
render(mol, bond_width=8, atom_scale=1.5, background="#f0f0f0") # individual overrides
render(mol, hide_bonds=True)                                   # hide all bonds
```

### Bond display rules

Selectively hide or add bonds using element categories, element symbols,
pi-coordination, or atom indices.  Specs are **1-indexed**.

```python
render(mol, unbond=["M-L"])                   # hide all metal-ligand bonds
render(mol, unbond=["sbm"])                   # hide all s-block metal bonds (Li, Na, K, etc.)
render(mol, unbond=["M-het"])                 # hide metal-heteroatom (not C, not H)
render(mol, unbond=["M-pi"])                  # hide metal pi-coordination (ferrocene-like)
render(mol, unbond=["pi"])                    # hide all pi-coordination bonds 
render(mol, unbond=["Fe-C"])                  # hide iron-carbon bonds specifically
render(mol, unbond=["Li"])                    # hide all bonds from lithium atoms
render(mol, unbond=["2"])                     # hide all bonds from atom 2
render(mol, unbond=["1-3"])                   # hide bond between atoms 1 and 3
render(mol, unbond=["M-L"], bond=["2-5"])     # hide M-L but keep bond 2-5
render(mol, unbond=["sbm", "M-pi", "1-3"])    # multiple specs
```

Available categories: `M` (all metals), `sbm` (s-block metals), `L` (non-metals),
`het` (heteroatoms: not C, not H, not metal), `pi` (eta-coordination to aromatic rings).
Any element symbol also works (`Fe`, `Li`, `O`, etc.).
NCI and TS overlay edges are never removed by rules.

### Haptic centroid bonds

Replace the fan of individual metal-to-ring bonds with a single dotted bond
from the metal to the ring centroid:

```python
render(mol, haptic=True)
```

### Hydrogen visibility

Atom indices are **1-indexed**.

```python
ethanol = load("ethanol.xyz")
render(ethanol, hy=True)            # show all H
render(ethanol, no_hy=True)         # hide all H
render(ethanol, hy=[7, 8, 9])       # show specific H atoms
```

### Atom filtering

Use `only=` or `exclude=` to remove atoms from the render-time graph before
orientation, canvas fitting, bond rules, annotations, hulls, and overlays are
resolved. Selectors use the same grammar as `highlight` and remain based on
the original input atom numbering after filtering.

```python
salt = load("salt.xyz")
render(salt, only="1-24")        # render one component of a salt
render(salt, exclude="Na,Cl")    # remove counterions
render(salt, exclude=["25-40"])  # repeatable/list form
```

Incident bonds are removed with excluded atoms. Cube/surface fields
(`mo`, `dens`, `esp`, `nci`) are not cropped by atom filtering.

### Overlays

```python
render(mol, vdw=True)               # vdW spheres on all atoms
render(mol, vdw=[1, 3, 5])          # vdW spheres on specific atoms
render(mol, ts_bonds=[(1, 6)])      # manual TS bond (1-indexed)
render(mol, ts_color="dodgerblue")  # color for dashed TS bonds
render(mol, nci_bonds=[(2, 8)])     # manual NCI bond (1-indexed)
render(mol, nci_color="teal")       # color for dotted NCI bonds
render(mol, nci_element=True)       # split-half atom colour on NCI/haptic dots (off by default; on in pmol/btube/tube/mtube)
render(mol, ts_element=True)        # same, for TS dashes (off by default; needs a by_element preset)
render(mol, ts_dash="2.0,1.5")      # (length, gap) as bond_width multiples (default 1.2,2.2). Tuple also OK.
render(mol, ts_width=0.5)           # TS line width as a bond_width multiple (default 1.2)
render(mol, nci_dash=(0.2, 1.5))    # NCI/haptic (length, gap) multipliers (default 0.08,2.0)
render(mol, nci_width=0.8)          # NCI/haptic line width as a bond_width multiple (default 1.0)
render(mol, idx=True)               # atom index labels ("C1", "N3", ...)
render(mol, idx="n")                # index only ("1", "3", ...)
render(mol, mol_color="gray")                            # flat color for all atoms + bonds
render(mol, highlight="1-3,7")                           # highlight atoms 1-3 and 7 (orchid)
render(mol, highlight="C,N")                             # element symbols / categories ("M", "het", ...)
render(mol, highlight=[1, 2, 3, 7])                      # 1-indexed list
render(mol, highlight=[("1-5", "blue"), ("10-15", "red")])  # multi-group with colors
render(mol, highlight=["1-5", "10-15"])                  # multi-group, auto-colors from palette
render(mol, mol_color="gray", highlight="1-5")           # gray base + orchid highlight on top
render(mol, dof=True)                                   # depth-of-field blur
render(mol, dof=True, dof_strength=6.0)                 # stronger blur
```

### Structural overlay

```python
mol1 = load("isothio_xtb.xyz", charge=1)
mol2 = load("isothio_uma.xyz", charge=1)
render(mol1, overlay=mol2)                         # overlay mol2 onto mol1
render(mol1, overlay=mol2, overlay_color="green")  # custom overlay color
render(mol1, overlay=mol2, align_atoms=[1, 2, 3])  # explicit 1-indexed subset
render(mol1, overlay=mol2, align_atoms="M,L")      # metal + coord shell (organometallic)
render_gif(mol1, overlay=mol2, gif_rot="y")        # spinning overlay GIF
```

`align_atoms` accepts a 1-indexed atom list/string or a symbol/category spec (`"M,L"`, `"Fe,P"`, …). Symbol specs are resolved per-graph so atom ordering doesn't need to match between the two structures. Metal-containing specs run metal-fragment overlay so paired metals coincide exactly; non-metal specs use MCS + K-subset Kabsch and pick the lowest-RMSD candidate.

See [Structural Overlay](examples/overlay.md) and [Conformer Ensemble](examples/ensemble.md) for more.

### Annotations

```python
render(mol, labels=["1 2 d", "1 2 3 a"])   # inline spec strings
render(mol, label_file="annot.txt")         # bulk annotation file
```

See [Annotations](examples/annotations.md) for the full spec syntax.

### Atom property colormap

`cmap` accepts a `{1-indexed atom: value}` dict or a path to a two-column file. Atoms absent from the mapping are drawn white.

```python
render(mol, cmap={1: 0.5, 2: -0.3}, cmap_range=(-1.0, 1.0))
render(mol, cmap="charges.txt", cmap_symm=True)   # symmetric range about 0
```

See [Atom Property Colormap](examples/cmap.md) for details on file format, palettes, and the colorbar.

### Surfaces (cube files)

```python
mol_cube = load("caffeine_homo.cube")
render(mol_cube, mo=True)                                          # MO lobes
render(mol_cube, mo=True, iso=0.03, mo_pos_color="maroon", mo_neg_color="teal")

dens_cube = load("caffeine_dens.cube")
render(dens_cube, dens=True)                       # density isosurface
render(dens_cube, esp="caffeine_esp.cube")         # ESP mapped onto density
render(dens_cube, nci="caffeine_grad.cube")        # NCI surface
render(load("phenol_di-sl2r.cub"), nci="phenol_di-dg_inter.cub", iso=0.005)  # high_field surface
```

See [Molecular Orbitals](examples/mo.md), [Electron Density and ESP](examples/dens_esp.md), and [NCI Surface](examples/nci_surf.md).

### Convex hull

```python
render(mol, hull=[1, 2, 3, 4, 5, 6],
       hull_color="steelblue", hull_opacity=0.35)
render(mol, hull="rings", hull_color="teal")       # auto-detect aromatic rings
```

See [Convex Hull](examples/hull.md) for multi-subset hulls and all options.

## Reusing a style config

`build_config()` builds a `RenderConfig` object you can pass to `render()` and `render_gif()`. Useful in notebooks or scripts that render several structures with the same style:

```python
from xyzrender import build_config

cfg = build_config("flat", atom_scale=1.5, gradient=False)
render(mol1, config=cfg)
render(mol2, config=cfg, ts_bonds=[(1, 6)])   # per-render overlay on shared style
render_gif("mol.xyz", gif_rot="y", config=cfg)
```

## Geometry measurements

`measure()` returns bonded distances, angles, and dihedrals as a dict. It does not render anything. Atom indices in the output are **0-indexed**.

```python
from xyzrender import measure

data = measure(mol)                    # all measurements
data = measure("mol.xyz")             # also accepts a path
data = measure(mol, modes=["d", "a"]) # distances and angles only

for i, j, d in data["distances"]:
    print(f"  {i+1}-{j+1}: {d:.3f} Å")
```

## Saving geometry

`Molecule.to_xyz()` writes the structure to an XYZ file. Ghost atoms are excluded. If the molecule has `cell_data` (loaded with `cell=True` or `crystal=...`), the output is extXYZ with a `Lattice=` header so it can be reloaded directly.

```python
mol = load("CC(=O)O", smiles=True)
mol.to_xyz("acetic_acid.xyz")                            # plain XYZ
mol.to_xyz("acetic_acid.xyz", title="acetic acid")       # with comment line

mol_cell = load("caffeine_cell.xyz", cell=True)
mol_cell.to_xyz("out.xyz")                               # extXYZ with Lattice= header
```

## Interactive orientation

`orient()` opens the 3D viewer ([**v**](https://github.com/briling/v)) so you can rotate a molecule manually, then locks the orientation for subsequent `render()` calls:

```python
from xyzrender import orient

mol = load("caffeine.xyz")
orient(mol)        # opens viewer — rotate, close to confirm
render(mol)        # renders in the manually chosen orientation
```

Requires `pip install xyzrender[v]` (Linux only).

## Orientation reference

Use `ref=` to save or load a reference orientation for consistent batch rendering:

```python
mol1 = load("homo.cube")
render(mol1, mo=True, ref="reference.xyz")   # first call saves oriented positions

mol2 = load("lumo.cube")
render(mol2, mo=True, ref="reference.xyz")   # subsequent calls Kabsch-align to the reference
```

When the reference file exists, `orient=True` is ignored — the reference is the orientation.

## GIF animations

```python
from xyzrender import render_gif

render_gif("caffeine.xyz", gif_rot="y")           # rotation GIF
render_gif("ts.out", gif_ts=True)                  # TS vibration GIF
render_gif("traj.xyz", gif_trj=True)               # trajectory GIF
render_gif("mep.xyz", gif_trj=True, trj_bonds=True)  # re-detect bonds per frame
render_gif("mol.xyz", gif_rot="y", config=cfg)     # with shared style config

# Diffuse / assembly GIF
render_gif("caffeine.xyz", gif_diffuse=True)
render_gif("caffeine.xyz", gif_diffuse=True, diffuse_noise=0.5, diffuse_bonds="hide")
render_gif("caffeine.xyz", gif_diffuse=True, gif_rot="y", diffuse_rot=90)
render_gif("caffeine.xyz", gif_diffuse=True, anchor="1-5,8")

# Surface in rotation GIF (cube file)
mol_cube = load("caffeine_homo.cube")
render_gif(mol_cube, gif_rot="y", mo=True, output="homo_rot.gif")
```

## Return types

### SVGResult

`render()` returns an `SVGResult` object. In Jupyter it displays inline automatically.

```python
result = render(mol)
str(result)              # raw SVG string
result.save("out.svg")   # write to file
```

### GIFResult

`render_gif()` returns a `GIFResult` object. In Jupyter it displays inline automatically.

```python
gif = render_gif("mol.xyz", gif_rot="y")
gif.path                 # pathlib.Path to the GIF on disk
bytes(gif)               # raw GIF bytes
gif.save("copy.gif")     # copy to another path
```
