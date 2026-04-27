# Orientation

## Auto-orientation

Auto-orientation is on by default. xyzrender aligns the molecule so the axis of largest positional variance lies along the x-axis (PCA), giving a consistent front-facing view.

```bash
xyzrender molecule.xyz            # auto-oriented (default)
xyzrender molecule.xyz --no-orient  # raw coordinates as-is
```

Auto-orientation is disabled automatically when reading from stdin.

## Interactive rotation (`-I`)

The `-I` flag opens the molecule in an interactive viewer for rotation.
`xyzrender` captures the rotated coordinates and renders from those.

Two viewer backends are supported via `--viewer`:

### vmol (default)

Opens the molecule in the [**v** molecular viewer](https://github.com/briling/v) by [Ksenia Briling **@briling**](https://github.com/briling).
Rotate with the mouse or arrow keys, then press `z` to output the orientation
and `q` or `Esc` to quit.

```bash
xyzrender molecule.xyz -I               # uses vmol by default
xyzrender molecule.xyz -I --viewer vmol # explicit
```

Requires `pip install xyzrender[v]` (or `pip install vmol`).

### ASE GUI

Opens the molecule in [ASE](https://wiki.fysik.dtu.dk/ase/)'s built-in graphical viewer.
Rotate with the mouse, then **close the window** to confirm the orientation.

```bash
xyzrender molecule.xyz -I --viewer ase
```

Requires `pip install xyzrender[cif]` (or `pip install ase`).

For periodic structures, the unit cell is shown in the viewer automatically.

## Orientation reference (`--ref`)

The `--ref` flag saves or loads a reference orientation for consistent rendering across multiple files (e.g. a batch of MO cube files).

**First render** — file does not exist yet, PCA-oriented positions are saved:
```bash
xyzrender homo.cube --mo --ref              # saves reference.xyz
xyzrender homo.cube --mo --ref custom.xyz   # saves custom.xyz
```

**Subsequent renders** — file exists, molecule is Kabsch-aligned to it:
```bash
xyzrender lumo.cube --mo --ref              # loads reference.xyz, same orientation
xyzrender lumo.cube --mo --ref custom.xyz   # loads custom.xyz
```

When loading an existing reference, `--orient` is ignored — the reference file IS the orientation.

### Combined with `-I`

Orient interactively once, then reuse:
```bash
xyzrender homo.cube --mo -I --ref           # orient in viewer, save
xyzrender lumo.cube --mo --ref              # load, same orientation
```

If the reference file already exists, `-I` is skipped (the viewer is not opened).

### Python API

```python
from xyzrender import render, load

mol1 = load("homo.cube")
render(mol1, mo=True, ref="reference.xyz")   # save

mol2 = load("lumo.cube")
render(mol2, mo=True, ref="reference.xyz")   # load, same orientation
```

### Consistent orientation across a chemical series

`--ref` works across related compounds with different substituents, atom counts, or conformations. The shared scaffold is detected automatically — molecules are aligned on their largest common connected substructure. This gives consistent orientations across a series of derivatives, useful for comparing substituent effects or building figure panels:

```bash
# Orient the first compound interactively and save the reference
xyzrender catalyst_a.xyz -I --ref series.xyz

# All derivatives align to the same scaffold, regardless of substitution
xyzrender catalyst_b.xyz --ref series.xyz   # different R-group
xyzrender catalyst_c.xyz --ref series.xyz   # different atom count
xyzrender catalyst_d.xyz --ref series.xyz   # different heterocycle
```

```{note}
`--ref` is not supported for periodic structures (inputs loaded with `cell=True` or crystal formats). Use `-I` for interactive orientation of crystals.
```

## Piping from v

We can also pipe from `v` (or `vmol`) directly when working with `.xyz` files:

```bash
v molecule.xyz | xyzrender
```

Orient the molecule, press `z` to output reoriented coordinates, then `q` or `esc` to close.

## Viewer installation

Both viewer backends are optional dependencies:

```bash
pip install 'xyzrender[v]'    # vmol (v viewer)
pip install 'xyzrender[cif]'  # ASE GUI
pip install 'xyzrender[all]'  # both
```