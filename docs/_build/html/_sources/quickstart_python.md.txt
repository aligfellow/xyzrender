# Python / Jupyter Quickstart

## Load and render

```python
from xyzrender import load, render

mol = load("caffeine.xyz")
render(mol, output="caffeine.svg")
```

Results display inline in Jupyter automatically — `render()` returns an `SVGResult` object with a rich HTML repr.

```python
# Inline display in notebook — no output= needed
render(mol)
```

Pass render options as keyword arguments:

```python
render(mol, output="caffeine.png", hy=True, config="paton")
```

## Orient interactively

`orient()` opens the molecule in the `v` viewer for interactive rotation. Rotate to the desired view, press `z`, then `q`. The rotated positions are written back to `mol` and `mol.oriented` is set to `True` so subsequent `render()` calls skip auto-orientation.

```python
from xyzrender import orient

orient(mol)
render(mol, output="oriented.svg")
```

## Save geometry

Write the current atom positions back to an XYZ file with `mol.to_xyz()`. For crystal structures the output is extXYZ format with a `Lattice=` header:

```python
mol.to_xyz("output.xyz")
mol.to_xyz("output.xyz", title="My molecule")
```

## GIF animations

```python
from xyzrender import render_gif

render_gif(mol, gif_output="caffeine.gif", gif_rot=True)
```

See [Core API](api/core.rst) for the full API reference.
