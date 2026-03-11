# Orientation

## Auto-orientation

Auto-orientation is on by default. xyzrender aligns the molecule so the axis of largest positional variance lies along the x-axis (PCA), giving a consistent front-facing view.

```bash
xyzrender molecule.xyz            # auto-oriented (default)
xyzrender molecule.xyz --no-orient  # raw coordinates as-is
```

Auto-orientation is disabled automatically when reading from stdin.

## Interactive rotation (`-I`)

The `-I` flag opens the molecule in the [**v** molecular viewer](https://github.com/briling/v) by [Ksenia Briling **@briling**](https://github.com/briling)
for interactive rotation. Rotate the molecule to the desired orientation
and close the window with `q` or `esc`.  
`xyzrender` captures the rotated coordinates and renders from those.

```bash
xyzrender molecule.xyz -I
```

## Piping from v

We can also pipe from `v` (or `vmol`) directly when working with `.xyz` files:

```bash
v molecule.xyz | xyzrender
```

Orient the molecule, press `z` to output reoriented coordinates, then `q` or `esc` to close.

## v installation

This is an *optional* dependency (Linux only) and should be installed by using either:
```bash
pip install xyzrender[v]
# or directly with
pip install vmol 
```