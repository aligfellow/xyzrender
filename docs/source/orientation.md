# Orientation

## Auto-orientation

Auto-orientation is on by default. xyzrender aligns the molecule so the axis of largest positional variance lies along the x-axis (PCA), giving a consistent front-facing view.

```bash
xyzrender molecule.xyz            # auto-oriented (default)
xyzrender molecule.xyz --no-orient  # raw coordinates as-is
```

Auto-orientation is disabled automatically when reading from stdin.

## Interactive rotation (`-I`)

The `-I` flag opens the molecule in the [v molecular viewer](https://github.com/briling/v) by [Ksenia Briling (@briling)](https://github.com/briling) for interactive rotation. Rotate to the desired view, press `z` to output coordinates, then `q` to close. xyzrender captures the rotated coordinates and renders from those.

```bash
xyzrender molecule.xyz -I
```

`v` must be installed separately and available on `$PATH` or in `~/bin/`.

## Piping from v

When working with `.xyz` files, you can pipe from `v` directly:

```bash
v molecule.xyz | xyzrender
```

Orient the molecule, press `z` to output coordinates, then `q` to close.
