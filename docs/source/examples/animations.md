# Animations

All GIF output defaults to `{input_basename}.gif`. Override with `-go`.

## Rotation GIF

| Rotation (y) | Rotation (xy) |
|-------------|--------------|
| ![Rotation (y)](../../../examples/images/caffeine.gif) | ![Rotation (xy)](../../../examples/images/caffeine_xy.gif) |

```bash
xyzrender caffeine.xyz --gif-rot -go caffeine.gif        # y-axis (default)
xyzrender caffeine.xyz --gif-rot xy -go caffeine_xy.gif  # xy axes
```

Available rotation axes: `x`, `y`, `z`, `xy`, `xz`, `yz`, `yx`, `zx`, `zy`. Prefix `-` to reverse (e.g. `-xy`). For crystal inputs, a 3-digit Miller index (e.g. `111`) rotates around the corresponding lattice direction.

Control speed and length:

```bash
xyzrender caffeine.xyz --gif-rot --gif-fps 20 --rot-frames 60 -go fast.gif
```

## TS vibration

| TS vibration (mn-h2) | TS + rotation (bimp) |
|---------------------|---------------------|
| ![TS vibration (mn-h2)](../../../examples/images/mn-h2.gif) | ![TS + rotation (bimp)](../../../examples/images/bimp.gif) |

```bash
xyzrender mn-h2.log --gif-ts -go mn-h2.gif
xyzrender bimp.out --gif-rot --gif-ts --vdw 84-169 -go bimp.gif
```

## Trajectory

```{image} ../../../examples/images/bimp_trj.gif
:width: 50%
:alt: Trajectory animation
```

```bash
xyzrender bimp.out --gif-trj --ts -go bimp_trj.gif
```

```{note}
`--gif-ts` and `--gif-trj` are mutually exclusive.
```

## Diffuse / assembly

```{image} ../../../examples/images/caffeine_diffuse.gif
:width: 50%
:alt: Diffuse assembly animation
```

````{tab-set}
```{tab-item} CLI
xyzrender caffeine.xyz --gif-diffuse -go caffeine_diffuse.gif
```
```{tab-item} Python
render_gif("caffeine.xyz", gif_diffuse=True, output="caffeine_diffuse.gif")
```
````

Options:

```bash
xyzrender caffeine.xyz --gif-diffuse --diffuse-noise 0.5       # more noise
xyzrender caffeine.xyz --gif-diffuse --diffuse-bonds hide       # no bonds
xyzrender caffeine.xyz --gif-diffuse --diffuse-bonds show       # keep bonds visible
xyzrender caffeine.xyz --gif-diffuse --diffuse-forward          # scatter (forward) instead of assembly
xyzrender caffeine.xyz --gif-diffuse --anchor 1-5               # keep atoms 1–5 fixed
xyzrender caffeine.xyz --gif-diffuse --gif-rot                  # rotation (360°, y-axis)
xyzrender caffeine.xyz --gif-diffuse --gif-rot xy               # rotation around xy
xyzrender caffeine.xyz --gif-diffuse --diffuse-rot              # partial rotation (180°)
xyzrender caffeine.xyz --gif-diffuse --diffuse-rot 90           # partial rotation (90°)
```

```{note}
`--gif-diffuse` cannot be combined with `--gif-ts` or `--gif-trj`.
`--gif-rot` gives a full 360° rotation; `--diffuse-rot` overrides the angle.
```

## Combined

Most options can be combined:

| TS animation (rot + ts + vdW + NCI) | Trajectory (trj + NCI + vdW) |
|------------------------------------|------------------------------|
| ![TS animation (rot + ts + vdW + NCI)](../../../examples/images/bimp_nci_ts.gif) | ![Trajectory (trj + NCI + vdW)](../../../examples/images/bimp_nci_trj.gif) |

```bash
xyzrender bimp.out --gif-ts --gif-rot --nci --vdw 84-169 -go bimp_nci_ts.gif
xyzrender bimp.out --gif-trj --nci --ts --vdw 84-169 -go bimp_nci_trj.gif
```
