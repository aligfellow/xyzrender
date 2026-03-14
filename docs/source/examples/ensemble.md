# Conformer Ensemble

Visualise multiple conformers from a multi-frame XYZ trajectory overlaid on a single reference frame. Each frame is RMSD-aligned onto the reference (frame 0) via the Kabsch algorithm. By default, conformers render with standard CPK atom colours. Use `--ensemble-color` to apply a continuous palette or a fixed colour.

| Default (CPK) | Spectral + opacity |
|---------------|--------------------|
| ![Default ensemble](../../../examples/images/triphenylbenzol_ensemble.svg) | ![Custom ensemble](../../../examples/images/triphenylbenzol_ensemble_custom.svg) |

```bash
xyzrender triphenylbenzol.xyz --ensemble -o triphenylbenzol_ensemble.svg
xyzrender triphenylbenzol.xyz --ensemble --align-atoms 21,22,23 --ensemble-color spectral --opacity 0.4 -o triphenylbenzol_ensemble_custom.svg
```

From Python:

```python
from xyzrender import render

render("triphenylbenzol.xyz", ensemble=True)                                          # CPK colours
render("triphenylbenzol.xyz", ensemble=True, ensemble_palette="spectral")             # spectral palette
render("triphenylbenzol.xyz", ensemble=True, ensemble_color="#FF0000")                # single colour
render("triphenylbenzol.xyz", ensemble=True, ensemble_palette="viridis", opacity=0.4) # faded
render("triphenylbenzol.xyz", ensemble=True, align_atoms=[21, 22, 23])               # align on subset
render("triphenylbenzol.xyz", ensemble=True, max_frames=10)                          # limit frames
```

## Alignment subset

By default the Kabsch fit uses all atoms. Use `--align-atoms` to fit on a subset (minimum 3 atoms to define a plane); the rotation is still applied to every atom. This works for both `--ensemble` and `--overlay`.

```bash
xyzrender triphenylbenzol.xyz --ensemble --align-atoms 21,22,23 -o ensemble_align.svg
xyzrender isothio_xtb.xyz --overlay isothio_uma.xyz --align-atoms 1-6 -o overlay_align.svg
```

| Flag | Description |
|------|-------------|
| `--ensemble` | Enable ensemble mode for multi-frame XYZ trajectories |
| `--ensemble-color VALUE` | Palette name (`viridis`, `spectral`, `coolwarm`), a single colour, or comma-separated colours |
| `--opacity FLOAT` | Opacity for non-reference conformers (0–1, default: 1.0) |
| `--align-atoms INDICES` | 1-indexed atom subset for alignment (min 3), e.g. `21,22,23` or `1-6` |
