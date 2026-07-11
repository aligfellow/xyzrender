# Conformer Ensemble

Visualise multiple conformers from a multi-frame XYZ trajectory overlaid on a single reference frame. Each frame is RMSD-aligned onto the reference (frame 0) via the Kabsch algorithm. By default, conformers are coloured across the `spectral` palette so each frame is distinguishable. Use `--ensemble-color` to choose a different palette or a fixed colour, or pass `cpk` for standard per-element atom colours.

| Default (spectral) | CPK per-element | Viridis + opacity |
|--------------------|-----------------|-------------------|
| ![Default ensemble](../../../examples/images/triphenylbenzol_ensemble.svg) | ![CPK ensemble](../../../examples/images/triphenylbenzol_ensemble_cpk.svg) | ![Custom ensemble](../../../examples/images/triphenylbenzol_ensemble_custom.svg) |

```bash
xyzrender triphenylbenzol.xyz --ensemble -o triphenylbenzol_ensemble.svg
xyzrender triphenylbenzol.xyz --ensemble --ensemble-color cpk -o triphenylbenzol_ensemble_cpk.svg
xyzrender triphenylbenzol.xyz --ensemble --align-atoms 21,22,23 --ensemble-color viridis --opacity 0.4 -o triphenylbenzol_ensemble_custom.svg
```

From Python, the ensemble options are passed to `load()`, which builds the multi-conformer `Molecule`; `render()` then draws it:

```python
from xyzrender import load, render

render(load("triphenylbenzol.xyz", ensemble=True))                                          # spectral palette (default)
render(load("triphenylbenzol.xyz", ensemble=True, ensemble_color="cpk"))                    # per-element CPK colours
render(load("triphenylbenzol.xyz", ensemble=True, ensemble_color="#FF0000"))                # single colour
render(load("triphenylbenzol.xyz", ensemble=True, ensemble_color="viridis"), opacity=0.4)   # faded palette
render(load("triphenylbenzol.xyz", ensemble=True, align_atoms=[21, 22, 23]))                # align on subset
render(load("triphenylbenzol.xyz", ensemble=True, max_frames=10))                           # limit frames
```

## Alignment subset

By default the Kabsch fit uses all atoms. Use `--align-atoms` to fit on a subset (minimum 3 atoms to define a plane); the rotation is still applied to every atom. This works for both `--ensemble` and `--overlay`.

```bash
xyzrender triphenylbenzol.xyz --ensemble --align-atoms 21,22,23 -o ensemble_align.svg
xyzrender isothio_xtb.xyz --overlay isothio_uma.xyz --align-atoms 1-6 -o overlay_align.svg
```

## Skip alignment

`--no-align` disables Kabsch entirely and renders each frame at its raw coordinates, preserving the trajectory's native geometry. Useful for trajectories where the absolute frame matters.

```bash
xyzrender triphenylbenzol.xyz --ensemble --no-align -o ensemble_raw.svg
```

| Flag | Description |
|------|-------------|
| `--ensemble` | Enable ensemble mode for multi-frame XYZ trajectories |
| `--ensemble-color VALUE` | Palette name (`viridis`, `plasma`, `spectral`, `coolwarm`, `RdBu`, `rainbow`), a single colour, comma-separated colours, or `cpk` for per-element atom colours (default: `spectral`) |
| `--opacity FLOAT` | Opacity for non-reference conformers (0–1, default: 1.0) |
| `--align-atoms INDICES` | 1-indexed atom subset for alignment (min 3), e.g. `21,22,23` or `1-6` |
| `--align` / `--no-align` | Force / skip Kabsch alignment (default: on) |
