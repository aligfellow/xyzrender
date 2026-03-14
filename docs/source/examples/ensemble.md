# Conformer Ensemble

Visualise multiple conformers from a multi-frame XYZ trajectory overlaid on a single reference frame. Each frame is RMSD-aligned onto the reference (frame 0) via the Kabsch algorithm. Non-reference conformers can be coloured by a continuous palette and faded with opacity.

| Default (viridis) | Spectral | With opacity |
|-------------------|----------|--------------|
| ![Viridis ensemble](../../../examples/images/sn2_ensemble_viridis.svg) | ![Spectral ensemble](../../../examples/images/sn2_ensemble_spectral.svg) | ![Opacity ensemble](../../../examples/images/sn2_ensemble_opacity.svg) |

```bash
xyzrender sn2.v000.xyz --ensemble -o sn2_ensemble.svg
xyzrender sn2.v000.xyz --ensemble --ensemble-color viridis -o sn2_ensemble_viridis.svg
xyzrender sn2.v000.xyz --ensemble --ensemble-color spectral -o sn2_ensemble_spectral.svg
xyzrender sn2.v000.xyz --ensemble --ensemble-color coolwarm --opacity 0.4 -o sn2_ensemble_opacity.svg
xyzrender mn-h2.v000.xyz --ensemble --ensemble-color viridis -o mn-h2_ensemble.svg --gif-rot -go mn-h2_ensemble.gif
```

From Python:

```python
from xyzrender import render

render("sn2.v000.xyz", ensemble=True)                                     # default viridis
render("sn2.v000.xyz", ensemble=True, ensemble_palette="spectral")        # spectral palette
render("sn2.v000.xyz", ensemble=True, ensemble_color="#FF0000")            # single color
render("sn2.v000.xyz", ensemble=True, opacity=0.4)                        # faded conformers
render("sn2.v000.xyz", ensemble=True, align_atoms=[1, 2, 3])              # align on subset
render("sn2.v000.xyz", ensemble=True, max_frames=10)                      # limit frames
```

## Alignment subset

By default the Kabsch fit uses all atoms. Use `--align-atoms` to fit on a subset (minimum 3 atoms to define a plane); the rotation is still applied to every atom. This works for both `--ensemble` and `--overlay`.

```bash
xyzrender sn2.v000.xyz --ensemble --align-atoms "1,2,3" -o sn2_ensemble_align.svg
xyzrender isothio_xtb.xyz --overlay isothio_uma.xyz --align-atoms "1-6" -o overlay_align.svg
```

| Flag | Description |
|------|-------------|
| `--ensemble` | Enable ensemble mode for multi-frame XYZ trajectories |
| `--ensemble-color VALUE` | Palette name (`viridis`, `spectral`, `coolwarm`), a single color, or comma-separated colors |
| `--opacity FLOAT` | Opacity for non-reference conformers (0–1, default: 1.0) |
| `--align-atoms INDICES` | 1-indexed atom subset for alignment (min 3), e.g. `1,2,3` or `1-6` |
