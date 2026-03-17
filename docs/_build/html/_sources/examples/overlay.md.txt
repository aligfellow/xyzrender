# Structural Overlay

Overlay two conformers or geometries to compare them. The second structure is RMSD-aligned onto the first via the Kabsch algorithm using index-based atom pairing, and rendered in a contrasting colour. Both molecules must have the same number of atoms in the same order.

| Default | Custom colour | Rotation GIF |
|---------|---------------|--------------|
| ![Default overlay](../../../examples/images/isothio_overlay.svg) | ![Custom colour overlay](../../../examples/images/isothio_overlay_custom.svg) | ![Overlay rotation GIF](../../../examples/images/isothio_overlay.gif) |

```bash
xyzrender isothio_xtb.xyz --overlay isothio_uma.xyz -c 1 --hy -o isothio_overlay_rot.svg --gif-rot -go isothio_overlay.gif
xyzrender isothio_xtb.xyz --overlay isothio_uma.xyz -c 1 --overlay-color green -a 2 --no-orient -o isothio_overlay_custom.svg
```

From Python:

```python
from xyzrender import load, render, render_gif

mol1 = load("isothio_xtb.xyz", charge=1)
mol2 = load("isothio_uma.xyz", charge=1)

render(mol1, overlay=mol2)                        # overlay mol2 onto mol1
render(mol1, overlay=mol2, overlay_color="green") # custom overlay color
render_gif(mol1, overlay=mol2, gif_rot="y")       # spinning overlay GIF
```

| Flag | Description |
|------|-------------|
| `--overlay FILE` | Second structure to overlay (RMSD-aligned onto the primary) |
| `--overlay-color COLOR` | Color for the overlay structure (hex or named, default: contrasting) |
| `--align-atoms INDICES` | 1-indexed atom subset for Kabsch alignment (min 3), e.g. `1,2,3` or `1-6` |
