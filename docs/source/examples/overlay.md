# Structural Overlay

Overlay two structures to compare them. When both molecules have the same atoms in the same order, alignment is direct (index-based Kabsch). When atom counts or elements differ, the largest shared connected substructure is found automatically and used as the alignment basis.

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

## Cross-molecule overlay

Molecules with different atom counts or elements can be overlaid directly. The shared scaffold is found automatically and used as the alignment basis:

| Cross-molecule overlay | Rotation |
|------------------------|----------|
| ![Cross-molecule overlay](../../../examples/images/isothio_overlay_cross.svg) | ![rotating](../../../examples/images/isothio_overlay_cross.gif) | 

```bash
xyzrender isothio_xtb.xyz --overlay isothio_bridged.xyz -c 1 --hy --gif-rot
```

```python
mol1 = load("isothio_xtb.xyz", charge=1)
mol2 = load("isothio_bridged.xyz")
render(mol1, overlay=mol2)  # aligns on largest shared connected substructure
```

| Flag | Description |
|------|-------------|
| `--overlay FILE` | Second structure to overlay (RMSD-aligned onto the primary). Molecules can have different atom counts — alignment uses the largest shared connected substructure |
| `--overlay-color COLOR` | Color for the overlay structure (hex or named, default: contrasting) |
| `--align-atoms INDICES` | 1-indexed atom subset for Kabsch alignment (min 3), e.g. `1,2,3` or `1-6`. Only for same-atom-count overlays |
