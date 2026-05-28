# Atom Colormap

Color atoms by a per-atom scalar value (e.g. partial charges, NMR shifts, Fukui indices) using a colormap palette.

> **Python.** All `xyzrender` flags below map 1:1 to keyword arguments on `render()`. Two shapes differ from the CLI:
>
> - `--cmap FILE` → either `cmap="charges.txt"` (path) **or** `cmap={1: 0.5, 2: -0.3}` (1-indexed dict, no file needed)
> - `--cmap-range VMIN VMAX` → `cmap_range=(-0.5, 0.5)` (tuple)
>
> ```python
> render(mol, cmap={1: 0.5, 2: -0.3, 3: 0.04}, cmap_range=(-0.5, 0.5), cbar=True)
> render(mol, cmap="charges.txt", cmap_symm=True, cmap_palette="coolwarm")
> ```

| Mulliken charges (rotation) | Symmetric range | With colorbar |
|----------------------------|----------------|---------------|
| ![Mulliken charges (rotation)](../../../examples/images/caffeine_cmap.gif) | ![Symmetric range](../../../examples/images/caffeine_cmap.svg) | ![With colorbar](../../../examples/images/caffeine_cmap_colorbar.svg) |

```bash
xyzrender caffeine.xyz --hy --cmap caffeine_charges.txt --gif-rot -go caffeine_cmap.gif
xyzrender caffeine.xyz --hy --cmap caffeine_charges.txt --cmap-range -0.5 0.5
xyzrender caffeine.xyz --hy --cmap caffeine_charges.txt --cbar
```

The colormap file has two columns — **1-indexed atom number** and value. Any extension works. Header lines (first token not an integer), blank lines, and `#` comment lines are silently skipped.

```text
# charges.txt
1  +0.512
2  -0.234
3   0.041
```

- Atoms **in the file**: colored by the selected palette (default: Viridis — dark purple → blue → green → bright yellow)
- Atoms **not in the file**: white (`#ffffff`). Override with `"cmap_unlabeled"` in a custom JSON preset
- Range defaults to min/max of provided values; use `--cmap-range vmin vmax` for an explicit range or `--cmap-symm` for a symmetric range about zero

| Flag | Description |
|------|-------------|
| `--cmap FILE` | Path to colormap data file (two-column: atom index, value) |
| `--cmap-range VMIN VMAX` | Override colormap range (useful for symmetric scales, e.g. `-0.5 0.5`) |
| `--cmap-symm` | Symmetric range about zero: `[-max(|v|), +max(|v|)]` |
| `--cmap-palette NAME` | Colormap palette (default: `viridis`) |
| `--cbar` | Add a vertical colorbar on the right showing the data range |
| `--label-size PT` | Font size for colorbar tick labels (and all other labels) |

Recommended palette set for `xyzrender`:

- Best for `--cmap`: `viridis`, `plasma`, `coolwarm`
- Best for ESP: `rainbow`, `coolwarm`, `RdBu`
