# Highlight & Molecule Color

Color specific atom groups to visualise partitioning, active sites, or any structural decomposition. Multiple highlight groups can be used simultaneously, each with its own color. A flat molecule color (`--mol-color`) can serve as a neutral base for highlights to paint on top of.

All atom indices are **1-indexed** and accept comma-range syntax (`"1-5,8,12"`).

## Single-group highlight

Color a set of atoms and their connecting bonds. Without an explicit color, the first palette color (orchid) is used.

| Default (orchid) | Custom colour | Rotation |
|------------------|---------------|----------|
| ![hl](../../../examples/images//caffeine_hl.svg) | ![hl custom](../../../examples/images//caffeine_hl_custom.svg) | ![hl rot](../../../examples/images//caffeine_hl.gif) |

```bash
xyzrender caffeine.xyz --hl "1-3,7"                    # orchid (default)
xyzrender caffeine.xyz --hl "1-3,7" lightseagreen      # custom colour
xyzrender caffeine.xyz --hl "1-3,7" --gif-rot -go hl.gif  # works in GIFs
```

### Python

```python
render(mol, highlight="1-3,7")                          # string
render(mol, highlight=[1, 2, 3, 7])                     # list
render(mol, highlight=[("1-3,7", "lightseagreen")])     # with explicit color
```

## Molecule color

Paint all atoms and bonds a single flat color, replacing the default CPK element coloring.

```bash
xyzrender caffeine.xyz --mol-color gray --hy
```

### Python

```python
render(mol, mol_color="gray", hy=True)
```

## Multi-group highlight

Highlight multiple atom groups with different colors. Each `--hl` flag specifies atoms and an optional color. Groups without a color are auto-assigned from the preset palette.

```bash
# Two groups with explicit colors
xyzrender caffeine.xyz --hl "1-3,5,10,11,15,16,19,21" maroon --hl "4,6-9,12-14,17,18,20,22-24" teal --hy

# Auto-color from palette (orchid, mediumseagreen, goldenrod, ...)
xyzrender caffeine.xyz --hl "1-5" --hl "6-10" --hl "11-14" --hy
```

| Multi-group (explicit colors) | Mol color + highlight + indices |
|-------------------------------|-------------------------------|
| ![multi hl](../../../examples/images/caffeine_multi_hl.svg) | ![mol color hl idx](../../../examples/images//caffeine_mol_color_hl_idx.svg) |

### Python

```python
# Multi-group with explicit colors
render(mol, highlight=[("1-3,5,10,11,15,16,19,21", "maroon"),
                       ("4,6-9,12-14,17,18,20,22-24", "teal")], hy=True)

# Auto-color from palette
render(mol, highlight=["1-5", "10-15"])

# List-of-lists form
render(mol, highlight=[[1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]])

# With explicit colors via list[int]
render(mol, highlight=[([1, 2, 3, 4, 5], "blue"), ([10, 11, 12], "red")])
```

## Molecule color + highlight

Use `--mol-color` as a neutral base, then `--hl` to pick out specific regions. Highlight overrides the molecule color for both atoms and bonds.

```bash
xyzrender caffeine.xyz --hl "1-3,5,10,11,15,16,19,21" --mol-color mediumseagreen --hy --idx n
```

### Python

```python
render(mol, mol_color="mediumseagreen", highlight=[1, 2, 3, 5, 10, 11, 15, 16, 19, 21],
       hy=True, idx="n")
```

## Preset palette

The default highlight palette is defined in `default.json` and can be customised in a preset file:

```json
"highlight_colors": ["orchid", "mediumseagreen", "goldenrod", "coral", "mediumpurple", "hotpink"]
```

Groups are assigned colors in order: first group gets `orchid`, second `mediumseagreen`, etc. The palette cycles if there are more groups than colors.
