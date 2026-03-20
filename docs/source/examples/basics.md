# Basics

## Presets

| Default | Flat | Paton (PyMOL-like) | Bubble |
|---------|------|-------------------|--------|
| ![Default](../../../examples/images/caffeine_default.svg) | ![Flat](../../../examples/images/caffeine_flat.svg) | ![Paton (PyMOL-like)](../../../examples/images/caffeine_paton.svg) | ![Bubble](../../../examples/images/caffeine_bubble.svg) |

| Tube | Wire |
|------|------|
| ![Tube](../../../examples/images/caffeine_tube.svg) | ![Wire](../../../examples/images/caffeine_wire.svg) |

```bash
xyzrender caffeine.xyz                        # default
xyzrender caffeine.xyz --config flat          # flat: no gradient
xyzrender caffeine.xyz --config paton         # paton: PyMOL-style
xyzrender caffeine.xyz --config bubble --hy   # space-filling-like
xyzrender caffeine.xyz --config tube          # tube: cylinder-shaded sticks
xyzrender caffeine.xyz --config wire          # wire: thin element-coloured lines
```

The `paton` style is inspired by the clean styling used by [Rob Paton](https://github.com/patonlab) through PyMOL.

The `tube` and `wire` presets hide atom circles and colour each bond by its endpoint atoms, with a cylinder shading gradient for a 3D look. The `tube` preset uses thick bonds; `wire` uses thin bonds.


## Hydrogen display

| All H | Some H | No H |
|-------|--------|------|
| ![All H](../../../examples/images/ethanol_all_h.svg) | ![Some H](../../../examples/images/ethanol_some_h.svg) | ![No H](../../../examples/images/ethanol_no_h.svg) |

```bash
xyzrender ethanol.xyz --hy              # all H
xyzrender ethanol.xyz --hy 7 8 9        # specific H atoms (1-indexed)
xyzrender ethanol.xyz --no-hy           # no H
```

## Bond orders

| Aromatic | Kekulé |
|----------|--------|
| ![Aromatic](../../../examples/images/benzene.svg) | ![Kekulé](../../../examples/images/caffeine_kekule.svg) |

```bash
xyzrender benzene.xyz --hy              # aromatic notation (default)
xyzrender caffeine.xyz --bo -k          # Kekulé bond orders
```

## vdW spheres

| All atoms | Selected atoms | Paton style |
|-----------|---------------|-------------|
| ![All atoms](../../../examples/images/asparagine_vdw.svg) | ![Selected atoms](../../../examples/images/asparagine_vdw_partial.svg) | ![Paton style](../../../examples/images/asparagine_vdw_paton.svg) |

```bash
xyzrender asparagine.xyz --hy --vdw                   # all atoms
xyzrender asparagine.xyz --hy --vdw "1-6"             # atoms 1–6 only
xyzrender asparagine.xyz --hy --vdw --config paton    # paton style
```

## Depth of field

Blur back atoms while keeping front atoms sharp. Uses SVG `feGaussianBlur` filters.

| DoF | Rotation | 
|-----|----------|
| ![dof](../../../examples/images/caffeine_dof.svg) | ![dof](../../../examples/images/caffeine_dof.gif) | 

```bash
xyzrender caffeine.xyz --dof --no-orient                    # default strength
xyzrender caffeine.xyz --dof --dof-strength 6.0 --no-orient # stronger blur
```

```python
render(mol, dof=True, orient=False)
render(mol, dof=True, dof_strength=6.0, orient=False)
```
