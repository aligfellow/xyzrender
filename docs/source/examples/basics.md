# Basics

## Presets

| Default | Flat | Paton (PyMOL-like) | Bubble |
|---------|------|-------------------|--------|
| ![Default](../../../examples/images/caffeine_default.svg) | ![Flat](../../../examples/images/caffeine_flat.svg) | ![Paton (PyMOL-like)](../../../examples/images/caffeine_paton.svg) | ![Bubble](../../../examples/images/caffeine_bubble.svg) |

```bash
xyzrender caffeine.xyz                        # default
xyzrender caffeine.xyz --config flat          # flat: no gradient
xyzrender caffeine.xyz --config paton         # paton: PyMOL-style
xyzrender caffeine.xyz --config bubble --hy   # space-filling-like
```

The `paton` style is inspired by the clean styling used by [Rob Paton](https://github.com/patonlab) through PyMOL.

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
