# Transition States and NCI

xyzrender uses [xyzgraph](https://github.com/aligfellow/xyzgraph) for molecular graph construction from Cartesian coordinates — determining bond connectivity, bond orders, detecting aromatic rings, and non-covalent interactions. It also provides element data (van der Waals radii, atomic numbers) used throughout rendering.

Transition state analysis uses [graphRC](https://github.com/aligfellow/graphRC) for internal coordinate vibrational mode analysis. Given a QM output file (ORCA, Gaussian, etc.), graphRC identifies which bonds are forming or breaking at the transition state with `--ts`. These are rendered as dashed bonds. graphRC is also used to generate TS vibration frames for `--gif-ts` animations.

## Transition states

`--ts` auto-detects forming/breaking bonds from QM output. TS bonds are rendered as dashed lines.

| Auto TS | Manual TS bond |
|---------|---------------|
| ![Auto TS](../../../examples/images/sn2_ts.svg) | ![Manual TS bond](../../../examples/images/sn2_ts_man.svg) |

```bash
xyzrender sn2.out --ts --hy -o sn2_ts.svg
xyzrender sn2.out --ts-bond "1-2" -o sn2_ts_man.svg    # specific bond only
xyzrender sn2.out --ts --ts-color dodgerblue -o sn2_ts_blue.svg
```

## QM output files

| ORCA output | Gaussian TS |
|-------------|------------|
| ![ORCA output](../../../examples/images/bimp_qm.svg) | ![Gaussian TS](../../../examples/images/mn-h2_qm.svg) |

```bash
xyzrender bimp.out -o bimp_qm.svg
xyzrender mn-h2.log --ts -o mn-h2_qm.svg
```

## NCI interactions (`--nci`)

`--nci` uses [xyzgraph](https://github.com/aligfellow/xyzgraph)'s `detect_ncis` to identify hydrogen bonds, halogen bonds, pi-stacking, and other non-covalent interactions from geometry. These are rendered as dotted bonds.

For pi-system interactions (e.g. pi-stacking, cation-pi), centroid dummy nodes are placed at the mean position of the pi-system atoms. For trajectory GIFs with `--nci`, interactions are re-detected per frame.

| Auto NCI | Manual NCI bond |
|----------|----------------|
| ![Auto NCI](../../../examples/images/nci.svg) | ![Manual NCI bond](../../../examples/images/nci_man.svg) |

```bash
xyzrender Hbond.xyz --hy --nci -o nci.svg                 # auto-detect all NCI
xyzrender Hbond.xyz --hy --nci-bond "8-9" -o nci_man.svg  # specific bond only
xyzrender Hbond.xyz --hy --nci --nci-color teal -o nci_teal.svg
```

## NCI + TS combined

| Default colours | Custom colours |
|-----------------|---------------|
| ![Default](../../../examples/images/bimp_ts_nci.svg) | ![Custom](../../../examples/images/bimp_ts_nci_custom.svg) |

```bash
xyzrender bimp.out --ts --nci --vdw 84-169 -o bimp_ts_nci.svg
xyzrender bimp.out --ts --nci -vdw 84-169 --ts-color magenta --nci-color teal -o bimp_ts_nci_custom.svg
```
