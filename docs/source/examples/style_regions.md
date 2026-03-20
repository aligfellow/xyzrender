# Style Regions

Render subsets of atoms with a different preset — useful for highlighting QM/MM regions, active sites, or multi-fragment systems. Each region specifies atom indices and a preset name (or JSON config path).

The base config controls global properties (canvas, fog, background). Regions override per-atom/bond properties (atom size, colors, bond width, gradient) for their atoms. Structural overlays (TS/NCI bonds, centroids) always use the base config.

| Tube + ball-stick region | Two regions | Multi-fragment with NCI |
|--------------------------|-------------|------------------------|
| ![region](../../../examples/images/caffeine_region.svg) | ![two regions](../../../examples/images/caffeine_two_region.svg) | ![bimp regions](../../../examples/images/bimp_regions.svg) |

```bash
# Ball-stick base, tube for atoms 84–165
xyzrender mol.xyz --region "84-165" tube

# Tube base, ball-stick for the QM region
xyzrender mol.xyz --config tube --region "1-20" default

# Multiple regions with different presets
xyzrender mol.xyz --config tube --region "1-20" default --region "21-40" flat

# Multi-fragment: NCI detection + highlight + vdW + two regions
xyzrender bimp.xyz --no-orient --region "84-165" tube --nci --hl "84-165" --vdw "84-165"
```

From Python:

```python
from xyzrender import load, render

mol = load("mol.xyz")

# Single region
render(mol, config="tube", regions=[("1-20", "default")])

# Multiple regions — indices are 1-indexed (strings or lists)
render(mol, config="tube", regions=[("1-20", "default"), ("21-40", "flat")])

# 1-indexed list form
render(mol, config="tube", regions=[([1, 2, 3, 4], "default")])
```

## Combining with other overlays

Style regions compose with all existing overlays — highlight, vdW spheres, NCI detection, TS bonds, and annotations. This makes it easy to build up complex visualisations from simple flags.

```bash
# QM/MM: tube for the MM region, ball-stick QM region with highlight + vdW + NCI
xyzrender complex.xyz --region "84-165" tube --hl "84-165" steelblue --vdw "84-165" --nci

# Active site: wire background, highlighted residue with vdW spheres
xyzrender protein.xyz --config wire --region "1-30" default --hl "10-15" --vdw "10-15"
```

```python
render(mol, config="wire",
       regions=[("84-165", "tube")],
       highlight=[("84-165", "steelblue")],
       vdw=list(range(84, 166)))
```

Highlight recolours atoms regardless of their region style, vdW spheres use the base config's sphere settings, and NCI/TS bonds always render in the base style — so everything stays visually consistent even with multiple regions active.

## Bond coloring

Element-coloured bonds and cylinder shading can be used with any preset.

```bash
xyzrender mol.xyz --bond-by-element                  # half-bond split by atom colour
xyzrender mol.xyz --bond-gradient                    # cylinder shading (3D tube look)
xyzrender mol.xyz --config tube --no-bond-by-element # uniform colour tube
```

```python
render(mol, bond_color_by_element=True)  # half-bond element colouring
render(mol, bond_gradient=True)          # cylinder shading
```

The tube and wire presets enable both by default. The cylinder shading uses the same `get_gradient_colors` system as atom radial gradients — controlled by `hue_shift_factor`, `light_shift_factor`, and `saturation_shift_factor` in the preset JSON.

## How two configs coexist

- Each atom maps to either a region config or the base config via a per-atom lookup
- Bonds between two atoms in the **same** region use that region's bond style
- **Boundary bonds** (one atom in region, one not) use the base config
- **TS/NCI bonds** always use the base config — they are structural overlays, not molecular skeleton
- **NCI centroids** (`*` nodes) always use the base config

| Flag | Description |
|------|-------------|
| `--region ATOMS CONFIG` | Render atom subset with a different preset (repeatable) |
| `--bond-by-element` / `--no-bond-by-element` | Color bonds by endpoint atom colors |
| `--bond-gradient` / `--no-bond-gradient` | Cylinder shading on bonds |
