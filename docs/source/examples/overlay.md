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
xyzrender isothio_xtb.xyz --overlay isothio_fused.xyz -c 1 --hy --gif-rot
```

```python
mol1 = load("isothio_xtb.xyz", charge=1)
mol2 = load("isothio_fused.xyz")
render(mol1, overlay=mol2)  # aligns on largest shared connected substructure
```

## Per-overlay style (independent of the primary)

The overlaid structure has its own style knobs that mirror the primary flags. All of them are absolute (not multipliers on the primary) and individually optional — unset fields inherit from the primary config.

![Styled overlay](../../../examples/images/isothio_overlay_styled.svg)

```bash
# Faded teal overlay, smaller atoms and thinner bonds than the primary
xyzrender isothio_xtb.xyz --overlay isothio_uma.xyz -c 1 --hy \
    --overlay-color teal --opacity 0.5 \
    --overlay-atom-scale 1.5 --overlay-bond-width 15
```

From Python — either keep using the flat `overlay_color=` / `opacity=` kwargs, or pass a full `OverlayConfig` for everything else:

```python
from xyzrender import load, render, OverlayConfig

mol1 = load("isothio_xtb.xyz", charge=1)
mol2 = load("isothio_uma.xyz", charge=1)

render(
    mol1,
    overlay=mol2,
    overlay_config=OverlayConfig(
        color="teal",
        opacity=0.5,
        atom_scale=1.5,
        bond_width=15,
        unbond=["all"],   # overlay-only bond rules (indices are 1-indexed on the overlay)
        # show=["N,O"],   # optional: render only these overlay atoms (post-align, per --overlay-show)
    ),
)
```

Preset JSON carries the same shape — drop it in your custom preset to apply everywhere:

```json
{
  "overlay": {
    "color": "teal",
    "opacity": 0.5,
    "atom_scale": 1.5,
    "bond_width": 15
  }
}
```

### Full config on the overlay

For `RenderConfig` fields that don't have a dedicated `--overlay-*` CLI shortcut (`atom_gradient_strength`, `bond_gap`, `fog`, `skeletal_style`, `bond_color_by_element`, ...), set `overlay.config` to a full `RenderConfig` and the renderer applies it to the overlay atoms via the same machinery as style regions. Scalar shortcuts above still win over whatever `config` sets.

```python
from xyzrender import load, render, OverlayConfig, build_config

render(
    load("isothio_xtb.xyz", charge=1),
    overlay=load("isothio_uma.xyz", charge=1),
    overlay_config=OverlayConfig(
        color="teal",
        config=build_config("tube"),   # full preset on the overlay only
    ),
)
```

Preset JSON supports the same via a nested `"config"` block — anything the main presets accept is valid here:

```json
{
  "overlay": {
    "color": "teal",
    "config": {
      "atom_gradient_strength": 2.0,
      "bond_gap": 0.3,
      "bond_color_by_element": true
    }
  }
}
```

## Two flat colours

To drop CPK entirely and paint each structure a single flat colour, pair `--mol-color` (primary) with `--overlay-color` (overlay). The overlay keeps its own colour — `--mol-color` only paints atoms that don't already carry a per-structure colour.

```bash
xyzrender isothio_xtb.xyz --overlay isothio_uma.xyz -c 1 --hy \
    --mol-color steelblue --overlay-color coral
```

## Skip alignment

`--no-align` disables Kabsch/MCS so the overlay keeps its raw coordinates relative to the primary. Useful when two optimisations already share a frame and you want to visualise the geometric difference directly. Works for both `--overlay` and `--ensemble`, and `-I` rotation applied to the base still propagates to the overlay under `--no-align`.

```bash
xyzrender isothio_xtb.xyz --overlay isothio_uma.xyz -c 1 --hy --no-align
```

| Flag | Description |
|------|-------------|
| `--overlay FILE` | Second structure to overlay (RMSD-aligned onto the primary). Molecules can have different atom counts — alignment uses the largest shared connected substructure |
| `--overlay-color COLOR` | Overlay atom colour (bonds auto-darkened 30 %) |
| `--opacity FLOAT` | Overlay transparency 0–1 (also used by ensemble/surfaces — the active mode wins) |
| `--overlay-atom-scale FLOAT` | Absolute atom radius scale for the overlay only |
| `--overlay-bond-width FLOAT` | Absolute bond width for the overlay only |
| *(preset / Python only)* | `atom_stroke_width`, `atom_stroke_color`, `bond_color`, `bond_outline_width`, `bond_outline_color` — set inside `"overlay"` in a preset JSON or on `OverlayConfig` directly |
| `--overlay-unbond SPEC [...]` | Hide bonds on the overlay only (same grammar as `--unbond`) |
| `--overlay-bond PAIR [...]` | Force-show / add bonds on the overlay only (1-indexed, overlay-local) |
| `--overlay-show SPEC [...]` | Render only these atoms of the overlay (same grammar as `--hl`: ranges, element symbols, `M`, `het`, `all`). Applied after alignment so the fit still uses the full scaffold |
| `--align` / `--no-align` | Force / skip Kabsch/MCS alignment (default: on). `--align` overrides a preset's `auto_align: false`; `--no-align` keeps raw coordinates |
| `--align-atoms INDICES` | 1-indexed atom subset for Kabsch alignment (min 3), e.g. `1,2,3` or `1-6`. Only for same-atom-count overlays |
