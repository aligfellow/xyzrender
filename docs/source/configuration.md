# Configuration

## Built-in presets

Use `--config` to load a styling preset. Built-in options: `default`, `flat`, `paton`, `skeletal`, `bubble`, `tube`, `wire`.

| Preset | Description |
|--------|-------------|
| `default` | Radial gradients, depth fog, CPK colors |
| `flat` | No gradients, no fog — clean flat look |
| `paton` | PyMOL-inspired style (see [Rob Paton](https://github.com/patonlab)) |
| `skeletal` | Skeletal formula diagram — thin bonds, minimal atoms |
| `bubble` | Space-filling (CPK) — large atoms, no bonds |
| `tube` | Tube/stick model — no atoms, thick element-coloured bonds with cylinder shading |
| `wire` | Wireframe — no atoms, thin element-coloured bonds with cylinder shading |

```bash
xyzrender caffeine.xyz --config flat
xyzrender caffeine.xyz --config paton
xyzrender caffeine.xyz --config skeletal
xyzrender caffeine.xyz --config bubble --hy
xyzrender caffeine.xyz --config tube
xyzrender caffeine.xyz --config wire
```

CLI flags override preset values:

```bash
xyzrender caffeine.xyz --config paton --bo   # paton preset but with bond orders on
xyzrender caffeine.xyz --config default --no-fog
```

## Custom presets (JSON)

Create a JSON file with any keys you want to override. Everything else falls back to the default. Load it with `--config`:

```bash
xyzrender caffeine.xyz --config my_style.json
```

All available keys:

```json
{
  "canvas_size": 800,
  "atom_scale": 2.5,
  "bond_width": 20,
  "bond_color": "#000000",
  "ts_color": "#1E90FF",
  "nci_color": "#228B22",
  "atom_stroke_width": 3,
  "gradient": true,
  "gradient_strength": 1.5,
  "fog": true,
  "fog_strength": 1.2,
  "bond_orders": true,
  "background": "#ffffff",
  "vdw_opacity": 0.25,
  "vdw_scale": 1.0,
  "vdw_gradient_strength": 0.845,
  "surface_opacity": 1.0,
  "mo_pos_color": "steelblue",
  "mo_neg_color": "maroon",
  "nci_mode": "avg",
  "dens_iso": 0.001,
  "dens_color": "steelblue",
  "label_font_size": 30,
  "label_color": "#222222",
  "label_offset": 1.5,
  "cmap_unlabeled": "#ffffff",
  "bond_color_by_element": false,
  "bond_gradient": false,
  "colors": {
    "C": "silver",
    "H": "whitesmoke",
    "N": "slateblue",
    "O": "red"
  }
}
```

The `colors` key maps element symbols to hex values (`#D9D9D9`) or [CSS4 named colors](https://matplotlib.org/stable/gallery/color/named_colors.html) (`steelblue`), overriding the default CPK palette.

Surface-related keys (`mo_pos_color`, `mo_neg_color`, `dens_iso`, `dens_color`) are only used when `--mo`, `--dens`, or `--esp` is active.

## Output formats

The output format is determined by the file extension of `-o`:

```bash
xyzrender caffeine.xyz -o out.svg   # SVG (default, scalable)
xyzrender caffeine.xyz -o out.png   # PNG (rasterised)
xyzrender caffeine.xyz -o out.pdf   # PDF (vector)
```

If no `-o` is given, output defaults to `{input_basename}.svg`.

## Styling flags

| Flag | Description |
|------|-------------|
| `-a`, `--atom-scale` | Atom radius scale factor |
| `-b`, `--bond-width` | Bond line width |
| `-s`, `--atom-stroke-width` | Atom outline width |
| `--bond-color` | Bond color (hex or named) |
| `--ts-color` | Color for dashed TS bonds (hex or named) |
| `--nci-color` | Color for dotted NCI bonds (hex or named) |
| `-S`, `--canvas-size` | Canvas size in pixels (default: 800) |
| `-B`, `--background` | Background color (hex or named, default: `#ffffff`) |
| `-t`, `--transparent` | Transparent background |
| `--grad` / `--no-grad` | Toggle radial gradients |
| `-G`, `--gradient-strength` | Gradient contrast |
| `--fog` / `--no-fog` | Toggle depth fog |
| `-F`, `--fog-strength` | Depth fog strength |
| `--bo` / `--no-bo` | Toggle bond order rendering |
| `--vdw-opacity` | vdW sphere opacity |
| `--vdw-scale` | vdW sphere radius scale |
| `--vdw-gradient` | vdW sphere gradient strength |
| `--bond-by-element` / `--no-bond-by-element` | Color bonds by endpoint atom colors |
| `--bond-gradient` / `--no-bond-gradient` | Cylinder shading on bonds (3D tube look) |
| `--region ATOMS CONFIG` | Render atom subset with a different preset (repeatable) |
