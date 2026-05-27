# Configuration

## Built-in presets

Use `--config` to load a styling preset. Built-in options: `default`, `flat`, `paton`, `pmol`, `skeletal`, `bubble`, `vdw`, `tube`, `mtube`, `btube`, `wire`, `graph`.

| Preset | Description |
|--------|-------------|
| `default` | Radial gradients, depth fog, CPK colors |
| `flat` | No gradients, no fog — clean flat look |
| `paton` | PyMOL-inspired style (see [Rob Paton](https://github.com/patonlab)) |
| `pmol` | Ball-and-stick with element-coloured split bonds and tube shading (PyMOL-inspired) |
| `skeletal` | Skeletal formula diagram — thin bonds, minimal atoms |
| `bubble` | Space-filling (CPK) — large atoms, no bonds |
| `vdw` | True space-filling — atoms at vdW radii, interlocked silhouettes (no gaps at contacts) |
| `tube` | Tube/stick model — no atoms, thick element-coloured bonds with cylinder shading |
| `mtube` | Metal tube — tube bonds with black edge stroke; metals auto-highlighted via preset region |
| `btube` | Ball-and-tube — ball-and-stick atoms with element-coloured tube bonds and outline stroke |
| `wire` | Wireframe — no atoms, thin element-coloured bonds with cylinder shading |
| `graph` | Minimal graph look — teal bonds, bold outlined nodes with light tinted centers |

```bash
xyzrender caffeine.xyz --config flat
xyzrender caffeine.xyz --config paton
xyzrender caffeine.xyz --config pmol
xyzrender caffeine.xyz --config skeletal
xyzrender caffeine.xyz --config bubble --hy
xyzrender caffeine.xyz --config vdw
xyzrender caffeine.xyz --config tube
xyzrender caffeine.xyz --config mtube
xyzrender caffeine.xyz --config btube
xyzrender caffeine.xyz --config wire
xyzrender caffeine.xyz --config graph
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
  "atom_gradient_strength": 1.0,
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
  "bond_gradient_strength": 0.3,
  "bond_outline_color": "#000000",
  "bond_outline_width": 0,
  "atom_wash": 0.0,
  "atoms_above_bonds": false,
  "colors": {
    "C": "silver",
    "H": "whitesmoke",
    "N": "slateblue",
    "O": "red"
  },
  "regions": {
    "M": "flat"
  }
}
```

The `colors` key maps element symbols to hex values (`#D9D9D9`) or [CSS4 named colors](https://matplotlib.org/stable/gallery/color/named_colors.html) (`steelblue`), overriding the default CPK palette.

`bond_outline_color` / `bond_outline_width` add a shadow edge behind bonds (visible as an outline). Set `bond_outline_width` > 0 to activate (color defaults to black). In styles with a visible atom disc (`pmol`, `btube`, `paton`, `default`), each bond's outline interleaves with the atoms so a front bond keeps its halo where it crosses a back bond. In flat styles (`tube`, `mtube`, `wire`) the outlines all sit in a single back-layer under the bonds.

`bond_gap` sets the spacing between stripes in double, triple, and aromatic bonds (default `0.6`, units = fraction of `bond_width`). Larger values widen the gap between stripes.

The `regions` key defines per-atom-group style overrides. Keys are atom selectors (`M` = metals, `Pt`, `sbm` = s-block metals, `het` = heteroatoms, or numeric `1-5`). Values are a preset name or an inline dict of overrides:

```json
"regions": {
  "M": "flat",
  "het": { "atom_scale": 3.0, "gradient": true }
}
```

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
| `--atom-gradient-strength` | Atom gradient strength (default: 1.0) |
| `--bond-gradient-strength` | Bond cylinder gradient strength (default: 0.3) |
| `--vdw-gradient-strength` | vdW sphere gradient strength (default: 1.6) |
| `--fog` / `--no-fog` | Toggle depth fog |
| `-F`, `--fog-strength` | Depth fog strength |
| `--bo` / `--no-bo` | Toggle bond order rendering |
| `--vdw-opacity` | vdW sphere opacity |
| `--vdw-scale` | vdW sphere radius scale |
| `--bond-by-element` / `--no-bond-by-element` | Color bonds by endpoint atom colors |
| `--bond-gradient` / `--no-bond-gradient` | Cylinder shading on bonds (3D tube look) |
| `--radius-scale ATOMS FACTOR` | Scale selected atoms (repeatable). Multiplies on top of `-a`. Selectors: `"1-5,M"` |
| `--region ATOMS CONFIG` | Render atom subset with a different preset (repeatable). Selectors: `"1-5"`, `"M"`, `"Pt"`, `"sbm"`, `"het"` |
