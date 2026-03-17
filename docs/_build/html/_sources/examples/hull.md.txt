# Convex hull

Draw the convex hull of selected atoms as semi-transparent facets — useful for aromatic rings, coordination spheres, or any subset of atoms. Facets are depth-sorted for correct occlusion. Hull edges that do not coincide with bonds are drawn as thin lines for better 3D perception; disable with `--no-hull-edge`.

Use `--hull` from the CLI (no args = all heavy atoms, `rings` to auto-detect aromatic rings, or 1-indexed atom ranges for subsets), or from Python pass `hull=` to `render()`: `True` for all heavy atoms, `"rings"` for automatic aromatic ring detection (one hull per ring), a flat list of 1-indexed atom indices for one hull, or a list of lists for multiple hulls with optional per-subset `hull_color=["red", "blue"]`. A default color palette cycles automatically for multiple subsets.

| Benzene ring | Anthracene (all ring carbons) | CoCl₆ octahedron |
|--------------|-------------------------------|------------------|
| ![benzene hull](../../../examples/images/benzene_ring_hull.svg) | ![anthracene hull](../../../examples/images/anthracene_hull.svg) | ![CoCl6 hull](../../../examples/images/CoCl6_octahedron_hull.svg) |

| Anthracene ring | Anthracene rot | Auto rings (`hull="rings"`) |
|--------------|------------------|----------------------------|
| ![anthracene hull](../../../examples/images/anthracene_hull_one.svg) | ![anthracene hull](../../../examples/images/anthracene_hull.gif) | ![mnh hull rings](../../../examples/images/mnh_hull_rings.svg) |

**CLI:**

```bash
# All heavy atoms:
xyzrender benzene.xyz --hull -o benzene_hull.svg

# Single subset (1-indexed atom range):
xyzrender benzene.xyz --hull 1-6 --hull-color steelblue --hull-opacity 0.35 -o benzene_ring_hull.svg

# Multiple subsets with per-hull colors:
xyzrender anthracene.xyz --hull 1-6 4,6-10 8,10-14 -o anthracene_hull.svg

# Auto-detect aromatic rings (one hull per ring, colours cycle automatically):
xyzrender mn-h2.log --ts --hull rings --hull-color teal -o mnh_hull_rings.svg
```

**Python:**

```python
from xyzrender import load, render, render_gif

# Single subset: one hull (e.g. benzene ring carbons, 1-indexed)
benzene = load("structures/benzene.xyz")
render(benzene, hull=[1, 2, 3, 4, 5, 6],
       hull_color="steelblue", hull_opacity=0.35, output="images/benzene_ring_hull.svg")
render_gif(benzene, gif_rot="y", hull=[1, 2, 3, 4, 5, 6],
           hull_color="steelblue", hull_opacity=0.35, output="images/benzene_ring_hull.gif")

# Multiple subsets with per-subset colors (1-indexed):
render(mol, hull=[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
       hull_color=["steelblue", "coral"], hull_opacity=0.35,
       output="anthracene_hull.svg")

# Auto-detect aromatic rings — each ring gets its own hull:
render(mol, hull="rings", hull_color="teal")
```

**Options (passed to `render()`):**

| Option | Description |
|--------|-------------|
| `hull` | `True` = all heavy atoms; `"rings"` = auto-detect aromatic rings (one hull per ring); flat list = one subset; list of lists = multiple hulls |
| `hull_color` | Single string or list of strings for per-subset colours (default palette cycles automatically) |
| `hull_opacity` | Fill opacity for all hull surfaces |
| `hull_edge` | Draw non-bond hull edges as thin lines (default: `True`) |
| `hull_edge_width_ratio` | Edge stroke width as fraction of bond width |

Examples in this section are generated from `examples/examples.ipynb` (benzene, anthracene, CoCl₆).
