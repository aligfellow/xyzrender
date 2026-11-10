# Agent notes for xyzrender

Publication-quality molecular graphics from XYZ, QM outputs, cubes, and related formats.

## MCP (optional)

```bash
pip install -e ".[mcp]"
python -m xyzrender.mcp
```

Cursor example config: [`.cursor/mcp.json`](.cursor/mcp.json).

| Tool | Use when |
|------|----------|
| `render_molecule` | Static SVG/PNG/PDF (preferred) |
| `render_gif` | Animation explicitly requested |
| `measure_geometry` | Distances / angles / dihedrals as JSON |
| `inspect_structure` | Atom count, formula, cube/cell/TS/NCI hints |
| `list_presets` | Built-in names for `config=` (JSON path also OK) |

Atom indices and bond pairs are **1-indexed**. `ts` / `nci` are graph overlays; `nci_surf` is a volumetric RDG cube. Do not open the interactive viewer.

## Python API

```python
from xyzrender import load, render, render_gif, measure

mol = load("mol.xyz")
render(mol, output="fig.png", hy=True, ts=True)
measure(mol)  # dict with distances / angles / dihedrals
```
