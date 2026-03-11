"""Render benzene with convex hull facets restricted to the 6 ring carbons.

This produces an SVG with a single coloured surface in the middle of the carbon ring.
Benzene XYZ from: https://github.com/nutjunkie/IQmol/blob/master/share/fragments/Molecules/Aromatics/Benzene.xyz

For multiple hulls (e.g. several rings), set cfg.hull_atom_indices to a list of index
lists: [[1, 2, 4, 6, 8, 10], [0, 3, 5]] for two separate hulls (0-based atom indices).
"""

from pathlib import Path

from xyzrender import load, render, render_gif
from xyzrender.config import build_config

# Benzene: 12 atoms — first 6 are C (indices 0-5), then 6 H. Ring = all 6 carbons (0-based).
# Single subset: one hull. Use list of lists for multiple hulls.
CARBON_INDICES = [0, 1, 2, 3, 4, 5]

ROOT = Path(__file__).resolve().parent.parent
BENZENE_XYZ = ROOT / "examples" / "structures" / "benzene.xyz"
OUT_SVG = ROOT / "examples" / "images" / "benzene_ring_hull.svg"
OUT_GIF = ROOT / "examples" / "images" / "benzene_ring_hull.gif"


def main() -> None:
    """Load benzene, render with ring hull (SVG + rotation GIF)."""
    mol = load(BENZENE_XYZ)
    cfg = build_config("default", hull=True, hull_color="steelblue", hull_opacity=0.35)
    cfg.hull_atom_indices = CARBON_INDICES
    render(mol, config=cfg, output=OUT_SVG)
    print(f"Wrote {OUT_SVG}")
    render_gif(mol, gif_rot="y", config=cfg, output=OUT_GIF)
    print(f"Wrote {OUT_GIF}")
    print("Open in a browser or viewer to see the coloured ring surface.")


if __name__ == "__main__":
    main()
