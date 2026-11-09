"""Atom and bond maps must assign equal colours to equal scalars and ranges."""

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from xyzrender import load, render, render_gif
from xyzrender.cmap import atom_colors, bond_color_hex, cmap_value_range
from xyzrender.types import RenderConfig


@pytest.mark.parametrize("palette", ["coolwarm", "viridis", "white_reverse_plasma"])
@pytest.mark.parametrize(("vmin", "vmax"), [(-1.0, 1.0), (2.0, 2.0), (-3.0, 5.0)])
def test_equal_atom_and_bond_values_share_color_including_boundaries_and_clamping(palette, vmin, vmax):
    for value in (-10.0, -1.0, 0.0, 0.25, 1.0, 2.0, 10.0):
        colors = atom_colors({0: value}, 2, palette, vmin, vmax, "#abcdef")
        assert colors[0].hex == bond_color_hex(value, palette, vmin, vmax)
        assert colors[1].hex == "#abcdef"


@pytest.mark.parametrize("symmetric", [False, True])
def test_equal_atom_and_bond_values_share_color_with_resolved_data_ranges(symmetric):
    values = [-2.0, 0.25, 1.0]
    vmin, vmax = cmap_value_range(values, cmap_range=None, cmap_symm=symmetric)
    colors = atom_colors(dict(enumerate(values)), len(values), "coolwarm", vmin, vmax, "#abcdef")
    assert [color.hex for color in colors] == [bond_color_hex(value, "coolwarm", vmin, vmax) for value in values]


@pytest.mark.parametrize("value", [-1.0, 0.25, 1.0])
def test_rendered_atom_and_bond_share_color_with_explicit_common_range(value):
    structure = Path(__file__).resolve().parent.parent / "examples/structures/ethanol.xyz"
    config = RenderConfig(
        cmap_range=(-1, 1), cmap_palette="coolwarm", gradient=False, fog=False, auto_orient=False, bond_orders=False
    )
    atom_svg = str(render(structure, cmap={1: value}, config=config))
    bond_svg = str(render(structure, bond_cmap={(1, 2): value}, config=config))
    atom_root = ET.fromstring(atom_svg)
    bond_root = ET.fromstring(bond_svg)
    namespaces = {"svg": "http://www.w3.org/2000/svg"}
    expected = bond_color_hex(value, "coolwarm", -1, 1)
    assert any(circle.attrib.get("fill") == expected for circle in atom_root.findall(".//svg:circle", namespaces))
    assert any(line.attrib.get("stroke") == expected for line in bond_root.findall(".//svg:line", namespaces))


@pytest.mark.parametrize(
    ("configured", "requested"),
    [("none", "both"), ("atom", "bond"), ("bond", "atom"), ("both", "none"), ("none", "empty")],
)
def test_conflicting_maps_rejected_before_resolution_or_graph_mutation(configured, requested):
    structure = Path(__file__).resolve().parent.parent / "examples/structures/ethanol.xyz"
    mol = load(structure)
    before = list(mol.graph.edges(data=True))
    config = RenderConfig(
        atom_cmap={0: 0.2} if configured in {"atom", "both"} else None,
        bond_cmap={(0, 2): 0.5} if configured in {"bond", "both"} else None,
    )
    atom = {} if requested == "empty" else ({1: 0.2} if requested in {"atom", "both"} else None)
    bond = {} if requested == "empty" else ({(1, 3): 0.5} if requested in {"bond", "both"} else None)
    with pytest.raises(ValueError, match="Atom and bond colormaps are mutually exclusive"):
        render(mol, config=config, cmap=atom, bond_cmap=bond)
    assert list(mol.graph.edges(data=True)) == before


@pytest.mark.parametrize("entrypoint", ["gif", "svg"])
def test_configured_conflicting_maps_rejected_at_render_boundaries(tmp_path, entrypoint):
    from xyzrender.renderer import render_svg

    structure = Path(__file__).resolve().parent.parent / "examples/structures/ethanol.xyz"
    mol = load(structure)
    config = RenderConfig(atom_cmap={0: 0.2}, bond_cmap={(0, 1): 0.5})
    output = tmp_path / "conflicting.gif"
    if entrypoint == "gif":
        with pytest.raises(ValueError, match="Atom and bond colormaps are mutually exclusive"):
            render_gif(mol, config=config, gif_rot="y", output=output)
    else:
        with pytest.raises(ValueError, match="Atom and bond colormaps are mutually exclusive"):
            render_svg(mol.graph, config)
    assert not output.exists()


@pytest.mark.parametrize("height", [80, 240, 800])
def test_colorbar_title_and_ticks_fit_legend_bounds(height):
    from xyzrender.cmap import colorbar_extra_width, colorbar_svg

    width = 240 + colorbar_extra_width(0, 3, 40, "Bond order")
    elements = colorbar_svg(0, 3, "viridis", 240, height, 40, "black", unit="Bond order")
    root = ET.fromstring(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">' + "".join(elements) + "</svg>"
    )
    ns = {"svg": "http://www.w3.org/2000/svg"}
    title = next(text for text in root.findall("svg:text", ns) if text.text == "Bond order")
    title_size = float(title.attrib["font-size"].removesuffix("px"))
    title_y = float(title.attrib["y"])
    assert title_size / 2 < title_y < height - title_size / 2
    assert 240 < float(title.attrib["x"]) < width
    for text in root.findall("svg:text", ns):
        size = float(text.attrib["font-size"].removesuffix("px"))
        assert size / 2 <= float(text.attrib["y"]) <= height - size / 2


def test_colorbar_title_visible_with_existing_cairo_renderer():
    from io import BytesIO

    import cairosvg
    import numpy as np
    from PIL import Image

    from xyzrender.cmap import colorbar_extra_width, colorbar_svg

    width = 240 + colorbar_extra_width(0, 3, 40, "Bond order")
    elements = colorbar_svg(0, 3, "viridis", 240, 240, 40, "black", unit="Bond order")
    root = ET.fromstring(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="240">' + "".join(elements) + "</svg>"
    )
    ns = {"svg": "http://www.w3.org/2000/svg"}
    title = next(text for text in root.findall("svg:text", ns) if text.text == "Bond order")
    size = float(title.attrib["font-size"].removesuffix("px"))
    center_y = float(title.attrib["y"])
    image = Image.open(BytesIO(cairosvg.svg2png(bytestring=ET.tostring(root)))).convert("RGBA")
    pixels = np.asarray(image)
    title_strip = pixels[max(0, int(center_y - size)) : int(center_y + size), 240:]
    assert np.any(title_strip[:, :, 3] > 128)
