"""Tests for cmap.py."""

from xml.etree import ElementTree as ET

import pytest

from xyzrender.cmap import atom_colors, bond_color_hex, cmap_value_range, colorbar_extra_width, colorbar_svg


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


@pytest.mark.parametrize("height", [80, 240, 800])
def test_colorbar_title_and_ticks_fit_legend_bounds(height):
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


@pytest.mark.parametrize(
    ("values", "bounds", "message"),
    [
        ([], None, "at least one scalar"),
        ([float("nan")], None, "scalar values must be finite"),
        ([float("inf")], (0, 1), "scalar values must be finite"),
        ([float("-inf")], None, "scalar values must be finite"),
        ([1], (float("nan"), 1), "range must have finite bounds"),
        ([1], (0, float("inf")), "range must have finite bounds"),
        ([1], (2, 1), "vmin <= vmax"),
    ],
)
def test_cmap_range_rejects_invalid_scalars_and_bounds(values, bounds, message):
    with pytest.raises(ValueError, match=message):
        cmap_value_range(values, cmap_range=bounds, cmap_symm=False)
