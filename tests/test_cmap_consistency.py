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
