"""Tests for renderer.py."""

import pytest


def test_render_svg_rejects_conflicting_maps():
    from pathlib import Path

    from xyzrender import load
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    mol = load(Path(__file__).resolve().parent.parent / "examples/structures/ethanol.xyz")
    config = RenderConfig(atom_cmap={0: 0.2}, bond_cmap={(0, 1): 0.5})
    with pytest.raises(ValueError, match="Atom and bond colormaps are mutually exclusive"):
        render_svg(mol.graph, config)


@pytest.mark.parametrize("priority", ["edge", "highlight"])
def test_bond_map_respects_explicit_colors_even_when_rgb_matches_molecule_color(priority):
    from xml.etree import ElementTree as ET

    import networkx as nx

    from xyzrender import render
    from xyzrender.api import Molecule
    from xyzrender.colors import Color, bond_color_from_atom
    from xyzrender.types import RenderConfig

    derived = bond_color_from_atom(Color.from_str("#00ffff"))
    graph = nx.Graph()
    graph.add_node(0, symbol="C", position=(-1.0, 0.0, 0.0))
    graph.add_node(1, symbol="C", position=(1.0, 0.0, 0.0))
    graph.add_edge(0, 1, bond_order=1.0)
    if priority == "edge":
        graph[0][1]["bond_color_override"] = derived
    config = RenderConfig(cmap_range=(0, 1), cmap_palette="viridis", fog=False, gradient=False, auto_orient=False)
    svg = str(
        render(
            Molecule(graph),
            config=config,
            mol_color="#00ffff",
            bond_cmap={(1, 2): 1.0},
            highlight=[("1-2", "#00ffff")] if priority == "highlight" else None,
        )
    )
    lines = [element for element in ET.fromstring(svg).iter() if element.tag.endswith("}line")]
    assert any(element.get("stroke") == derived for element in lines)
    assert all(element.get("stroke") != "#fde725" for element in lines)
