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
        with pytest.raises(ValueError, match="Atom and bond colormaps are mutually exclusive"):
            render_gif(mol, config=RenderConfig(atom_cmap={0: 0.2}), bond_cmap={}, gif_rot="y", output=output)
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


@pytest.mark.parametrize(
    ("source", "scene", "detect"),
    [
        ("dict", "single", True),
        ("file", "single", True),
        ("config", "single", True),
        ("config", "ensemble", True),
        ("config", "overlay", True),
        ("dict", "ensemble", False),
    ],
)
def test_explicit_missing_bond_survives_scene_preparation_without_mutating_input(tmp_path, source, scene, detect):
    import networkx as nx
    import numpy as np

    from xyzrender.api import EnsembleFrames, Molecule

    graph = nx.Graph()
    graph.add_node(0, symbol="C", position=(-2.0, 0.0, 0.0))
    graph.add_node(1, symbol="O", position=(2.0, 0.0, 0.0))
    graph.graph["ncis"] = []  # Re-detection removes old NCI edges before rebuilding.
    mol = Molecule(graph)
    config = RenderConfig(cmap_range=(0, 1), cmap_palette="viridis", fog=False, gradient=False, auto_orient=False)
    kwargs = {}
    if source == "config":
        config.bond_cmap = {(0, 1): 1.0}
    elif source == "file":
        path = tmp_path / "pairs.txt"
        path.write_text("1 2 1.0\n")
        kwargs["bond_cmap"] = path
    else:
        kwargs["bond_cmap"] = {(1, 2): 1.0}
    if scene == "ensemble":
        positions = np.array([[[-2.0, 0, 0], [2.0, 0, 0]], [[-2.0, 1, 0], [2.0, 1, 0]]])
        mol.ensemble = EnsembleFrames(positions, [None, None], [None, None])
    elif scene == "overlay":
        kwargs.update(overlay=mol.copy(), auto_align=False)
    svg = str(render(mol, config=config, detect_nci=detect, **kwargs))
    expected = bond_color_hex(1, "viridis", 0, 1)
    root = ET.fromstring(svg)
    dotted = [el for el in root.iter() if "stroke-dasharray" in el.attrib]
    assert len(dotted) == 1  # No uncoloured copies on other conformers.
    assert dotted[0].get("stroke") == expected
    assert not graph.edges
    assert graph.graph["ncis"] == []


def test_missing_bond_maps_render_in_rotation_and_rebuilt_trajectory_frames(tmp_path, monkeypatch):
    import copy
    from io import BytesIO

    import numpy as np
    from PIL import Image

    from xyzrender import gif

    svgs = []
    original_render = gif.render_svg

    def record_svg(*args, **kwargs):
        svg = original_render(*args, **kwargs)
        svgs.append(svg)
        return svg

    def serial(worker, items, total):
        pngs = [b""] * total
        for item in items:
            owned_worker, owned_item = copy.deepcopy((worker, item))
            idx, png = owned_worker(owned_item)
            pngs[idx] = png
        return pngs

    monkeypatch.setattr(gif, "render_svg", record_svg)
    monkeypatch.setattr(gif, "_parallel_render", serial)
    path = tmp_path / "separated.xyz"
    frame = "3\nseparated\nH 0 6 0\nC -2 0 0\nO 2 0 0\n"
    path.write_text(frame * 2)
    cmap_file = tmp_path / "pairs.txt"
    cmap_file.write_text("2 3 1\n")
    config = RenderConfig(
        canvas_size=320, cmap_range=(0, 1), cmap_palette="viridis", fog=False, gradient=False, auto_orient=False
    )
    mol = load(path)
    for name in ("rotation", "trajectory"):
        svgs.clear()
        output = tmp_path / f"{name}.gif"
        if name == "rotation":
            render_gif(mol, config=config, gif_rot="z", rot_frames=2, only="2,3", bond_cmap=cmap_file, output=output)
        else:
            config.bond_cmap = {(1, 2): 1.0}
            render_gif(path, config=config, gif_trj=True, trj_bonds=True, detect_nci=True, output=output)
        assert len(svgs) == 2
        expected = bond_color_hex(1, "viridis", 0, 1)
        for svg in svgs:
            root = ET.fromstring(svg)
            assert any(el.get("stroke") == expected and "stroke-dasharray" in el.attrib for el in root.iter())
        image = Image.open(BytesIO(output.read_bytes())).convert("RGB")
        r, g, b = tuple(int(expected[i : i + 2], 16) for i in (1, 3, 5))
        pixels = np.asarray(image, dtype=int)
        assert np.abs(pixels - (r, g, b)).sum(axis=2).min() < 60
    assert not mol.graph.edges


@pytest.mark.parametrize("priority", ["edge", "highlight", "overlay-edge"])
def test_local_bond_color_priority_uses_source_not_rgb_equality(priority):
    import networkx as nx

    from xyzrender.api import Molecule
    from xyzrender.colors import Color, bond_color_from_atom

    derived = bond_color_from_atom(Color.from_str("#00ffff"))
    graph = nx.Graph()
    graph.add_node(0, symbol="C", position=(-1.0, 0.0, 0.0))
    graph.add_node(1, symbol="C", position=(1.0, 0.0, 0.0))
    graph.add_edge(0, 1, bond_order=1.0)
    kwargs = {}
    expected = derived
    if priority == "edge":
        graph[0][1]["bond_color_override"] = derived
    elif priority == "highlight":
        kwargs["highlight"] = [("1-2", "#00ffff")]
    else:
        extra = graph.copy()
        extra[0][1]["bond_color_override"] = expected = "#ff8800"
        for i in extra:
            x, y, z = extra.nodes[i]["position"]
            extra.nodes[i]["position"] = (x, y + 2.0, z)
        kwargs.update(overlay=Molecule(extra), overlay_color="#00ff00", auto_align=False)
    config = RenderConfig(cmap_range=(0, 1), cmap_palette="viridis", fog=False, gradient=False, auto_orient=False)
    svg = str(render(Molecule(graph), config=config, mol_color="#00ffff", bond_cmap={(1, 2): 1.0}, **kwargs))
    lines = [el for el in ET.fromstring(svg).iter() if el.tag.endswith("}line")]
    assert any(el.get("stroke") == expected for el in lines)
    if priority != "overlay-edge":
        assert all(el.get("stroke") != "#fde725" for el in lines)
