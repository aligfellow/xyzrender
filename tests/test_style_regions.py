"""Tests for style regions, element-coloured bonds, and cylinder shading."""

from __future__ import annotations

import copy
import json
import re
import tempfile
from pathlib import Path

import pytest

from xyzrender import load, render
from xyzrender.colors import get_color
from xyzrender.types import Color, RenderConfig, StyleRegion, resolve_color
from xyzrender.utils import parse_atom_indices

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


@pytest.fixture(scope="module")
def caffeine():
    return load(STRUCTURES / "caffeine.xyz")


@pytest.fixture(scope="module")
def ethanol():
    return load(STRUCTURES / "ethanol.xyz")


# ---------------------------------------------------------------------------
# parse_atom_indices
# ---------------------------------------------------------------------------


class TestParseAtomIndices:
    def test_simple_range(self):
        assert parse_atom_indices("1-5") == [0, 1, 2, 3, 4]

    def test_mixed(self):
        assert parse_atom_indices("1-3,7,10-12") == [0, 1, 2, 6, 9, 10, 11]

    def test_list_1indexed(self):
        assert parse_atom_indices([1, 2, 3]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Tube preset
# ---------------------------------------------------------------------------


class TestTubePreset:
    def test_tube_no_visible_atom_circles(self, caffeine):
        """Tube preset should produce no visible atom circles (r=0.0)."""
        svg = str(render(caffeine, config="tube", orient=False))
        radii = re.findall(r'<circle[^>]*r="([^"]+)"', svg)
        for r in radii:
            assert float(r) == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# Element-coloured bonds
# ---------------------------------------------------------------------------


class TestElementColouredBonds:
    def test_heteroatom_bond_shows_both_colours(self, ethanol):
        """C-O bond should show the oxygen CPK colour in the SVG."""
        svg = str(render(ethanol, config="tube", bond_gradient=False, fog=False, orient=False, hy=True))
        o_color = get_color(8, None).hex  # atomic number 8 = O
        assert o_color in svg

    def test_nci_bonds_never_element_coloured(self):
        """NCI (dotted) bonds should use uniform colour even with bond_color_by_element."""
        mol = load(STRUCTURES / "Hbond.xyz", nci_detect=True)
        svg = str(render(mol, bond_color_by_element=True, fog=False, gradient=False, orient=False))
        dotted = [line for line in svg.split("\n") if "stroke-dasharray" in line]
        assert len(dotted) > 0
        for line in dotted:
            assert "url(#" not in line

    def test_nci_bond_color_override(self):
        """nci_color should control dotted NCI bond color."""
        import networkx as nx

        from xyzrender.renderer import render_svg

        g = nx.Graph()
        # Use centroid dummy nodes ("*") so this test is independent of
        # element-radius tables from optional xyzgraph versions.
        g.add_node(0, symbol="*", position=[0.0, 0.0, 0.0])
        g.add_node(1, symbol="*", position=[1.2, 0.0, 0.0])
        g.add_edge(0, 1, bond_order=1.0, bond_type="NCI")

        cfg = RenderConfig(
            fog=False,
            gradient=False,
            auto_orient=False,
            nci_color="#ff00ff",
        )
        svg = render_svg(g, cfg, _unique_ids=False)
        dotted = [line for line in svg.split("\n") if "stroke-dasharray" in line and "<line" in line]
        assert len(dotted) > 0
        assert any("#ff00ff" in line for line in dotted)


# ---------------------------------------------------------------------------
# Cylinder shading (bond_gradient)
# ---------------------------------------------------------------------------


class TestCylinderShading:
    def test_gradient_uses_five_stops(self, caffeine):
        """Cylinder shading should use 5-stop gradient (lo->me->hi->me->lo)."""
        svg = str(render(caffeine, bond_gradient=True, fog=False, orient=False))
        grad_match = re.search(r"<linearGradient[^>]*>(.*?)</linearGradient>", svg)
        assert grad_match is not None
        stops = grad_match.group(1).count("stop offset")
        assert stops == 5

    def test_shading_not_applied_to_dashed_bonds(self):
        """TS (dashed) bonds should not get cylinder shading."""
        mol = load(STRUCTURES / "sn2.out", ts_detect=True)
        svg = str(render(mol, bond_gradient=True, fog=False, orient=False, hy=True))
        dashed = [line for line in svg.split("\n") if "stroke-dasharray" in line and "<line" in line]
        for line in dashed:
            assert "url(#" not in line


# ---------------------------------------------------------------------------
# Style regions
# ---------------------------------------------------------------------------


class TestStyleRegions:
    def test_region_atoms_get_different_radius(self, caffeine):
        """Atoms in a tube region should have r~0 while base atoms have r>0."""
        svg = str(
            render(caffeine, config="default", regions=[("1-5", "tube")], fog=False, gradient=False, orient=False)
        )
        radii = [float(m) for m in re.findall(r'<circle[^>]*r="([^"]+)"', svg)]
        has_zero = any(r < 0.1 for r in radii)
        has_nonzero = any(r > 1.0 for r in radii)
        assert has_zero, "Tube region atoms should have near-zero radius"
        assert has_nonzero, "Base atoms should have visible radius"

    def test_region_atoms_get_region_colours(self, ethanol):
        """Atoms in a region with custom color_overrides should use those colours."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"colors": {"O": "#00ff00"}}, f)
            path = f.name
        svg = str(
            render(ethanol, config="default", regions=[("1-9", path)], fog=False, gradient=False, orient=False, hy=True)
        )
        assert "#00ff00" in svg

    def test_bond_orders_per_region(self, caffeine):
        """Region with bond_orders=False should flatten bond orders (fewer lines)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"bond_orders": False, "bond_color_by_element": False}, f)
            path = f.name
        svg_base = str(render(caffeine, config="default", fog=False, gradient=False, orient=False))
        svg_flat_bo = str(
            render(caffeine, config="default", regions=[("1-24", path)], fog=False, gradient=False, orient=False)
        )
        assert svg_flat_bo.count("<line") < svg_base.count("<line"), "Flattened bond orders should produce fewer lines"

    def test_no_mutation(self, caffeine):
        """Molecule graph must not be mutated by region rendering."""
        pos_before = {n: copy.deepcopy(caffeine.graph.nodes[n]["position"]) for n in caffeine.graph.nodes()}
        nodes_before = set(caffeine.graph.nodes())
        render(caffeine, config="tube", regions=[("1-10", "default")], orient=False)
        assert set(caffeine.graph.nodes()) == nodes_before
        for n in caffeine.graph.nodes():
            assert caffeine.graph.nodes[n]["position"] == pos_before[n]

    def test_overlapping_regions_raises(self, caffeine):
        """Atoms appearing in multiple regions should raise ValueError."""
        with pytest.raises(ValueError, match="appear in multiple style regions"):
            render(caffeine, regions=[("1-5", "flat"), ("3-8", "tube")], orient=False)


# ---------------------------------------------------------------------------
# Structural overlay isolation
# ---------------------------------------------------------------------------


class TestOverlayIsolation:
    def test_centroid_nodes_use_base_config(self):
        """NCI centroid (*) nodes should use base config, not region config.

        Synthetic graph: two atoms + one centroid. Real atoms in tube region
        (atom_scale=0), centroid should keep base config's visible radius.
        """
        import networkx as nx

        from xyzrender.renderer import render_svg

        g = nx.Graph()
        g.add_node(0, symbol="C", position=[0.0, 0.0, 0.0])
        g.add_node(1, symbol="N", position=[1.5, 0.0, 0.0])
        g.add_node(2, symbol="*", position=[0.75, 0.8, 0.0])  # NCI centroid
        g.add_edge(0, 1, bond_order=1.0)
        g.add_edge(0, 2, bond_order=1.0, bond_type="NCI")
        g.add_edge(1, 2, bond_order=1.0, bond_type="NCI")

        tube_cfg = RenderConfig(atom_scale=0, atom_stroke_width=0)
        cfg = RenderConfig(
            fog=False,
            gradient=False,
            auto_orient=False,
            style_regions=[StyleRegion(indices=[0, 1], config=tube_cfg)],
        )

        svg = render_svg(g, cfg, _unique_ids=False)
        radii = [float(m) for m in re.findall(r'<circle[^>]*r="([^"]+)"', svg)]
        assert any(r < 0.1 for r in radii), "Tube atoms should have near-zero radius"
        assert any(r > 0.5 for r in radii), "Centroid (*) node should keep base config radius"

    def test_ts_bonds_width_capped(self):
        """TS bonds should have width capped even when base is tube (bond_width=50)."""
        mol = load(STRUCTURES / "sn2.out", ts_detect=True)
        svg = str(render(mol, config="tube", fog=False, orient=False, hy=True))
        dashed = re.findall(
            r'<line[^>]*stroke-dasharray="[^"]*"[^>]*stroke-width="([^"]+)"',
            svg,
        )
        for w in dashed:
            assert float(w) < 30, f"TS bond width {w} should be capped below tube's bond_width"

    def test_highlight_does_not_colour_nci_bonds(self):
        """Highlight should not recolour NCI (dotted) bonds."""
        mol = load(STRUCTURES / "Hbond.xyz", nci_detect=True)
        n = mol.graph.number_of_nodes()
        dark = Color.from_str(resolve_color("orchid")).blend(Color(0, 0, 0), 0.3).hex
        svg = str(render(mol, highlight=list(range(1, n + 1)), fog=False, gradient=False, orient=False, hy=True))
        dotted_lines = [line for line in svg.split("\n") if "stroke-dasharray" in line and "<line" in line]
        for line in dotted_lines:
            if 'stroke-dasharray="0.' in line or 'stroke-dasharray="1.' in line:
                assert dark not in line, "NCI dotted bond should not get highlight colour"


# ---------------------------------------------------------------------------
# Config edge cases
# ---------------------------------------------------------------------------


class TestConfigEdgeCases:
    def test_style_regions_stripped_from_json(self):
        """style_regions key in JSON should not cause an error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"gradient": False, "style_regions": [{"bad": "data"}]}, f)
            path = f.name
        from xyzrender.config import build_region_config

        cfg = build_region_config(path)
        assert cfg.style_regions == []
