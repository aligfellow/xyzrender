"""Tests for style regions, element-coloured bonds, and cylinder shading."""

from __future__ import annotations

import copy
import json
import re
import tempfile
from pathlib import Path

import pytest

from xyzrender import load, render
from xyzrender.colors import Color, get_color, resolve_color
from xyzrender.types import RenderConfig, StyleRegion
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

    def test_mtube_has_flat_bonds_and_outline(self, caffeine):
        """mtube should disable gradient shading and emit edge stroke shadow."""
        svg = str(render(caffeine, config="mtube", orient=False))
        assert 'stroke="#000000"' in svg


# ---------------------------------------------------------------------------
# Element-coloured bonds
# ---------------------------------------------------------------------------


class TestElementColouredBonds:
    def test_heteroatom_bond_shows_both_colours(self, ethanol):
        """C-O bond should show the oxygen CPK colour in the SVG."""
        svg = str(render(ethanol, config="tube", bond_gradient=False, fog=False, orient=False, hy=True))
        o_color = get_color(8, None).hex  # atomic number 8 = O
        assert o_color in svg

    def test_nci_bonds_uniform_when_nci_element_off(self):
        """With nci_element=False (default) NCI (dotted) bonds use uniform colour
        even when bond_color_by_element is on. The split-half element path is
        opt-in via nci_element=True (pre-enabled in pmol/btube/tube/mtube/wire)."""
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
        g.add_edge(0, 1, bond_order=1.0, NCI=True)

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
# TS / NCI dash + width tuning
# ---------------------------------------------------------------------------


class TestDashAndWidthTuning:
    def test_coerce_dash_accepts_string_and_tuple(self):
        from xyzrender.config import _coerce_dash

        assert _coerce_dash(None) is None
        assert _coerce_dash("0.8,2.2") == (0.8, 2.2)
        assert _coerce_dash((0.8, 2.2)) == (0.8, 2.2)
        assert _coerce_dash([0.8, 2.2]) == (0.8, 2.2)
        with pytest.raises(ValueError, match="dash spec"):
            _coerce_dash("oops")
        with pytest.raises(ValueError, match="dash spec"):
            _coerce_dash("1.0")

    def test_build_config_coerces_dash_strings(self):
        from xyzrender import build_config

        cfg = build_config("default", ts_dash="0.8,2.2", nci_dash="0.1,3.0")
        assert cfg.ts_dash == (0.8, 2.2)
        assert cfg.nci_dash == (0.1, 3.0)

    def test_ts_dash_and_width_propagate_to_svg(self):
        """Custom ts_dash and ts_width should produce a TS line whose dash and
        width match the configured multipliers against the same bond_width
        reference."""
        import re

        mol = load(STRUCTURES / "sn2.out", ts_detect=True)
        dash_mults = (2.0, 1.5)
        width_mult = 0.5
        svg = str(
            render(
                mol,
                config="pmol",
                ts_dash=f"{dash_mults[0]},{dash_mults[1]}",
                ts_width=width_mult,
                orient=False,
                fog=False,
            )
        )
        dashed = [line for line in svg.split("\n") if "stroke-dasharray" in line and "<line" in line]
        assert dashed, "no TS dashed lines in SVG"

        def _grab(pattern: str, s: str) -> str:
            m = re.search(pattern, s)
            assert m is not None, f"no match for {pattern!r}"
            return m.group(1)

        # Dash ratio is independent of width / outline padding — robust check.
        line = dashed[0]
        dd, gg = (float(x) for x in _grab(r'stroke-dasharray="([0-9.,]+)"', line).split(","))
        assert dd / gg == pytest.approx(dash_mults[0] / dash_mults[1], rel=0.01)

        # Inner (fill) stroke width should equal dash_value * (ts_width / ts_dash[0]).
        # Pick the smaller of the two stroke widths (fill, not outline).
        widths = sorted(float(_grab(r'stroke-width="([0-9.]+)"', ln)) for ln in dashed[:2])
        fill_w = widths[0]
        bw_from_width = fill_w / width_mult
        bw_from_dash = dd / dash_mults[0]
        assert abs(bw_from_width - bw_from_dash) / bw_from_dash < 0.05


# ---------------------------------------------------------------------------
# Cylinder shading (bond_gradient)
# ---------------------------------------------------------------------------


class TestCylinderShading:
    def test_gradient_uses_three_stops(self, caffeine):
        """Cylinder shading should use 3-stop gradient (lo->hi->lo)."""
        svg = str(render(caffeine, bond_gradient=True, fog=False, orient=False))
        grad_match = re.search(r"<linearGradient[^>]*>(.*?)</linearGradient>", svg)
        assert grad_match is not None
        stops = grad_match.group(1).count("stop offset")
        assert stops == 3

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

    def test_element_selector_in_region(self, ethanol):
        """Element selectors like 'O' should work in render(regions=...)."""
        svg = str(render(ethanol, config="tube", regions=[("O", "default")], fog=False, orient=False, hy=True))
        radii = [float(m) for m in re.findall(r'<circle[^>]*r="([^"]+)"', svg)]
        assert any(r > 1.0 for r in radii), "O atom in default region should have visible radius"

    def test_bond_outline_in_render(self, caffeine):
        """bond_outline_width > 0 should produce outline strokes in the SVG."""
        svg = str(render(caffeine, bond_outline_width=5, fog=False, gradient=False, orient=False))
        assert 'stroke="#000000"' in svg

    def test_outline_drawn_per_bond_segment(self):
        """Outline strokes should follow visible bond segments for multi-bond orders."""
        import networkx as nx

        from xyzrender.renderer import render_svg

        g = nx.Graph()
        g.add_node(0, symbol="*", position=[0.0, 0.0, 0.0])
        g.add_node(1, symbol="*", position=[1.6, 0.0, 0.0])  # double bond
        g.add_node(2, symbol="*", position=[0.0, 1.7, 0.0])  # triple bond
        g.add_node(3, symbol="*", position=[-1.6, 0.2, 0.0])
        g.add_node(4, symbol="*", position=[-0.8, 1.6, 0.0])  # aromatic bond
        g.add_edge(0, 1, bond_order=2.0)
        g.add_edge(0, 2, bond_order=3.0)
        g.add_edge(3, 4, bond_order=1.5)

        cfg = RenderConfig(
            atom_scale=0.0,
            atom_stroke_width=0.0,
            bond_width=18.0,
            bond_outline_color="#000000",
            bond_outline_width=2.0,
            bond_orders=True,
            bond_gradient=False,
            fog=False,
            gradient=False,
            auto_orient=False,
        )
        svg = render_svg(g, cfg, _unique_ids=False)
        outlines = [line for line in svg.split("\n") if "<line" in line and 'stroke="#000000"' in line]

        # Expected visible segment count: double=2, triple=3, aromatic=2.
        assert len(outlines) == 7
        assert all('stroke-linecap="round"' in line for line in outlines)
        assert any("stroke-dasharray" in line for line in outlines), "Aromatic dashed segment should be outlined"

    def test_aromatic_dashed_element_split_uses_single_dashed_line(self):
        """Element-split aromatic dashed segment should stay one dashed line (no midpoint dash reset)."""
        import networkx as nx

        from xyzrender.renderer import render_svg

        g = nx.Graph()
        g.add_node(0, symbol="N", position=[0.0, 0.0, 0.0])
        g.add_node(1, symbol="C", position=[1.6, 0.0, 0.0])
        g.add_edge(0, 1, bond_order=1.5)

        cfg = RenderConfig(
            atom_scale=0.0,
            atom_stroke_width=0.0,
            bond_width=18.0,
            bond_outline_width=0.0,
            bond_orders=True,
            bond_color_by_element=True,
            bond_gradient=False,
            fog=False,
            gradient=False,
            auto_orient=False,
            color_overrides={"N": "#1111ff", "C": "#dddddd"},
        )
        svg = render_svg(g, cfg, _unique_ids=False)
        dashed = [line for line in svg.split("\n") if "<line" in line and "stroke-dasharray" in line]
        assert len(dashed) == 1
        assert 'stroke="url(#' in dashed[0]
        assert "#1111ff" in svg
        assert "#dddddd" in svg

    def test_bond_outline_fog_blended(self):
        """With fog enabled, bond outline should fog-blend from one endpoint to the other."""
        import networkx as nx

        from xyzrender.renderer import render_svg

        g = nx.Graph()
        g.add_node(0, symbol="*", position=[0.0, 0.0, 1.0])
        g.add_node(1, symbol="*", position=[1.6, 0.0, -3.0])
        g.add_edge(0, 1, bond_order=1.0)

        cfg = RenderConfig(
            atom_scale=0.0,
            atom_stroke_width=0.0,
            bond_width=18.0,
            bond_color="#ff0000",
            bond_outline_color="#000000",
            bond_outline_width=2.0,
            bond_orders=True,
            bond_color_by_element=False,
            bond_gradient=False,
            fog=True,
            fog_strength=1.2,
            gradient=False,
            auto_orient=False,
        )
        svg = render_svg(g, cfg, _unique_ids=False)
        round_lines = [line for line in svg.split("\n") if "<line" in line and 'stroke-linecap="round"' in line]
        widths = []
        for line in round_lines:
            m = re.search(r'stroke-width="([^"]+)"', line)
            if m is not None:
                widths.append((float(m.group(1)), line))
        assert widths
        # Outline is the widest line (bond width + 2*outline width).
        _, outline_line = max(widths, key=lambda t: t[0])
        assert 'stroke="url(#' in outline_line
        gid = re.search(r'stroke="url\(#([^)]+)\)"', outline_line)
        assert gid is not None
        gmatch = re.search(rf'<linearGradient id="{re.escape(gid.group(1))}"[^>]*>(.*?)</linearGradient>', svg)
        assert gmatch is not None
        stops = re.findall(r'stop-color="([^"]+)"', gmatch.group(1))
        assert len(stops) == 2
        assert stops[0] != stops[1]

    def test_element_split_order_matches_bond_endpoints_when_drawn_reverse(self):
        """Element split should stay bound to endpoints even when drawn as (high_idx -> low_idx)."""
        import networkx as nx

        from xyzrender.renderer import render_svg

        g = nx.Graph()
        # node 0 is front (higher z), node 1 is back (lower z) so render loop draws 1->0
        g.add_node(0, symbol="O", position=[1.0, 0.0, 1.0])
        g.add_node(1, symbol="N", position=[-1.0, 0.0, 0.0])
        g.add_edge(0, 1, bond_order=1.0)

        cfg = RenderConfig(
            atom_scale=0.0,
            atom_stroke_width=0.0,
            bond_width=14.0,
            bond_orders=True,
            bond_color_by_element=True,
            bond_gradient=False,
            fog=False,
            gradient=False,
            auto_orient=False,
            color_overrides={"N": "#1111ff", "O": "#ff1111"},
        )
        svg = render_svg(g, cfg, _unique_ids=False)
        split_lines = [
            line
            for line in svg.split("\n")
            if "<line" in line
            and ('stroke="#1111ff"' in line or 'stroke="#ff1111"' in line)
            and "dasharray" not in line
        ]
        assert len(split_lines) == 2

        # Determine colour at left-most endpoint by checking which segment touches min-x.
        segs = []
        for ln in split_lines:
            m = re.search(r'x1="([^"]+)" y1="([^"]+)" x2="([^"]+)" y2="([^"]+)" stroke="([^"]+)"', ln)
            assert m is not None
            x1 = float(m.group(1))
            x2 = float(m.group(3))
            col = m.group(5)
            segs.append((x1, x2, col))
        min_x = min(min(x1, x2) for x1, x2, _ in segs)
        left_cols = [col for x1, x2, col in segs if abs(x1 - min_x) < 1e-6 or abs(x2 - min_x) < 1e-6]
        assert "#1111ff" in left_cols

    def test_preset_region_creates_style_region(self):
        """Preset with 'regions' key should load region_specs on config."""
        from xyzrender.config import build_config

        cfg = build_config("mtube")
        assert cfg.region_specs is not None
        assert "M" in cfg.region_specs

    def test_user_region_overrides_preset_region(self):
        """User region wins over preset region for overlapping atoms."""
        import networkx as nx

        from xyzrender.api import _apply_style_regions

        g = nx.Graph()
        g.add_node(0, symbol="Fe")
        g.add_node(1, symbol="C")

        cfg = RenderConfig(region_specs={"M": {"atom_scale": 5.0}})
        _apply_style_regions(cfg, g, regions=[("Fe", "flat")])
        # Fe should get flat (gradient=False), not the preset's atom_scale=5.0
        fe_region = [r for r in cfg.style_regions if 0 in r._index_set]
        assert len(fe_region) == 1
        assert fe_region[0].config.gradient is False  # flat preset
        assert fe_region[0].config.atom_scale != 5.0  # not the preset override

    def test_preset_region_not_resolved_twice(self):
        """Preset regions cleared after resolution — no duplicates on second call."""
        import networkx as nx

        from xyzrender.api import _apply_style_regions

        g = nx.Graph()
        g.add_node(0, symbol="Fe")
        g.add_node(1, symbol="C")

        cfg = RenderConfig(region_specs={"M": {"atom_scale": 5.0}})
        _apply_style_regions(cfg, g)
        assert len(cfg.style_regions) == 1
        assert cfg.region_specs is None

        _apply_style_regions(cfg, g)
        assert len(cfg.style_regions) == 1  # not doubled


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
        g.add_edge(0, 2, bond_order=1.0, NCI=True)
        g.add_edge(1, 2, bond_order=1.0, NCI=True)

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

    def test_highlight_does_not_colour_nci_bonds_default(self):
        """Default config (nci_element=False) keeps NCI dotted bonds flat even
        when endpoint atoms are highlighted."""
        mol = load(STRUCTURES / "Hbond.xyz", nci_detect=True)
        n = mol.graph.number_of_nodes()
        dark = Color.from_str(resolve_color("orchid")).blend(Color(0, 0, 0), 0.3).hex
        svg = str(render(mol, highlight=list(range(1, n + 1)), fog=False, gradient=False, orient=False, hy=True))
        dotted_lines = [line for line in svg.split("\n") if "stroke-dasharray" in line and "<line" in line]
        for line in dotted_lines:
            if 'stroke-dasharray="0.' in line or 'stroke-dasharray="1.' in line:
                assert dark not in line, "NCI dotted bond should not get highlight colour by default"


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
