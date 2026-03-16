"""Tests for --hl (highlight) atom group coloring."""

from pathlib import Path

import pytest

from xyzrender import load, render
from xyzrender.types import Color, resolve_color

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


@pytest.fixture(scope="module")
def caffeine():
    return load(STRUCTURES / "caffeine.xyz")


def test_highlight_color_in_svg(caffeine):
    """Orchid (default) appears as fill on highlighted atoms."""
    svg = str(render(caffeine, highlight="1-3", gradient=False, fog=False, orient=False))
    assert resolve_color("orchid") in svg


def test_highlight_custom_color(caffeine):
    """Custom color replaces default."""
    svg = str(render(caffeine, highlight="1-3", highlight_color="steelblue", gradient=False, fog=False, orient=False))
    assert resolve_color("steelblue") in svg
    assert resolve_color("orchid") not in svg


def test_highlight_bond_darkened(caffeine):
    """Bonds between highlighted atoms get darkened color."""
    svg = str(render(caffeine, highlight="1-3", gradient=False, fog=False, orient=False))
    orchid = Color.from_str(resolve_color("orchid"))
    dark = orchid.blend(Color(0, 0, 0), 0.3).hex
    assert dark in svg


def test_highlight_no_mutation(caffeine):
    """Molecule graph is not mutated by highlight rendering."""
    pos_before = {n: caffeine.graph.nodes[n]["position"] for n in caffeine.graph.nodes()}
    render(caffeine, highlight="1-5", orient=False)
    for n in caffeine.graph.nodes():
        assert caffeine.graph.nodes[n]["position"] == pos_before[n]


def test_highlight_list_api(caffeine):
    """Python API accepts 0-indexed list."""
    svg = str(render(caffeine, highlight=[0, 1, 2], gradient=False, fog=False, orient=False))
    assert resolve_color("orchid") in svg
