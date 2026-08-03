"""Tests for atom glow groups (`--glow`, `render(glow=...)`)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from xyzrender import load, render
from xyzrender.types import GlowGroup, RenderConfig

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


@pytest.fixture(scope="module")
def caffeine():
    return load(STRUCTURES / "caffeine.xyz")


def _glow_circles(svg: str) -> list[str]:
    """Every <circle> carrying a glow blur filter."""
    return [m.group(0) for m in re.finditer(r"<circle[^>]*filter=\"url\(#\w*glow\d+\)\"[^>]*/>", svg)]


def _svg_attribute(fragment: str, pattern: str) -> str:
    match = re.search(pattern, fragment)
    assert match is not None
    return match.group(1)


# ---------------------------------------------------------------------------
# Backward compatibility with the single-group form
# ---------------------------------------------------------------------------


def test_element_selector_still_accepted(caffeine):
    """`--glow "N"` is the documented spelling and must keep working."""
    svg = str(render(caffeine, glow="N", orient=False))
    assert len(_glow_circles(svg)) == 4  # caffeine has four nitrogens


def test_index_range_selector(caffeine):
    svg = str(render(caffeine, glow="1-5", orient=False))
    assert len(_glow_circles(svg)) == 5


def test_uncoloured_glow_uses_each_atoms_own_colour(caffeine):
    """No colour given => the glow inherits the atom colour, as it always has."""
    svg = str(render(caffeine, glow="N,O", orient=False, fog=False))
    fills = {_svg_attribute(c, r'fill="([^"]+)"').lower() for c in _glow_circles(svg)}
    assert len(fills) == 2, f"expected one fill per element, got {fills}"


def test_glow_indices_field_still_populated(caffeine):
    """Downstream code reading cfg.glow_indices keeps seeing the selection."""
    from xyzrender.api import _apply_glow

    cfg = RenderConfig()
    _apply_glow(cfg, caffeine.graph, glow="1-3")
    assert cfg.glow_indices == [0, 1, 2]


def test_legacy_glow_indices_without_groups_still_renders(caffeine):
    """A cfg built by hand, pre-dating glow_groups, must still draw glows."""
    cfg = RenderConfig(glow_indices=[0, 1], glow_strength=7.0)
    svg = str(render(caffeine, config=cfg, orient=False))
    assert len(_glow_circles(svg)) == 2
    assert 'stdDeviation="7.00"' in svg


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


def test_two_groups_get_their_own_colours(caffeine):
    svg = str(render(caffeine, glow=[("N", "gold"), ("O", "cyan")], orient=False, fog=False))
    circles = _glow_circles(svg)
    assert len(circles) == 6  # 4 N + 2 O
    fills = [_svg_attribute(c, r'fill="([^"]+)"').lower() for c in circles]
    assert fills.count("#ffd700") == 4
    assert fills.count("#00ffff") == 2


def test_scale_widens_the_glow_circle(caffeine):
    """The glow radius is a multiple of the atom radius, so scale must change r."""

    def _radii(scale):
        svg = str(render(caffeine, glow="N", glow_scale=scale, orient=False))
        return [float(_svg_attribute(c, r'r="([\d.]+)"')) for c in _glow_circles(svg)]

    small, big = _radii(1.0), _radii(2.5)
    assert all(b > s for s, b in zip(small, big, strict=True))
    assert big[0] == pytest.approx(small[0] * 2.5, rel=1e-2)


def test_opacity_reaches_the_circle(caffeine):
    svg = str(render(caffeine, glow="N", glow_opacity=0.4, orient=False))
    assert all('fill-opacity="0.40"' in c for c in _glow_circles(svg))


def test_per_group_strength_emits_one_filter_each():
    """Two groups blurred differently need two filter defs, not one."""
    cfg = RenderConfig(
        glow_groups=[
            GlowGroup(indices=[0], strength=3.0),
            GlowGroup(indices=[1], strength=9.0),
        ]
    )
    svg = str(render(load(STRUCTURES / "caffeine.xyz"), config=cfg, orient=False))
    assert 'stdDeviation="3.00"' in svg
    assert 'stdDeviation="9.00"' in svg
    assert svg.count("<feGaussianBlur") == 2


def test_glow_blur_is_pixel_space_not_relative_to_atom_size(caffeine):
    """One stdDeviation in user units => every glow is equally soft.

    This is the property kept from the released --glow over the draft-PR halo,
    which used objectBoundingBox units and so blurred large atoms more.
    """
    svg = str(render(caffeine, glow="all", glow_strength=6.0, orient=False))
    filters = re.findall(r"<filter id=\"\w*glow\d+\"[^>]*>", svg)
    assert len(filters) == 1, "one strength should need exactly one filter"
    assert "objectBoundingBox" not in filters[0]
    assert "primitiveUnits" not in filters[0]
    assert svg.count('stdDeviation="6.00"') == 1


def test_config_glow_strength_survives_a_selector(caffeine):
    """A RenderConfig carrying glow_strength must not be overridden by the default.

    render(config=cfg, glow=...) reaches _apply_glow with glow_strength=None;
    the group has to inherit cfg.glow_strength rather than fall back to 5.0.
    """
    svg = str(render(caffeine, config=RenderConfig(glow_strength=9.0), glow="N", orient=False))
    assert 'stdDeviation="9.00"' in svg
    assert 'stdDeviation="5.00"' not in svg


def test_glow_is_occluded_by_nearer_atoms(caffeine):
    """The glow sits in its atom's own depth slot, so painter's order holds.

    Each glow circle must be immediately followed by that atom's own graphics
    rather than being hoisted into a single layer behind everything.
    """
    svg = str(render(caffeine, glow="N", orient=False))
    positions = [m.start() for m in re.finditer(r'filter="url\(#\w*glow\d+\)"', svg)]
    assert len(positions) == 4
    # The glows are spread through the body of the SVG, not clustered at the top.
    assert max(positions) - min(positions) > 200


def test_empty_selection_yields_no_groups(caffeine):
    from xyzrender.api import _apply_glow

    cfg = RenderConfig()
    _apply_glow(cfg, caffeine.graph, glow="Xe")
    assert cfg.glow_groups == []


def test_selection_uses_original_indices_under_an_atom_filter(caffeine):
    """--glow resolves twice (CLI, then render); both must use original indices.

    With `only=`, the render-time graph is renumbered, so a stale second
    resolution would glow the wrong atoms.
    """
    svg = str(render(caffeine, glow=[("9-12", "gold")], only="9-24", orient=False, fog=False))
    circles = _glow_circles(svg)
    assert len(circles) == 4
    assert all("#ffd700" in c for c in circles)
