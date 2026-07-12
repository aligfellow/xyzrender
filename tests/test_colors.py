"""Tests for color blending logic."""

import numpy as np

from xyzrender.colors import _FOG_MIN_DELTA_E, _MAX_FOG, Color, blend_fog, delta_e, fog_alpha, fog_target

WHITE = Color(255, 255, 255)
BLACK = Color(0, 0, 0)


def _plain_lerp(hex_c: str, fog: Color, alpha: float) -> str:
    """What blend_fog would return with no legibility floor."""
    return Color.from_str(hex_c).blend(fog, alpha).hex


def test_fog_alpha_ramp():
    a = fog_alpha(np.linspace(0.0, 1.0, 50), 1.2)
    assert a[0] == 0.0
    assert np.all(np.diff(a) >= 0)
    assert fog_alpha(0.5, 1.0) == 0.25  # quadratic ease-in keeps the front crisp
    assert fog_alpha(1.0, 100.0) == _MAX_FOG


def test_blend_fog_clamps_alpha():
    assert blend_fog("#abcdef", WHITE, 0.0) == "#abcdef"
    assert blend_fog("#abcdef", WHITE, -1.0) == "#abcdef"
    assert blend_fog("#000000", WHITE, 5.0) == blend_fog("#000000", WHITE, 1.0)


def test_light_colour_cannot_wash_into_the_page():
    """#154: pmol carbon washed to #f4f4f4, and its hydrogen would go grey if fogged."""
    carbon = blend_fog("#d9d9d9", WHITE, 1.0)
    assert carbon != "#ffffff"
    assert delta_e(Color.from_str(carbon), WHITE) >= _FOG_MIN_DELTA_E - 0.5
    # already at the floor: no contrast left to spend, so it must not move at all
    assert all(blend_fog("#fafafa", WHITE, a) == "#fafafa" for a in (0.35, _MAX_FOG, 1.0))


def test_floor_is_a_backstop_only():
    """It may bind only on colours already near the page. #ffff30 is the ΔE-vs-lightness
    case: ~3 L* from white, but with chroma to spend — judged on lightness it would freeze."""
    for hex_c in ("#000000", "#989898", "#ff0d0d", "#7f7fbf", "#ffff30"):
        assert blend_fog(hex_c, WHITE, _MAX_FOG) == _plain_lerp(hex_c, WHITE, _MAX_FOG)
    assert blend_fog("#d9d9d9", WHITE, _MAX_FOG) != _plain_lerp("#d9d9d9", WHITE, _MAX_FOG)


def test_fog_follows_the_background():
    """Fog was hardcoded white, so on a dark page atoms got *brighter* as they receded."""
    assert fog_target("black") == BLACK
    assert fog_target("#ffffff", "steelblue") == Color.from_str("steelblue")
    assert fog_target("none") == WHITE  # a legal SVG fill, but not a colour
    assert int(blend_fog("#d9d9d9", fog_target("black"), _MAX_FOG)[1:3], 16) < 0xD9
