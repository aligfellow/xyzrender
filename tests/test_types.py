"""Tests for Color arithmetic and resolve_color."""

import pytest

from xyzrender.colors import Color, resolve_color


def test_hex_roundtrip():
    c = Color(173, 42, 99)
    assert Color.from_hex(c.hex) == c


def test_blend():
    assert Color(0, 0, 0).blend(Color(200, 200, 200), 0.5) == Color(100, 100, 100)


def test_frozen():
    c = Color(1, 2, 3)
    with pytest.raises(AttributeError):
        c.r = 5  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("input_color", "expected"),
    [
        ("#FF0000", "#ff0000"),
        ("2554A5", "#2554a5"),
        ("red", "#ff0000"),
        ("SteelBlue", "#4682b4"),
        ("  cornflowerblue  ", "#6495ed"),
    ],
)
def test_resolve_color(input_color, expected):
    assert resolve_color(input_color) == expected


def test_resolve_color_unknown_raises():
    with pytest.raises(ValueError, match="Unknown color"):
        resolve_color("notarealcolor")


def test_from_str_named():
    assert Color.from_str("steelblue") == Color(70, 130, 180)
