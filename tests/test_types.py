"""Tests for Color arithmetic."""

import pytest

from xyzrender.types import Color


def test_hex_roundtrip():
    c = Color(173, 42, 99)
    assert Color.from_hex(c.hex) == c


def test_blend():
    assert Color(0, 0, 0).blend(Color(200, 200, 200), 0.5) == Color(100, 100, 100)


def test_frozen():
    c = Color(1, 2, 3)
    with pytest.raises(AttributeError):
        c.r = 5  # type: ignore[misc]
