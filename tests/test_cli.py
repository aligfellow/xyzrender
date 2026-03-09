"""Tests for CLI helpers."""

from xyzrender.cli import _basename, _parse_indices, _parse_pairs


def test_basename_from_xyz():
    assert _basename("molecule.xyz", from_stdin=False) == "molecule"


def test_basename_from_path():
    assert _basename("/path/to/caffeine.xyz", from_stdin=False) == "caffeine"


def test_basename_from_out_file():
    assert _basename("calc.out", from_stdin=False) == "calc"


def test_basename_stdin():
    assert _basename(None, from_stdin=True) == "graphic"


def test_basename_stdin_overrides_input():
    assert _basename("molecule.xyz", from_stdin=True) == "graphic"


def test_basename_none_not_stdin():
    assert _basename(None, from_stdin=False) == "graphic"


# ---------------------------------------------------------------------------
# _parse_pairs
# ---------------------------------------------------------------------------


def test_parse_pairs_single():
    assert _parse_pairs("1-6") == [(0, 5)]


def test_parse_pairs_multiple():
    assert _parse_pairs("1-6,3-4") == [(0, 5), (2, 3)]


def test_parse_pairs_empty():
    assert _parse_pairs("") == []
    assert _parse_pairs("   ") == []


# ---------------------------------------------------------------------------
# _parse_indices
# ---------------------------------------------------------------------------


def test_parse_indices_single():
    assert _parse_indices("1") == [0]


def test_parse_indices_range():
    assert _parse_indices("1-3") == [0, 1, 2]


def test_parse_indices_mixed():
    assert _parse_indices("1-3,5,7") == [0, 1, 2, 4, 6]


def test_parse_indices_empty():
    assert _parse_indices("") == []
