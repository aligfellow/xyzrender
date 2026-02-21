"""Tests for CLI helpers."""

from xyzrender.cli import _basename


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
