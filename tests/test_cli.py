"""Tests for CLI helpers."""

import networkx as nx
import pytest

from xyzrender.cli import _basename, _parse_indices, _parse_pairs, _resolve_atom_spec


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


def _toy_graph():
    g = nx.Graph()
    g.add_node(0, symbol="Pt", position=[0.0, 0.0, 0.0])
    g.add_node(1, symbol="Ni", position=[1.0, 0.0, 0.0])
    g.add_node(2, symbol="C", position=[2.0, 0.0, 0.0])
    g.add_node(3, symbol="H", position=[3.0, 0.0, 0.0])
    return g


def test_resolve_atom_spec_element_single():
    assert _resolve_atom_spec("el:Pt", _toy_graph()) == [1]


def test_resolve_atom_spec_element_list_with_comma():
    assert _resolve_atom_spec("el:Pt,Ni", _toy_graph()) == [1, 2]


def test_resolve_atom_spec_metal_keyword():
    assert _resolve_atom_spec("el:metal", _toy_graph()) == [1, 2]


def test_resolve_atom_spec_mixed_numeric_and_element():
    assert _resolve_atom_spec("2-3,el:Pt", _toy_graph()) == [1, 2, 3]


def test_resolve_atom_spec_unknown_element_raises():
    with pytest.raises(ValueError, match="unknown element selector"):
        _resolve_atom_spec("el:Xx", _toy_graph())
