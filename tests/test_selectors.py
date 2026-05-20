"""Tests for selectors — element-category atom selection."""

import networkx as nx
import pytest

from xyzrender.selectors import normalize_token, resolve_atom_indices, resolve_element_set


def _metal_graph():
    """Pt-O, Pt-N, Ni-C, plus isolated H — covers metals, ligands, organic."""
    g = nx.Graph()
    g.add_node(0, symbol="Pt")
    g.add_node(1, symbol="Ni")
    g.add_node(2, symbol="C")
    g.add_node(3, symbol="H")
    g.add_node(4, symbol="O")
    g.add_node(5, symbol="N")
    # Pt's coord shell: O, N.  Ni's coord shell: C.  H is unbonded.
    g.add_edges_from([(0, 4), (0, 5), (1, 2)])
    return g


# ---------------------------------------------------------------------------
# normalize_token
# ---------------------------------------------------------------------------


def test_normalize_token():
    assert normalize_token("M") == "M"
    assert normalize_token("m") == "M"
    assert normalize_token("sbm") == "sbm"
    assert normalize_token("Fe") == "Fe"
    assert normalize_token("fe") == "Fe"


def test_unknown_token_raises():
    with pytest.raises(ValueError, match="Unknown category or element"):
        normalize_token("Xx")


def test_number_token_raises():
    with pytest.raises(ValueError, match="Unknown category or element"):
        normalize_token("42")


# ---------------------------------------------------------------------------
# resolve_element_set
# ---------------------------------------------------------------------------


def test_element_set_categories():
    metals = resolve_element_set("M")
    assert "Fe" in metals
    assert "Pt" in metals
    assert "C" not in metals

    sbm = resolve_element_set("sbm")
    assert sbm <= metals
    assert "Li" in sbm

    nonmetals = resolve_element_set("L")
    assert not (nonmetals & metals)
    assert "C" in nonmetals

    het = resolve_element_set("het")
    assert "O" in het
    assert "N" in het
    assert "C" not in het
    assert "Fe" not in het


def test_element_set_single():
    assert resolve_element_set("Fe") == frozenset({"Fe"})


def test_element_set_pi_raises():
    with pytest.raises(ValueError, match="cannot be resolved to an element set"):
        resolve_element_set("pi")


# ---------------------------------------------------------------------------
# resolve_atom_indices
# ---------------------------------------------------------------------------


def test_resolve_categories():
    g = _metal_graph()
    # M is element-based (all metals), no graph-context narrowing.
    assert resolve_atom_indices("M", g) == {0, 1}  # Pt, Ni
    # L and het are graph-context-aware: "ligand" = bonded to a metal.
    # Pt has O, N as ligands; Ni has C as ligand.  H is unbonded so excluded.
    assert resolve_atom_indices("L", g) == {2, 4, 5}  # C(Ni), O(Pt), N(Pt)
    assert resolve_atom_indices("het", g) == {4, 5}  # O(Pt), N(Pt)


def test_resolve_l_het_falls_back_to_element_set_without_metals():
    """On pure-organic graphs (no metals), L/het use the full element set."""
    g = nx.Graph()
    g.add_node(0, symbol="C")
    g.add_node(1, symbol="H")
    g.add_node(2, symbol="O")
    assert resolve_atom_indices("L", g) == {0, 1, 2}  # all non-metals
    assert resolve_atom_indices("het", g) == {2}  # all non-CHM


def test_resolve_element():
    g = _metal_graph()
    assert resolve_atom_indices("Pt", g) == {0}
    assert resolve_atom_indices("pt", g) == {0}  # case-insensitive


def test_resolve_numeric():
    g = _metal_graph()
    assert resolve_atom_indices("1", g) == {0}
    assert resolve_atom_indices("2-4", g) == {1, 2, 3}
    assert resolve_atom_indices(" 1 ", g) == {0}  # whitespace


def test_resolve_no_matches():
    assert resolve_atom_indices("Fe", _metal_graph()) == set()


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown category or element"):
        resolve_atom_indices("Xx", _metal_graph())
