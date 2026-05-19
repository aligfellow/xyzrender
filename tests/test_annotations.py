"""Tests for annotations.py — label parsing."""

import networkx as nx

from xyzrender.annotations import AtomValueLabel, BondLabel, parse_annotations


def _two_atom_graph():
    g = nx.Graph()
    g.add_node(0, symbol="O", position=(0.0, 0.0, 0.0))
    g.add_node(1, symbol="H", position=(1.0, 0.0, 0.0))
    g.add_edge(0, 1, bond_order=1.0)
    return g


# ---------------------------------------------------------------------------
# Custom label case preservation
# ---------------------------------------------------------------------------


def test_atom_value_label_preserves_case():
    """Custom atom labels must keep their original case — e.g. '1 HOH' stays 'HOH'."""
    [lab] = parse_annotations([["1", "HOH"]], None, _two_atom_graph())
    assert isinstance(lab, AtomValueLabel)
    assert lab.text == "HOH"


def test_bond_label_preserves_case():
    """Custom bond labels must keep their original case."""
    [lab] = parse_annotations([["1", "2", "C-alpha-N"]], None, _two_atom_graph())
    assert isinstance(lab, BondLabel)
    assert lab.text == "C-alpha-N"
