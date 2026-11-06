"""Tests for annotations.py — labels and colormap file loaders."""

from pathlib import Path

import networkx as nx
import pytest

from xyzrender import load
from xyzrender.annotations import AtomValueLabel, BondLabel, load_bond_cmap, parse_annotations

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


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


# ---------------------------------------------------------------------------
# Bond property colormap file loader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ethanol():
    return load(STRUCTURES / "ethanol.xyz")


def test_load_bond_cmap_valid(ethanol, tmp_path):
    path = tmp_path / "bonds.txt"
    path.write_text("2 3 0.5\n# comment\n1 2 1.0\n")
    result = load_bond_cmap(str(path), ethanol.graph)
    assert result[(1, 2)] == 0.5  # 0-indexed 1-2 from line "2 3"
    assert result[(0, 1)] == 1.0


def test_load_bond_cmap_canonicalizes_pair(ethanol, tmp_path):
    path = tmp_path / "bonds.txt"
    path.write_text("3 2 0.25\n")
    assert load_bond_cmap(str(path), ethanol.graph)[(1, 2)] == 0.25


def test_load_bond_cmap_missing_atom(ethanol, tmp_path):
    path = tmp_path / "bonds.txt"
    path.write_text("2 99 1.0\n")
    with pytest.raises(ValueError, match="not found"):
        load_bond_cmap(str(path), ethanol.graph)


def test_load_bond_cmap_adds_missing_edge(ethanol, tmp_path, caplog):
    import logging

    path = tmp_path / "bonds.txt"
    path.write_text("3 4 1.0\n")
    graph = ethanol.graph
    assert not graph.has_edge(2, 3)
    with caplog.at_level(logging.WARNING):
        result = load_bond_cmap(str(path), graph)
    assert result[(2, 3)] == 1.0
    assert graph.has_edge(2, 3)
    assert graph.edges[2, 3].get("NCI") is True
    assert any("adding NCI-style edge" in r.message for r in caplog.records)


def test_load_bond_cmap_same_atom(ethanol, tmp_path):
    path = tmp_path / "bonds.txt"
    path.write_text("2 2 1.0\n")
    with pytest.raises(ValueError, match="must differ"):
        load_bond_cmap(str(path), ethanol.graph)
