"""Tests for geometric MCS (Maximum Common Substructure) matching."""

import copy
from pathlib import Path

import pytest

from xyzrender import load
from xyzrender.mcs import find_mcs_mapping

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


@pytest.fixture(scope="module")
def caffeine():
    return load(STRUCTURES / "caffeine.xyz")


@pytest.fixture(scope="module")
def benzene():
    return load(STRUCTURES / "benzene.xyz")


@pytest.fixture(scope="module")
def anthracene():
    return load(STRUCTURES / "anthracene.xyz")


@pytest.fixture(scope="module")
def ethanol():
    return load(STRUCTURES / "ethanol.xyz")


# ---------------------------------------------------------------------------
# Core matching tests
# ---------------------------------------------------------------------------


def test_mcs_identical_molecules(caffeine):
    """Two copies of the same molecule → full match."""
    g = caffeine.graph
    mapping = find_mcs_mapping(g, copy.deepcopy(g))
    assert mapping is not None
    g1_ids, g2_ids = mapping
    assert len(g1_ids) == len(g2_ids)
    assert len(g1_ids) == g.number_of_nodes()


def test_mcs_substructure(benzene, anthracene):
    """Benzene vs anthracene → benzene ring found in anthracene."""
    mapping = find_mcs_mapping(benzene.graph, anthracene.graph)
    assert mapping is not None
    g1_ids, _ = mapping
    # At least the 6C ring must match; H count depends on ring junction
    heavy = sum(1 for n in g1_ids if benzene.graph.nodes[n]["symbol"] != "H")
    assert heavy >= 6


def test_mcs_no_common(caffeine, ethanol):
    """Caffeine vs ethanol → None or very small match."""
    mapping = find_mcs_mapping(caffeine.graph, ethanol.graph)
    if mapping is not None:
        g1_ids, _ = mapping
        assert len(g1_ids) < 6


def test_mcs_element_matching(benzene, anthracene):
    """Matched node pairs must have the same element symbol."""
    mapping = find_mcs_mapping(benzene.graph, anthracene.graph)
    assert mapping is not None
    g1_ids, g2_ids = mapping
    for n1, n2 in zip(g1_ids, g2_ids, strict=True):
        assert benzene.graph.nodes[n1]["symbol"] == anthracene.graph.nodes[n2]["symbol"]


def test_mcs_min_atoms_respected(benzene, anthracene):
    """Setting min_atoms higher than molecule size → None."""
    mapping = find_mcs_mapping(benzene.graph, anthracene.graph, min_atoms=100)
    assert mapping is None


def test_mcs_shuffled_ids(caffeine):
    """Renumbered node IDs → same-size mapping found."""
    import networkx as nx

    g = caffeine.graph
    mapping_relabel = {n: n + 100 for n in g.nodes()}
    g2 = nx.relabel_nodes(g, mapping_relabel)

    result = find_mcs_mapping(g, g2)
    assert result is not None
    g1_ids, _ = result
    assert len(g1_ids) == g.number_of_nodes()


def test_mcs_returns_valid_node_ids(benzene, anthracene):
    """Returned IDs must exist in the original graphs."""
    mapping = find_mcs_mapping(benzene.graph, anthracene.graph)
    assert mapping is not None
    g1_ids, g2_ids = mapping
    for n in g1_ids:
        assert n in benzene.graph.nodes()
    for n in g2_ids:
        assert n in anthracene.graph.nodes()


def test_mcs_edge_preserving(benzene, anthracene):
    """Matched pairs form an edge-preserving subgraph in both molecules."""
    mapping = find_mcs_mapping(benzene.graph, anthracene.graph)
    assert mapping is not None
    g1_ids, g2_ids = mapping

    # Every edge in the matched subgraph of g1 must correspond to an edge in g2
    edge_count = 0
    for i in range(len(g1_ids)):
        for j in range(i + 1, len(g1_ids)):
            if benzene.graph.has_edge(g1_ids[i], g1_ids[j]):
                assert anthracene.graph.has_edge(g2_ids[i], g2_ids[j]), (
                    f"Edge ({g1_ids[i]}, {g1_ids[j]}) in g1 has no counterpart ({g2_ids[i]}, {g2_ids[j]}) in g2"
                )
                edge_count += 1
    assert edge_count > 0


def test_mcs_connected(caffeine):
    """Matched atoms must form a connected subgraph."""
    import networkx as nx

    g = caffeine.graph
    g2 = copy.deepcopy(g)
    mapping = find_mcs_mapping(g, g2)
    assert mapping is not None
    g1_ids, _ = mapping
    sub = g.subgraph(g1_ids)
    assert nx.is_connected(sub)


# ---------------------------------------------------------------------------
# Cross-molecule tests with real structures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def isothio():
    return load(STRUCTURES / "isothio_uma.xyz", charge=1)


@pytest.fixture(scope="module")
def isothio_xtb():
    return load(STRUCTURES / "isothio_xtb.xyz", charge=1)


def test_mcs_different_atom_counts(isothio):
    """Isothio (52 atoms) vs a different conformer/molecule → ≥ 10 heavy atoms."""
    bridged = load(STRUCTURES / "isothio_bridged.xyz")
    mapping = find_mcs_mapping(isothio.graph, bridged.graph)
    assert mapping is not None
    g1_ids, _g2_ids = mapping
    heavy = sum(1 for n in g1_ids if isothio.graph.nodes[n]["symbol"] != "H")
    assert heavy >= 10
    # All heteroatoms should be matched
    matched_elements = {isothio.graph.nodes[n]["symbol"] for n in g1_ids}
    assert "S" in matched_elements
    assert "N" in matched_elements
    assert "O" in matched_elements


def test_mcs_same_molecule_different_conformer(isothio, isothio_xtb):
    """Same molecule, different geometry (UMA vs xTB) → nearly full match."""
    mapping = find_mcs_mapping(isothio.graph, isothio_xtb.graph)
    assert mapping is not None
    g1_ids, _ = mapping
    total = isothio.graph.number_of_nodes()
    assert len(g1_ids) >= total * 0.8  # at least 80% matched
