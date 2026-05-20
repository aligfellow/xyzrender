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
    fused = load(STRUCTURES / "isothio_fused.xyz")
    mapping = find_mcs_mapping(isothio.graph, fused.graph)
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


# ---------------------------------------------------------------------------
# Type-aware matching (M/het class collapse)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mnh():
    return load(STRUCTURES / "mnh.xyz")


@pytest.fixture(scope="module")
def cocl6():
    return load(STRUCTURES / "CoCl6.xyz")


def test_mcs_type_aware_allows_metal_class_match(mnh, cocl6):
    """type_aware=True lets Fe/Mn ↔ Co match via the shared 'M' class.

    Strict matching would miss the metal pair (Fe ≠ Co, Mn ≠ Co); type-aware
    collapses all metals into one class so they pair up.
    """
    from xyzgraph import DATA

    strict = find_mcs_mapping(mnh.graph, cocl6.graph, type_aware=False)
    typed = find_mcs_mapping(mnh.graph, cocl6.graph, type_aware=True)
    # Strict mode shouldn't have any metal-metal pair (different elements).
    if strict is not None:
        for a, b in zip(strict[0], strict[1], strict=True):
            assert not (mnh.graph.nodes[a]["symbol"] in DATA.metals and cocl6.graph.nodes[b]["symbol"] in DATA.metals)
    # Type-aware should include at least one metal pair.
    assert typed is not None
    metal_pairs = [
        (a, b)
        for a, b in zip(typed[0], typed[1], strict=True)
        if mnh.graph.nodes[a]["symbol"] in DATA.metals and cocl6.graph.nodes[b]["symbol"] in DATA.metals
    ]
    assert metal_pairs, "type-aware MCS should pair at least one metal-metal"


def test_mcs_type_aware_rejects_small_alkyl_matches(caffeine):
    """type_aware=True rejects trivial all-C/H matches (< 5 heavy atoms).

    A propyl-sized C/H fragment is too unspecific to anchor an alignment;
    the caller (overlay) falls through to a different strategy.
    """
    import networkx as nx

    g1 = nx.Graph()
    for i, (s, p) in enumerate(
        [
            ("C", (0.0, 0.0, 0.0)),
            ("C", (1.5, 0.0, 0.0)),
            ("C", (3.0, 0.0, 0.0)),
            ("H", (0.0, 1.0, 0.0)),
            ("H", (1.5, 1.0, 0.0)),
            ("H", (3.0, 1.0, 0.0)),
        ]
    ):
        g1.add_node(i, symbol=s, position=p)
    g1.add_edges_from([(0, 1), (1, 2), (0, 3), (1, 4), (2, 5)])

    g2 = copy.deepcopy(g1)
    # Strict matches the propyl fragment (3 heavy, ≥ min_atoms).
    assert find_mcs_mapping(g1, g2, type_aware=False) is not None
    # Type-aware rejects: too few heavy atoms (3 < 5) AND no heteroatom.
    assert find_mcs_mapping(g1, g2, type_aware=True) is None


def test_mcs_type_aware_accepts_benzene_ring_match(benzene):
    """Benzene-benzene is all C/H but has 6 heavy atoms — accepted as anchor.

    Aromatic rings are real structural anchors even without heteroatoms; the
    alkyl-rejection threshold (≥ 5 heavy atoms) is what distinguishes them
    from trivial methyl/ethyl/propyl fragments.
    """
    mapping = find_mcs_mapping(benzene.graph, benzene.graph, type_aware=True)
    assert mapping is not None
    # Should include all 6 ring carbons (and likely all 6 hydrogens too).
    heavy = [n for n in mapping[0] if benzene.graph.nodes[n]["symbol"] != "H"]
    assert len(heavy) == 6


def test_mcs_aromatic_ring_seeds_land_benzene_on_aromatic_ring(benzene):
    """Benzene overlaid on a mixed-ring molecule must land on an aromatic ring.

    isothio_uma has 3 six-membered rings: one non-aromatic (pyrazine-like,
    4 C + 2 N) and two aromatic (a benzannulated core and a Ph substituent).
    Without aromatic-ring seeds the MCS BFS grows from whichever close pairs
    happen to fall within the threshold after PCA — often a non-aromatic ring.
    The aromatic-ring seeds in find_mcs_mapping guarantee that an aromatic
    benzene-on-benzene alignment is in the candidate set.
    """
    from xyzrender import load

    target = load(STRUCTURES / "isothio_uma.xyz", charge=1)
    aromatic_atoms: set[int] = {n for r in target.graph.graph.get("aromatic_rings", []) for n in r}

    mapping = find_mcs_mapping(target.graph, benzene.graph, type_aware=True)
    assert mapping is not None
    target_matched = set(mapping[0])
    # All 6 ring carbons matched, and they're all from an aromatic ring.
    heavy = [n for n in target_matched if target.graph.nodes[n]["symbol"] != "H"]
    assert len(heavy) == 6
    assert set(heavy).issubset(aromatic_atoms), (
        f"benzene MCS landed on non-aromatic atoms: {set(heavy) - aromatic_atoms}"
    )
