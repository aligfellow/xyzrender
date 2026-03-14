"""Tests for the overlay module and render(overlay=)."""

import copy
from pathlib import Path

import numpy as np
import pytest

from xyzrender import load, render
from xyzrender.overlay import align, kabsch_align, merge_graphs

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


@pytest.fixture(scope="module")
def caffeine():
    return load(STRUCTURES / "caffeine.xyz")


@pytest.fixture(scope="module")
def ethanol():
    return load(STRUCTURES / "ethanol.xyz")


# ---------------------------------------------------------------------------
# align()
# ---------------------------------------------------------------------------


def test_align_identity_zero_rmsd(caffeine):
    """Aligning a molecule with itself returns near-zero RMSD."""
    g = caffeine.graph
    aligned = align(g, copy.deepcopy(g))
    pos1 = np.array([g.nodes[n]["position"] for n in g.nodes()], dtype=float)
    assert float(np.sqrt(np.mean((aligned - pos1) ** 2))) < 1e-6


def test_align_rotated_molecule(caffeine):
    """Kabsch rotation recovers the original frame after a 90° rotation."""
    g = caffeine.graph
    g2 = copy.deepcopy(g)
    nodes = list(g2.nodes())
    pos = np.array([g2.nodes[n]["position"] for n in nodes], dtype=float)
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    pos_rot = pos @ rot.T
    for k, nid in enumerate(nodes):
        g2.nodes[nid]["position"] = tuple(float(v) for v in pos_rot[k])
    aligned = align(g, g2)
    pos1 = np.array([g.nodes[n]["position"] for n in g.nodes()], dtype=float)
    assert float(np.sqrt(np.mean((aligned - pos1) ** 2))) < 1e-4


def test_align_mismatched_atoms_raises(caffeine, ethanol):
    with pytest.raises(ValueError, match="counts must match"):
        align(caffeine.graph, ethanol.graph)


# ---------------------------------------------------------------------------
# merge_graphs()
# ---------------------------------------------------------------------------


def test_merge_graphs_structure(caffeine):
    """Merged graph has 2xn nodes; mol2 nodes carry overlay=True and bond_color_override."""
    g = caffeine.graph
    aligned = align(g, copy.deepcopy(g))
    merged = merge_graphs(g, copy.deepcopy(g), aligned)

    n = g.number_of_nodes()
    assert merged.number_of_nodes() == 2 * n

    mol2_nodes = [nid for nid in merged.nodes() if nid >= n]
    assert all(merged.nodes[nid]["overlay"] for nid in mol2_nodes)

    mol2_edges = [(i, j, d) for i, j, d in merged.edges(data=True) if i >= n or j >= n]
    assert all(d.get("bond_color_override", "").startswith("#") for _, _, d in mol2_edges)


# ---------------------------------------------------------------------------
# render(overlay=)
# ---------------------------------------------------------------------------


def test_render_overlay_produces_svg(caffeine):
    svg = str(render(caffeine, overlay=STRUCTURES / "caffeine.xyz", orient=False))
    assert svg.startswith("<svg")
    assert "</svg>" in svg


def test_render_overlay_color_appears_in_svg(caffeine):
    from xyzrender.types import resolve_color

    svg = str(render(caffeine, overlay=caffeine, overlay_color="steelblue", gradient=False, fog=False, orient=False))
    assert resolve_color("steelblue") in svg


def test_render_overlay_mutual_exclusion_surface(caffeine):
    with pytest.raises(ValueError, match="mutually exclusive"):
        render(caffeine, overlay=caffeine, dens=True, orient=False)


def test_render_overlay_mutual_exclusion_cell():
    mol = load(STRUCTURES / "caffeine_cell.xyz", cell=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        render(mol, overlay=mol, orient=False)


def test_render_overlay_atom_count_mismatch(caffeine, ethanol):
    with pytest.raises(ValueError, match="counts must match"):
        render(caffeine, overlay=ethanol, orient=False)


# ---------------------------------------------------------------------------
# kabsch_align() — shared alignment function
# ---------------------------------------------------------------------------


def test_kabsch_align_identity():
    """Aligning identical positions returns them unchanged."""
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    aligned = kabsch_align(pos, pos.copy())
    assert float(np.sqrt(np.mean((aligned - pos) ** 2))) < 1e-10


def test_kabsch_align_with_align_atoms():
    """Subset alignment: fit on 3 atoms, transform applies to all."""
    ref = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]], dtype=float)
    # Rotate the mobile by 90 deg around z
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    mobile = ref @ rot.T
    aligned = kabsch_align(ref, mobile, align_atoms=[0, 1, 2])
    # The first 3 atoms should be well-aligned
    assert float(np.sqrt(np.mean((aligned[:3] - ref[:3]) ** 2))) < 1e-4
    # All atoms should be aligned since it's a rigid rotation
    assert float(np.sqrt(np.mean((aligned - ref) ** 2))) < 1e-4


def test_kabsch_align_too_few_atoms_raises():
    """align_atoms with fewer than 3 indices should raise."""
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    with pytest.raises(ValueError, match="at least 3"):
        kabsch_align(pos, pos.copy(), align_atoms=[0, 1])


def test_render_overlay_with_align_atoms(caffeine):
    """Overlay with align_atoms renders without error."""
    # Use 1-indexed atoms for the render API
    svg = str(render(caffeine, overlay=caffeine, align_atoms=[1, 2, 3], orient=False))
    assert svg.startswith("<svg")
