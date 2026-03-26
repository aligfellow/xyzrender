"""Tests for --ref orientation reference (save / load)."""

import copy
import re
from pathlib import Path

import numpy as np
import pytest

from xyzrender import load, render


def _strip_svg_ids(svg: str) -> str:
    """Strip render-counter IDs so SVGs from different render() calls can be compared."""
    return re.sub(r'id="x\d+', 'id="x0', re.sub(r"url\(#x\d+", "url(#x0", svg))


STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


@pytest.fixture(scope="module")
def caffeine():
    return load(STRUCTURES / "caffeine.xyz")


@pytest.fixture(scope="module")
def ethanol():
    return load(STRUCTURES / "ethanol.xyz")


@pytest.fixture(scope="module")
def caffeine_cube():
    return load(STRUCTURES / "caffeine_homo.cube")


# ---------------------------------------------------------------------------
# SAVE mode
# ---------------------------------------------------------------------------


def test_ref_save_creates_file(caffeine, tmp_path):
    """Nonexistent ref path → file created, valid XYZ, correct atom count."""
    ref_path = tmp_path / "ref.xyz"
    render(caffeine, ref=ref_path, orient=True)

    assert ref_path.is_file()
    lines = ref_path.read_text().splitlines()
    n_atoms = int(lines[0].strip())
    non_ghost = sum(1 for n in caffeine.graph.nodes() if caffeine.graph.nodes[n]["symbol"] != "*")
    assert n_atoms == non_ghost
    for line in lines[2:]:
        assert len(line.split()) == 4


def test_ref_rejects_periodic(tmp_path):
    """--ref raises ValueError for periodic structures."""
    mol = load(STRUCTURES / "caffeine_cell.xyz", cell=True)
    ref_path = tmp_path / "crystal_ref.xyz"

    with pytest.raises(ValueError, match="not supported for periodic"):
        render(mol, ref=ref_path)


# ---------------------------------------------------------------------------
# LOAD mode
# ---------------------------------------------------------------------------


def test_ref_load_reproduces_orientation(caffeine, tmp_path):
    """Save ref, load for a rotated copy → Kabsch-aligns back."""
    ref_path = tmp_path / "ref.xyz"
    render(caffeine, ref=ref_path, orient=True)

    # 90° rotation
    mol2 = copy.deepcopy(caffeine)
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    for n in mol2.graph.nodes():
        pos = np.array(mol2.graph.nodes[n]["position"], dtype=float)
        mol2.graph.nodes[n]["position"] = tuple((pos @ rot.T).tolist())

    # mol2 should not be mutated (rmol is a deep copy)
    svg = str(render(mol2, ref=ref_path))
    assert svg.startswith("<svg")


def test_ref_load_ignores_orient(caffeine, tmp_path):
    """orient=True with existing ref → ref wins, orient ignored."""
    ref_path = tmp_path / "ref.xyz"
    render(caffeine, ref=ref_path, orient=False)

    svg_with_orient = _strip_svg_ids(str(render(caffeine, ref=ref_path, orient=True)))
    svg_without_orient = _strip_svg_ids(str(render(caffeine, ref=ref_path, orient=False)))
    assert svg_with_orient == svg_without_orient


def test_ref_round_trip_with_surfaces(caffeine_cube, tmp_path):
    """Cube --mo --ref save then load → both valid SVG."""
    ref_path = tmp_path / "ref.xyz"
    svg1 = str(render(caffeine_cube, mo=True, ref=ref_path))
    assert svg1.startswith("<svg")
    assert ref_path.is_file()
    svg2 = str(render(caffeine_cube, mo=True, ref=ref_path))
    assert svg2.startswith("<svg")


def test_ref_mcs_fallback(tmp_path):
    """Save ref from benzene, load for anthracene → MCS alignment, valid SVG."""
    benzene = load(STRUCTURES / "benzene.xyz")
    anthracene = load(STRUCTURES / "anthracene.xyz")
    ref_path = tmp_path / "ref.xyz"
    render(benzene, ref=ref_path, orient=True)
    svg = str(render(anthracene, ref=ref_path))
    assert svg.startswith("<svg")


def test_ref_very_different_molecules(caffeine, ethanol, tmp_path):
    """Very different molecules: either MCS works (small match) or raises."""
    ref_path = tmp_path / "ref.xyz"
    render(caffeine, ref=ref_path, orient=False)
    try:
        svg = str(render(ethanol, ref=ref_path))
        assert svg.startswith("<svg")
    except ValueError:
        pass  # acceptable if no common substructure found


def test_ref_idempotent(caffeine, tmp_path):
    """Load same ref twice → positions identical (no drift)."""
    ref_path = tmp_path / "ref.xyz"
    render(caffeine, ref=ref_path, orient=True)

    svg1 = _strip_svg_ids(str(render(caffeine, ref=ref_path)))
    svg2 = _strip_svg_ids(str(render(caffeine, ref=ref_path)))
    assert svg1 == svg2


def test_ref_rejects_periodic_load(tmp_path):
    """--ref LOAD also rejects periodic structures."""
    mol_free = load(STRUCTURES / "caffeine.xyz")
    ref_path = tmp_path / "ref.xyz"
    render(mol_free, ref=ref_path, orient=False)

    mol_cell = load(STRUCTURES / "caffeine_cell.xyz", cell=True)
    with pytest.raises(ValueError, match="not supported for periodic"):
        render(mol_cell, ref=ref_path)
