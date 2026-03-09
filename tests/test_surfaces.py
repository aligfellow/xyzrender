"""Tests for surface modules: dens, esp, nci, mo (via surfaces.py)."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from xyzrender.cube import parse_cube
from xyzrender.surfaces import (
    compute_dens_surface,
    compute_esp_surface,
    compute_mo_surface,
    compute_nci_surface,
)
from xyzrender.types import DensParams, ESPParams, MOParams, NCIParams, RenderConfig

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def graph_from_cube(cube_data):
    """Build a minimal nx.Graph from the atom list embedded in a cube file."""
    g = nx.Graph()
    for i, (sym, pos) in enumerate(cube_data.atoms):
        g.add_node(i, symbol=sym, position=list(pos))
    return g


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def caffeine_graph():
    from xyzrender.readers import load_molecule

    g, _ = load_molecule(str(STRUCTURES / "caffeine.xyz"))
    return g


@pytest.fixture(scope="module")
def caffeine_dens_cube():
    return parse_cube(STRUCTURES / "caffeine_dens.cube")


@pytest.fixture(scope="module")
def caffeine_esp_cube():
    return parse_cube(STRUCTURES / "caffeine_esp.cube")


@pytest.fixture(scope="module")
def caffeine_homo_cube():
    return parse_cube(STRUCTURES / "caffeine_homo.cube")


@pytest.fixture(scope="module")
def nci_dens_cube():
    return parse_cube(STRUCTURES / "base-pair-dens.cube")


@pytest.fixture(scope="module")
def nci_grad_cube():
    return parse_cube(STRUCTURES / "base-pair-grad.cube")


@pytest.fixture(scope="module")
def nci_graph(nci_dens_cube):
    return graph_from_cube(nci_dens_cube)


# ---------------------------------------------------------------------------
# nci.find_nci_regions — unit tests with synthetic data
# ---------------------------------------------------------------------------


def test_find_nci_regions_detects_low_rdg_blob():
    from xyzrender.nci import find_nci_regions

    # 10x10x10 grid with a small low-RDG blob in the centre
    grad = np.ones((10, 10, 10), dtype=float)
    grad[4:6, 4:6, 4:6] = 0.1  # 2x2x2 low region
    steps = np.eye(3) * 0.5  # 0.5 Bohr spacing

    regions = find_nci_regions(grad, steps, isovalue=0.3)
    assert len(regions) == 1
    assert len(regions[0].flat_indices) > 0


def test_find_nci_regions_two_blobs():
    from xyzrender.nci import find_nci_regions

    grad = np.ones((15, 15, 15), dtype=float)
    grad[2:4, 2:4, 2:4] = 0.1
    grad[11:13, 11:13, 11:13] = 0.1
    steps = np.eye(3) * 0.5

    regions = find_nci_regions(grad, steps, isovalue=0.3)
    assert len(regions) == 2


def test_find_nci_regions_empty_when_all_above():
    from xyzrender.nci import find_nci_regions

    grad = np.ones((8, 8, 8), dtype=float)
    regions = find_nci_regions(grad, np.eye(3), isovalue=0.3)
    assert regions == []


# ---------------------------------------------------------------------------
# compute_dens_surface
# ---------------------------------------------------------------------------


def test_compute_dens_surface_sets_contours(caffeine_graph, caffeine_dens_cube):
    cfg = RenderConfig(auto_orient=False)
    compute_dens_surface(caffeine_graph, caffeine_dens_cube, cfg, DensParams())
    assert cfg.dens_contours is not None
    assert len(cfg.dens_contours.lobes) > 0


def test_dens_layers_svg_returns_paths(caffeine_graph, caffeine_dens_cube):
    from xyzrender.dens import dens_layers_svg

    cfg = RenderConfig(auto_orient=False)
    compute_dens_surface(caffeine_graph, caffeine_dens_cube, cfg, DensParams())
    assert cfg.dens_contours is not None
    elems = dens_layers_svg(cfg.dens_contours, 0.7, 100.0, 400.0, 400.0, 800, 800)
    assert len(elems) > 0
    assert all("<" in e for e in elems)


# ---------------------------------------------------------------------------
# compute_mo_surface
# ---------------------------------------------------------------------------


def test_compute_mo_surface_sets_contours(caffeine_graph, caffeine_homo_cube):
    cfg = RenderConfig(auto_orient=False)
    compute_mo_surface(caffeine_graph, caffeine_homo_cube, cfg, MOParams())
    assert cfg.mo_contours is not None


# ---------------------------------------------------------------------------
# compute_esp_surface
# ---------------------------------------------------------------------------


def test_compute_esp_surface_sets_surface(caffeine_graph, caffeine_dens_cube, caffeine_esp_cube):
    cfg = RenderConfig(auto_orient=False)
    compute_esp_surface(caffeine_graph, caffeine_dens_cube, caffeine_esp_cube, cfg, ESPParams())
    assert cfg.esp_surface is not None
    assert cfg.esp_surface.png_data_uri.startswith("data:image/png;base64,")


def test_esp_surface_svg_returns_elements(caffeine_graph, caffeine_dens_cube, caffeine_esp_cube):
    from xyzrender.esp import esp_surface_svg

    cfg = RenderConfig(auto_orient=False)
    compute_esp_surface(caffeine_graph, caffeine_dens_cube, caffeine_esp_cube, cfg, ESPParams())
    assert cfg.esp_surface is not None
    elems = esp_surface_svg(cfg.esp_surface, 100.0, 400.0, 400.0, 800, 800, 0.9)
    assert len(elems) > 0
    assert all(isinstance(e, str) for e in elems)


# ---------------------------------------------------------------------------
# compute_nci_surface
# ---------------------------------------------------------------------------


def test_compute_nci_surface_sets_contours(nci_graph, nci_dens_cube, nci_grad_cube):
    cfg = RenderConfig(auto_orient=False)
    compute_nci_surface(nci_graph, nci_dens_cube, nci_grad_cube, cfg, NCIParams())
    assert cfg.nci_contours is not None


def test_nci_loops_svg_returns_paths(nci_graph, nci_dens_cube, nci_grad_cube):
    from xyzrender.nci import nci_loops_svg

    cfg = RenderConfig(auto_orient=False)
    compute_nci_surface(nci_graph, nci_dens_cube, nci_grad_cube, cfg, NCIParams())
    assert cfg.nci_contours is not None
    elems = nci_loops_svg(cfg.nci_contours, 0.7, 100.0, 400.0, 400.0, 800, 800)
    assert len(elems) > 0
    assert all("<path" in e for e in elems)
