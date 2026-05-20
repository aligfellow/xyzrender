"""Tests for surface modules: dens, esp, nci, mo (via surfaces.py)."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from xyzrender.cube import CubeData, parse_cube
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


def cube_from_array(
    data: np.ndarray,
    *,
    shape: tuple[int, int, int] | None = None,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    steps: np.ndarray | None = None,
) -> CubeData:
    """Construct a minimal CubeData for synthetic surface tests."""
    arr = np.array(data, dtype=float)
    if shape is not None:
        arr = arr.reshape(shape)
    if steps is None:
        steps = np.eye(3) * 0.5
    return CubeData(
        atoms=[("H", (0.0, 0.0, 0.0))],
        origin=np.array(origin, dtype=float),
        steps=np.array(steps, dtype=float),
        grid_shape=arr.shape,
        grid_data=arr,
        mo_index=None,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def caffeine_graph():
    from xyzrender.readers import load_molecule

    g, _ = load_molecule(str(STRUCTURES / "caffeine.xyz"))
    return g


@pytest.fixture(scope="module")
def caffeine_mol(caffeine_graph):
    from xyzrender.api import Molecule

    return Molecule(graph=caffeine_graph)


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


@pytest.fixture(scope="module")
def nci_mol(nci_graph):
    from xyzrender.api import Molecule

    return Molecule(graph=nci_graph)


@pytest.fixture(scope="module")
def igmh_inter_cube():
    return parse_cube(STRUCTURES / "phenol_di-dg_inter.cub")


@pytest.fixture(scope="module")
def igmh_intra_cube():
    return parse_cube(STRUCTURES / "phenol_di-dg_intra.cub")


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


def test_find_nci_regions_detects_high_dg_blob_for_igmh():
    from xyzrender.nci import find_nci_regions

    dg = np.zeros((10, 10, 10), dtype=float)
    dg[4:6, 4:6, 4:6] = 0.4
    steps = np.eye(3) * 0.5

    regions = find_nci_regions(dg, steps, isovalue=0.3, mode="high_field")
    assert len(regions) == 1
    assert len(regions[0].flat_indices) > 0


def test_classify_surface_field_identifies_nci_grad_as_low_field(nci_grad_cube):
    from xyzrender.nci import classify_surface_field

    assert classify_surface_field(nci_grad_cube.grid_data) == "low_field"


def test_classify_surface_field_identifies_igmh_cubes_as_high_field(igmh_inter_cube, igmh_intra_cube):
    from xyzrender.nci import classify_surface_field

    assert classify_surface_field(igmh_inter_cube.grid_data) == "high_field"
    assert classify_surface_field(igmh_intra_cube.grid_data) == "high_field"


def test_build_nci_contours_rejects_mismatched_grids():
    from xyzrender.nci import build_nci_contours

    color_cube = cube_from_array(np.zeros((8, 8, 8), dtype=float))
    surface_cube = cube_from_array(np.zeros((7, 8, 8), dtype=float))

    with pytest.raises(ValueError, match="same grid shape"):
        build_nci_contours(surface_cube, color_cube, NCIParams())


def test_build_nci_contours_supports_igmh_surface_mode():
    from xyzrender.nci import build_nci_contours

    sl2r = np.zeros((12, 12, 12), dtype=float)
    sl2r[4:8, 4:8, 4:8] = -0.2
    dg = np.zeros((12, 12, 12), dtype=float)
    dg[4:8, 4:8, 4:8] = 0.4

    color_cube = cube_from_array(sl2r)
    surface_cube = cube_from_array(dg)
    contours = build_nci_contours(surface_cube, color_cube, NCIParams(isovalue=0.3), surface_mode="high_field")

    assert contours.lobes


def test_build_nci_contours_uses_igmh_default_isovalue_when_nci_default_would_hide_surface():
    from xyzrender.nci import build_nci_contours

    sl2r = np.zeros((12, 12, 12), dtype=float)
    sl2r[4:8, 4:8, 4:8] = -0.2
    dg = np.zeros((12, 12, 12), dtype=float)
    dg[4:8, 4:8, 4:8] = 0.02

    color_cube = cube_from_array(sl2r)
    surface_cube = cube_from_array(dg)
    contours = build_nci_contours(surface_cube, color_cube, NCIParams(), surface_mode="high_field")

    assert contours.lobes


def test_build_nci_contours_auto_classifies_high_field_surface():
    from xyzrender.nci import build_nci_contours

    sl2r = np.zeros((12, 12, 12), dtype=float)
    sl2r[4:8, 4:8, 4:8] = -0.2
    dg = np.zeros((12, 12, 12), dtype=float)
    dg[4:8, 4:8, 4:8] = 0.02

    color_cube = cube_from_array(sl2r)
    surface_cube = cube_from_array(dg)
    contours = build_nci_contours(surface_cube, color_cube, NCIParams())

    assert contours.lobes


# ---------------------------------------------------------------------------
# compute_dens_surface
# ---------------------------------------------------------------------------


def test_compute_dens_surface_sets_contours(caffeine_mol, caffeine_dens_cube):
    cfg = RenderConfig(auto_orient=False)
    compute_dens_surface(caffeine_mol, caffeine_dens_cube, cfg, DensParams())
    assert cfg.dens_contours is not None
    assert len(cfg.dens_contours.lobes) > 0


def test_dens_layers_svg_returns_paths(caffeine_mol, caffeine_dens_cube):
    from xyzrender.dens import dens_layers_svg

    cfg = RenderConfig(auto_orient=False)
    compute_dens_surface(caffeine_mol, caffeine_dens_cube, cfg, DensParams())
    assert cfg.dens_contours is not None
    elems = dens_layers_svg(cfg.dens_contours, 0.7, 100.0, 400.0, 400.0, 800, 800)
    assert len(elems) > 0
    assert all("<" in e for e in elems)


# ---------------------------------------------------------------------------
# compute_mo_surface
# ---------------------------------------------------------------------------


def test_compute_mo_surface_sets_contours(caffeine_mol, caffeine_homo_cube):
    cfg = RenderConfig(auto_orient=False)
    compute_mo_surface(caffeine_mol, caffeine_homo_cube, cfg, MOParams())
    assert cfg.mo_contours is not None


# ---------------------------------------------------------------------------
# compute_esp_surface
# ---------------------------------------------------------------------------


def test_compute_esp_surface_sets_surface(caffeine_mol, caffeine_dens_cube, caffeine_esp_cube):
    cfg = RenderConfig(auto_orient=False)
    compute_esp_surface(caffeine_mol, caffeine_dens_cube, caffeine_esp_cube, cfg, ESPParams())
    assert cfg.esp_surface is not None
    assert cfg.esp_surface.png_data_uri.startswith("data:image/png;base64,")


def test_esp_surface_svg_returns_elements(caffeine_mol, caffeine_dens_cube, caffeine_esp_cube):
    from xyzrender.esp import esp_surface_svg

    cfg = RenderConfig(auto_orient=False)
    compute_esp_surface(caffeine_mol, caffeine_dens_cube, caffeine_esp_cube, cfg, ESPParams())
    assert cfg.esp_surface is not None
    elems = esp_surface_svg(cfg.esp_surface, 100.0, 400.0, 400.0, 800, 800, 0.9)
    assert len(elems) > 0
    assert all(isinstance(e, str) for e in elems)


def test_render_svg_includes_esp_colorbar(caffeine_mol, caffeine_dens_cube, caffeine_esp_cube):
    from xyzrender.renderer import render_svg

    cfg = RenderConfig(auto_orient=False, cbar=True)
    compute_esp_surface(caffeine_mol, caffeine_dens_cube, caffeine_esp_cube, cfg, ESPParams())
    svg = render_svg(caffeine_mol.graph, cfg)

    assert "linearGradient" in svg
    assert "\u2212" in svg
    assert ".000" in svg


def test_render_svg_esp_palette_changes_colorbar(caffeine_mol, caffeine_dens_cube, caffeine_esp_cube):
    from xyzrender.renderer import render_svg

    cfg = RenderConfig(auto_orient=False, cbar=True, cmap_palette="coolwarm")
    compute_esp_surface(caffeine_mol, caffeine_dens_cube, caffeine_esp_cube, cfg, ESPParams())
    svg = render_svg(caffeine_mol.graph, cfg)

    assert "#b40426" in svg
    assert "#3b4cc0" in svg


def test_render_svg_esp_colorbar_uses_actual_range(caffeine_mol):
    from xyzrender.esp import ESPSurface
    from xyzrender.renderer import render_svg

    cfg = RenderConfig(auto_orient=False, cbar=True)
    cfg.esp_surface = ESPSurface(
        png_data_uri="data:image/png;base64,",
        resolution=10,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        esp_vmin=-0.029,
        esp_vmax=0.185,
    )

    svg = render_svg(caffeine_mol.graph, cfg)

    assert ">0</text>" in svg
    assert ">.185</text>" in svg
    assert ">\u22120</text>" in svg
    assert ">.029</text>" in svg


def test_esp_surface_uses_manual_cmap_range(caffeine_mol, caffeine_dens_cube, caffeine_esp_cube):
    cfg = RenderConfig(auto_orient=False, cmap_range=(-0.003, 0.003))
    compute_esp_surface(caffeine_mol, caffeine_dens_cube, caffeine_esp_cube, cfg, ESPParams())
    assert cfg.esp_surface is not None
    assert cfg.esp_surface.esp_vmin == pytest.approx(-0.003)
    assert cfg.esp_surface.esp_vmax == pytest.approx(0.003)


def test_esp_surface_uses_symmetric_cmap_range(caffeine_mol, caffeine_dens_cube, caffeine_esp_cube):
    cfg = RenderConfig(auto_orient=False, cmap_symm=True)
    compute_esp_surface(caffeine_mol, caffeine_dens_cube, caffeine_esp_cube, cfg, ESPParams())
    assert cfg.esp_surface is not None
    assert cfg.esp_surface.esp_vmin == pytest.approx(-cfg.esp_surface.esp_vmax)


def test_render_svg_esp_colorbar_uses_manual_cmap_range(caffeine_mol, caffeine_dens_cube, caffeine_esp_cube):
    from xyzrender.renderer import render_svg

    cfg = RenderConfig(auto_orient=False, cbar=True, cmap_range=(-0.003, 0.003))
    compute_esp_surface(caffeine_mol, caffeine_dens_cube, caffeine_esp_cube, cfg, ESPParams())
    svg = render_svg(caffeine_mol.graph, cfg)

    assert "\u22120</text>" in svg
    assert ">.003</text>" in svg


def test_esp_cmap_range_and_symm_are_mutually_exclusive():
    from xyzrender.api import load, render

    mol = load(str(STRUCTURES / "caffeine_dens.cube"))
    with pytest.raises(ValueError, match="mutually exclusive"):
        render(mol, esp=str(STRUCTURES / "caffeine_esp.cube"), cmap_range=(-0.003, 0.003), cmap_symm=True)


# ---------------------------------------------------------------------------
# compute_nci_surface
# ---------------------------------------------------------------------------


def test_compute_nci_surface_sets_contours(nci_mol, nci_dens_cube, nci_grad_cube):
    cfg = RenderConfig(auto_orient=False)
    compute_nci_surface(nci_mol, nci_dens_cube, nci_grad_cube, cfg, NCIParams())
    assert cfg.nci_contours is not None


def test_nci_loops_svg_returns_paths(nci_mol, nci_dens_cube, nci_grad_cube):
    from xyzrender.nci import nci_loops_svg

    cfg = RenderConfig(auto_orient=False)
    compute_nci_surface(nci_mol, nci_dens_cube, nci_grad_cube, cfg, NCIParams())
    assert cfg.nci_contours is not None
    elems = nci_loops_svg(cfg.nci_contours, 0.7, 100.0, 400.0, 400.0, 800, 800)
    assert len(elems) > 0
    assert all("<path" in e for e in elems)


def test_compute_nci_surface_supports_explicit_igmh_mode():
    from xyzrender.api import Molecule

    cfg = RenderConfig(auto_orient=False)
    sl2r = cube_from_array(np.zeros((12, 12, 12), dtype=float))
    dg = cube_from_array(np.pad(np.full((4, 4, 4), 0.02, dtype=float), 4))
    mol = Molecule(graph=graph_from_cube(sl2r))

    compute_nci_surface(mol, sl2r, dg, cfg, NCIParams(), surface_mode="high_field")

    assert cfg.nci_contours is not None
    assert cfg.nci_contours.lobes


def test_compute_nci_surface_auto_classifies_high_field_surface():
    from xyzrender.api import Molecule

    cfg = RenderConfig(auto_orient=False)
    sl2r = cube_from_array(np.zeros((12, 12, 12), dtype=float))
    dg = cube_from_array(np.pad(np.full((4, 4, 4), 0.02, dtype=float), 4))
    mol = Molecule(graph=graph_from_cube(sl2r))

    compute_nci_surface(mol, sl2r, dg, cfg, NCIParams())

    assert cfg.nci_contours is not None
    assert cfg.nci_contours.lobes
