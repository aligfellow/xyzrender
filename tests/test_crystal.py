"""Tests for crystal structure loading and rendering."""

import copy
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).parent.parent / "examples" / "structures"
VASP_FILE = EXAMPLES / "NV63.vasp"
QE_FILE = EXAMPLES / "NV63.in"
EXTXYZ_FILE = EXAMPLES / "caffeine_cell.xyz"


@pytest.fixture(scope="module")
def vasp_crystal():
    from xyzrender.crystal import load_crystal

    return load_crystal(VASP_FILE, "vasp")


@pytest.fixture(scope="module")
def qe_crystal():
    from xyzrender.crystal import load_crystal

    return load_crystal(QE_FILE, "qe")


# ---------------------------------------------------------------------------
# I/O tests
# ---------------------------------------------------------------------------


def test_load_crystal_vasp(vasp_crystal):
    graph, cell_data = vasp_crystal
    assert graph.number_of_nodes() == 63
    assert cell_data.lattice.shape == (3, 3)


def test_load_crystal_qe(qe_crystal):
    graph, cell_data = qe_crystal
    assert graph.number_of_nodes() == 63
    assert cell_data.lattice.shape == (3, 3)


def test_load_crystal_vasp_qe_same_lattice(vasp_crystal, qe_crystal):
    """VASP and QE files describe the same structure — lattices must match."""
    _, cd_vasp = vasp_crystal
    _, cd_qe = qe_crystal
    np.testing.assert_allclose(cd_vasp.lattice, cd_qe.lattice, atol=1e-3)


def test_crystal_images(vasp_crystal):
    """add_crystal_images produces image nodes each bonded to ≥1 cell atom."""
    from xyzrender.crystal import add_crystal_images

    graph, cell_data = copy.deepcopy(vasp_crystal)
    n_cell = graph.number_of_nodes()
    n_added = add_crystal_images(graph, cell_data)

    assert n_added > 0, "Expected at least some image atoms"
    cell_ids = set(range(n_cell))

    for node_id, attrs in graph.nodes(data=True):
        if not attrs.get("image", False):
            continue
        # Every image atom must have at least one bond to a cell atom
        neighbors = list(graph.neighbors(node_id))
        cell_neighbors = [nb for nb in neighbors if nb in cell_ids]
        assert cell_neighbors, f"Image node {node_id} (sym={attrs['symbol']}) has no bond to a cell atom"


def test_crystal_images_no_orphans(vasp_crystal):
    """No image node may exist without at least one image_bond=True edge to a cell atom."""
    from xyzrender.crystal import add_crystal_images

    graph, cell_data = copy.deepcopy(vasp_crystal)
    n_cell = graph.number_of_nodes()
    add_crystal_images(graph, cell_data)

    cell_ids = set(range(n_cell))
    for node_id, attrs in graph.nodes(data=True):
        if not attrs.get("image", False):
            continue
        image_bonds_to_cell = [
            nb
            for nb in graph.neighbors(node_id)
            if nb in cell_ids and graph.edges[node_id, nb].get("image_bond", False)
        ]
        assert image_bonds_to_cell, f"Image node {node_id} has no image_bond edge to a cell atom"


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------


def test_render_crystal_cell_box(vasp_crystal):
    """render_svg with cell_data + show_cell=True produces exactly 12 cell edges."""
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, cell_data = vasp_crystal
    cfg = RenderConfig(cell_data=cell_data, show_cell=True)
    svg = render_svg(graph, cfg)

    # Count lines tagged as cell edges
    cell_lines = [ln for ln in svg.splitlines() if 'class="cell-edge"' in ln]
    assert len(cell_lines) == 12, f"Expected 12 cell-box lines, got {len(cell_lines)}"


def test_render_no_cell(vasp_crystal):
    """render_svg with show_cell=False produces no cell-edge lines."""
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, cell_data = vasp_crystal
    cfg = RenderConfig(cell_data=cell_data, show_cell=False)
    svg = render_svg(graph, cfg)

    cell_lines = [ln for ln in svg.splitlines() if 'class="cell-edge"' in ln]
    assert len(cell_lines) == 0


def test_render_crystal_no_cell_data(vasp_crystal):
    """Crystal-specific SVG elements are absent when cell_data is None."""
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, _cell_data = vasp_crystal
    cfg = RenderConfig()
    svg = render_svg(graph, cfg)
    assert 'class="cell-edge"' not in svg


def test_render_crystal_with_images(vasp_crystal):
    """Image atoms render with opacity and produce a valid SVG."""
    from xyzrender.crystal import add_crystal_images
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, cell_data = copy.deepcopy(vasp_crystal)
    add_crystal_images(graph, cell_data)
    cfg = RenderConfig(cell_data=cell_data, show_cell=True, periodic_image_opacity=0.5)
    svg = render_svg(graph, cfg)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert 'opacity="0.50"' in svg


def test_render_crystal_no_images(vasp_crystal):
    """Without add_crystal_images, no opacity attributes appear in atoms/bonds."""
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph, cell_data = vasp_crystal
    cfg = RenderConfig(cell_data=cell_data, show_cell=True, periodic_image_opacity=0.5)
    svg = render_svg(graph, cfg)
    assert 'opacity="0.50"' not in svg


# ---------------------------------------------------------------------------
# extXYZ Lattice= tests (--cell path, no phonopy)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extxyz_graph():
    from xyzrender.readers import load_molecule

    graph, _ = load_molecule(EXTXYZ_FILE)
    return graph


def test_extxyz_lattice_parsed(extxyz_graph):
    """extXYZ file with Lattice= stores a (3, 3) lattice on graph.graph."""
    lat = np.array(extxyz_graph.graph["lattice"])
    assert lat.shape == (3, 3)


def test_extxyz_lattice_values(extxyz_graph):
    """Lattice= row-major values are parsed correctly."""
    lat = np.array(extxyz_graph.graph["lattice"])
    # caffeine_cell.xyz: Lattice="14.8 0.0 0.0  0.0 16.7 0.0  -0.484 0.0 3.940"
    np.testing.assert_allclose(lat[0, 0], 14.8, atol=1e-3)
    np.testing.assert_allclose(lat[1, 1], 16.7, atol=1e-3)
    np.testing.assert_allclose(lat[2, 2], 3.940, atol=1e-3)


def test_extxyz_cell_box_renders(extxyz_graph):
    """extXYZ --cell path: CellData from graph.graph produces 12 cell edges."""
    from xyzrender.renderer import render_svg
    from xyzrender.types import CellData, RenderConfig

    cfg = RenderConfig(
        cell_data=CellData(lattice=np.array(extxyz_graph.graph["lattice"], dtype=float)),
        show_cell=True,
    )
    svg = render_svg(extxyz_graph, cfg)
    cell_lines = [ln for ln in svg.splitlines() if 'class="cell-edge"' in ln]
    assert len(cell_lines) == 12


def test_orient_hkl_cell_corotates_with_atoms(vasp_crystal):
    """orient_hkl_to_view keeps lattice and atom positions mutually consistent."""
    import copy

    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig
    from xyzrender.viewer import orient_hkl_to_view

    graph, cell_data = copy.deepcopy(vasp_crystal)
    lat_before = cell_data.lattice.copy()
    cfg = RenderConfig(cell_data=cell_data, show_cell=True)
    orient_hkl_to_view(graph, cell_data, "100", cfg)

    # Lattice must have been rotated
    assert not np.allclose(cell_data.lattice, lat_before, atol=1e-6)
    # Render must still produce exactly 12 cell edges
    svg = render_svg(graph, cfg)
    cell_lines = [ln for ln in svg.splitlines() if 'class="cell-edge"' in ln]
    assert len(cell_lines) == 12


def test_orient_hkl_cell_origin_updated_from_zero(vasp_crystal):
    """orient_hkl_to_view updates cell_origin from its zero default value."""
    import copy

    from xyzrender.types import RenderConfig
    from xyzrender.viewer import orient_hkl_to_view

    graph, cell_data = copy.deepcopy(vasp_crystal)
    assert np.allclose(cell_data.cell_origin, np.zeros(3))

    cfg = RenderConfig()
    orient_hkl_to_view(graph, cell_data, "100", cfg)

    # cell_origin is rotated around the atom centroid; since the centroid is
    # not at the origin the rotated zero-origin is non-zero
    assert not np.allclose(cell_data.cell_origin, np.zeros(3), atol=1e-6), (
        f"cell_origin should be non-zero after HKL rotation (got {cell_data.cell_origin})"
    )


def test_orient_hkl_fractional_coords_preserved(vasp_crystal):
    """orient_hkl_to_view preserves the fractional coordinates of all atoms."""
    import copy

    from xyzrender.types import RenderConfig
    from xyzrender.viewer import orient_hkl_to_view

    graph, cell_data = copy.deepcopy(vasp_crystal)
    node_ids = list(graph.nodes())
    pos0 = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
    frac_before = np.linalg.solve(cell_data.lattice.T, (pos0 - cell_data.cell_origin).T).T

    cfg = RenderConfig()
    orient_hkl_to_view(graph, cell_data, "111", cfg)

    pos1 = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
    frac_after = np.linalg.solve(cell_data.lattice.T, (pos1 - cell_data.cell_origin).T).T

    np.testing.assert_allclose(
        frac_after,
        frac_before,
        atol=1e-9,
        err_msg="Fractional coordinates must be preserved after orient_hkl_to_view",
    )


def test_orient_hkl_cell_corotates_with_ghost_atoms(vasp_crystal):
    """orient_hkl_to_view rotates all ghost atoms and keeps 12 cell edges."""
    import copy

    from xyzrender.crystal import add_crystal_images
    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig
    from xyzrender.viewer import orient_hkl_to_view

    graph, cell_data = copy.deepcopy(vasp_crystal)
    n_real = graph.number_of_nodes()
    add_crystal_images(graph, cell_data)
    assert graph.number_of_nodes() > n_real, "Ghost atoms must have been added"

    cfg = RenderConfig(cell_data=cell_data, show_cell=True, auto_orient=False)
    orient_hkl_to_view(graph, cell_data, "110", cfg)

    svg = render_svg(graph, cfg)
    cell_lines = [ln for ln in svg.splitlines() if 'class="cell-edge"' in ln]
    assert len(cell_lines) == 12

    # COM of real atoms must be inside the cell (fractional coords in [0, 1])
    real_pos = np.array([graph.nodes[i]["position"] for i in range(n_real)], dtype=float)
    com = real_pos.mean(axis=0)
    frac = np.linalg.solve(cell_data.lattice.T, com - cell_data.cell_origin)
    assert np.all(frac > -0.5), f"COM fractional coords {frac} are far outside the cell after rotation"
    assert np.all(frac < 1.5), f"COM fractional coords {frac} are far outside the cell after rotation"
