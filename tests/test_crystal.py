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


def test_cell_corotates_with_atoms(extxyz_graph):
    import copy

    from xyzrender.types import CellData, RenderConfig
    from xyzrender.viewer import apply_rotation

    graph = copy.deepcopy(extxyz_graph)
    lat_before = np.array(graph.graph["lattice"], dtype=float)
    cell_data = CellData(
        lattice=lat_before.copy(), cell_origin=np.array(graph.graph.get("lattice_origin", [0.0, 0.0, 0.0]), dtype=float)
    )
    cfg = RenderConfig(cell_data=cell_data, show_cell=True)
    assert cfg.cell_data is not None
    apply_rotation(graph, rx=30.0, ry=45.0, rz=15.0)
    lat_graph = np.array(graph.graph["lattice"], dtype=float)
    assert not np.allclose(lat_graph, lat_before, atol=1e-6)
    assert np.allclose(cfg.cell_data.lattice, lat_before, atol=1e-6)
    cfg.cell_data.lattice = lat_graph.copy()
    cfg.cell_data.cell_origin = np.array(graph.graph.get("lattice_origin", [0.0, 0.0, 0.0]), dtype=float)
    np.testing.assert_allclose(cfg.cell_data.lattice, lat_graph, atol=1e-9)
    node_ids = list(graph.nodes())
    positions = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
    centroid = positions.mean(axis=0)
    cell_norm = np.linalg.norm(cfg.cell_data.lattice, axis=1).max()
    origin_dist = np.linalg.norm(cfg.cell_data.cell_origin - centroid)
    assert origin_dist < 2 * cell_norm
    from xyzrender.renderer import render_svg

    svg = render_svg(graph, cfg)
    q = chr(34)
    cell_lines = [ln for ln in svg.splitlines() if "class=" + q + "cell-edge" + q in ln]
    assert len(cell_lines) == 12


def test_apply_rotation_sets_lattice_origin_when_absent(extxyz_graph):
    """apply_rotation must write graph.graph['lattice_origin'] even when the
    file had no explicit origin — the origin is (0,0,0) implicitly but must
    be updated to the rotated value so the cell box stays aligned."""
    import copy

    from xyzrender.viewer import apply_rotation

    graph = copy.deepcopy(extxyz_graph)
    # Confirm the fixture has no explicit origin (caffeine_cell.xyz has none)
    assert "lattice_origin" not in graph.graph

    apply_rotation(graph, rx=45.0, ry=0.0, rz=0.0)

    # After rotation the key must exist and must NOT be zeros
    assert "lattice_origin" in graph.graph, "apply_rotation must write lattice_origin"
    origin = np.array(graph.graph["lattice_origin"], dtype=float)
    assert not np.allclose(origin, np.zeros(3), atol=1e-6), (
        f"lattice_origin should be non-zero after a 45° rotation (got {origin})"
    )


def test_fractional_coords_preserved_after_rotation(extxyz_graph):
    """Fractional coordinates of all atoms must be unchanged after a consistent
    rotation of both atoms and cell (lattice + origin)."""
    import copy

    from xyzrender.viewer import apply_rotation

    graph = copy.deepcopy(extxyz_graph)
    lat0 = np.array(graph.graph["lattice"], dtype=float)
    orig0 = np.array(graph.graph.get("lattice_origin", np.zeros(3)), dtype=float)
    node_ids = list(graph.nodes())
    pos0 = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
    # Fractional coords before: solve lat.T @ f = (pos - origin) for each atom
    frac_before = np.linalg.solve(lat0.T, (pos0 - orig0).T).T  # shape (n, 3)

    apply_rotation(graph, rx=30.0, ry=20.0, rz=50.0)

    lat1 = np.array(graph.graph["lattice"], dtype=float)
    orig1 = np.array(graph.graph["lattice_origin"], dtype=float)
    pos1 = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
    frac_after = np.linalg.solve(lat1.T, (pos1 - orig1).T).T

    np.testing.assert_allclose(
        frac_after,
        frac_before,
        atol=1e-9,
        err_msg="Fractional coordinates must be preserved after rotation",
    )


def test_cell_corotates_with_ghost_atoms(extxyz_graph):
    """Full --cell -I pipeline: add ghost atoms, rotate, re-sync cell_data,
    render — cell box must still have exactly 12 edges and be consistent."""
    import copy

    from xyzrender.crystal import add_crystal_images
    from xyzrender.renderer import render_svg
    from xyzrender.types import CellData, RenderConfig
    from xyzrender.viewer import apply_rotation

    graph = copy.deepcopy(extxyz_graph)
    lat0 = np.array(graph.graph["lattice"], dtype=float)
    cell_data = CellData(lattice=lat0.copy())
    add_crystal_images(graph, cell_data)
    n_after_ghosts = graph.number_of_nodes()
    assert n_after_ghosts > extxyz_graph.number_of_nodes(), "Ghost atoms must have been added"

    # Simulate what rotate_with_viewer / --cell -I does: rotate (all nodes
    # including ghosts), then re-sync cell_data from updated graph.graph.
    apply_rotation(graph, rx=0.0, ry=90.0, rz=0.0)

    cell_data.lattice = np.array(graph.graph["lattice"], dtype=float)
    cell_data.cell_origin = np.array(graph.graph["lattice_origin"], dtype=float)

    cfg = RenderConfig(
        cell_data=cell_data,
        show_cell=True,
        auto_orient=False,
    )
    svg = render_svg(graph, cfg)
    cell_lines = [ln for ln in svg.splitlines() if 'class="cell-edge"' in ln]
    assert len(cell_lines) == 12

    # COM of real atoms must be within a reasonable distance of the cell box
    real_ids = [i for i in graph.nodes() if not graph.nodes[i].get("image", False)]
    real_pos = np.array([graph.nodes[i]["position"] for i in real_ids], dtype=float)
    com = real_pos.mean(axis=0)
    lat = cell_data.lattice
    orig = cell_data.cell_origin
    frac = np.linalg.solve(lat.T, com - orig)
    # All fractional coordinates of the COM should be roughly in [0, 1]
    # (within one cell width, allowing for atoms at the boundary)
    assert np.all(frac > -0.5), f"COM fractional coords {frac} are far outside the cell after rotation"
    assert np.all(frac < 1.5), f"COM fractional coords {frac} are far outside the cell after rotation"
