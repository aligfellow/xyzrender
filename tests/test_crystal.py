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


def test_build_supercell_repeats_atoms(vasp_crystal):
    """build_supercell replicates the unit cell correctly."""
    import copy

    from xyzrender.crystal import build_supercell

    graph, cell_data = copy.deepcopy(vasp_crystal)
    n0 = graph.number_of_nodes()
    a = cell_data.lattice[0].copy()

    g2 = build_supercell(graph, cell_data, (2, 1, 1))
    assert g2.number_of_nodes() == 2 * n0

    # For at least one atom, there must be a copy at +a.
    p0 = np.array(g2.nodes[0]["position"], dtype=float)
    target = p0 + a
    pos_all = np.array([g2.nodes[i]["position"] for i in g2.nodes()], dtype=float)
    dists = np.linalg.norm(pos_all - target[None, :], axis=1)
    assert float(dists.min()) < 1e-6, "Expected a replicated atom at +a in the (2,1,1) supercell"


def test_build_supercell_preserves_intra_edges(vasp_crystal):
    """Intra-replica edges and their attributes are preserved from the unit cell."""
    import copy

    from xyzrender.crystal import build_supercell

    graph, cell_data = copy.deepcopy(vasp_crystal)
    e0 = graph.number_of_edges()
    n0 = graph.number_of_nodes()

    g2 = build_supercell(graph, cell_data, (2, 1, 1))
    # Must have at least 2x original edges (intra-replica) plus some cross-boundary
    assert g2.number_of_edges() >= 2 * e0

    # First replica edges (indices 0..n0-1) should match original edge count
    replica0_edges = [(u, v) for u, v in g2.edges() if u < n0 and v < n0]
    assert len(replica0_edges) == e0


def test_build_supercell_cross_boundary_bonds(vasp_crystal):
    """Cross-boundary bonds exist between adjacent replicas in the supercell."""
    import copy

    from xyzrender.crystal import build_supercell

    graph, cell_data = copy.deepcopy(vasp_crystal)
    n0 = graph.number_of_nodes()

    g2 = build_supercell(graph, cell_data, (2, 1, 1))
    # Cross-boundary edges: one endpoint in [0, n0) and the other in [n0, 2*n0)
    cross_edges = [(u, v) for u, v in g2.edges() if (u < n0) != (v < n0)]
    assert len(cross_edges) > 0, "Expected cross-boundary bonds between replicas"


def test_build_supercell_different_axes():
    """Non-trivial supercell along different axes produces correct atom counts."""
    from xyzrender.crystal import build_supercell
    from xyzrender.readers import load_molecule
    from xyzrender.types import CellData

    g, _ = load_molecule(EXTXYZ_FILE)
    n0 = g.number_of_nodes()
    lat = np.array(g.graph["lattice"], dtype=float)
    cd = CellData(lattice=lat)

    for repeats, factor in [((1, 2, 1), 2), ((1, 1, 3), 3), ((2, 2, 2), 8)]:
        g2 = build_supercell(g, cd, repeats)
        assert g2.number_of_nodes() == factor * n0, f"repeats={repeats}"


def _reference_bonded_pairs(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    eidx_a: np.ndarray,
    eidx_b: np.ndarray,
    elem_thresh: np.ndarray,
) -> set[tuple[int, int]]:
    """All-pairs distance check (ground truth for _find_bonded_pairs tests)."""
    dists = np.linalg.norm(pos_a[:, None, :] - pos_b[None, :, :], axis=2)
    thresh_mat = elem_thresh[eidx_a[:, None], eidx_b[None, :]]
    ii, jj = np.where(dists < thresh_mat)
    return set(zip(ii.tolist(), jj.tolist(), strict=False))


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_find_bonded_pairs_spatial_hash_square_grid(seed: int):
    """_find_bonded_pairs uses the cell-list path when na * nb > 50_000 (square na == nb).

    Covers the bulk of that function: B key sort, 27-offset key lookup, repeat /
    b_order expansion, and final d² threshold filter.  Regression for MOF-scale
    cells where a broken mask broadcast raised ValueError.
    """
    from xyzrender.crystal import _build_elem_thresh, _find_bonded_pairs

    rng = np.random.default_rng(seed)
    n = 240
    assert n * n > 50_000
    pos = rng.uniform(0.0, 50.0, size=(n, 3))
    syms = ["C"] * n
    elem_thresh, eidx, max_cutoff = _build_elem_thresh(syms)
    shift = np.array([0.05, 0.0, 0.0], dtype=float)
    pos_b = pos
    pos_a = pos + shift

    pairs_cell = set(_find_bonded_pairs(pos_a, pos_b, eidx, eidx, elem_thresh, max_cutoff))
    pairs_ref = _reference_bonded_pairs(pos_a, pos_b, eidx, eidx, elem_thresh)
    assert pairs_cell == pairs_ref


def test_find_bonded_pairs_spatial_hash_rectangular_na_nb():
    """Same spatial-hash branch with na != nb but na * nb still > 50_000."""
    from xyzrender.crystal import _build_elem_thresh, _find_bonded_pairs

    rng = np.random.default_rng(7)
    na, nb = 300, 200
    assert na * nb > 50_000
    pos_a = rng.uniform(0.0, 40.0, size=(na, 3))
    pos_b = rng.uniform(0.0, 40.0, size=(nb, 3))
    # One (E, E) matrix for C/O pairs; indices match a=all C, b=all O.
    elem_thresh, eidx_stacked, max_cutoff = _build_elem_thresh(["C"] * na + ["O"] * nb)
    eidx_a = eidx_stacked[:na]
    eidx_b = eidx_stacked[na:]

    pairs_cell = set(_find_bonded_pairs(pos_a, pos_b, eidx_a, eidx_b, elem_thresh, max_cutoff))
    pairs_ref = _reference_bonded_pairs(pos_a, pos_b, eidx_a, eidx_b, elem_thresh)
    assert pairs_cell == pairs_ref


# ---------------------------------------------------------------------------
# build_supercell node ordering contract + ghost generation consistency
# ---------------------------------------------------------------------------


def test_build_supercell_node_ordering_matches_ids():
    """Nodes must be inserted in ascending-id order so list(graph.nodes())[:n_base]
    is the unit cell. _add_crystal_images_supercell slices that to find base positions."""
    from xyzrender.crystal import build_supercell
    from xyzrender.readers import load_molecule

    g, _ = load_molecule(EXTXYZ_FILE)
    lat = np.array(g.graph["lattice"], dtype=float)
    from xyzrender.types import CellData

    cd = CellData(lattice=lat)

    sc = build_supercell(g, cd, (2, 2, 1))
    nodes = list(sc.nodes())
    assert nodes == sorted(nodes), "node insertion order must be ascending id"


def test_add_crystal_images_supercell_matches_generic():
    """Supercell-optimized and generic ghost paths must agree on a real crystal."""
    from xyzrender.crystal import (
        _add_crystal_images_generic,
        _add_crystal_images_supercell,
        build_supercell,
    )
    from xyzrender.readers import load_molecule
    from xyzrender.types import CellData

    src = EXAMPLES / "NV63_cell.xyz"
    g, _ = load_molecule(src)
    lat = np.array(g.graph["lattice"], dtype=float)
    cd = CellData(lattice=lat)

    repeats = (2, 2, 1)
    sc_cell = CellData(lattice=lat * np.array(repeats)[:, None])

    sc_generic = build_supercell(g, cd, repeats)
    n_generic = _add_crystal_images_generic(sc_generic, sc_cell)

    sc_opt = build_supercell(g, cd, repeats)
    n_opt = _add_crystal_images_supercell(sc_opt, sc_cell, repeats, cd, g.number_of_nodes())

    assert n_opt == n_generic, f"ghost count mismatch: optimized={n_opt}, generic={n_generic}"
    assert sc_opt.number_of_nodes() == sc_generic.number_of_nodes()
    assert sc_opt.number_of_edges() == sc_generic.number_of_edges()


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
# extXYZ Lattice= tests (--cell path)
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


def test_extxyz_supercell_builds(extxyz_graph):
    """Supercell expansion works for any input that carries a lattice (extXYZ)."""
    from xyzrender.crystal import build_supercell
    from xyzrender.types import CellData

    g = extxyz_graph
    n0 = g.number_of_nodes()
    lat = np.array(g.graph["lattice"], dtype=float)
    origin = np.array(g.graph.get("lattice_origin", np.zeros(3)), dtype=float)
    cd = CellData(lattice=lat, cell_origin=origin)

    g2 = build_supercell(g, cd, (2, 2, 1))
    assert g2.number_of_nodes() == 4 * n0


def test_supercell_ghosts_no_overlap():
    """Ghost atoms on a supercell must not overlap with real supercell atoms."""
    from xyzrender.crystal import add_crystal_images, build_supercell
    from xyzrender.readers import load_molecule
    from xyzrender.types import CellData

    g, _ = load_molecule(EXTXYZ_FILE)
    lat = np.array(g.graph["lattice"], dtype=float)
    cd = CellData(lattice=lat)

    repeats = (2, 1, 1)
    g2 = build_supercell(g, cd, repeats)
    real_pos = np.array([g2.nodes[i]["position"] for i in g2.nodes()], dtype=float)

    # Use supercell lattice for ghosts (as api.py now does)
    sc_lat = np.vstack([repeats[0] * lat[0], repeats[1] * lat[1], repeats[2] * lat[2]])
    ghost_cd = CellData(lattice=sc_lat, cell_origin=cd.cell_origin)
    add_crystal_images(g2, ghost_cd)

    # Check no ghost overlaps with any real atom
    for nid in g2.nodes():
        if not g2.nodes[nid].get("image", False):
            continue
        gpos = np.array(g2.nodes[nid]["position"], dtype=float)
        dists = np.linalg.norm(real_pos - gpos[None, :], axis=1)
        assert dists.min() > 0.1, f"Ghost {nid} at {gpos} overlaps with real atom (min dist {dists.min():.3f})"


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


def test_apply_axis_angle_rotation_keeps_ghost_atoms_with_parent_offset():
    """After rotation, each ghost atom's offset from its source atom must equal the
    pre-rotation offset rotated by the same matrix (i.e. ghosts move rigidly with
    their source). Catches a class of bug where ghosts get treated as independent
    atoms during rotation rather than rigid extensions of their unit-cell parent."""
    import networkx as nx

    from xyzrender.crystal import _add_crystal_images_generic
    from xyzrender.types import CellData
    from xyzrender.utils import apply_axis_angle_rotation

    # Tight periodic chain — two carbons inside a 1.5 Å cell so the
    # nearest-image bond crosses every face → guaranteed ghost atoms.
    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.2, 0.2, 0.2))
    g.add_node(1, symbol="C", position=(0.5, 0.5, 0.5))
    g.add_edge(0, 1, bond_order=1.0)
    lat = np.array([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]], dtype=float)
    g.graph["lattice"] = lat
    g.graph["lattice_origin"] = np.zeros(3)
    cell = CellData(lattice=lat.copy(), cell_origin=np.zeros(3))
    _add_crystal_images_generic(g, cell)

    ghosts = [n for n in g.nodes() if g.nodes[n].get("image", False)]
    if not ghosts:
        pytest.skip("Periodic chain fixture failed to produce ghost atoms")

    offsets_before = {}
    for gh in ghosts:
        src = g.nodes[gh]["source"]
        offsets_before[gh] = np.asarray(g.nodes[gh]["position"], dtype=float) - np.asarray(
            g.nodes[src]["position"], dtype=float
        )

    axis = np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0)
    angle = 47.0
    apply_axis_angle_rotation(g, axis, angle)

    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = c * np.eye(3) + s * k + (1 - c) * np.outer(axis, axis)

    for gh, off_before in offsets_before.items():
        src = g.nodes[gh]["source"]
        off_after = np.asarray(g.nodes[gh]["position"], dtype=float) - np.asarray(g.nodes[src]["position"], dtype=float)
        expected = rot @ off_before
        assert np.allclose(off_after, expected, atol=1e-9), (
            f"Ghost atom {gh} (source={src}) drifted relative to its source under rotation. "
            f"expected offset={expected.tolist()}, got={off_after.tolist()}"
        )
