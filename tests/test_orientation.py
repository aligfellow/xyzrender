"""Cross-cutting orientation invariants.

System-wide invariants for orient / lattice / PCA that span multiple modules.
Feature-level orientation tests live with their feature (test_gif.py,
test_crystal.py, test_overlay.py, test_api.py).
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

from xyzrender.types import CellData
from xyzrender.utils import pca_matrix, pca_orient

_STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_periodic_graph() -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    """3-atom triclinic graph with a non-zero lattice origin.

    Returns ``(graph, lattice, origin)`` so callers can recompute fractional
    coords after a rotation.
    """
    g = nx.Graph()
    lattice = np.array([[5.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.5, 0.3, 4.5]], dtype=float)
    origin = np.array([0.2, -0.1, 0.05], dtype=float)
    frac = np.array(
        [[0.10, 0.25, 0.40], [0.60, 0.55, 0.30], [0.30, 0.80, 0.70]],
        dtype=float,
    )
    cart = frac @ lattice + origin
    for i, (s, p) in enumerate(zip(["C", "O", "N"], cart, strict=True)):
        g.add_node(i, symbol=s, position=tuple(p.tolist()))
    g.add_edges_from([(0, 1), (1, 2)])
    g.graph["lattice"] = lattice.copy()
    g.graph["lattice_origin"] = origin.copy()
    return g, lattice, origin


def _build_4atom_graph() -> nx.Graph:
    """Asymmetric 4-atom system with distinct PCA axes."""
    g = nx.Graph()
    positions = [
        ("C", (0.0, 0.0, 0.0)),
        ("C", (3.0, 0.5, 0.1)),
        ("O", (0.5, 1.5, -0.2)),
        ("N", (-0.3, 0.2, 1.0)),
    ]
    for i, (sym, p) in enumerate(positions):
        g.add_node(i, symbol=sym, position=p)
    g.add_edges_from([(0, 1), (0, 2), (0, 3)])
    return g


def _positions(graph: nx.Graph) -> np.ndarray:
    return np.array([graph.nodes[n]["position"] for n in graph.nodes()], dtype=float)


# ---------------------------------------------------------------------------
# PCA primitives
# ---------------------------------------------------------------------------


def test_pca_orient_double_application_stable():
    """After one PCA orient the variance axes are already aligned; applying
    PCA again must give back the same positions (up to per-axis sign)."""
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((20, 3)) * np.array([4.0, 2.0, 0.5])
    once = pca_orient(pos)
    twice = pca_orient(once)
    assert np.allclose(np.abs(twice), np.abs(once), atol=1e-9)


def test_pca_matrix_and_pca_orient_agree():
    """``pca_matrix(pos)`` and ``pca_orient(pos, return_matrix=True)`` must
    return the same rotation, so callers can mix the two primitives safely."""
    rng = np.random.default_rng(1)
    pos = rng.standard_normal((10, 3)) * np.array([3.0, 1.5, 0.5])
    vt_only = pca_matrix(pos)
    _, vt_with_orient = pca_orient(pos, return_matrix=True)
    assert np.allclose(vt_only, vt_with_orient, atol=1e-9)


def test_priority_pairs_influence_orientation():
    """``pca_orient(priority_pairs=...)`` must change the result.

    Atoms 0 and 3 are far apart along z; without priority pairs PCA picks z
    as least-variance.  With (0, 3) weighted, that bond is pulled into xy.
    """
    pos = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]],
        dtype=float,
    )
    out_no_pp = pca_orient(pos)
    out_with_pp = pca_orient(pos, priority_pairs=[(0, 3)])
    assert not np.allclose(out_no_pp, out_with_pp, atol=1e-6)


# ---------------------------------------------------------------------------
# Molecule.orient() — the canonical orientation method
# ---------------------------------------------------------------------------


def test_molecule_orient_preserves_centroid_relative_fractional_coords():
    """Fractional coords relative to the molecular centroid must survive a
    rotation — atoms and lattice were rotated by the same matrix."""
    from xyzrender.api import Molecule

    graph, lat0, _ = _build_periodic_graph()
    pos_before = _positions(graph)
    centroid_before = pos_before.mean(axis=0)
    frac_before = (pos_before - centroid_before) @ np.linalg.inv(lat0)

    mol = Molecule(graph=graph, cell_data=CellData(lattice=lat0.copy()))
    mol.orient()
    assert mol.cell_data is not None

    pos_after = _positions(mol.graph)
    centroid_after = pos_after.mean(axis=0)
    rotated_lat = np.asarray(mol.cell_data.lattice, dtype=float)
    frac_after = (pos_after - centroid_after) @ np.linalg.inv(rotated_lat)

    assert np.allclose(frac_after, frac_before, atol=1e-9)


def test_molecule_orient_syncs_lattice_storage_locations():
    """``cell_data.lattice`` and ``graph.graph['lattice']`` must agree after
    orient — the bug the old ``resolve_orientation`` had.
    """
    from xyzrender.api import Molecule

    graph, lat0, _ = _build_periodic_graph()
    mol = Molecule(graph=graph, cell_data=CellData(lattice=lat0.copy()))

    mol.orient()
    assert mol.cell_data is not None
    assert np.allclose(
        np.asarray(mol.cell_data.lattice, dtype=float),
        np.asarray(mol.graph.graph["lattice"], dtype=float),
        atol=1e-12,
    )


def test_molecule_orient_centers_atoms_at_origin():
    """Atoms are written back centred at the origin (mean = 0) — matches the
    surface builders' centred-at-0 frame so cube-grid math stays unchanged."""
    from xyzrender.api import Molecule

    graph, _, _ = _build_periodic_graph()
    mol = Molecule(graph=graph)
    mol.orient()

    centroid_after = _positions(mol.graph).mean(axis=0)
    assert np.allclose(centroid_after, np.zeros(3), atol=1e-9)


def test_molecule_orient_with_tilt_changes_z():
    """``tilt_degrees=-30`` applies an extra x-axis rotation after PCA: x is
    preserved, z is not."""
    from xyzrender.api import Molecule

    graph_a, _, _ = _build_periodic_graph()
    graph_b, _, _ = _build_periodic_graph()
    Molecule(graph=graph_a).orient(tilt_degrees=None)
    Molecule(graph=graph_b).orient(tilt_degrees=-30.0)

    pos_a = _positions(graph_a)
    pos_b = _positions(graph_b)
    assert np.allclose(pos_a[:, 0], pos_b[:, 0], atol=1e-9)
    assert not np.allclose(pos_a[:, 2], pos_b[:, 2], atol=1e-6)


def test_molecule_orient_is_idempotent():
    """Second ``orient()`` is a no-op once ``mol.oriented`` is True."""
    from xyzrender.api import Molecule

    graph, lat0, _ = _build_periodic_graph()
    mol = Molecule(graph=graph, cell_data=CellData(lattice=lat0.copy()))

    mol.orient()
    assert mol.lattice is not None
    pos_first = _positions(mol.graph)
    lat_first = mol.lattice.copy()

    mol.orient()
    assert np.allclose(_positions(mol.graph), pos_first, atol=1e-12)
    assert np.allclose(mol.lattice, lat_first, atol=1e-12)


def test_molecule_orient_force_reorients():
    """``force=True`` overrides the idempotence check."""
    from xyzrender.api import Molecule

    graph, _, _ = _build_periodic_graph()
    mol = Molecule(graph=graph, oriented=True)
    pos_before = _positions(mol.graph)

    mol.orient()
    assert np.allclose(_positions(mol.graph), pos_before, atol=1e-12)

    mol.orient(force=True)
    assert not np.allclose(_positions(mol.graph), pos_before, atol=1e-3)


def test_molecule_copy_isolates_mutations():
    """``mol.copy()`` deep-copies the graph and cell_data so orienting the
    copy doesn't leak back to the original."""
    from xyzrender.api import Molecule

    graph, lat0, _ = _build_periodic_graph()
    mol = Molecule(graph=graph, cell_data=CellData(lattice=lat0.copy()))
    assert mol.lattice is not None
    pos_orig = _positions(mol.graph)
    lat_orig = mol.lattice.copy()

    mol.copy().orient()

    assert np.allclose(_positions(mol.graph), pos_orig, atol=1e-12)
    assert np.allclose(mol.lattice, lat_orig, atol=1e-12)


# ---------------------------------------------------------------------------
# render() — orient state contracts
# ---------------------------------------------------------------------------


def test_molecule_oriented_flag_suppresses_pca_in_render():
    """``mol.oriented = True`` short-circuits PCA in render(), protecting
    callers that orient via the interactive viewer first."""
    from xyzrender.api import load, render

    mol = load(str(_STRUCTURES / "caffeine.xyz"))
    pos_before = _positions(mol.graph)

    mol.oriented = True
    render(mol)

    assert np.allclose(_positions(mol.graph), pos_before, atol=1e-12)


def test_render_does_not_mutate_caller_molecule_graph():
    """render() must deep-copy at entry so PCA / ghost-atom additions during
    rendering do not leak back to the caller's mol — render(mol) must be
    safe to call repeatedly with the same input."""
    from xyzrender.api import load, render

    mol = load(str(_STRUCTURES / "caffeine.xyz"))
    pos_before = _positions(mol.graph).copy()

    render(mol)

    assert np.allclose(_positions(mol.graph), pos_before, atol=1e-12)


# ---------------------------------------------------------------------------
# Static render vs GIF frame 0: the two paths use different orient functions
# (``Molecule.orient`` vs ``gif._orient_graph``); they must leave the graph
# in the same state.  Cheap mock-based checks first; full-render integration
# tests for the surface cases below.
# ---------------------------------------------------------------------------


def test_static_orient_and_gif_orient_graph_agree_plain():
    """``Molecule.orient()`` vs ``gif._orient_graph(g, pca_matrix(pos))`` on a
    plain PCA orient.  If these diverge, the GIF's first frame would show
    the molecule at a different orientation from the static SVG."""
    from xyzrender.api import Molecule
    from xyzrender.gif import _orient_graph

    g_static = _build_4atom_graph()
    Molecule(graph=g_static).orient()

    g_gif = _build_4atom_graph()
    _orient_graph(g_gif, pca_matrix(_positions(g_gif)), None)

    assert np.allclose(_positions(g_static), _positions(g_gif), atol=1e-12)


def test_static_orient_and_gif_orient_graph_agree_with_lattice():
    """Same as above with a lattice: both paths must rotate ``cell_data.lattice``
    and ``graph.graph['lattice']`` and keep them in sync."""
    from xyzrender.api import Molecule
    from xyzrender.gif import _orient_graph

    g_static, lat, _ = _build_periodic_graph()
    mol = Molecule(graph=g_static, cell_data=CellData(lattice=lat.copy()))
    mol.orient()

    g_gif, lat_gif, _ = _build_periodic_graph()
    cd_gif = CellData(lattice=lat_gif.copy())
    _orient_graph(g_gif, pca_matrix(_positions(g_gif)), cd_gif)

    assert np.allclose(_positions(g_static), _positions(g_gif), atol=1e-12)
    assert mol.lattice is not None
    assert np.allclose(mol.lattice, cd_gif.lattice, atol=1e-12)
    assert np.allclose(g_static.graph["lattice"], g_gif.graph["lattice"], atol=1e-12)


def test_static_orient_and_gif_orient_with_mo_tilt_agree():
    """The MO -30° tilt block in render_rotation_gif must match
    ``Molecule.orient(tilt_degrees=-30)``."""
    from xyzrender.api import Molecule
    from xyzrender.gif import _orient_graph

    g_static = _build_4atom_graph()
    Molecule(graph=g_static).orient(tilt_degrees=-30.0)

    # Replicate the GIF path: PCA via _orient_graph, then the manual tilt
    # block at gif.py:326-333.
    g_gif = _build_4atom_graph()
    _orient_graph(g_gif, pca_matrix(_positions(g_gif)), None)
    theta = np.radians(-30.0)
    c, s = np.cos(theta), np.sin(theta)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    pos = _positions(g_gif) @ rx.T
    for i, nid in enumerate(g_gif.nodes()):
        g_gif.nodes[nid]["position"] = tuple(pos[i].tolist())

    assert np.allclose(_positions(g_static), _positions(g_gif), atol=1e-12)


# ---------------------------------------------------------------------------
# Integration: static render vs GIF frame 0 (full pipeline, ~0.5s each).
# Catches orchestration-layer bugs that the mock checks above can't —
# e.g. the path-based render_gif silently dropping cube_data.
# ---------------------------------------------------------------------------

_CIRCLE_CXY = re.compile(r'<circle\s+cx="([\d.\-]+)"\s+cy="([\d.\-]+)"')


def _atom_xy(svg: str) -> np.ndarray:
    return np.array([(float(x), float(y)) for x, y in _CIRCLE_CXY.findall(svg)], dtype=float)


def _normalise_layout(xy: np.ndarray) -> np.ndarray:
    """Centre on centroid, scale so max-norm = 1, sort lexicographically.

    Invariant to viewport translation + uniform scale (static auto-fit vs
    GIF bounding-sphere) and to SVG draw order (z-sort tie-breaking).
    """
    centred = xy - xy.mean(axis=0)
    scale = float(np.linalg.norm(centred, axis=1).max())
    normed = centred / max(scale, 1e-12)
    return normed[np.lexsort((normed[:, 1], normed[:, 0]))]


def _capture_gif_frame_svg(mol_path: str, **render_gif_kw) -> str:
    """Render a 1-frame rotation GIF; return the SVG passed to PNG conversion.

    Forces serial rendering (multiprocessing workers don't see monkey-patches)
    and intercepts ``svg_to_png_bytes`` to grab the frame SVG before any
    PNG conversion.
    """
    from xyzrender import render_gif

    captured: list[str] = []

    def _capture(svg: str, *, size=None):
        captured.append(svg)
        return b""

    def _serial(worker, items, total):
        results = [b""] * total
        for item in items:
            idx, png = worker(item)
            results[idx] = png
        return results

    with (
        patch("xyzrender.gif.svg_to_png_bytes", side_effect=_capture),
        patch("xyzrender.gif._parallel_render", side_effect=_serial),
        patch("xyzrender.gif._stitch_gif", lambda *a, **kw: None),
    ):
        render_gif(mol_path, gif_rot="y", rot_frames=1, output="/tmp/_unused.gif", **render_gif_kw)
    assert captured, "no SVG captured from GIF frame 0"
    return captured[0]


def _assert_static_matches_gif_frame0(path: str, **kw) -> None:
    from xyzrender import render

    static_svg = str(render(path, **kw))
    frame_svg = _capture_gif_frame_svg(path, **kw)
    s = _normalise_layout(_atom_xy(static_svg))
    f = _normalise_layout(_atom_xy(frame_svg))
    assert s.shape == f.shape, f"atom count differs: static={s.shape}, frame={f.shape}"
    diff = float(np.abs(s - f).max())
    assert diff < 1e-3, f"static / GIF-frame-0 atom layout diverged (max diff {diff:.4g})"


def test_integration_static_matches_gif_frame0_mo_surface():
    """MO surface with -30° tilt — regression for the path-based render_gif
    bug where ``cube_data`` was dropped, silently skipping the MO tilt block."""
    _assert_static_matches_gif_frame0(str(_STRUCTURES / "caffeine_homo.cube"), mo=True)


def test_integration_static_matches_gif_frame0_dens_surface():
    """Density surface across the full render pipeline."""
    _assert_static_matches_gif_frame0(str(_STRUCTURES / "caffeine_dens.cube"), dens=True)


def test_integration_static_matches_gif_frame0_overlay_with_align_atoms():
    """Overlay + ``align_atoms`` (Kabsch alignment) must apply identically.

    ``_apply_overlay`` is shared between the paths so this guards future
    divergence in the order of overlay vs PCA orient.
    """
    main = str(_STRUCTURES / "caffeine.xyz")
    _assert_static_matches_gif_frame0(main, overlay=main, align_atoms=[1, 2, 3])


def test_integration_static_matches_gif_frame0_overlay_auto_path():
    """Overlay with no align_atoms (auto MCS/best-fit path).

    Uses benzene (small) as main + caffeine (larger) as overlay so the
    merged-graph PCA is dominated by caffeine and would diverge from
    benzene's own PCA — catches GIF-path re-PCA on the merged graph
    instead of using the base's pre-merge PCA.
    """
    main = str(_STRUCTURES / "benzene.xyz")
    overlay = str(_STRUCTURES / "caffeine.xyz")
    _assert_static_matches_gif_frame0(main, overlay=overlay)
