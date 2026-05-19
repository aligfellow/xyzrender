"""Tests for viewer integration.

Includes orientation invariants — atom/lattice fractional coords must stay
co-registered through every rotation path. If a `orient_hkl_to_view` or
`rotate_with_viewer` test fails, the cell box has drifted relative to the atoms
(a class of bug that's been seen in earlier versions and is now pinned down).
"""

import numpy as np
import pytest


def test_rotate_with_viewer_missing_vmol(monkeypatch):
    """ImportError with helpful message when vmol is not installed."""
    import builtins

    import networkx as nx

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "vmol":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from xyzrender.viewer import rotate_with_viewer

    g = nx.Graph()
    g.add_node(0, symbol="H", position=(0.0, 0.0, 0.0))

    with pytest.raises(ImportError, match="Interactive viewer requires vmol"):
        rotate_with_viewer(g)


# ---------------------------------------------------------------------------
# Orientation invariants — atom ↔ lattice fractional-coord preservation
# ---------------------------------------------------------------------------


def _triclinic_graph_with_lattice():
    """Build a small graph with a non-orthogonal lattice — orthorhombic would
    accidentally pass tests that lose the off-diagonal structure."""
    import networkx as nx

    g = nx.Graph()
    lattice = np.array(
        [
            [5.0, 0.0, 0.0],
            [1.0, 5.0, 0.0],
            [0.5, 0.3, 4.5],
        ],
        dtype=float,
    )
    origin = np.array([0.2, -0.1, 0.05], dtype=float)
    frac = np.array(
        [
            [0.10, 0.25, 0.40],
            [0.60, 0.55, 0.30],
            [0.30, 0.80, 0.70],
        ],
        dtype=float,
    )
    cart = frac @ lattice + origin
    for i, (s, p) in enumerate(zip(["C", "O", "N"], cart, strict=True)):
        g.add_node(i, symbol=s, position=tuple(p.tolist()))
    g.add_edges_from([(0, 1), (1, 2)])
    g.graph["lattice"] = lattice
    g.graph["lattice_origin"] = origin
    return g, frac.copy()


def _fractional_coords(graph) -> np.ndarray:
    lat = np.asarray(graph.graph["lattice"], dtype=float)
    origin = np.asarray(graph.graph.get("lattice_origin", np.zeros(3)), dtype=float)
    pos = np.array([graph.nodes[n]["position"] for n in graph.nodes()], dtype=float)
    return (pos - origin) @ np.linalg.inv(lat)


def test_orient_hkl_to_view_co_rotates_cfg_vectors():
    """Vector arrows in cfg.vectors must co-rotate with atoms+lattice
    (regression guard for viewer.py:189-191)."""
    from xyzrender.types import CellData, RenderConfig, VectorArrow
    from xyzrender.utils import kabsch_rotation
    from xyzrender.viewer import orient_hkl_to_view

    g, _ = _triclinic_graph_with_lattice()
    cell = CellData(lattice=g.graph["lattice"].copy(), cell_origin=g.graph["lattice_origin"].copy())

    arrow = VectorArrow(vector=np.array([1.0, 0.0, 0.0]), origin=np.array([0.0, 0.0, 0.0]), color="red")
    cfg = RenderConfig()
    cfg.vectors = [arrow]

    pos_before = np.array([g.nodes[n]["position"] for n in g.nodes()], dtype=float)
    centroid = pos_before.mean(axis=0)

    orient_hkl_to_view(g, cell, "110", cfg)

    pos_after = np.array([g.nodes[n]["position"] for n in g.nodes()], dtype=float)
    rot = kabsch_rotation(pos_before - centroid, pos_after - centroid)

    expected_vec = rot @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(cfg.vectors[0].vector, expected_vec, atol=1e-9), (
        f"cfg.vectors[0].vector not co-rotated — expected ≈ {expected_vec.tolist()}, "
        f"got {cfg.vectors[0].vector.tolist()}"
    )


def _make_fake_viewer_output(symbols, original_pos, rot, centroid):
    """Build a fake vmol output string for a known rotation matrix."""
    rotated = (rot @ (original_pos - centroid).T).T + centroid
    rot_flat = ",".join(f"{v:.10f}" for v in rot.ravel())
    lines = [f"rot:{rot_flat}", f"{len(symbols)}", "comment"]
    lines.extend(f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in zip(symbols, rotated, strict=True))
    return "\n".join(lines)


def test_rotate_with_viewer_preserves_atom_lattice_fractional_coords(monkeypatch):
    """rotate_with_viewer must rotate the lattice by the same matrix it applies to atoms
    (viewer.py:114-120)."""
    from xyzrender import viewer as viewer_mod

    g, frac_before = _triclinic_graph_with_lattice()
    symbols = [g.nodes[n]["symbol"] for n in g.nodes()]
    pos = np.array([g.nodes[n]["position"] for n in g.nodes()], dtype=float)
    centroid = pos.mean(axis=0)

    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    theta = np.radians(60.0)
    c, s = np.cos(theta), np.sin(theta)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = c * np.eye(3) + s * k + (1 - c) * np.outer(axis, axis)

    fake_output = _make_fake_viewer_output(symbols, pos, rot, centroid)
    monkeypatch.setattr(viewer_mod, "_run_viewer_with_atoms", lambda *_a, **_kw: fake_output)
    monkeypatch.setitem(__import__("sys").modules, "vmol", type("FakeVmol", (), {"vmol": object()}))

    out_rot, _c1, _c2 = viewer_mod.rotate_with_viewer(g, backend="vmol")

    assert out_rot is not None
    frac_after = _fractional_coords(g)
    assert np.allclose(frac_after, frac_before, atol=1e-9), (
        f"rotate_with_viewer drifted atoms relative to lattice. "
        f"before={frac_before.tolist()}, after={frac_after.tolist()}"
    )
