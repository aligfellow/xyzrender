"""Tests for shared utilities."""

import numpy as np

from xyzrender.utils import kabsch_rotation, pca_matrix, pca_orient


def test_pca_orient_shape():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    result = pca_orient(pos)
    assert result.shape == (3, 3)


def test_pca_orient_centered():
    pos = np.array([[10, 20, 30], [11, 20, 30], [10, 21, 30]], dtype=float)
    result = pca_orient(pos)
    # Result should be centered (mean ~ 0)
    assert np.allclose(result.mean(axis=0), 0, atol=1e-10)


def test_pca_orient_largest_variance_on_x():
    # Spread along z in input — after PCA, largest variance should be on x
    pos = np.array([[0, 0, 0], [0, 0, 5], [0, 0.1, 2.5]], dtype=float)
    result = pca_orient(pos)
    x_var = np.var(result[:, 0])
    y_var = np.var(result[:, 1])
    z_var = np.var(result[:, 2])
    assert x_var >= y_var
    assert y_var >= z_var


def test_pca_matrix_shape():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    vt = pca_matrix(pos)
    assert vt.shape == (3, 3)


def test_pca_matrix_orthogonal():
    pos = np.random.randn(10, 3)
    vt = pca_matrix(pos)
    # Vt should be orthogonal: Vt @ Vt.T = I
    assert np.allclose(vt @ vt.T, np.eye(3), atol=1e-10)


def test_pca_orient_monoatomic():
    pos = np.array([[5.0, 3.0, 1.0]])
    oriented, rot = pca_orient(pos, return_matrix=True)
    assert np.allclose(rot, np.eye(3))
    assert np.allclose(oriented, [[0.0, 0.0, 0.0]])


def test_pca_orient_diatomic():
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    oriented, rot = pca_orient(pos, return_matrix=True)
    assert rot.shape == (3, 3)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-10)
    # Bond should be along x after orientation
    assert np.var(oriented[:, 0]) >= np.var(oriented[:, 1])


def test_pca_orient_coincident():
    pos = np.array([[1.0, 2.0, 3.0]] * 5)
    oriented, rot = pca_orient(pos, return_matrix=True)
    assert np.allclose(rot, np.eye(3))
    assert np.allclose(oriented, 0.0)


def test_pca_matrix_monoatomic():
    assert np.allclose(pca_matrix(np.array([[5.0, 3.0, 1.0]])), np.eye(3))


def test_pca_matrix_diatomic():
    vt = pca_matrix(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]))
    assert vt.shape == (3, 3)
    assert np.allclose(vt @ vt.T, np.eye(3), atol=1e-10)


def test_kabsch_recovers_rotation():
    """Apply a known 90-degree rotation and verify Kabsch recovers it."""
    rng = np.random.default_rng(42)
    original = rng.standard_normal((8, 3))
    # 90-degree rotation around z-axis
    theta = np.pi / 2
    expected = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    target = (original - original.mean(axis=0)) @ expected.T + original.mean(axis=0)
    recovered = kabsch_rotation(original, target)
    assert np.allclose(recovered, expected, atol=1e-10)


def test_kabsch_identity():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    rot = kabsch_rotation(pos, pos)
    assert np.allclose(rot, np.eye(3), atol=1e-10)


# ---------------------------------------------------------------------------
# apply_axis_angle_rotation — atom AND lattice must rotate together so that
# atom↔lattice fractional coords stay invariant. Used by gif_rot trajectory
# rendering ([gif.py:807](src/xyzrender/gif.py#L807)).
# ---------------------------------------------------------------------------


import pytest  # noqa: E402


def _frac_pair_with_lattice():
    """Triclinic test graph — non-orthogonal lattice to surface drift bugs."""
    import networkx as nx

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
    g.graph["lattice"] = lattice
    g.graph["lattice_origin"] = origin
    return g, frac


def _frac_of(graph):
    lat = np.asarray(graph.graph["lattice"], dtype=float)
    origin = np.asarray(graph.graph.get("lattice_origin", np.zeros(3)), dtype=float)
    pos = np.array([graph.nodes[n]["position"] for n in graph.nodes()], dtype=float)
    return (pos - origin) @ np.linalg.inv(lat)


@pytest.mark.parametrize(
    ("axis", "angle"),
    [
        (np.array([0.0, 0.0, 1.0]), 30.0),
        (np.array([1.0, 0.0, 0.0]), 45.0),
        (np.array([1.0, 1.0, 1.0]), 60.0),
        (np.array([0.3, -0.7, 0.5]), 75.0),
    ],
)
def test_apply_axis_angle_rotation_preserves_atom_lattice_fractional_coords(axis, angle):
    """apply_axis_angle_rotation rotates atoms AND lattice — fractional coords
    must be invariant (utils.py:264 → _apply_rot_to_vecs at utils.py:291)."""
    from xyzrender.utils import apply_axis_angle_rotation

    g, frac_before = _frac_pair_with_lattice()
    apply_axis_angle_rotation(g, axis, angle)
    frac_after = _frac_of(g)

    assert np.allclose(frac_after, frac_before, atol=1e-9), (
        f"apply_axis_angle_rotation(axis={axis.tolist()}, angle={angle}) drifted atoms "
        f"relative to lattice. before={frac_before.tolist()}, after={frac_after.tolist()}"
    )
