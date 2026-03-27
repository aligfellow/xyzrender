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
