"""Tests for shared utilities."""

import numpy as np

from xyzrender.utils import pca_matrix, pca_orient


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
