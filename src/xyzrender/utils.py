"""Shared utilities for xyzrender."""

from __future__ import annotations

import numpy as np


def pca_orient(
    pos: np.ndarray,
    priority_pairs: list[tuple[int, int]] | None = None,
    priority_weight: float = 5.0,
) -> np.ndarray:
    """Align molecule: largest variance along x, then y, smallest along z (depth).

    If *priority_pairs* are given (e.g. TS bonds), those atom positions are
    up-weighted so their bond vectors preferentially lie in the xy (visible) plane.
    """
    c = pos - pos.mean(axis=0)
    if priority_pairs:
        # Duplicate priority atom positions to bias PCA towards their plane
        extra = []
        for i, j in priority_pairs:
            extra.extend([c[i], c[j]])
        extra = np.array(extra) * priority_weight
        c_weighted = np.vstack([c, extra])
    else:
        c_weighted = c
    _, _, vt = np.linalg.svd(c_weighted, full_matrices=False)
    oriented = c @ vt.T

    # For TS bonds: rotate around z to align TS bond vectors along x (horizontal)
    if priority_pairs:
        vecs = np.array([oriented[j, :2] - oriented[i, :2] for i, j in priority_pairs])
        avg_dir = vecs.mean(axis=0)
        mag = np.linalg.norm(avg_dir)
        if mag > 1e-6:
            theta = -np.arctan2(avg_dir[1], avg_dir[0])
            ct, st = np.cos(theta), np.sin(theta)
            rz = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
            oriented = oriented @ rz.T

    return oriented


def pca_matrix(pos: np.ndarray) -> np.ndarray:
    """Compute PCA rotation matrix (Vt) without applying it."""
    c = pos - pos.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt
