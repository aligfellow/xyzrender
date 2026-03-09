"""Shared utilities for xyzrender."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

    from xyzrender.cube import CubeData
    from xyzrender.types import RenderConfig


@overload
def pca_orient(
    pos: np.ndarray,
    priority_pairs: list[tuple[int, int]] | None = ...,
    priority_weight: float = ...,
    *,
    fit_mask: np.ndarray | None = ...,
    return_matrix: Literal[False] = ...,
) -> np.ndarray: ...


@overload
def pca_orient(
    pos: np.ndarray,
    priority_pairs: list[tuple[int, int]] | None = ...,
    priority_weight: float = ...,
    *,
    fit_mask: np.ndarray | None = ...,
    return_matrix: Literal[True] = ...,
) -> tuple[np.ndarray, np.ndarray]: ...


def pca_orient(
    pos: np.ndarray,
    priority_pairs: list[tuple[int, int]] | None = None,
    priority_weight: float = 5.0,
    *,
    fit_mask: np.ndarray | None = None,
    return_matrix: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Align molecule: largest variance along x, then y, smallest along z (depth).

    If *priority_pairs* are given (e.g. TS bonds), those atom positions are
    up-weighted so their bond vectors preferentially lie in the xy (visible) plane.
    If *fit_mask* is given, only those positions are used to compute the PCA
    axes; the rotation is still applied to all positions.  This prevents NCI
    centroid dummy nodes from influencing the orientation.
    """
    fit = pos[fit_mask] if fit_mask is not None else pos
    centroid = fit.mean(axis=0)
    c = pos - centroid  # center all positions around fit centroid
    c_fit = fit - centroid
    if priority_pairs:
        # Duplicate priority atom positions to bias PCA towards their plane
        extra = []
        for i, j in priority_pairs:
            extra.extend([c_fit[i], c_fit[j]])
        extra = np.array(extra) * priority_weight
        c_weighted = np.vstack([c_fit, extra])
    else:
        c_weighted = c_fit
    _, _, vt = np.linalg.svd(c_weighted, full_matrices=False)
    # Ensure proper rotation (det=+1); SVD can return a reflection.
    if np.linalg.det(vt) < 0:
        vt[-1] *= -1
    rot = vt  # cumulative rotation matrix
    oriented = c @ rot.T  # apply rotation to ALL positions

    # For TS bonds: rotate around z to align TS bond vectors along x (horizontal)
    if priority_pairs:
        vecs = np.array([oriented[j, :2] - oriented[i, :2] for i, j in priority_pairs])
        avg_dir = vecs.mean(axis=0)
        mag = np.linalg.norm(avg_dir)
        if mag > 1e-6:
            theta = -np.arctan2(avg_dir[1], avg_dir[0])
            ct, st = np.cos(theta), np.sin(theta)
            rz = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
            rot = rz @ rot
            oriented = oriented @ rz.T

    if return_matrix:
        return oriented, rot
    return oriented


def pca_matrix(pos: np.ndarray) -> np.ndarray:
    """Compute PCA rotation matrix (Vt) without applying it."""
    c = pos - pos.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[-1] *= -1
    return vt


def resolve_orientation(
    graph: nx.Graph,
    cube: CubeData | None,
    cfg: RenderConfig,
    *,
    tilt_degrees: float | None = None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Apply PCA auto-orient (if enabled) and compute Kabsch rotation for cube alignment.

    When ``cfg.auto_orient`` is ``True``, atom positions in *graph* are
    rotated in-place by PCA so the largest variance lies along **x** and
    depth along **z**.  If ``tilt_degrees`` is given an additional rotation
    around the **x**-axis is applied (MO surfaces use ``-30.0`` to separate
    above/below-plane lobes).  Crystal lattice vectors and cell origin stored
    on *cfg* are co-rotated by the same matrix.

    After any PCA rotation the Kabsch algorithm is used to compute the
    rotation that maps the original cube atom positions to the current
    (possibly PCA-rotated) graph positions.  This rotation is then passed
    to the surface builders so that the volumetric grid is aligned with the
    rendered molecule.

    Parameters
    ----------
    graph:
        Molecular graph whose node ``"position"`` attributes may be updated.
    cube:
        Gaussian cube data whose atom positions serve as the reference frame.
        Pass ``None`` when no cube is available; in this case *rot* is
        ``None`` and both centroids equal the molecular centroid.
    cfg:
        Render configuration.  ``cfg.auto_orient`` is cleared to ``False``
        once PCA has been applied.
    tilt_degrees:
        Extra rotation around the x-axis applied **after** PCA, in degrees.
        Pass ``None`` (default) to skip the tilt.

    Returns
    -------
    rot : numpy.ndarray or None
        3x3 rotation matrix that maps cube-grid coordinates to the current
        view orientation, or ``None`` when no rotation was needed.
    atom_centroid : numpy.ndarray
        Centroid (Å) of the cube atom positions (or graph centroid when
        *cube* is ``None``).
    curr_centroid : numpy.ndarray
        Centroid (Å) of the current graph atom positions.
    """
    node_ids = list(graph.nodes())
    curr_pos = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
    curr_centroid = curr_pos.mean(axis=0)

    rot: np.ndarray | None = None

    if cfg.auto_orient:
        oriented, rot = pca_orient(curr_pos, return_matrix=True)

        if tilt_degrees is not None:
            theta = np.radians(tilt_degrees)
            rx = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(theta), -np.sin(theta)],
                    [0.0, np.sin(theta), np.cos(theta)],
                ]
            )
            rot = rx @ rot
            oriented = oriented @ rx.T

        centroid_before = curr_pos.mean(axis=0)
        curr_centroid = oriented.mean(axis=0)

        for idx, nid in enumerate(node_ids):
            graph.nodes[nid]["position"] = tuple(oriented[idx].tolist())

        # Co-rotate crystal lattice and cell origin by the same matrix
        if cfg.cell_data is not None:
            lat = cfg.cell_data.lattice
            cfg.cell_data.lattice = (rot @ lat.T).T
            cfg.cell_data.cell_origin = rot @ (cfg.cell_data.cell_origin - centroid_before) + curr_centroid

        cfg.auto_orient = False  # already applied; renderer must not re-apply

    if cube is None:
        atom_centroid = curr_centroid
        return rot, atom_centroid, curr_centroid

    atom_centroid = np.array([p for _, p in cube.atoms], dtype=float).mean(axis=0)

    # If no PCA was applied but positions may have been modified (e.g. interactive
    # viewer), compute the Kabsch rotation from original→current atom positions.
    if rot is None:
        orig = np.array([p for _, p in cube.atoms], dtype=float)
        curr = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
        if not np.allclose(orig, curr, atol=1e-6):
            rot = kabsch_rotation(orig, curr)

    # Recompute curr_centroid after potential in-place updates above
    curr_centroid = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float).mean(axis=0)

    return rot, atom_centroid, curr_centroid


def apply_axis_angle_rotation(graph: nx.Graph, axis: np.ndarray, angle: float) -> None:
    """Rotate all atom positions in-place around an arbitrary axis (degrees).

    Uses Rodrigues' rotation formula for a clean rotation around a single
    axis vector. Rotation is around the molecular centroid.

    Parameters
    ----------
    graph:
        Molecular graph whose node positions are updated in-place.
    axis:
        3-vector defining the rotation axis (need not be normalised).
    angle:
        Rotation angle in degrees.
    """
    from xyzrender.viewer import _apply_rot_to_lattice

    nodes = list(graph.nodes())
    theta = np.radians(angle)
    k = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    k_cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot = c * np.eye(3) + s * k_cross + (1 - c) * np.outer(k, k)

    positions = np.array([graph.nodes[n]["position"] for n in nodes])
    centroid = positions.mean(axis=0)
    rotated = (rot @ (positions - centroid).T).T + centroid
    for i, nid in enumerate(nodes):
        graph.nodes[nid]["position"] = tuple(rotated[i].tolist())
    _apply_rot_to_lattice(graph, rot, centroid)


def kabsch_rotation(original: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute optimal rotation matrix from *original* to *target* positions.

    Both arrays must have shape ``(N, 3)``.  They are centered internally
    (centroids subtracted) before computing the rotation via SVD.
    Handles reflections by correcting the sign of the determinant.

    Returns the 3x3 rotation matrix R such that ``(original - centroid) @ R.T``
    best aligns with ``(target - centroid)``.
    """
    oc = original - original.mean(axis=0)
    tc = target - target.mean(axis=0)
    h = oc.T @ tc
    u, _, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    return vt.T @ np.diag([1.0, 1.0, np.sign(d)]) @ u.T
