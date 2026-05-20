"""Shared utilities for xyzrender."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

    from xyzrender.cube import CubeData


def parse_atom_indices(spec: str | list[int], *, one_indexed: bool = False) -> list[int]:
    """Parse an atom specifier into a list of atom indices.

    Accepts a 1-indexed string (``"1-5,8,12"``) or a 1-indexed
    ``list[int]``.  By default converts to 0-indexed output.
    Pass ``one_indexed=True`` to keep 1-indexed (for passing to API
    functions that expect user-facing numbering).
    """
    offset = 0 if one_indexed else -1
    if isinstance(spec, list):
        return [i + offset for i in spec]
    if not isinstance(spec, str) or not spec.strip():
        return []
    indices: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            indices.extend(range(int(a) + offset, int(b) + offset + 1))
        else:
            indices.append(int(part) + offset)
    return indices


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

    # Degenerate: single atom or all coincident
    if len(c_fit) < 2 or np.allclose(c_fit, 0, atol=1e-12):
        return (c, np.eye(3)) if return_matrix else c

    # Diatomic: align bond along x
    if len(c_fit) == 2:
        ax = c_fit[1] - c_fit[0]
        ax /= np.linalg.norm(ax)
        ref = np.eye(3)[np.argmin(np.abs(ax))]
        z = np.cross(ax, ref)
        z /= np.linalg.norm(z)
        y = np.cross(z, ax)
        rot = np.vstack([ax, y, z])
        oriented = c @ rot.T
        return (oriented, rot) if return_matrix else oriented

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
    if len(c) < 2 or np.allclose(c, 0, atol=1e-12):
        return np.eye(3)
    if len(c) == 2:
        ax = c[1] - c[0]
        ax /= np.linalg.norm(ax)
        ref = np.eye(3)[np.argmin(np.abs(ax))]
        z = np.cross(ax, ref)
        z /= np.linalg.norm(z)
        y = np.cross(z, ax)
        return np.vstack([ax, y, z])
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[-1] *= -1
    return vt


def align_cube_to_atoms(
    cube: CubeData | None,
    graph: nx.Graph,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Kabsch rotation from cube atom positions to current graph positions.

    Replaces the cube-alignment half of the old ``resolve_orientation``
    function.  Pure read: no mutation of either *cube* or *graph*.

    Returns ``(rot, atom_centroid, target_centroid)`` where:

    - ``rot`` is the 3x3 rotation that maps cube-frame to graph-frame, or
      ``None`` when no rotation is needed (positions already match within
      ``1e-6``, or *cube* is ``None``).
    - ``atom_centroid`` is the centroid of *cube*'s atom positions (or the
      graph centroid if *cube* is ``None``) — used by surface builders to
      translate cube-grid points before rotating.
    - ``target_centroid`` is the centroid of the current graph positions —
      used by surface builders as the translation after rotating.
    """
    node_ids = list(graph.nodes())
    curr = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
    target_centroid = curr.mean(axis=0)

    if cube is None:
        return None, target_centroid, target_centroid

    orig = np.array([p for _, p in cube.atoms], dtype=float)
    atom_centroid = orig.mean(axis=0)
    if np.allclose(orig, curr, atol=1e-6):
        return None, atom_centroid, target_centroid
    rot = kabsch_rotation(orig, curr)
    return rot, atom_centroid, target_centroid


def _apply_rot_to_vecs(
    rot: np.ndarray,
    directions: np.ndarray,
    origins: np.ndarray,
    centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate direction vectors and translate origins around *centroid* by *rot*.

    Works for shape ``(3,)`` (single vector) or ``(N, 3)`` (row-vectors).
    Returns ``(rotated_directions, rotated_origins)``.
    """
    return (rot @ directions.T).T, (rot @ (origins - centroid).T).T + centroid


def rotation_to_align(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Return the rotation matrix mapping unit vector *v_from* onto *v_to*.

    Uses Rodrigues' formula on the cross-product axis.  Handles the parallel
    and anti-parallel edge cases (180° flip picks an arbitrary perpendicular).
    """
    a = v_from / np.linalg.norm(v_from)
    b = v_to / np.linalg.norm(v_to)
    cos = float(np.dot(a, b))
    if cos > 0.9999:
        return np.eye(3)
    if cos < -0.9999:
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
    else:
        axis = np.cross(a, b)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(cos, -1.0, 1.0))
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


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
    if "lattice" in graph.graph:
        origin = np.asarray(graph.graph.get("lattice_origin", np.zeros(3)), dtype=float)
        graph.graph["lattice"], graph.graph["lattice_origin"] = _apply_rot_to_vecs(
            rot, graph.graph["lattice"], origin, centroid
        )


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


def kabsch_align(
    ref_positions: np.ndarray,
    mobile_positions: np.ndarray,
    align_atoms: list[int] | None = None,
) -> np.ndarray:
    """Kabsch RMSD alignment of *mobile_positions* onto *ref_positions*.

    Parameters
    ----------
    ref_positions, mobile_positions:
        (N, 3) arrays of matching atom positions.  Must have the same shape.
    align_atoms:
        Optional list of 0-indexed atom indices to fit on.  When given (min 3),
        the rotation and translation are computed from this subset only, then
        applied to *all* atoms.  ``None`` (default) fits on every atom.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Aligned positions for *mobile_positions*.
    """
    if ref_positions.shape != mobile_positions.shape:
        msg = f"kabsch_align: shape mismatch — ref {ref_positions.shape} vs mobile {mobile_positions.shape}"
        raise ValueError(msg)

    if align_atoms is not None:
        if len(align_atoms) < 3:
            msg = "kabsch_align: align_atoms must contain at least 3 indices to define a plane"
            raise ValueError(msg)
        n = ref_positions.shape[0]
        for idx in align_atoms:
            if not (0 <= idx < n):
                msg = f"kabsch_align: align_atoms index {idx} out of range for {n} atoms"
                raise ValueError(msg)
        ref_sub = ref_positions[align_atoms]
        mob_sub = mobile_positions[align_atoms]
    else:
        ref_sub = ref_positions
        mob_sub = mobile_positions

    c_ref = ref_sub.mean(axis=0)
    c_mob = mob_sub.mean(axis=0)
    # kabsch_rotation(mobile, ref) → h = mobile_centered.T @ ref_centered → R s.t. mobile @ R.T ≈ ref
    rot = kabsch_rotation(mob_sub, ref_sub)
    return (mobile_positions - c_mob) @ rot.T + c_ref


def mcs_kabsch_align(
    ref_positions: np.ndarray,
    mobile_positions: np.ndarray,
    ref_indices: list[int],
    mobile_indices: list[int],
) -> np.ndarray:
    """Kabsch alignment using a matched subset (MCS) of atoms.

    Unlike :func:`kabsch_align`, the two position arrays may have different
    shapes.  The rotation is fitted on the paired subset and applied to all
    of *mobile_positions*.

    Parameters
    ----------
    ref_positions:
        (N1, 3) reference positions.
    mobile_positions:
        (N2, 3) mobile positions.
    ref_indices, mobile_indices:
        Paired indices into the respective position arrays (same length, >= 3).

    Returns
    -------
    np.ndarray, shape (N2, 3)
    """
    ref_sub = ref_positions[ref_indices]
    mob_sub = mobile_positions[mobile_indices]
    c_ref = ref_sub.mean(axis=0)
    c_mob = mob_sub.mean(axis=0)
    rot = kabsch_rotation(mob_sub, ref_sub)
    return (mobile_positions - c_mob) @ rot.T + c_ref
