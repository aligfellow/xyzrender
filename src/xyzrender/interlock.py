"""VDW interlocking-spheres silhouette.

Using the approach by David Meijer, github.com/moltools/cinemol.git.

For each atom we sample points around its outline (a circle perpendicular
to the view), drop the ones hidden by neighbouring spheres, and take the
2D hull of what remains.  Where two spheres overlap, the cut is sampled
once and added to both polygons so they meet on identical vertices.
Only clusters of overlapping atoms run through this path; isolated atoms
return None and the caller draws a plain circle.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from xyzrender.hull import _convex_hull_2d

_ARC_POINTS = 32  # samples per pairwise intersection circle


_PERIMETER_CACHE: dict[int, np.ndarray] = {}


def _perimeter_unit(n: int) -> np.ndarray:
    """``n`` unit-circle points in the z=0 plane, shape (n, 3)."""
    pts = _PERIMETER_CACHE.get(n)
    if pts is None:
        thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        pts = np.column_stack((np.cos(thetas), np.sin(thetas), np.zeros(n)))
        _PERIMETER_CACHE[n] = pts
    return pts


def _intersection_circles(
    c_a: np.ndarray,
    r_a: np.ndarray,
    c_b: np.ndarray,
    r_b: np.ndarray,
    n_arc: int = _ARC_POINTS,
) -> np.ndarray:
    """Intersection-circle points for P sphere pairs at once, shape (P, n_arc, 3).

    Pairs the caller already filtered to actually-intersecting (so no per-pair
    early-exit needed); numerically degenerate pairs collapse to a point.
    """
    d_vec = c_b - c_a  # (P, 3)
    d = np.linalg.norm(d_vec, axis=1)  # (P,)
    d_safe = np.where(d > 1e-9, d, 1.0)
    a = (d * d + r_a * r_a - r_b * r_b) / (2.0 * d_safe)
    h = np.sqrt(np.maximum(r_a * r_a - a * a, 0.0))
    normal = d_vec / d_safe[:, None]  # (P, 3)
    # Pick a helper axis not parallel to each normal (vectorised).
    helper = np.where(np.abs(normal[:, 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(normal, u)
    centre = c_a + a[:, None] * normal  # (P, 3)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_arc, endpoint=False)
    cos_t = np.cos(thetas)[None, :, None]  # (1, n_arc, 1)
    sin_t = np.sin(thetas)[None, :, None]
    return centre[:, None, :] + h[:, None, None] * (cos_t * u[:, None, :] + sin_t * v[:, None, :])


def compute_interlock_polygons(
    centers_3d: np.ndarray,
    radii_3d: np.ndarray,
    *,
    samples: int = 64,
    min_clip_fraction: float = 0.03,
) -> list[np.ndarray | None]:
    """Per-atom (K, 2) silhouette polygons in xy, or ``None`` if no overlap."""
    n = centers_3d.shape[0]
    out: list[np.ndarray | None] = [None] * n
    if n == 0:
        return out

    # Squared distances throughout: d² < r² avoids per-element sqrt.
    c2d = centers_3d[:, :2]
    delta = c2d[:, None, :] - c2d[None, :, :]
    d2d_sq = (delta * delta).sum(-1)
    r_sum_2d_sq = (radii_3d[:, None] + radii_3d[None, :]) ** 2
    overlap_2d = d2d_sq < r_sum_2d_sq
    np.fill_diagonal(overlap_2d, False)
    if not overlap_2d.any():
        return out

    g = nx.Graph()
    g.add_nodes_from(range(n))
    ii, jj = np.where(np.triu(overlap_2d, k=1))
    g.add_edges_from(zip(ii.tolist(), jj.tolist(), strict=True))

    perim = _perimeter_unit(samples)  # (S, 3)

    for comp in nx.connected_components(g):
        if len(comp) < 2:
            continue
        members = sorted(comp)
        cc = centers_3d[members]
        rc = radii_3d[members]
        k_n = len(members)

        # Pair pre-filter: only seed arcs for pairs whose 3D spheres really
        # intersect.  Then compute all intersection circles in one batch.
        d3d_sq = ((cc[:, None, :] - cc[None, :, :]) ** 2).sum(-1)
        r_sum_sq = (rc[:, None] + rc[None, :]) ** 2
        r_diff_sq = (rc[:, None] - rc[None, :]) ** 2
        intersects = (d3d_sq < r_sum_sq) & (d3d_sq > r_diff_sq)
        ia, ib = np.where(np.triu(intersects, k=1))
        pair_arcs: dict[tuple[int, int], np.ndarray] = {}
        if ia.size:
            for (a_idx, b_idx), arc in zip(
                zip(ia.tolist(), ib.tolist(), strict=True),
                _intersection_circles(cc[ia], rc[ia], cc[ib], rc[ib]),
                strict=True,
            ):
                pair_arcs[(a_idx, b_idx)] = arc

        for k, atom_idx in enumerate(members):
            sil = cc[k] + perim * rc[k]  # (S, 3) screen-stable perimeter

            arcs = [arc for (a, b), arc in pair_arcs.items() if k in (a, b)]
            pts = np.concatenate([sil, *arcs], axis=0) if arcs else sil

            mask = np.ones(k_n, dtype=bool)
            mask[k] = False
            others_c = cc[mask]
            others_r = rc[mask]
            eps = 1e-6 * float(rc[k])
            others_r_eps_sq = (others_r - eps) ** 2
            diff = sil[:, None, :] - others_c[None, :, :]
            d_perim_sq = np.einsum("ijk,ijk->ij", diff, diff)
            perim_inside = (d_perim_sq < others_r_eps_sq[None, :]).any(axis=1)
            # Skip atoms whose clipped arc is too small to distinguish from a
            # plain circle — the polygon path is much more expensive.
            if perim_inside.sum() < min_clip_fraction * len(sil):
                continue
            if arcs:
                arc_pts = pts[len(sil) :]
                diff_a = arc_pts[:, None, :] - others_c[None, :, :]
                d_arc_sq = np.einsum("ijk,ijk->ij", diff_a, diff_a)
                arc_inside = (d_arc_sq < others_r_eps_sq[None, :]).any(axis=1)
                inside_any = np.concatenate([perim_inside, arc_inside])
            else:
                inside_any = perim_inside
            visible = pts[~inside_any]

            if visible.shape[0] < 3:
                continue
            hull_idx = _convex_hull_2d(visible[:, :2])
            if hull_idx.size < 3:
                continue
            out[atom_idx] = visible[hull_idx][:, :2]

    return out
