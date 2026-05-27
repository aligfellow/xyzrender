"""Tests for VDW interlocking-spheres silhouette."""

from __future__ import annotations

import numpy as np

from xyzrender.api import load, render
from xyzrender.config import build_render_config
from xyzrender.interlock import _intersection_circles, compute_interlock_polygons


def test_intersection_circles_on_both_spheres():
    """Sampled points sit exactly on both spheres — the no-gap precondition."""
    c_a = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    c_b = np.array([[1.4, 0.0, 0.0], [0.0, 1.4, 0.0]])
    r_a = np.array([1.0, 1.0])
    r_b = np.array([1.0, 1.0])
    pts = _intersection_circles(c_a, r_a, c_b, r_b, n_arc=24)
    assert pts.shape == (2, 24, 3)
    for p in range(2):
        assert np.allclose(np.linalg.norm(pts[p] - c_a[p], axis=1), r_a[p], atol=1e-9)
        assert np.allclose(np.linalg.norm(pts[p] - c_b[p], axis=1), r_b[p], atol=1e-9)


def test_interlock_no_overlap_returns_all_none():
    """Disjoint atoms fall through to the plain-circle path."""
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    radii = np.array([1.0, 1.0, 1.0])
    polys = compute_interlock_polygons(centers, radii)
    assert all(p is None for p in polys)


def test_interlock_polygons_share_intersection_vertices():
    """Adjacent polygons share vertices on the cut — no white gap."""
    centers = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    radii = np.array([1.0, 1.0])
    polys = compute_interlock_polygons(centers, radii)
    assert polys[0] is not None
    assert polys[1] is not None
    set_a = {tuple(np.round(v, 6)) for v in polys[0]}
    set_b = {tuple(np.round(v, 6)) for v in polys[1]}
    # At least the two endpoints of the visible cut should coincide.
    assert len(set_a & set_b) >= 2


def test_interlock_skip_when_clip_tiny():
    """Sub-threshold overlap stays a circle — keeps file size sane on bubble-like presets."""
    # Centres 1.999 apart, radii 1.0: the visible cut covers ~1° of each
    # perimeter, well below the 3% default threshold.
    centers = np.array([[0.0, 0.0, 0.0], [1.999, 0.0, 0.0]])
    radii = np.array([1.0, 1.0])
    polys = compute_interlock_polygons(centers, radii, min_clip_fraction=0.03)
    assert all(p is None for p in polys)


def test_render_vdw_preset_emits_polygons():
    """Space-filling benzene → all six carbons render as polygons."""
    svg = str(render(load("examples/structures/benzene.xyz"), config="vdw"))
    assert svg.count("<polygon") >= 6


def test_render_overlay_with_interlock_emits_polygons():
    """--vdw overlay with vdw_interlocking=True draws interlocked spheres."""
    cfg = build_render_config({"vdw_indices": [], "vdw_opacity": 1.0, "vdw_interlocking": True}, {})
    svg = str(render(load("examples/structures/benzene.xyz"), config=cfg))
    assert "<polygon" in svg


def test_render_default_emits_no_polygons():
    """Default preset stays circles-only — catches accidental always-on interlocking."""
    svg = str(render(load("examples/structures/benzene.xyz")))
    assert "<polygon" not in svg
