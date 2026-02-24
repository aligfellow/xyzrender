"""Tests for cube file parsing and MO math: marching squares, segment chaining, 3D lobe finding."""

from __future__ import annotations

import numpy as np

from xyzrender.cube import BOHR_TO_ANG, CubeData, parse_cube
from xyzrender.mo import chain_segments, cube_corners_ang, find_3d_lobes, marching_squares

# ---------------------------------------------------------------------------
# marching_squares
# ---------------------------------------------------------------------------


def _circle_grid(radius: float = 3.0, size: int = 20) -> np.ndarray:
    """Create a 2D grid with a filled circle (values > 0 inside)."""
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size / 2, size / 2
    return radius**2 - ((x - cx) ** 2 + (y - cy) ** 2)


def test_marching_squares_circle():
    grid = _circle_grid()
    segments = marching_squares(grid, threshold=0.0)
    assert len(segments) > 10  # circle should produce many segments


def test_marching_squares_empty():
    grid = np.zeros((10, 10))
    assert marching_squares(grid, threshold=1.0) == []


def test_marching_squares_all_above():
    grid = np.ones((5, 5))
    assert marching_squares(grid, threshold=0.5) == []


def test_marching_squares_tiny_grid():
    assert marching_squares(np.array([[1.0]]), threshold=0.5) == []


# ---------------------------------------------------------------------------
# chain_segments
# ---------------------------------------------------------------------------


def test_chain_segments_circle():
    grid = _circle_grid()
    segs = marching_squares(grid, threshold=0.0)
    loops = chain_segments(segs)
    assert len(loops) >= 1
    # A circle should produce one closed loop
    for loop in loops:
        assert len(loop) >= 4  # minimum meaningful loop


def test_chain_segments_empty():
    assert chain_segments([]) == []


# ---------------------------------------------------------------------------
# find_3d_lobes
# ---------------------------------------------------------------------------


def _two_blob_grid() -> np.ndarray:
    """Small 10x10x10 grid with two separated positive blobs."""
    g = np.zeros((10, 10, 10))
    g[1:4, 1:4, 1:4] = 1.0  # blob A at corner
    g[6:9, 6:9, 6:9] = 1.0  # blob B at opposite corner
    return g


def test_find_lobes_two_positive():
    lobes = find_3d_lobes(_two_blob_grid(), isovalue=0.5)
    pos_lobes = [lb for lb in lobes if lb.phase == "pos"]
    assert len(pos_lobes) == 2


def test_find_lobes_pos_and_neg():
    g = _two_blob_grid()
    g[6:9, 6:9, 6:9] = -1.0  # make second blob negative
    lobes = find_3d_lobes(g, isovalue=0.5)
    phases = {lb.phase for lb in lobes}
    assert phases == {"pos", "neg"}


def test_find_lobes_empty():
    g = np.zeros((5, 5, 5))
    assert find_3d_lobes(g, isovalue=0.5) == []


# ---------------------------------------------------------------------------
# cube_corners_ang
# ---------------------------------------------------------------------------


def test_cube_corners():
    cube = CubeData(
        atoms=[],
        origin=np.array([0.0, 0.0, 0.0]),
        steps=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        grid_shape=(3, 3, 3),
        grid_data=np.zeros((3, 3, 3)),
        mo_index=None,
    )
    corners = cube_corners_ang(cube)
    assert corners.shape == (8, 3)
    # Origin corner should be (0,0,0) in Angstrom
    assert np.allclose(corners[0], [0.0, 0.0, 0.0])
    # Far corner: (2,2,2) Bohr -> Angstrom
    assert np.allclose(corners[-1], np.array([2.0, 2.0, 2.0]) * BOHR_TO_ANG)


# ---------------------------------------------------------------------------
# parse_cube (synthetic file)
# ---------------------------------------------------------------------------


def test_parse_cube_synthetic(tmp_path):
    """Parse a minimal synthetic cube file with 1 atom and a 2x2x2 grid."""
    cube_file = tmp_path / "test.cube"
    cube_file.write_text(
        "title\n"
        "comment\n"
        " -1   0.0  0.0  0.0\n"
        "  2   1.0  0.0  0.0\n"
        "  2   0.0  1.0  0.0\n"
        "  2   0.0  0.0  1.0\n"
        "  6   6.0   0.0  0.0  0.0\n"
        "    1   42\n"
        " 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8\n"
    )
    cube = parse_cube(cube_file)
    assert len(cube.atoms) == 1
    assert cube.atoms[0][0] == "C"  # Z=6 -> carbon
    assert cube.grid_shape == (2, 2, 2)
    assert cube.grid_data.shape == (2, 2, 2)
    assert cube.mo_index == 42
    assert np.isclose(cube.grid_data[0, 0, 0], 0.1)
    assert np.isclose(cube.grid_data[1, 1, 1], 0.8)
