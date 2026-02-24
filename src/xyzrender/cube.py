"""Gaussian cube file parsing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from xyzgraph import DATA

logger = logging.getLogger(__name__)

BOHR_TO_ANG = 0.52918


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CubeData:
    """Parsed Gaussian cube file."""

    atoms: list[tuple[str, tuple[float, float, float]]]  # symbol, (x,y,z) Angstrom
    origin: np.ndarray  # (3,) Bohr
    steps: np.ndarray  # (3, 3) row = step vector, Bohr
    grid_shape: tuple[int, int, int]  # (N1, N2, N3)
    grid_data: np.ndarray  # (N1, N2, N3) orbital values
    mo_index: int | None


# ---------------------------------------------------------------------------
# Cube file parser
# ---------------------------------------------------------------------------


def parse_cube(path: str | Path) -> CubeData:
    """Parse a Gaussian cube file.

    Handles both standard cube files (positive natoms) and MO cube files
    (negative natoms with extra MO-index line after atoms).
    """
    path = Path(path)
    with open(path) as f:
        lines = f.readlines()

    title = lines[0].strip()
    comment = lines[1].strip()
    logger.debug("Cube file: %s / %s", title, comment)

    # Line 3: natoms, origin
    parts = lines[2].split()
    natoms_raw = int(parts[0])
    is_mo = natoms_raw < 0
    natoms = abs(natoms_raw)
    origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

    # Lines 4-6: grid dimensions and step vectors
    grid_ns = []
    step_vecs = []
    for i in range(3):
        parts = lines[3 + i].split()
        grid_ns.append(int(parts[0]))
        step_vecs.append([float(parts[1]), float(parts[2]), float(parts[3])])
    grid_shape = tuple(grid_ns)
    steps = np.array(step_vecs)

    # Atom lines
    atoms = []
    for i in range(natoms):
        parts = lines[6 + i].split()
        z = int(parts[0])
        sym = DATA.n2s.get(z, "X")
        x, y, zc = float(parts[2]), float(parts[3]), float(parts[4])
        atoms.append((sym, (x * BOHR_TO_ANG, y * BOHR_TO_ANG, zc * BOHR_TO_ANG)))

    # MO index line (only for MO cube files)
    mo_index = None
    data_start = 6 + natoms
    if is_mo:
        mo_parts = lines[data_start].split()
        if len(mo_parts) >= 2:
            mo_index = int(mo_parts[1])
        data_start += 1

    # Volumetric data — parse all remaining floats in one numpy call
    expected = grid_shape[0] * grid_shape[1] * grid_shape[2]
    raw = " ".join(lines[data_start:])
    grid_data = np.fromstring(raw, dtype=np.float64, sep=" ")

    if grid_data.size != expected:
        raise ValueError(f"Cube data count mismatch: got {grid_data.size}, expected {expected}")

    grid_data = grid_data.reshape(grid_shape)

    logger.debug(
        "Parsed cube: %d atoms, grid %s, MO index %s, range [%.4g, %.4g]",
        natoms,
        grid_shape,
        mo_index,
        grid_data.min(),
        grid_data.max(),
    )
    return CubeData(
        atoms=atoms,
        origin=origin,
        steps=steps,
        grid_shape=grid_shape,
        grid_data=grid_data,
        mo_index=mo_index,
    )
