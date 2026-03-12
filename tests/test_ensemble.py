"""Tests for ensemble (multi-frame XYZ) rendering."""

from pathlib import Path

import numpy as np
import pytest

from xyzrender import ensemble


def _write_multiframe_xyz(path: Path, frames: list[list[tuple[str, tuple[float, float, float]]]]) -> None:
    lines: list[str] = []
    for atoms in frames:
        lines.append(f"{len(atoms)}\n")
        lines.append("frame\n")
        for sym, (x, y, z) in atoms:
            lines.append(f"{sym} {x:.8f} {y:.8f} {z:.8f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_ensemble_renders_svg(tmp_path: Path):
    # Simple 3-atom molecule; second frame is rotated + translated.
    f0 = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.96, 0.0, 0.0)),
        ("H", (-0.24, 0.93, 0.0)),
    ]
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    trans = np.array([2.0, -1.5, 0.4], dtype=float)
    f1 = []
    for sym, xyz in f0:
        p = np.array(xyz, dtype=float) @ rot.T + trans
        f1.append((sym, (float(p[0]), float(p[1]), float(p[2]))))

    xyz_path = tmp_path / "water_trj.xyz"
    _write_multiframe_xyz(xyz_path, [f0, f1])

    svg = str(ensemble(xyz_path, orient=False))
    assert svg.startswith("<svg")
    assert "</svg>" in svg


def test_ensemble_mismatched_symbols_raises(tmp_path: Path):
    f0 = [("H", (0.0, 0.0, 0.0)), ("H", (0.7, 0.0, 0.0))]
    f1 = [("H", (0.0, 0.0, 0.0)), ("F", (0.7, 0.0, 0.0))]

    xyz_path = tmp_path / "bad.xyz"
    _write_multiframe_xyz(xyz_path, [f0, f1])

    with pytest.raises(ValueError, match="identical element ordering"):
        ensemble(xyz_path, orient=False)

