"""Tests for the optional MCP / agent helpers."""

from pathlib import Path

import pytest

from xyzrender.agent import (
    inspect_structure,
    list_presets,
    measure_geometry,
    render_animation,
    render_molecule,
)
from xyzrender.config import list_presets as config_list_presets

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"
CAFFEINE = STRUCTURES / "caffeine.xyz"
ETHANOL = STRUCTURES / "ethanol.xyz"


def test_list_presets():
    result = list_presets()
    assert result["success"] is True
    assert result["action"] == "list_presets"
    assert "default" in result["presets"]
    assert "flat" in result["presets"]
    assert "named_colors" not in result["presets"]
    assert result["presets"] == config_list_presets()


def test_inspect_structure_caffeine():
    result = inspect_structure(str(CAFFEINE))
    assert result["success"] is True
    assert result["action"] == "inspect"
    assert result["n_atoms"] > 0
    assert "C" in result["formula"]
    assert result["has_cube"] is False


def test_inspect_structure_missing_file():
    result = inspect_structure("/no/such/molecule.xyz")
    assert result["success"] is False
    error = result["error"]
    assert error is not None
    assert "not found" in error.lower() or "No such" in error


def test_measure_geometry_distances_only():
    result = measure_geometry(str(ETHANOL), modes=["d"])
    assert result["success"] is True
    assert "distances" in result
    assert "angles" not in result
    assert isinstance(result["distances"][0][2], float)


def test_render_molecule_svg(tmp_path):
    out = tmp_path / "caffeine.svg"
    result = render_molecule(str(CAFFEINE), str(out), config="flat")
    assert result["success"] is True
    assert result["output"] == str(out.resolve())
    assert out.is_file()
    assert out.read_text().lstrip().startswith("<svg")


def test_render_molecule_rejects_bad_suffix(tmp_path):
    out = tmp_path / "caffeine.jpg"
    result = render_molecule(str(CAFFEINE), str(out))
    assert result["success"] is False
    error = result["error"]
    assert error is not None
    assert "Unsupported output suffix" in error


def test_render_animation_rot(tmp_path):
    out = tmp_path / "caffeine.gif"
    result = render_animation(
        str(CAFFEINE),
        str(out),
        mode="rot",
        rot_frames=4,
        gif_fps=2,
        config="flat",
    )
    assert result["success"] is True
    assert out.is_file()
    assert out.stat().st_size > 0


def test_build_server_registers_tools():
    pytest.importorskip("mcp", reason="mcp extra required")
    from xyzrender.mcp import build_server

    server = build_server()
    assert server is not None
