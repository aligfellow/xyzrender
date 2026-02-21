"""Tests for config loading and merging."""

import json
import tempfile

import pytest

from xyzrender.config import build_render_config, load_config


def test_load_default_preset():
    data = load_config("default")
    assert data["gradient"] is True
    assert data["fog"] is True
    assert "canvas_size" in data


def test_load_flat_preset():
    data = load_config("flat")
    assert data["gradient"] is False


def test_load_paton_preset():
    data = load_config("paton")
    assert "colors" in data
    assert data["bond_orders"] is False


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_preset_xyz")


def test_load_custom_json_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"canvas_size": 500, "gradient": False}, f)
        f.flush()
        data = load_config(f.name)
    assert data["canvas_size"] == 500
    assert data["gradient"] is False


def test_build_render_config_defaults():
    cfg = build_render_config({}, {})
    assert cfg.canvas_size == 800  # dataclass default
    assert cfg.gradient is False


def test_build_render_config_from_preset():
    data = load_config("default")
    cfg = build_render_config(data, {})
    assert cfg.gradient is True
    assert cfg.fog is True


def test_cli_overrides_win():
    data = {"gradient": True, "fog": True, "canvas_size": 800}
    overrides = {"gradient": False, "canvas_size": 400}
    cfg = build_render_config(data, overrides)
    assert cfg.gradient is False
    assert cfg.canvas_size == 400
    assert cfg.fog is True  # not overridden


def test_colors_mapped_to_color_overrides():
    data = {"colors": {"C": "#333333", "N": "#0000ff"}}
    cfg = build_render_config(data, {})
    assert cfg.color_overrides == {"C": "#333333", "N": "#0000ff"}


def test_colors_absent_gives_none():
    cfg = build_render_config({}, {})
    assert cfg.color_overrides is None
