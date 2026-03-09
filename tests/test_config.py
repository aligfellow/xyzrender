"""Tests for config loading and merging via the public build_config() API."""

import json
import tempfile

import pytest

from xyzrender import build_config


def test_default_preset_has_gradient_and_fog():
    cfg = build_config("default")
    assert cfg.gradient is True
    assert cfg.fog is True


def test_flat_preset_has_no_gradient():
    cfg = build_config("flat")
    assert cfg.gradient is False


def test_paton_preset_has_no_bond_orders():
    cfg = build_config("paton")
    assert cfg.bond_orders is False


def test_nonexistent_preset_raises():
    with pytest.raises(FileNotFoundError):
        build_config("nonexistent_preset_xyz")


def test_custom_json_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"canvas_size": 500, "gradient": False}, f)
        path = f.name
    cfg = build_config(path)
    assert cfg.canvas_size == 500
    assert cfg.gradient is False


def test_kwarg_overrides_preset():
    """Explicit kwargs win over preset values."""
    cfg = build_config("default", gradient=False, canvas_size=400)
    assert cfg.gradient is False
    assert cfg.canvas_size == 400
    assert cfg.fog is True  # not overridden


def test_named_colors_resolved_to_hex():
    cfg = build_config("default", background="ivory")
    assert cfg.background == "#fffff0"


def test_paton_colors_resolved():
    """Paton preset uses named colors — verify they are resolved to hex."""
    cfg = build_config("paton")
    if cfg.color_overrides:
        for v in cfg.color_overrides.values():
            assert v.startswith("#"), f"Color not resolved to hex: {v}"


def test_build_config_returns_render_config():
    from xyzrender.types import RenderConfig

    cfg = build_config("default")
    assert isinstance(cfg, RenderConfig)
