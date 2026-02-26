"""Configuration loading for xyzrender."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from xyzrender.types import RenderConfig, resolve_color

logger = logging.getLogger(__name__)

_PRESET_DIR = Path(__file__).parent / "presets"


def load_config(name_or_path: str) -> dict:
    """Load config from a built-in preset name or a JSON file path.

    Built-in presets: ``default``, ``flat``, ``custom``.
    """
    # Built-in preset?
    preset_file = _PRESET_DIR / f"{name_or_path}.json"
    if preset_file.exists():
        logger.debug("Loading preset: %s", preset_file)
        return json.loads(preset_file.read_text())

    # User-provided file path
    path = Path(name_or_path)
    if path.exists():
        logger.debug("Loading config file: %s", path)
        return json.loads(path.read_text())

    available = ", ".join(p.stem for p in sorted(_PRESET_DIR.glob("*.json")) if p.stem != "named_colors")
    msg = f"Config not found: {name_or_path!r} (built-in presets: {available})"
    raise FileNotFoundError(msg)


def build_render_config(config_data: dict, cli_overrides: dict) -> RenderConfig:
    """Merge config dict with CLI overrides into a RenderConfig.

    ``config_data`` is the base layer (from JSON).
    ``cli_overrides`` contains only explicitly-set CLI values (non-None).
    CLI values win over config file values.
    """
    merged = {**config_data}
    for k, v in cli_overrides.items():
        if v is not None:
            merged[k] = v

    # "colors" key in JSON maps to color_overrides on RenderConfig
    colors = merged.pop("colors", None)
    if colors:
        merged["color_overrides"] = {sym: resolve_color(c) for sym, c in colors.items()}

    # MO/density keys are stored in config but not passed to RenderConfig
    # directly (they're used at build time, not render time). Strip them to
    # avoid TypeError from unexpected kwargs.
    merged.pop("mo_pos_color", None)
    merged.pop("mo_neg_color", None)
    merged.pop("dens_iso", None)
    merged.pop("dens_color", None)

    # Resolve any named colors to hex for fields that downstream code parses as hex
    for key in ("background", "bond_color", "atom_stroke_color"):
        if key in merged:
            merged[key] = resolve_color(merged[key])

    return RenderConfig(**merged)
