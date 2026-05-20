"""Configuration loading for xyzrender."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from xyzrender.colors import resolve_color
from xyzrender.types import (
    DensParams,
    ESPParams,
    MOParams,
    NCIParams,
    OverlayConfig,
    RenderConfig,
)


@dataclass(frozen=True)
class SurfaceOverrides:
    """Per-render surface override values from ``render()``/``render_gif()`` kwargs.

    Constructed once at the public-API boundary and passed through internal
    surface-pipeline calls as a single object — replaces a kwargs-dict hop.
    Non-``None`` fields supersede preset defaults on ``cfg`` inside
    :func:`build_surface_params`.

    ``nci_mode`` accepts ``'avg'``, ``'pixel'``, ``'uniform'``, or a colour
    name/hex (implying uniform mode).  ``flat_mo=True`` overrides
    ``cfg.flat_mo``; ``False`` defers to it.
    """

    iso: float | None = None
    mo_pos_color: str | None = None
    mo_neg_color: str | None = None
    mo_blur: float | None = None
    mo_upsample: int | None = None
    flat_mo: bool = False
    dens_color: str | None = None
    nci_mode: str | None = None
    nci_cutoff: float | None = None


logger = logging.getLogger(__name__)

_PRESET_DIR = Path(__file__).parent / "presets"
_DEFAULT_CONFIG: dict | None = None


def _load_default() -> dict:
    """Return the built-in default preset."""
    global _DEFAULT_CONFIG  # noqa: PLW0603
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = json.loads((_PRESET_DIR / "default.json").read_text())
    # Always return a deep copy so nested mappings (e.g. "colors") cannot
    # leak mutations between preset loads.
    return deepcopy(_DEFAULT_CONFIG)


def _merge_onto_default(overrides: dict) -> dict:
    """Merge *overrides* on top of default.json, deep-merging nested dicts."""
    base = _load_default()
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    return base


def load_config(name_or_path: str) -> dict:
    """Load config from a built-in preset name or a JSON file path.

    All presets (including named built-ins like ``"flat"`` or ``"paton"``) are
    merged on top of ``default.json`` so unspecified keys always inherit the
    standard defaults.  The ``"default"`` preset itself is returned as-is.
    """
    # Built-in default — returned as-is (it IS the base)
    if name_or_path == "default":
        return _load_default()

    # Other built-in presets — merge on top of default
    preset_file = _PRESET_DIR / f"{name_or_path}.json"
    if preset_file.exists():
        logger.debug("Loading preset: %s (on top of default)", preset_file)
        return _merge_onto_default(json.loads(preset_file.read_text()))

    # User-provided file path — same merge
    path = Path(name_or_path)
    if path.exists():
        logger.debug("Loading config file: %s (on top of default)", path)
        return _merge_onto_default(json.loads(path.read_text()))

    available = ", ".join(p.stem for p in sorted(_PRESET_DIR.glob("*.json")) if p.stem != "named_colors")
    msg = f"Config not found: {name_or_path!r} (built-in presets: {available})"
    raise FileNotFoundError(msg)


_PASSTHROUGH_COLORS = frozenset({"atom"})  # literal markers the renderer handles specially


def _resolve_color_fields(kw: dict, fields: tuple[str, ...]) -> None:
    """Resolve each CSS-name / hex entry in *kw* for the given *fields* to a hex string.

    Used to normalise preset JSON colour values (``"teal"`` → ``"#008080"``)
    both for top-level :class:`RenderConfig` fields and for the nested
    :class:`OverlayConfig` block.  ``None`` values and passthrough literals
    (e.g. ``"atom"`` = use per-element CPK) are left untouched.
    """
    for key in fields:
        v = kw.get(key)
        if v is not None and v not in _PASSTHROUGH_COLORS:
            kw[key] = resolve_color(v)


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

    # style_regions can't be deserialised from plain JSON dicts
    merged.pop("style_regions", None)

    # Extract preset-defined regions before RenderConfig (resolved at render time)
    region_specs = merged.pop("regions", None)
    if region_specs is not None:
        merged["region_specs"] = region_specs

    # "colors" key in JSON maps to color_overrides on RenderConfig
    colors = merged.pop("colors", None)
    if colors:
        merged["color_overrides"] = {sym: resolve_color(c) for sym, c in colors.items()}

    # Rename JSON keys → RenderConfig field names
    for old, new in (
        ("mo_iso", "mo_isovalue"),
        ("mo_blur", "mo_blur_sigma"),
        ("mo_upsample", "mo_upsample_factor"),
        ("dens_iso", "dens_isovalue"),
        ("nci_iso", "nci_isovalue"),
    ):
        if old in merged:
            merged[new] = merged.pop(old)

    # Resolve all color fields to hex
    _color_fields = (
        "background",
        "bond_color",
        "bond_outline_color",
        "ts_color",
        "nci_color",
        "atom_stroke_color",
        "label_color",
        "cmap_unlabeled",
        "cell_color",
        "mo_pos_color",
        "mo_neg_color",
        "dens_color",
        "mol_color",
    )
    _resolve_color_fields(merged, _color_fields)

    # Hydrate the preset JSON "overlay" block into an OverlayConfig instance.
    # Colour fields go through the same CSS-name → hex resolution as the top-level
    # fields above; a nested "config" sub-block is built through this same
    # preset machinery so any RenderConfig key is valid inside it.
    overlay_raw = merged.pop("overlay", None)
    if isinstance(overlay_raw, dict):
        overlay_kw: dict = dict(overlay_raw)
        _resolve_color_fields(overlay_kw, ("color", "atom_stroke_color", "bond_color", "bond_outline_color"))
        ov_inner = overlay_kw.get("config")
        if isinstance(ov_inner, dict):
            overlay_kw["config"] = build_render_config(ov_inner, {})
        merged["overlay"] = OverlayConfig(**overlay_kw)

    # nci_mode can be a mode name ("avg", "pixel", "uniform") or a color — resolve if color
    if "nci_mode" in merged and merged["nci_mode"] not in ("avg", "pixel", "uniform", None):
        merged["nci_mode"] = resolve_color(merged["nci_mode"])

    # axis_colors comes from JSON as a list of 3 strings; convert to tuple and resolve
    if "axis_colors" in merged:
        raw = merged["axis_colors"]
        merged["axis_colors"] = tuple(resolve_color(c) for c in raw)

    # hull_colors: list of color strings (one per subset) → resolve to hex
    if "hull_colors" in merged and merged["hull_colors"] is not None:
        merged["hull_colors"] = [resolve_color(c) for c in merged["hull_colors"]]

    # highlight_colors: list of color strings → resolve to hex
    # Backward compat: old presets may have "highlight_color" (single string)
    if "highlight_color" in merged:
        old = merged.pop("highlight_color")
        merged.setdefault("highlight_colors", [resolve_color(old)])
    if "highlight_colors" in merged and merged["highlight_colors"] is not None:
        merged["highlight_colors"] = [resolve_color(c) for c in merged["highlight_colors"]]

    # radius_scale: JSON dict {"H": 0.8, ...} → list of (selector, factor) tuples
    if "radius_scale" in merged and isinstance(merged["radius_scale"], dict):
        merged["radius_scale"] = [(sel, factor) for sel, factor in merged["radius_scale"].items()]

    # Validate surface_style
    if "surface_style" in merged and merged["surface_style"] not in ("solid", "mesh", "contour", "dot"):
        msg = f"Invalid surface_style: {merged['surface_style']!r} (must be 'solid', 'mesh', 'contour', or 'dot')"
        raise ValueError(msg)

    return RenderConfig(**merged)


def apply_hydrogen_flags(cfg: RenderConfig, *, hy: bool | list[int] | None, no_hy: bool = False) -> None:
    """Single source of truth for --hy / --no-hy logic. Called by CLI and Python API.

    hy=None → hide C-H (default), hy=True → show all, hy=[1,3] → show specific (1-indexed).
    """
    if no_hy:
        cfg.hide_h = True
        cfg.show_h_indices = []
    elif hy is None:
        cfg.hide_h = True
    elif hy is True:
        cfg.hide_h = False
    elif isinstance(hy, list):
        cfg.hide_h = True
        cfg.show_h_indices = [i - 1 for i in hy]


def build_config(
    config_name: str = "default",
    *,
    canvas_size=None,
    atom_scale=None,
    bond_width=None,
    atom_stroke_width=None,
    bond_color=None,
    bond_outline_color=None,
    bond_outline_width=None,
    ts_color=None,
    nci_color=None,
    background=None,
    transparent: bool = False,
    gradient=None,
    hue_shift_factor=None,
    light_shift_factor=None,
    saturation_shift_factor=None,
    fog=None,
    fog_strength=None,
    bo=None,
    label_font_size=None,
    vdw_opacity=None,
    vdw_scale=None,
    atom_gradient_strength=None,
    bond_gradient_strength=None,
    vdw_gradient_strength=None,
    hide_bonds: bool = False,
    unbond: list[str] | None = None,
    bond: list[str] | None = None,
    hy: bool | list[int] | None = None,
    no_hy: bool = False,
    orient: bool | None = None,
    opacity=None,
    ts_bonds: list[tuple[int, int]] | None = None,
    nci_bonds: list[tuple[int, int]] | None = None,
    vdw_indices: list[int] | None = None,
    show_indices: bool = False,
    idx_format: str = "sn",
    atom_cmap: dict[int, float] | None = None,
    cmap_range: tuple[float, float] | None = None,
    cmap_palette: str | None = None,
    cbar: bool = False,
    cmap_symm: bool = False,
) -> RenderConfig:
    """Build a :class:`~xyzrender.types.RenderConfig` from a preset and style kwargs.

    Parameters
    ----------
    config_name:
        Preset name (``"default"``, ``"flat"``, ``"paton"``, …) or path to a
        custom JSON config file.
    canvas_size, atom_scale, bond_width, …:
        Style overrides; any ``None`` value falls back to the preset default.
    orient:
        ``True`` / ``False`` to force / suppress PCA auto-orientation.
        ``None`` (default) enables auto-orientation.
    ts_bonds, nci_bonds:
        Manual TS / NCI bond overlays as 0-indexed atom pairs.
    vdw_indices:
        VdW sphere atom list (0-indexed). ``[]`` = all atoms, ``None`` = off.
    show_indices:
        Enable atom index labels.
    atom_cmap:
        Atom property colour map (0-indexed keys).

    Returns
    -------
    RenderConfig
        Ready to pass to :func:`~xyzrender.render` as ``config=``.

    Example
    -------
    ::

        cfg = build_config("flat", atom_scale=1.5, gradient=False)
        render(mol1, config=cfg)
        render(mol2, config=cfg)

    Bond/index/cmap params use **0-indexed** atom numbering (the internal
    convention). The Python API converts from user-facing 1-indexed values
    before calling this function; the CLI passes _parse_pairs() output directly.
    """
    config_data = load_config(config_name)
    overrides: dict = {}
    for key, val in [
        ("canvas_size", canvas_size),
        ("atom_scale", atom_scale),
        ("bond_width", bond_width),
        ("atom_stroke_width", atom_stroke_width),
        ("bond_color", bond_color),
        ("bond_outline_color", bond_outline_color),
        ("bond_outline_width", bond_outline_width),
        ("ts_color", ts_color),
        ("nci_color", nci_color),
        ("background", background),
        ("gradient", gradient),
        ("hue_shift_factor", hue_shift_factor),
        ("light_shift_factor", light_shift_factor),
        ("saturation_shift_factor", saturation_shift_factor),
        ("fog", fog),
        ("fog_strength", fog_strength),
        ("bond_orders", bo),
        ("label_font_size", label_font_size),
        ("vdw_opacity", vdw_opacity),
        ("vdw_scale", vdw_scale),
        ("atom_gradient_strength", atom_gradient_strength),
        ("bond_gradient_strength", bond_gradient_strength),
        ("vdw_gradient_strength", vdw_gradient_strength),
    ]:
        if val is not None:
            overrides[key] = val
    if transparent:
        overrides["transparent"] = True
    if hide_bonds:
        overrides["hide_bonds"] = True

    cfg = build_render_config(config_data, overrides)
    cfg.auto_orient = orient if orient is not None else True
    apply_hydrogen_flags(cfg, hy=hy, no_hy=no_hy)

    if opacity is not None:
        cfg.surface_opacity = opacity
    if ts_bonds is not None:
        cfg.ts_bonds = list(ts_bonds)
    if nci_bonds is not None:
        cfg.nci_bonds = list(nci_bonds)
    if unbond is not None:
        cfg.unbond = list(unbond)
    if bond is not None:
        cfg.bond = list(bond)
    if vdw_indices is not None:
        cfg.vdw_indices = vdw_indices
    if show_indices:
        cfg.show_indices = True
        cfg.idx_format = idx_format
    if atom_cmap is not None:
        cfg.atom_cmap = atom_cmap
    if cmap_range is not None:
        cfg.cmap_range = cmap_range
    cfg.cmap_palette = cmap_palette
    if cbar:
        cfg.cbar = True
    if cmap_symm:
        cfg.cmap_symm = True

    return cfg


def build_region_config(config_name: str = "default", **overrides) -> RenderConfig:
    """Build a :class:`RenderConfig` for use as a :class:`StyleRegion` config.

    Only per-atom/bond fields are meaningful; global fields (canvas, fog,
    surfaces) are ignored by the renderer for region configs.
    """
    config_data = load_config(config_name)
    return build_render_config(config_data, {k: v for k, v in overrides.items() if v is not None})


def build_surface_params(
    cfg: RenderConfig,
    overrides: SurfaceOverrides,
    *,
    has_mo: bool = False,
    has_dens: bool = False,
    has_esp: bool = False,
    has_nci: bool = False,
) -> tuple[MOParams | None, DensParams | None, ESPParams | None, NCIParams | None]:
    """Merge ``cfg`` defaults with per-render ``overrides`` into typed ``*Params``.

    Returns ``None`` for any surface that is not active (``has_*`` flag is
    ``False``), so callers can use simple ``if params:`` checks.
    """
    mo_params: MOParams | None = None
    dens_params: DensParams | None = None
    esp_params: ESPParams | None = None
    nci_params: NCIParams | None = None

    iso = overrides.iso

    if has_mo:
        mo_params = MOParams(
            isovalue=iso if iso is not None else cfg.mo_isovalue,
            pos_color=overrides.mo_pos_color or cfg.mo_pos_color,
            neg_color=overrides.mo_neg_color or cfg.mo_neg_color,
            blur_sigma=overrides.mo_blur if overrides.mo_blur is not None else cfg.mo_blur_sigma,
            upsample_factor=overrides.mo_upsample if overrides.mo_upsample is not None else cfg.mo_upsample_factor,
            flat=bool(overrides.flat_mo or cfg.flat_mo),
        )

    if has_dens:
        dens_params = DensParams(
            isovalue=iso if iso is not None else cfg.dens_isovalue,
            color=overrides.dens_color or cfg.dens_color,
        )

    if has_esp:
        esp_params = ESPParams(
            isovalue=iso if iso is not None else cfg.dens_isovalue,
        )

    if has_nci:
        nci_mode = overrides.nci_mode or cfg.nci_mode
        if nci_mode in ("avg", "pixel"):
            _nci_color_mode = nci_mode
            _nci_color = "forestgreen"
        elif nci_mode == "uniform":
            _nci_color_mode = "uniform"
            _nci_color = "forestgreen"
        else:
            _nci_color_mode = "uniform"
            _nci_color = resolve_color(nci_mode)
        nci_params = NCIParams(
            isovalue=iso if iso is not None else cfg.nci_isovalue,
            color=_nci_color,
            color_mode=_nci_color_mode,
            dens_cutoff=overrides.nci_cutoff,
        )

    return mo_params, dens_params, esp_params, nci_params
