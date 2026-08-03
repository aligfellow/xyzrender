"""Protein cartoon geometry and SVG generation.

Produces depth-sorted ``(z_depth, svg_lines)`` tuples consumed by the main
renderer via the same drain pattern as NCI patches.

Geometry model
--------------
The backbone of each chain segment is turned into a *swept surface*:

1. A per-residue orientation frame is derived from chemistry — the Carson-Bugg
   carbonyl direction ``O - CA``, for every secondary structure — and then
   flip-corrected against its predecessor so the frame never rolls arbitrarily
   along a run.  In a strand that vector lies in the sheet plane; in a helix it
   points along the helix axis, so the tape's width runs axially and successive
   turns nearly close up, which is what makes a helix read as a coiled solid.
2. The centreline is splined **once per backbone segment** (centripetal
   Catmull-Rom) and the frame is interpolated onto the same samples.  Splining
   per secondary-structure run instead would leave a tangent discontinuity at
   every H/E/C boundary.
3. A closed cross-section is swept along that frame.  The section is a
   superellipse whose exponent morphs from a rounded rectangle (helix, sheet)
   to a circle (coil), and whose half-width/half-thickness are interpolated
   per sample, so secondary-structure transitions are a continuous change of
   shape rather than a butt joint between two differently-sized strips.
4. Each quad of the resulting surface is emitted as its own depth-sorted item
   and shaded by its own normal.  Per-quad depth is what lets a helix occlude
   its own turns; per-quad shading is what makes it read as a solid object.

Secondary structure labels: ``"H"`` helix, ``"E"`` strand, ``"C"`` coil.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from xyzrender.colors import _FOG_NEAR, Color, blend_fog, palette_color

if TYPE_CHECKING:
    from collections.abc import Iterable

    from xyzrender.types import ProteinData, ProteinSemantics, RenderConfig, ResidueData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-8

_WHITE = Color(255, 255, 255)
_BLACK = Color(0, 0, 0)

# Spline samples per residue along the backbone.  Uniform across a segment:
# per-run tessellation would reintroduce seams at SS boundaries.
_SPLINE_STEPS = 7
# Vertices around the swept cross-section.  8 gives a flat tape a bevelled
# edge without exploding the polygon count.
_PROFILE_N = 8

# Cross-section superellipse exponents: 1.0 is a circle, lower is boxier.
_SECTION_P_FLAT = 0.38  # helix / sheet
_SECTION_P_ROUND = 1.0  # coil

# Cross-section extents as multiples of the ribbon half-width.  Strands are a
# touch narrower than helices so the arrowhead flange reads as an overhang, and
# both are thin tapes.  Coils use the loop half-width and stay circular.
# `_section_dims` is the single source of truth -- both the swept surface and
# the sidechain anchor solve must agree, or sidechains float off the tape.
_SEC_HELIX_HW = 1.0
_SEC_SHEET_HW = 0.92
_SEC_HELIX_HT = 0.13
_SEC_SHEET_HT = 0.12

# Backbone discontinuity thresholds (metadata + geometry)
_MAX_SEQ_STEP = 1
_MAX_CA_GAP = 5.0  # Å

# Arrowhead: flange half-width multiple, and the fraction of the final
# residue interval over which the strand tapers to its point.
_ARROW_FLANGE = 1.9
_ARROW_TIP_FRAC = 0.06

# Directional light in eye space (x right, y up, z toward viewer), matching
# the upper-left focal point of the atom sphere gradients.
_LIGHT_DIR = np.array([-0.40, 0.55, 1.0])
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)
# Lambert term is remapped into [_SHADE_FLOOR, 1] before tinting so unlit
# faces stay readable rather than going black.  Kept low: the floor sets how
# dark a shadowed face can get, and that contrast is what separates one strand
# of a sheet from the one behind it.
_SHADE_FLOOR = 0.15

# Ribbon fog is attenuated relative to atom fog.  A protein spans a far larger
# depth range than a small molecule, so applying the full atom-strength fog
# makes rear chains vanish entirely.  mo.py attenuates its own fog for the same
# reason (_MO_FOG_FACTOR).
_RIBBON_FOG_FACTOR = 0.45

_DEFAULT_RIBBON_STYLE = "gloss"


@dataclass(frozen=True)
class RibbonStyleProfile:
    """Per-style cartoon rendering controls.

    Replaces the previous 17-float profile.  ``width_scale`` and
    ``thickness_scale`` shape the swept section; ``shade_gain`` sets how hard
    the Lambert term tints the base colour; ``outline_px`` is the silhouette
    stroke width in pixels (0 disables outlines).
    """

    name: str
    width_scale: float
    thickness_scale: float
    shade_gain: float
    outline_px: float
    outline_dark: float


_RIBBON_STYLE_PROFILES: dict[str, RibbonStyleProfile] = {
    # Shaded, glossy — the default.  Strong Lambert tinting, hairline outline.
    "gloss": RibbonStyleProfile(
        name="gloss",
        width_scale=1.0,
        thickness_scale=1.0,
        shade_gain=0.95,
        outline_px=0.35,
        outline_dark=0.25,
    ),
    # Flatter, textbook look: weak shading, visible dark contour.
    "illustration": RibbonStyleProfile(
        name="illustration",
        width_scale=0.95,
        thickness_scale=0.85,
        shade_gain=0.45,
        outline_px=0.5,
        outline_dark=0.30,
    ),
}


# Accepted alternative spellings, resolved to a real profile.
_RIBBON_STYLE_ALIASES: dict[str, str] = {"cartoon": "gloss"}


def default_ribbon_style() -> str:
    """Canonical default style used for --protein and protein=True."""
    return _DEFAULT_RIBBON_STYLE


def normalize_ribbon_style(style: str) -> str:
    """Normalize and validate ribbon style names."""
    key = style.strip().lower()
    key = _RIBBON_STYLE_ALIASES.get(key, key)
    if key not in _RIBBON_STYLE_PROFILES:
        raise ValueError(f"unknown ribbon style {style!r}")
    return key


def ribbon_style_profile(style: str) -> RibbonStyleProfile:
    """Return the resolved style profile for a ribbon style name."""
    return _RIBBON_STYLE_PROFILES[normalize_ribbon_style(style)]


def ribbon_style_names(*, include_aliases: bool = True) -> tuple[str, ...]:
    """Return valid ribbon style names for API/CLI validation."""
    names = set(_RIBBON_STYLE_PROFILES)
    if include_aliases:
        names |= set(_RIBBON_STYLE_ALIASES)
    return tuple(sorted(names))


def ribbon_style_uses_gradients(style: str) -> bool:
    """Whether a ribbon style requires SVG gradient defs.

    Always ``False``: shading is now computed per quad from its own normal and
    baked into a solid fill, so no ``<linearGradient>`` defs are needed.  Kept
    so the renderer's def-emitting branch stays a no-op rather than a crash.
    """
    normalize_ribbon_style(style)
    return False


def ribbon_gradient_defs(chain_colors: dict[str, str], style: str = "gloss") -> list[str]:
    """Return an empty list — ribbon gradients are deprecated.

    The previous implementation emitted one ``<linearGradient>`` per chain
    colour with ``gradientUnits="objectBoundingBox"``, which ramped across each
    polygon's *screen bounding box* rather than across the ribbon's local
    width.  Per-quad Lambert shading replaces it.
    """
    normalize_ribbon_style(style)
    return []


# ---------------------------------------------------------------------------
# Chain colour assignment
# ---------------------------------------------------------------------------


def _muted_pastel_color(hex_color: str) -> str:
    """Desaturate/lighten chain colours for textbook illustration styling."""
    c = Color.from_str(hex_color)
    h, lightness, s = c.to_hls()
    lightness = min(1.0, lightness + 0.18 * (1.0 - lightness))
    s = max(0.20, s * 0.58)
    return Color.from_hls(h, lightness, s).hex


def assign_chain_colors(cfg: "RenderConfig", chain_ids: list[str], style: str | None = None) -> dict[str, str]:
    """Return ``chain_id → hex colour`` mapping.

    Uses ``cfg.chain_colors`` overrides where given, filling the rest from
    ``cfg.protein_palette``.  Past the end of the palette the hue repeats but
    the lightness is shifted, so chain 1 and chain 9 remain distinguishable
    instead of being identical.
    """
    overrides = cfg.chain_colors or {}
    palette = cfg.protein_palette
    style_name = normalize_ribbon_style(style or cfg.protein_style)
    pastel_auto = style_name == "illustration"
    result: dict[str, str] = {}
    auto_idx = 0
    for cid in chain_ids:
        if cid in overrides:
            result[cid] = overrides[cid]
            continue
        raw = palette[auto_idx % len(palette)]
        lap = auto_idx // len(palette)
        if lap:
            c = Color.from_str(raw)
            h, lightness, s = c.to_hls()
            # Alternate darker / lighter on successive laps.
            shift = 0.16 * lap
            lightness = max(0.18, lightness - shift) if lap % 2 else min(0.9, lightness + shift)
            raw = Color.from_hls(h, lightness, s).hex
        result[cid] = _muted_pastel_color(raw) if pastel_auto else raw
        auto_idx += 1
    return result


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > _EPS else v


def _unit_rows(a: np.ndarray) -> np.ndarray:
    """Row-wise normalise an ``(n, 3)`` array, leaving degenerate rows alone."""
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return np.divide(a, n, out=np.zeros_like(a), where=n > _EPS)


def _fallback_perp(tangent: np.ndarray) -> np.ndarray:
    """Return a stable arbitrary unit vector perpendicular to *tangent*."""
    arb = np.array([1.0, 0.0, 0.0]) if abs(tangent[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = np.cross(tangent, arb)
    if np.linalg.norm(perp) <= _EPS:
        perp = np.cross(tangent, np.array([0.0, 0.0, 1.0]))
    return _unit(perp)


def _central_tangents(p: np.ndarray) -> np.ndarray:
    """Return unit tangents by central difference, with one-sided ends."""
    n = len(p)
    if n < 2:
        return np.tile(np.array([1.0, 0.0, 0.0]), (max(n, 1), 1))
    t = np.empty_like(p)
    t[1:-1] = p[2:] - p[:-2]
    t[0] = p[1] - p[0]
    t[-1] = p[-1] - p[-2]
    t = _unit_rows(t)
    # Repair any zero rows (coincident points) by carrying the previous tangent.
    for i in range(n):
        if np.linalg.norm(t[i]) <= _EPS:
            t[i] = t[i - 1] if i else np.array([1.0, 0.0, 0.0])
    return t


# ---------------------------------------------------------------------------
# Splines
# ---------------------------------------------------------------------------


def _catmull_rom_3d(points: np.ndarray, steps: int = _SPLINE_STEPS, *, alpha: float = 0.5) -> np.ndarray:
    """Centripetal Catmull-Rom spline through *points*.

    ``alpha=0.5`` (centripetal) is used rather than the uniform ``alpha=0``
    parameterisation: uniform Catmull-Rom overshoots and can cusp where three
    control points are nearly collinear-but-tight, which happens constantly in
    a tight backbone turn.

    Returns an array of shape ``(n_segments * steps + 1, 3)``.
    """
    n = len(points)
    if n < 2:
        return points.copy()
    if n == 2:
        ts = np.linspace(0.0, 1.0, steps + 1)[:, None]
        return points[0] * (1 - ts) + points[1] * ts

    # Reflected phantom endpoints so the curve stays tangent to the end chords.
    ext = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])

    # Knot sequence for the centripetal parameterisation.
    d = np.linalg.norm(np.diff(ext, axis=0), axis=1)
    d = np.maximum(d, _EPS) ** alpha
    t_knots = np.concatenate([[0.0], np.cumsum(d)])

    out: list[np.ndarray] = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        t0, t1, t2, t3 = t_knots[i - 1], t_knots[i], t_knots[i + 1], t_knots[i + 2]
        if t2 - t1 <= _EPS:
            continue
        t = np.linspace(t1, t2, steps, endpoint=False)[:, None]
        # Barry-Goldman pyramidal formulation of the non-uniform spline.
        a1 = (t1 - t) / max(t1 - t0, _EPS) * p0 + (t - t0) / max(t1 - t0, _EPS) * p1
        a2 = (t2 - t) / max(t2 - t1, _EPS) * p1 + (t - t1) / max(t2 - t1, _EPS) * p2
        a3 = (t3 - t) / max(t3 - t2, _EPS) * p2 + (t - t2) / max(t3 - t2, _EPS) * p3
        b1 = (t2 - t) / max(t2 - t0, _EPS) * a1 + (t - t0) / max(t2 - t0, _EPS) * a2
        b2 = (t3 - t) / max(t3 - t1, _EPS) * a2 + (t - t1) / max(t3 - t1, _EPS) * a3
        out.append((t2 - t) / max(t2 - t1, _EPS) * b1 + (t - t1) / max(t2 - t1, _EPS) * b2)
    out.append(points[[-1]])
    return np.vstack(out)


def _adaptive_spline_steps(ca_pos: np.ndarray, scale: float) -> int:
    """Choose spline density from residue count and projected span.

    Uniform across a whole backbone segment.  Density is reduced for very
    large structures purely to keep the emitted polygon count sane.
    """
    n = len(ca_pos)
    if n <= 1:
        return _SPLINE_STEPS
    if n > 2000:
        steps = 3
    elif n > 900:
        steps = 4
    else:
        steps = _SPLINE_STEPS
    span_px = float(np.max(np.ptp(ca_pos[:, :2], axis=0))) * max(scale, _EPS)
    if span_px < 500.0:
        steps = max(3, steps - 1)
    return steps


# ---------------------------------------------------------------------------
# Orientation frame
# ---------------------------------------------------------------------------


def _preferred_normals(
    ca_pos: np.ndarray,
    o_pos: list[np.ndarray | None],
    tangents: np.ndarray,
    ss: list[str],
) -> np.ndarray:
    """Chemically-meaningful ribbon *width* direction at every residue.

    This is the Carson-Bugg construction: the carbonyl direction ``O - CA``,
    projected perpendicular to the chain tangent, for **every** secondary
    structure.  In a beta strand that vector lies in the sheet plane; in an
    alpha helix the carbonyl points nearly along the helix axis, so the tape's
    width runs axially and successive turns nearly close up — which is what
    makes a cartoon helix read as a coiled solid rather than a spiral fin.

    The previous implementation special-cased helices to use the curvature
    vector, which points *at* the helix axis.  That put the tape's width along
    the radius, producing a fin spiralling around the axis.

    Only when no carbonyl is available does this fall back to curvature (or to
    the binormal of neighbouring tangents).  Unresolvable rows are left zero
    and filled in by :func:`_frame_normals`.
    """
    n = len(ca_pos)
    out = np.zeros((n, 3), dtype=float)
    for i in range(n):
        if o_pos[i] is not None:
            raw = o_pos[i] - ca_pos[i]
        elif 0 < i < n - 1:
            raw = ca_pos[i - 1] + ca_pos[i + 1] - 2.0 * ca_pos[i]
        else:
            raw = np.cross(tangents[max(0, i - 1)], tangents[min(n - 1, i + 1)])
        raw = raw - np.dot(raw, tangents[i]) * tangents[i]
        if np.linalg.norm(raw) > _EPS:
            out[i] = raw
    return _unit_rows(out)


def _frame_normals(
    ca_pos: np.ndarray,
    o_pos: list[np.ndarray | None],
    tangents: np.ndarray,
    ss: list[str],
) -> np.ndarray:
    """Per-residue ribbon normals, flip-corrected for continuity.

    The old implementation parallel-transported a single seed normal along the
    whole segment and discarded the per-residue chemistry.  Transport does not
    track the helix axis, so helix faces drifted and strand faces did not lie
    in the sheet plane.  Here every residue keeps its own chemical normal and
    only the *sign* is propagated — successive carbonyls alternate by ~180° in
    a strand, and that alternation is exactly what must be cancelled.
    """
    n = len(ca_pos)
    if n == 0:
        return np.zeros((0, 3), dtype=float)

    pref = _preferred_normals(ca_pos, o_pos, tangents, ss)

    normals = np.zeros_like(pref)
    prev: np.ndarray | None = None
    for i in range(n):
        v = pref[i]
        if np.linalg.norm(v) <= _EPS:
            # No chemistry available: carry the previous normal, re-projected.
            v = prev if prev is not None else _fallback_perp(tangents[i])
            v = v - np.dot(v, tangents[i]) * tangents[i]
            v = _unit(v) if np.linalg.norm(v) > _EPS else _fallback_perp(tangents[i])
        if prev is not None and np.dot(v, prev) < 0.0:
            v = -v
        normals[i] = v
        prev = v

    # Light 3-tap smoothing.  Deliberately weaker than the previous 5-tap
    # [1,2,3,2,1]: that damped out most of the chemical signal it was applied
    # to.  Re-orthogonalise afterwards so the frame stays perpendicular.
    if n >= 3:
        sm = normals.copy()
        sm[1:-1] = normals[:-2] * 0.25 + normals[1:-1] * 0.5 + normals[2:] * 0.25
        sm -= np.einsum("ij,ij->i", sm, tangents)[:, None] * tangents
        sm = _unit_rows(sm)
        bad = np.linalg.norm(sm, axis=1) <= _EPS
        sm[bad] = normals[bad]
        normals = sm
    return normals


def _smooth_helix_trace(ca_pos: np.ndarray, ss: list[str], strength: float) -> np.ndarray:
    """Pull helix (and, weakly, strand) CA positions toward the local axis.

    A helical CA trace spirals ~2.3 Å off the helix axis with a ~5.4 Å period.
    Splining it faithfully reproduces that spiral, which projects as a
    triangular sawtooth band instead of a smooth coil — the single most
    damaging visual defect in the previous implementation.  A windowed mean
    over +/-2 residues collapses one full turn onto the axis.

    *strength* is the blend toward the smoothed trace (0 = raw CA).  The blend
    is ramped down over the two residues at each end of a run so that the
    smoothed interior rejoins the raw coil continuously.

    **Defaults to off.**  It was measured against real renders of 8UWL and
    turned out to be unnecessary once the frame used the Carson-Bugg carbonyl
    direction (see :func:`_preferred_normals`) and the tape was widened to
    ``ribbon_width`` ~4.5 Å: the sawtooth those two changes removed was the
    only thing this was needed for, and at full strength it flattens the coil
    into a near-straight strap.  Retained as a knob for coarse or CA-only
    models, where no carbonyl is available and the trace can be genuinely
    noisy.
    """
    n = len(ca_pos)
    if n < 5 or strength <= 0.0:
        return ca_pos

    # Windowed mean over +/-2 with edge clamping.
    idx = np.arange(n)
    win = np.stack([ca_pos[np.clip(idx + k, 0, n - 1)] for k in (-2, -1, 0, 1, 2)])
    smoothed = win.mean(axis=0)

    # Per-residue target strength: full inside H, reduced inside E, none in C.
    target = np.array([strength if s == "H" else (0.35 * strength if s == "E" else 0.0) for s in ss])
    # Ramp across run boundaries: a windowed mean of the weights fades the
    # blend in and out over ~2 residues, so a smoothed helix interior rejoins
    # the raw coil continuously instead of stepping.
    idxc = np.arange(n)
    w = np.stack([target[np.clip(idxc + k, 0, n - 1)] for k in (-2, -1, 0, 1, 2)]).mean(axis=0)
    w = np.clip(w, 0.0, 1.0)[:, None]
    return ca_pos * (1.0 - w) + smoothed * w


# ---------------------------------------------------------------------------
# Cross-section sweep
# ---------------------------------------------------------------------------


def _fog_at(z: float, enabled: bool, strength: float, z_front: float, z_range: float) -> float:
    """Fog factor for a face at depth *z*, matching the renderer's atom formula.

    Same ``_FOG_NEAR`` dead zone and normalisation as renderer.py, attenuated
    by :data:`_RIBBON_FOG_FACTOR`.
    """
    if not enabled:
        return 0.0
    return float(strength * _RIBBON_FOG_FACTOR * np.clip((z_front - z - _FOG_NEAR) / z_range, 0.0, 1.0))


def _section_offsets(p_exp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Superellipse cross-section coordinates for a per-sample exponent array.

    Returns ``(u, v)`` each of shape ``(m, _PROFILE_N)`` in unit half-width /
    half-thickness coordinates.  ``p_exp == 1`` is a circle; smaller values
    square it off.  Morphing the exponent is what turns a coil tube smoothly
    into a flat helix tape across a transition.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, _PROFILE_N, endpoint=False)[None, :]
    c, s = np.cos(theta), np.sin(theta)
    e = p_exp[:, None]
    u = np.sign(c) * np.abs(c) ** e
    v = np.sign(s) * np.abs(s) ** e
    return u, v


def _residue_section_params(
    ss: list[str],
    dims: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-residue (half_width, half_thickness, section exponent)."""
    n = len(ss)
    hw = np.empty(n)
    ht = np.empty(n)
    pe = np.empty(n)
    for i, s in enumerate(ss):
        key = s if s in ("H", "E") else "C"
        hw[i], ht[i] = dims[key]
        pe[i] = _SECTION_P_FLAT if key in ("H", "E") else _SECTION_P_ROUND
    return hw, ht, pe


def _smooth3(a: np.ndarray) -> np.ndarray:
    """3-tap [1,2,1] smoothing, used to morph section size across SS joins."""
    if len(a) < 3:
        return a
    out = a.copy()
    out[1:-1] = a[:-2] * 0.25 + a[1:-1] * 0.5 + a[2:] * 0.25
    return out


def _runs_of_type(ss: list[str], target: str) -> list[tuple[int, int]]:
    """Return inclusive [start, end] index runs for *target* SS label."""
    out: list[tuple[int, int]] = []
    i = 0
    n = len(ss)
    while i < n:
        if ss[i] != target:
            i += 1
            continue
        j = i
        while j + 1 < n and ss[j + 1] == target:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def _sample_widths(
    ss: list[str],
    res_t: np.ndarray,
    dims: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample half-width / half-thickness / section exponent.

    Strand arrowheads are built here rather than as a separate overlapping
    triangle: the half-width jumps to the flange at the penultimate residue and
    tapers to a point at the last one, so the arrow is part of the same swept
    surface and shares its depth ordering, normals and stroke.
    """
    hw_r, ht_r, pe_r = _residue_section_params(ss, dims)
    hw_r, ht_r, pe_r = _smooth3(hw_r), _smooth3(ht_r), _smooth3(pe_r)

    n = len(ss)
    idx = np.arange(n, dtype=float)
    hw = np.interp(res_t, idx, hw_r)
    ht = np.interp(res_t, idx, ht_r)
    pe = np.interp(res_t, idx, pe_r)

    # Arrowheads, applied after interpolation so the flange edge stays crisp.
    for s, e in _runs_of_type(ss, "E"):
        if e <= s:
            continue  # single-residue strand: no room for an arrow, leave as-is
        base = float(e - 1)
        sel = (res_t >= base) & (res_t <= e)
        if not sel.any():
            continue
        frac = np.clip((res_t[sel] - base) / max(float(e) - base, _EPS), 0.0, 1.0)
        hw[sel] = dims["E"][0] * (_ARROW_FLANGE + (_ARROW_TIP_FRAC - _ARROW_FLANGE) * frac)
    return hw, ht, pe


# ---------------------------------------------------------------------------
# Shading
# ---------------------------------------------------------------------------


class _Shader:
    """Cached per-colour Lambert shading with optional fog."""

    def __init__(
        self,
        base_hex: str,
        cfg: "RenderConfig",
        gain: float,
        *,
        fog_enabled: bool = False,
        fog_rgb: Color | None = None,
        levels: int = 24,
    ) -> None:
        # Explicit blends rather than get_gradient_colors(): that helper is
        # tuned for the subtle falloff of a sphere radial gradient and gives a
        # washed-out, near-flat ramp when used as a Lambert term across a
        # large surface.
        base = Color.from_str(base_hex)
        lighter = base.blend(_WHITE, 0.40 * gain)
        darker = base.blend(_BLACK, 0.45 * gain)
        self._gain = gain
        self._levels = levels
        self._fog_enabled = fog_enabled
        self._fog_rgb = fog_rgb
        # Pre-bake the shading ramp; quads then only do an index lookup.
        self._ramp: list[str] = []
        for k in range(levels):
            t = k / (levels - 1)
            if t >= 0.5:
                c = base.blend(lighter, (t - 0.5) * 2.0)
            else:
                c = base.blend(darker, (0.5 - t) * 2.0)
            self._ramp.append(c.hex)
        self._fog_cache: dict[tuple[int, int], str] = {}

    def shade(self, lambert: float, fog: float = 0.0) -> str:
        """Colour for a face with the given Lambert term and fog factor."""
        k = round(float(np.clip(lambert, 0.0, 1.0)) * (self._levels - 1))
        if not self._fog_enabled or fog <= 0.01 or self._fog_rgb is None:
            return self._ramp[k]
        fk = round(float(np.clip(fog, 0.0, 1.0)) * 20)
        key = (k, fk)
        hit = self._fog_cache.get(key)
        if hit is None:
            hit = blend_fog(self._ramp[k], self._fog_rgb, fk / 20.0)
            self._fog_cache[key] = hit
        return hit


# ---------------------------------------------------------------------------
# Segment sweep → SVG items
# ---------------------------------------------------------------------------


def _section_dims(
    half_ribbon: float,
    loop_half: float,
    style: RibbonStyleProfile,
) -> dict[str, tuple[float, float]]:
    """Build the SS -> (half-width, half-thickness) table for the swept cross-section."""
    ws, ts_ = style.width_scale, style.thickness_scale
    return {
        "H": (half_ribbon * _SEC_HELIX_HW * ws, half_ribbon * _SEC_HELIX_HT * ts_),
        "E": (half_ribbon * _SEC_SHEET_HW * ws, half_ribbon * _SEC_SHEET_HT * ts_),
        "C": (loop_half * ws, loop_half * ts_),
    }


def _validate_ribbon_dimensions(cfg: "RenderConfig") -> None:
    for name in ("ribbon_width", "loop_width"):
        value = float(getattr(cfg, name))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {value!r}")


def ribbon_fit_margin(cfg: "RenderConfig") -> float:
    """Maximum world-space radius of a ribbon cross-section for canvas fitting."""
    _validate_ribbon_dimensions(cfg)
    style = ribbon_style_profile(cfg.protein_style)
    dims = _section_dims(cfg.ribbon_width / 2.0, cfg.loop_width / 2.0, style)
    radii = [float(np.hypot(hw, ht)) for hw, ht in dims.values()]
    sheet_hw, sheet_ht = dims["E"]
    radii.append(float(np.hypot(sheet_hw * _ARROW_FLANGE, sheet_ht)))
    return max(radii)


def _silhouette_edges(
    px: np.ndarray,
    py: np.ndarray,
    front: np.ndarray,
    i: int,
    j: int,
    m: int,
    outline: str,
    sw: float,
) -> list[str]:
    """Stroke the edges of quad ``(i, j)`` that lie on the surface silhouette.

    An edge is on the silhouette when the quad sharing it is back-facing (and so
    culled) or absent at a segment end.  Stroking the quad itself instead draws
    every interior mesh edge, which is what rendered `illustration` as a
    wireframe.
    """
    n = front.shape[1]
    k = (j + 1) % n
    # (vertex, vertex, quad on the far side) for the two ring edges and the two
    # rails.  The ring is closed, so only the rows can run out.
    edges = (
        ((i, j), (i, k), None if i == 0 else (i - 1, j)),
        ((i + 1, j), (i + 1, k), None if i == m - 2 else (i + 1, j)),
        ((i, j), (i + 1, j), (i, (j - 1) % n)),
        ((i, k), (i + 1, k), (i, k)),
    )
    out: list[str] = []
    for a, b, nb in edges:
        if nb is not None and front[nb]:
            continue
        out.append(
            f'  <line x1="{px[a]:.1f}" y1="{py[a]:.1f}" x2="{px[b]:.1f}" y2="{py[b]:.1f}" '
            f'stroke="{outline}" stroke-width="{sw:.2f}" stroke-linecap="round"/>'
        )
    return out


def _segment_items(
    ca_pos: np.ndarray,
    o_pos: list[np.ndarray | None],
    ss: list[str],
    shaders: list[_Shader],
    style: RibbonStyleProfile,
    outline: str | None,
    half_ribbon: float,
    loop_half: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: float,
    canvas_h: float,
    *,
    helix_smoothing: float,
    fog_enabled: bool,
    fog_strength: float,
    z_front: float,
    z_range: float,
) -> list[tuple[float, list[str]]]:
    """Sweep one continuous backbone segment and emit per-quad items."""
    n = len(ca_pos)
    if n < 2:
        return []

    # --- 1. trace conditioning + frame -------------------------------------
    # The centreline is smoothed toward the helix axis, but the orientation
    # frame is derived from the RAW backbone: the carbonyl direction is a
    # property of the real geometry, and a flattened helix has no meaningful
    # curvature left to read an orientation from.
    traced = _smooth_helix_trace(ca_pos, ss, helix_smoothing)
    res_tangents = _central_tangents(traced)
    res_normals = _frame_normals(ca_pos, o_pos, res_tangents, ss)

    # --- 2. one spline for the whole segment, frame carried along ----------
    steps = _adaptive_spline_steps(traced, scale)
    centers = _catmull_rom_3d(traced, steps)
    m = len(centers)
    res_t = np.linspace(0.0, n - 1, m)

    normals = np.empty((m, 3))
    for k in range(3):
        normals[:, k] = np.interp(res_t, np.arange(n, dtype=float), res_normals[:, k])
    tangents = _central_tangents(centers)
    normals -= np.einsum("ij,ij->i", normals, tangents)[:, None] * tangents
    normals = _unit_rows(normals)
    for i in range(m):  # repair any degenerate rows
        if np.linalg.norm(normals[i]) <= _EPS:
            normals[i] = normals[i - 1] if i else _fallback_perp(tangents[i])
    binormals = _unit_rows(np.cross(tangents, normals))

    # --- 3. cross-section size per sample ----------------------------------
    dims = _section_dims(half_ribbon, loop_half, style)
    hw, ht, pe = _sample_widths(ss, res_t, dims)
    u, v = _section_offsets(pe)

    # --- 4. vertices: (m, _PROFILE_N, 3) -----------------------------------
    verts = (
        centers[:, None, :]
        + normals[:, None, :] * (u * hw[:, None])[:, :, None]
        + binormals[:, None, :] * (v * ht[:, None])[:, :, None]
    )

    # --- 5. quads, one depth-sorted item each ------------------------------
    items: list[tuple[float, list[str]]] = []
    px = canvas_w / 2 + scale * (verts[:, :, 0] - cx)
    py = canvas_h / 2 - scale * (verts[:, :, 1] - cy)
    pz = verts[:, :, 2]
    sw = style.outline_px
    draw_outline = bool(outline) and sw > 0
    # A cap is one closed polygon whose boundary is the silhouette, so stroking
    # it directly is correct.  Quads are not — see _silhouette_edges.
    cap_stroke = f' stroke="{outline}" stroke-width="{sw:.2f}"' if draw_outline else ' stroke="none"'
    # Each spline sample belongs to the nearest residue, so per-residue colour
    # modes (rainbow, by-SS, by-B-factor) shade the right stretch of tape.
    sample_res = np.clip(np.rint(res_t).astype(int), 0, len(shaders) - 1)

    nxt = np.roll(np.arange(_PROFILE_N), -1)
    # Everything per-quad is computed for all quads at once: np.cross and
    # np.linalg.norm cost far more in call overhead than in arithmetic when
    # handed single 3-vectors.  Shapes are (m - 1, _PROFILE_N[, 3]).
    _ax = px[:-1, nxt] - px[:-1, :]
    _ay = py[:-1, nxt] - py[:-1, :]
    _bx = px[1:, :] - px[:-1, :]
    _by = py[1:, :] - py[:-1, :]
    # Screen-space signed area; negative faces the viewer.  Needed twice:
    # back-face culling and silhouette edges.
    front = (_ax * _by - _ay * _bx) < 0.0

    e1 = verts[:-1, nxt, :] - verts[:-1, :, :]
    e2 = verts[1:, :, :] - verts[:-1, :, :]
    fn = np.cross(e1, e2)
    fl = np.linalg.norm(fn, axis=2)
    ok = fl > _EPS
    # True Lambert, not |Lambert|: the winding is consistently outward, so faces
    # turned from the light belong in shadow.  abs() lit the tape's thin edge
    # faces as brightly as its top.  Degenerate quads fall back to a flat 0.5.
    lam_raw = np.einsum("ijk,k->ij", fn, _LIGHT_DIR) / np.where(ok, fl, 1.0)
    lam_all = np.where(ok, np.maximum(lam_raw, 0.0), 0.5)
    lam_all = _SHADE_FLOOR + (1.0 - _SHADE_FLOOR) * lam_all

    zq_all = (pz[:-1, :] + pz[:-1, nxt] + pz[1:, nxt] + pz[1:, :]) * 0.25

    for i in range(m - 1):
        for j in range(_PROFILE_N):
            if not front[i, j]:
                continue  # back face of a closed tube — culled
            k = nxt[j]
            zq = zq_all[i, j]
            lam = lam_all[i, j]
            fog = _fog_at(zq, fog_enabled, fog_strength, z_front, z_range)
            fill = shaders[sample_res[i]].shade(lam, fog)
            pts = (
                f"{px[i, j]:.1f},{py[i, j]:.1f} {px[i, k]:.1f},{py[i, k]:.1f} "
                f"{px[i + 1, k]:.1f},{py[i + 1, k]:.1f} {px[i + 1, j]:.1f},{py[i + 1, j]:.1f}"
            )
            item = [f'  <polygon points="{pts}" fill="{fill}"/>']
            if draw_outline:
                # Same item, after the polygon: over its own quad, still under
                # anything the z-order puts in front.
                item.extend(_silhouette_edges(px, py, front, i, j, m, outline or "", sw))
            items.append((float(zq), item))

    # --- 6. end caps so open segment ends are not hollow -------------------
    view_z = np.array([0.0, 0.0, 1.0])
    for i, sgn in ((0, -1.0), (m - 1, 1.0)):
        cap_n = tangents[i] * sgn
        if float(np.dot(cap_n, view_z)) <= 0.0:
            continue
        pts = " ".join(f"{px[i, j]:.1f},{py[i, j]:.1f}" for j in range(_PROFILE_N))
        zc = float(pz[i].mean())
        fog = _fog_at(zc, fog_enabled, fog_strength, z_front, z_range)
        # Shaded like any other face; a flat base colour reads as unlit against
        # the surrounding Lambert range.
        cap_lam = _SHADE_FLOOR + (1.0 - _SHADE_FLOOR) * max(0.0, float(np.dot(cap_n, _LIGHT_DIR)))
        fill = shaders[sample_res[i]].shade(cap_lam, fog)
        items.append((zc, [f'  <polygon points="{pts}" fill="{fill}"{cap_stroke}/>']))

    return items


# ---------------------------------------------------------------------------
# Segment extraction helpers
# ---------------------------------------------------------------------------


def _extract_ca_o_ss(
    residues: list["ResidueData"],
    pos: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray | None], list[str]]:
    """Extract CA positions, O positions and normalized ss labels."""
    ca_list: list[np.ndarray] = []
    o_list: list[np.ndarray | None] = []
    ss_list: list[str] = []
    for res in residues:
        if res.ca_index is None:
            continue
        ca_list.append(pos[res.ca_index])
        o_list.append(pos[res.o_index] if res.o_index is not None else None)
        ss_list.append(res.ss_type if res.ss_type in {"H", "E", "C"} else "C")
    return np.array(ca_list), o_list, ss_list


def _split_backbone_segments(
    residues: list["ResidueData"],
    pos: np.ndarray,
) -> list[list["ResidueData"]]:
    """Split a chain into continuous CA segments using sequence + geometry.

    This prevents rendering across true backbone discontinuities (missing
    residues, unresolved loops) while keeping style transitions continuous
    inside each segment.
    """
    ca_res = [r for r in residues if r.ca_index is not None]
    if len(ca_res) < 2:
        return [ca_res] if ca_res else []

    segments: list[list["ResidueData"]] = [[ca_res[0]]]
    prev = ca_res[0]
    for cur in ca_res[1:]:
        seq_step = int(cur.res_seq) - int(prev.res_seq)
        # An inserted residue (100, 100A, 100B ...) repeats res_seq, giving
        # seq_step == 0.  That is a continuation of the chain, not a break.
        if seq_step == 0 and getattr(cur, "i_code", "") != getattr(prev, "i_code", ""):
            seq_step = _MAX_SEQ_STEP
        gap = float(np.linalg.norm(pos[cur.ca_index] - pos[prev.ca_index]))
        if (seq_step != _MAX_SEQ_STEP) or (gap > _MAX_CA_GAP):
            segments.append([cur])
        else:
            segments[-1].append(cur)
        prev = cur
    return [seg for seg in segments if len(seg) >= 2]


def has_renderable_ribbon(
    protein_data: "ProteinData | ProteinSemantics",
    cfg: "RenderConfig",
    pos: np.ndarray,
) -> bool:
    """Whether protein mode can emit at least one ribbon or trace segment."""
    from xyzrender.types import ProteinConfidence

    if protein_data.confidence_tier == ProteinConfidence.INSUFFICIENT:
        return False
    for chain_id, chain in protein_data.chains.items():
        if chain_id in cfg.exclude_chains:
            continue
        if protein_data.confidence_tier == ProteinConfidence.TRACE_ONLY:
            trace = protein_data.trace_chains.get(chain_id, [])
            if len(trace) < 2:
                trace = [res.ca_index for res in chain.residues if res.ca_index is not None]
            if len(trace) >= 2:
                return True
        elif any(len(segment) >= 2 for segment in _split_backbone_segments(chain.residues, pos)):
            return True
    return False


# ---------------------------------------------------------------------------
# Colour modes
# ---------------------------------------------------------------------------

COLOR_BY_MODES: tuple[str, ...] = ("chain", "rainbow", "ss", "bfactor")

# Secondary-structure palette.  Distinct hues rather than a ramp: H/E/C is a
# categorical distinction, not an ordered one.
_SS_COLORS: dict[str, str] = {"H": "#d95f5f", "E": "#e0c060", "C": "#7f9fbf"}

_RAINBOW_PALETTE = "spectral"
_BFACTOR_PALETTE = "viridis"


def normalize_color_by(mode: str | None) -> str:
    """Validate a ``--color-by`` mode name."""
    key = (mode or "chain").strip().lower()
    if key not in COLOR_BY_MODES:
        opts = ", ".join(COLOR_BY_MODES)
        raise ValueError(f"unknown color-by mode {mode!r} (expected one of: {opts})")
    return key


def _residue_colors(
    residues: list["ResidueData"],
    mode: str,
    chain_color: str,
    *,
    b_range: tuple[float, float] | None,
) -> list[str]:
    """One base colour per residue for the given mode."""
    n = len(residues)
    if mode == "ss":
        return [_SS_COLORS.get(r.ss_type, _SS_COLORS["C"]) for r in residues]
    if mode == "rainbow":
        # N-terminus to C-terminus across the chain.
        if n == 1:
            return [palette_color(_RAINBOW_PALETTE, 0.5).hex]
        return [palette_color(_RAINBOW_PALETTE, i / (n - 1)).hex for i in range(n)]
    if mode == "bfactor":
        if b_range is None:
            return [chain_color] * n
        lo, hi = b_range
        span = max(hi - lo, _EPS)
        return [palette_color(_BFACTOR_PALETTE, (r.b_factor - lo) / span).hex for r in residues]
    return [chain_color] * n


def bfactor_range(protein_data: "ProteinData | ProteinSemantics", exclude: set[str]) -> tuple[float, float] | None:
    """Min/max residue B-factor across displayed chains, or ``None`` if flat."""
    vals = [r.b_factor for cid, chain in protein_data.chains.items() if cid not in exclude for r in chain.residues]
    if not vals or max(vals) - min(vals) <= _EPS:
        return None
    return min(vals), max(vals)


def excluded_atom_indices(protein_data: "ProteinData | ProteinSemantics", exclude_chains: Iterable[str]) -> set[int]:
    """Every atom of an excluded chain — polymer residues and heteroatoms alike.

    Keyed off the requested chain IDs rather than ``protein_data.chains``: a
    chain with fewer than two peptide residues is dropped from ``chains``
    entirely, so a ligand-only chain would otherwise be silently unexcludable.
    """
    out: set[int] = set()
    het_chains = protein_data.het_chains
    for cid in exclude_chains:
        chain = protein_data.chains.get(cid)
        if chain is not None:
            for res in chain.residues:
                out.update(res.atom_indices)
        out.update(het_chains.get(cid, ()))
    return out


# ---------------------------------------------------------------------------
# Sidechain attachment
# ---------------------------------------------------------------------------


def select_sidechain_atoms(
    protein_data: "ProteinData | ProteinSemantics",
    spec: str | None,
    *,
    exclude_chains: set[str] | frozenset[str] = frozenset(),
) -> set[int]:
    """Resolve a residue selector to the sidechain atom indices it covers.

    *spec* is a comma-separated list of residue numbers and ranges, each
    optionally chain-qualified: ``"45"``, ``"102-108"``, ``"A:45"``,
    ``"A:102-108"``.  ``None`` selects every residue in the displayed chains.
    """
    wanted: list[tuple[str | None, int, int]] = []
    if spec is not None:
        for raw_token in str(spec).split(","):
            token = raw_token.strip()
            if not token:
                continue
            chain: str | None = None
            if ":" in token:
                chain_part, _, rest = token.partition(":")
                chain = chain_part.strip() or None
                token = rest.strip()
            try:
                if "-" in token.lstrip("-"):
                    lo_s, _, hi_s = token.partition("-")
                    lo, hi = int(lo_s), int(hi_s)
                else:
                    lo = hi = int(token)
            except ValueError as exc:
                msg = f"invalid residue selector {raw_token.strip()!r} (expected e.g. '45', '102-108' or 'A:45')"
                raise ValueError(msg) from exc
            wanted.append((chain, min(lo, hi), max(lo, hi)))
        if not wanted:
            raise ValueError("residue selector must not be empty")

    out: set[int] = set()
    for cid, chain_data in protein_data.chains.items():
        if cid in exclude_chains:
            continue
        for res in chain_data.residues:
            if wanted and not any((c is None or c == cid) and lo <= res.res_seq <= hi for c, lo, hi in wanted):
                continue
            out.update(res.atom_indices)
    return out


# The swept cross-section is a superellipse, which is boxier than the ellipse
# used to solve for the exit point below; inflate slightly so the sidechain
# starts just clear of the surface rather than just inside it.
_SECTION_INFLATE = 1.06


def sidechain_anchors(
    protein_data: "ProteinData | ProteinSemantics",
    cfg: "RenderConfig",
    pos: np.ndarray,
) -> dict[int, np.ndarray]:
    """Map ``CA atom index -> point where the ribbon surface is exited``.

    With the cartoon a solid tape rather than a flat strip, a sidechain bond
    drawn from CA starts *inside* the ribbon: CA is the centreline, and the
    tape is ~2.25 A half-width.  Anchoring the bond where it leaves the swept
    surface instead makes the sidechain read as growing out of the ribbon.

    The exit point is solved against an ellipse with the local half-width and
    half-thickness, which is closed-form and a close fit to the superellipse
    actually swept.  The direction used is the residue's own CA->CB vector
    where a CB exists, falling back to the frame's thickness axis (a sidechain
    leaves the tape face, not its edge).

    The half-extents come from :func:`_sample_widths` evaluated at the residues'
    own parameter values, i.e. the same call the sweep makes.  Reading the raw
    per-SS table instead would miss the cross-run smoothing and the strand
    arrowhead flange/taper, and sidechains at strand C-termini would float off
    the tip.
    """
    style = ribbon_style_profile(cfg.protein_style)
    dims = _section_dims(cfg.ribbon_width / 2.0, cfg.loop_width / 2.0, style)

    anchors: dict[int, np.ndarray] = {}
    for cid, chain in protein_data.chains.items():
        if cid in cfg.exclude_chains:
            continue
        for seg in _split_backbone_segments(chain.residues, pos):
            ca_pos, o_pos, ss = _extract_ca_o_ss(seg, pos)
            if len(ca_pos) < 2:
                continue
            tangents = _central_tangents(ca_pos)
            normals = _frame_normals(ca_pos, o_pos, tangents, ss)
            binormals = _unit_rows(np.cross(tangents, normals))
            hw_r, ht_r, _ = _sample_widths(ss, np.arange(len(ss), dtype=float), dims)

            res_with_ca = [(r, r.ca_index) for r in seg if r.ca_index is not None]
            for k, (res, ca_index) in enumerate(res_with_ca):
                hw, ht = float(hw_r[k]), float(ht_r[k])

                # Direction out of the tape: CA->CB when available.
                d = None
                cb = next(
                    (i for i in res.atom_indices if i not in (ca_index, res.c_index, res.o_index, res.n_index)),
                    None,
                )
                if cb is not None:
                    d = pos[cb] - ca_pos[k]
                if d is None or np.linalg.norm(d) <= _EPS:
                    d = binormals[k]
                # Only the in-section components matter.
                du = float(np.dot(d, normals[k]))
                dv = float(np.dot(d, binormals[k]))
                denom = np.hypot(du / max(hw, _EPS), dv / max(ht, _EPS))
                if denom <= _EPS:
                    continue
                t = _SECTION_INFLATE / denom
                anchors[ca_index] = ca_pos[k] + (normals[k] * du + binormals[k] * dv) * t
    return anchors


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def ribbon_svg_items(
    protein_data: "ProteinData | ProteinSemantics",
    cfg: "RenderConfig",
    pos: np.ndarray,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: float,
    canvas_h: float,
    *,
    fog_enabled: bool = False,
    fog_strength: float = 0.0,
    fog_rgb: Color | None = None,
    helix_smoothing: float = 0.0,
) -> list[tuple[float, list[str]]]:
    """Return ``(z_depth, svg_lines)`` tuples for all cartoon elements.

    Sorted ascending by z_depth so the caller can drain them into the main
    painter's-algorithm render loop (same pattern as NCI patches).  Items are
    emitted at *quad* granularity, which is what allows a helix to occlude its
    own turns.
    """
    _validate_ribbon_dimensions(cfg)
    chain_ids = [cid for cid in protein_data.chains if cid not in cfg.exclude_chains]
    # Allocate over every chain, then filter: excluding one must not renumber
    # the rest, or the same chain changes colour between figures in a series.
    chain_colors = assign_chain_colors(cfg, list(protein_data.chains), style=cfg.protein_style)
    half_ribbon = cfg.ribbon_width / 2.0
    loop_half = cfg.loop_width / 2.0
    style = ribbon_style_profile(cfg.protein_style)
    mode = normalize_color_by(cfg.color_by)
    b_range = bfactor_range(protein_data, set(cfg.exclude_chains)) if mode == "bfactor" else None
    if mode == "bfactor" and b_range is None:
        logger.warning("color_by='bfactor' requested but every residue has the same B-factor; using chain colours")

    z_front = float(pos[:, 2].max()) if len(pos) else 0.0
    z_range = max(float(pos[:, 2].max() - pos[:, 2].min()), 1e-6) if len(pos) else 1.0

    # One shader per distinct colour, shared across residues that repeat it.
    shader_cache: dict[str, _Shader] = {}

    def _shader_for(hex_color: str) -> _Shader:
        got = shader_cache.get(hex_color)
        if got is None:
            got = _Shader(hex_color, cfg, style.shade_gain, fog_enabled=fog_enabled, fog_rgb=fog_rgb)
            shader_cache[hex_color] = got
        return got

    items: list[tuple[float, list[str]]] = []
    for cid in chain_ids:
        color = chain_colors[cid]
        outline = Color.from_str(color).blend(_BLACK, style.outline_dark).hex if style.outline_px > 0 else None
        for backbone_seg in _split_backbone_segments(protein_data.chains[cid].residues, pos):
            ca_pos, o_pos, ss = _extract_ca_o_ss(backbone_seg, pos)
            if len(ca_pos) < 2:
                continue
            seg_res = [r for r in backbone_seg if r.ca_index is not None]
            seg_shaders = [_shader_for(c) for c in _residue_colors(seg_res, mode, color, b_range=b_range)]
            items.extend(
                _segment_items(
                    ca_pos,
                    o_pos,
                    ss,
                    seg_shaders,
                    style,
                    outline,
                    half_ribbon,
                    loop_half,
                    scale,
                    cx,
                    cy,
                    canvas_w,
                    canvas_h,
                    helix_smoothing=helix_smoothing,
                    fog_enabled=fog_enabled,
                    fog_strength=fog_strength,
                    z_front=z_front,
                    z_range=z_range,
                )
            )

    items.sort(key=lambda x: x[0])
    return items


def trace_svg_items(
    protein_data: "ProteinData | ProteinSemantics",
    cfg: "RenderConfig",
    pos: np.ndarray,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: float,
    canvas_h: float,
    *,
    fog_enabled: bool = False,
    fog_strength: float = 0.0,
    fog_rgb: Color | None = None,
) -> list[tuple[float, list[str]]]:
    """Fallback CA-trace rendering for TRACE_ONLY confidence.

    Sweeps the same round coil section along the raw trace; no secondary
    structure is claimed.
    """
    chain_ids = [cid for cid in protein_data.chains if cid not in cfg.exclude_chains]
    # Allocate over every chain, then filter: excluding one must not renumber
    # the rest, or the same chain changes colour between figures in a series.
    chain_colors = assign_chain_colors(cfg, list(protein_data.chains), style=cfg.protein_style)
    style = ribbon_style_profile(cfg.protein_style)
    loop_half = cfg.loop_width / 2.0

    z_front = float(pos[:, 2].max()) if len(pos) else 0.0
    z_range = max(float(pos[:, 2].max() - pos[:, 2].min()), 1e-6) if len(pos) else 1.0

    items: list[tuple[float, list[str]]] = []
    for cid in chain_ids:
        trace = protein_data.trace_chains.get(cid, [])
        if len(trace) < 2:
            trace = [r.ca_index for r in protein_data.chains[cid].residues if r.ca_index is not None]
        if len(trace) < 2:
            continue
        ca_pos = np.array([pos[i] for i in trace], dtype=float)
        color = chain_colors[cid]
        shader = _Shader(color, cfg, style.shade_gain, fog_enabled=fog_enabled, fog_rgb=fog_rgb)
        outline = Color.from_str(color).blend(_BLACK, style.outline_dark).hex if style.outline_px > 0 else None
        items.extend(
            _segment_items(
                ca_pos,
                [None] * len(ca_pos),
                ["C"] * len(ca_pos),
                [shader] * len(ca_pos),
                style,
                outline,
                loop_half,
                loop_half,
                scale,
                cx,
                cy,
                canvas_w,
                canvas_h,
                helix_smoothing=0.0,
                fog_enabled=fog_enabled,
                fog_strength=fog_strength,
                z_front=z_front,
                z_range=z_range,
            )
        )

    items.sort(key=lambda x: x[0])
    return items
