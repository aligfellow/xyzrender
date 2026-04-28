"""CPK colors, Color type, and fog blending for xyzrender."""

from __future__ import annotations

import colorsys
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xyzrender.types import RenderConfig

# ---------------------------------------------------------------------------
# Color type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Color:
    """RGB color (0-255).

    Examples
    --------
    >>> Color(255, 0, 0).hex
    '#ff0000'
    >>> Color(100, 100, 100).blend(Color(200, 200, 200), 0.5)
    Color(r=150, g=150, b=150)
    """

    r: int
    g: int
    b: int

    # ---------- conversions ----------

    def to_hls(self) -> tuple[float, float, float]:
        """Convert to (hue 0-360, lightness 0-1, saturation 0-1)."""
        r, g, b = self.r / 255, self.g / 255, self.b / 255
        h_val, l_val, s_val = colorsys.rgb_to_hls(r, g, b)
        return h_val * 360, l_val, s_val

    @staticmethod
    def from_hls(h_val: float, l_val: float, s_val: float) -> "Color":
        """Create from (hue 0-360, lightness 0-1, saturation 0-1)."""
        r, g, b = colorsys.hls_to_rgb((h_val % 360) / 360, l_val, s_val)
        return Color(int(r * 255), int(g * 255), int(b * 255))

    @property
    def hex(self) -> str:
        """CSS hex string."""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def blend(self, other: Color, t: float) -> Color:
        """Lerp toward ``other`` by ``t`` (0=self, 1=other), clamped to 0-255."""
        return Color(
            min(255, max(0, int(self.r + t * (other.r - self.r)))),
            min(255, max(0, int(self.g + t * (other.g - self.g)))),
            min(255, max(0, int(self.b + t * (other.b - self.b)))),
        )

    def darken(
        self,
        strength: float = 1.0,
        hue_shift_factor: float = 0.2,
        light_shift_factor: float = 0.2,
        saturation_shift_factor: float = 0.2,
    ) -> "Color":
        """Darken toward blue, scaled by *strength*."""
        h_val, l_val, s_val = self.to_hls()

        # decrease lightness
        new_l = l_val * (1 - light_shift_factor * strength * 3)
        new_l = max(0.0, min(1.0, new_l))

        # hue shift toward blue (240°)
        d = ((240 - h_val + 180) % 360) - 180
        new_h = (h_val + d * hue_shift_factor * strength) % 360

        # increase saturation
        new_s = s_val + (1 - s_val) * saturation_shift_factor * strength
        new_s = max(0.0, min(1.0, new_s))

        return Color.from_hls(new_h, new_l, new_s)

    def lighten(
        self,
        strength: float = 1.0,
        hue_shift_factor: float = 0.2,
        light_shift_factor: float = 0.2,
        saturation_shift_factor: float = 0.2,
    ) -> "Color":
        """Lighten toward yellow, scaled by *strength*."""
        h_val, l_val, s_val = self.to_hls()

        # increase lightness
        new_l = l_val + light_shift_factor * strength * (1 - l_val)
        new_l = max(0.0, min(1.0, new_l))

        # hue shift toward yellow (60°)
        d = ((60 - h_val + 180) % 360) - 180  # shortest direction
        new_h = (h_val + d * hue_shift_factor * strength) % 360

        # decrease saturation
        new_s = s_val * (1 - saturation_shift_factor * strength)
        new_s = max(0.0, min(1.0, new_s))

        return Color.from_hls(new_h, new_l, new_s)

    @classmethod
    def from_hex(cls, hex_str: str) -> Color:
        """From ``'#ff0000'`` or ``'ff0000'``.

        Examples
        --------
        >>> Color.from_hex("#ff0000")
        Color(r=255, g=0, b=0)
        """
        h = hex_str.lstrip("#")
        return cls(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    @classmethod
    def from_str(cls, color: str) -> Color:
        """From hex (``'#ff0000'``) or CSS4 name (``'red'``).

        Examples
        --------
        >>> Color.from_str("#ff0000")
        Color(r=255, g=0, b=0)
        """
        return cls.from_hex(resolve_color(color))

    @classmethod
    def from_int(cls, value: int) -> Color:
        """From ``0xff0000``.

        Examples
        --------
        >>> Color.from_int(0xFF0000)
        Color(r=255, g=0, b=0)
        """
        return cls((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)


# ---------------------------------------------------------------------------
# Named color resolution
# ---------------------------------------------------------------------------

_NAMED_COLORS: dict[str, str] | None = None


def _load_named_colors() -> dict[str, str]:
    """Load CSS4 named colors from bundled JSON (cached on first call)."""
    global _NAMED_COLORS  # noqa: PLW0603
    if _NAMED_COLORS is None:
        path = Path(__file__).parent / "presets" / "named_colors.json"
        with path.open() as f:
            _NAMED_COLORS = json.load(f)
    return _NAMED_COLORS


def resolve_color(color: str) -> str:
    """Resolve hex (``'#FF0000'``) or CSS4 name (``'red'``) to ``'#rrggbb'``.

    Examples
    --------
    >>> resolve_color("#FF0000")
    '#ff0000'
    >>> resolve_color("FF0000")
    '#ff0000'
    >>> resolve_color("red")
    '#ff0000'
    """
    s = color.strip()
    h = s.lstrip("#")
    # Fast path: already a 6-digit hex string
    if len(h) == 6 and all(c in "0123456789abcdefABCDEF" for c in h):
        return f"#{h.lower()}"
    # Named color lookup
    named = _load_named_colors()
    key = s.lower()
    if key in named:
        return named[key]
    msg = f"Unknown color {color!r}. Use hex (#rrggbb) or a named color (e.g. 'steelblue')."
    raise ValueError(msg)


WHITE = Color(255, 255, 255)

# CPK colors by atomic number (from xyz2svg). Index 0 unused.
_CPK: list[int] = [
    0x999999,
    0xFFFFFF, 0xD9FFFF,                                                         # H, He
    0xCC80FF, 0xC2FF00, 0xFFB5B5, 0x909090, 0x3050F8, 0xFF0D0D, 0x90E050, 0xB3E3F5,  # Li-Ne
    0xAB5CF2, 0x8AFF00, 0xBFA6A6, 0xF0C8A0, 0xFF8000, 0xFFFF30, 0x1FF01F, 0x80D1E3,  # Na-Ar
    0x8F40D4, 0x3DFF00, 0xE6E6E6, 0xBFC2C7, 0xA6A6AB, 0x8A99C7, 0x9C7AC7, 0xE06633, 0xF090A0, 0x50D050,  # K-Ni
    0xC88033, 0x7D80B0, 0xC28F8F, 0x668F8F, 0xBD80E3, 0xFFA100, 0xA62929, 0x5CB8D1,  # Cu-Kr
    0x702EB0, 0x00FF00, 0x94FFFF, 0x94E0E0, 0x73C2C9, 0x54B5B5, 0x3B9E9E, 0x248F8F, 0x0A7D8C, 0x006985,  # Rb-Pd
    0xC0C0C0, 0xFFD98F, 0xA67573, 0x668080, 0x9E63B5, 0xD47A00, 0x940094, 0x429EB0,  # Ag-Xe
    0x57178F, 0x00C900, 0x70D4FF, 0xFFFFC7, 0xD9FFC7, 0xC7FFC7, 0xA3FFC7, 0x8FFFC7, 0x61FFC7,  # Cs-Eu
    0x45FFC7, 0x30FFC7, 0x1FFFC7, 0x00FF9C, 0x00E675, 0x00D452, 0x00BF38,  # Gd-Yb
    0x00AB24, 0x4DC2FF, 0x4DA6FF, 0x2194D6, 0x267DAB, 0x266696, 0x175487, 0xD0D0E0,  # Lu-Pt
    0xFFD123, 0xB8B8D0, 0xA6544D, 0x575961, 0x9E4FB5, 0xAB5C00, 0x754F45, 0x428296,  # Au-Rn
    0x420066, 0x007D00, 0x70ABFA, 0x00BAFF, 0x00A1FF, 0x008FFF, 0x0080FF, 0x006BFF, 0x545CF2,  # Fr-Am
    0x785CE3, 0x8A4FE3, 0xA136D4, 0xB31FD4, 0xB31FBA, 0xB30DA6, 0xBD0D87, 0xC70066, 0xCC0059, 0xA0A0A0,
] + [0xA0A0A0] * 14  # fmt: skip

_DEFAULT_COLOR = 0xA0A0A0
_CENTROID_COLOR = 0x008080  # teal for NCI pi-system centroids


def get_color(atomic_number: int, overrides: dict[str, str] | None = None) -> Color:
    """Get element color by atomic number, with optional per-element overrides.

    Atomic number 0 is used for NCI pi-system centroid dummy nodes.
    """
    if overrides:
        from xyzgraph import DATA

        sym = DATA.n2s.get(atomic_number, 0)
        if sym in overrides:
            return Color.from_hex(overrides[sym])
    if atomic_number == 0:
        return Color.from_int(_CENTROID_COLOR)
    if 0 < atomic_number < len(_CPK):
        return Color.from_int(_CPK[atomic_number])
    return Color.from_int(_DEFAULT_COLOR)


def get_gradient_colors(
    color: Color, config: RenderConfig | None = None, strength: float = 1.0
) -> tuple[Color, Color, Color]:
    """Compute gradient triplet from a base color: (lighter center, base, darker edge).

    *strength* scales the lighten/darken shift (see ``atom_gradient_strength``,
    ``bond_gradient_strength``, ``vdw_gradient_strength``).
    """
    cfg = config or RenderConfig()
    return (
        color.lighten(
            strength=strength,
            hue_shift_factor=cfg.hue_shift_factor,
            light_shift_factor=cfg.light_shift_factor,
            saturation_shift_factor=cfg.saturation_shift_factor,
        ),
        color,
        color.darken(
            strength=strength,
            hue_shift_factor=cfg.hue_shift_factor,
            light_shift_factor=cfg.light_shift_factor,
            saturation_shift_factor=cfg.saturation_shift_factor,
        ),
    )


_FOG_NEAR = 1.0  # Å of depth before fog kicks in
_MAX_FOG = 0.70  # deepest atoms retain at least 30% of their color

# ---------------------------------------------------------------------------
# Colormap palettes
# ---------------------------------------------------------------------------

PALETTES: dict[str, list[Color]] = {
    "viridis": [
        Color.from_str("#440154"),
        Color.from_str("#31688e"),
        Color.from_str("#35b779"),
        Color.from_str("#90d743"),
        Color.from_str("#fde725"),
    ],
    "plasma": [
        Color.from_str("#0d0887"),
        Color.from_str("#7e03a8"),
        Color.from_str("#cc4778"),
        Color.from_str("#f89441"),
        Color.from_str("#f0f921"),
    ],
    "spectral": [
        Color.from_str("#9e0142"),
        Color.from_str("#d53e4f"),
        Color.from_str("#f46d43"),
        Color.from_str("#fdae61"),
        Color.from_str("#abdda4"),
        Color.from_str("#45aba3"),
        Color.from_str("#3288bd"),
    ],
    "coolwarm": [
        Color.from_str("#b40426"),
        Color.from_str("#e8836b"),
        Color.from_str("#d2d2d2"),
        Color.from_str("#7c9fed"),
        Color.from_str("#3b4cc0"),
    ],
    "RdBu": [
        Color.from_str("#67001f"),
        Color.from_str("#d6604d"),
        Color.from_str("#f7f7f7"),
        Color.from_str("#4393c3"),
        Color.from_str("#053061"),
    ],
    "rainbow": [
        Color.from_str("maroon"),
        Color.from_str("peru"),
        Color.from_str("darkseagreen"),
        Color.from_str("steelblue"),
        Color.from_str("midnightblue"),
    ],
    "managua": [
        Color.from_str("#ffcf67"),
        Color.from_str("#cc824d"),
        Color.from_str("#92463b"),
        Color.from_str("#572949"),
        Color.from_str("#4e5593"),
        Color.from_str("#6498ce"),
        Color.from_str("#81e7ff"),
    ],
    "bam": [
        Color.from_str("#65024b"),
        Color.from_str("#b5539c"),
        Color.from_str("#e4aed6"),
        Color.from_str("#f6f1f0"),
        Color.from_str("#c1daa2"),
        Color.from_str("#5e903d"),
        Color.from_str("#0d4c00"),
    ],
    "vik": [
        Color.from_str("#590008"),
        Color.from_str("#a94512"),
        Color.from_str("#d39774"),
        Color.from_str("#ece5e0"),
        Color.from_str("#70a8c4"),
        Color.from_str("#06548b"),
        Color.from_str("#001261"),
    ],
    "roma": [
        Color.from_str("#7d1700"),
        Color.from_str("#a5681f"),
        Color.from_str("#c8b455"),
        Color.from_str("#c0eac3"),
        Color.from_str("#5dc1d3"),
        Color.from_str("#287ab8"),
        Color.from_str("#023198"),
    ],
    "batlow": [
        Color.from_str("#faccfb"),
        Color.from_str("#fdac9e"),
        Color.from_str("#d39443"),
        Color.from_str("#838231"),
        Color.from_str("#3c6d56"),
        Color.from_str("#144d62"),
        Color.from_str("#011a59"),
    ],
}

PALETTE_NAMES: list[str] = list(PALETTES)

DEFAULT_CMAP_PALETTE = "viridis"
DEFAULT_ESP_PALETTE = "rainbow"


def palette_color(name: str, t: float) -> Color:
    """Sample a single point ``t ∈ [0, 1]`` from a named palette."""
    stops = PALETTES[name]
    t = max(0.0, min(1.0, float(t)))
    n_segs = len(stops) - 1
    seg = min(int(t * n_segs), n_segs - 1)
    return stops[seg].blend(stops[seg + 1], t * n_segs - seg)


def sample_palette(name: str, n: int) -> list[str]:
    """Sample *n* evenly-spaced hex colours from a named palette."""
    if n <= 1:
        return [palette_color(name, 0.5).hex]
    return [palette_color(name, i / (n - 1)).hex for i in range(n)]


_BOND_DARKEN_T: float = 0.3  # blend toward black — shared by mol_color, highlight, overlay


def bond_color_from_atom(atom: Color) -> str:
    """Derive a bond colour from its atom colour: 30 % blend toward black.

    Shared by the three "set one colour, darken for bonds" paths (mol_color,
    highlight groups, overlay / ensemble per-structure).  Inline callers used
    to duplicate this with two different maths (``.darken`` vs ``.blend``);
    consolidating here keeps the three sites visually identical.
    """
    return atom.blend(Color(0, 0, 0), _BOND_DARKEN_T).hex


def blend_fog(hex_color: str, fog_rgb: np.ndarray, strength: float) -> str:
    """Blend color toward fog using strength**2, capped so atoms stay visible."""
    s = min(strength**2, _MAX_FOG)
    hex_color = resolve_color(hex_color)
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)])
    blended = (1 - s) * rgb + s * fog_rgb
    r, g, b = np.clip(blended, 0, 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"
