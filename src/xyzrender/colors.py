"""CPK colors and fog blending for xyzrender."""

from __future__ import annotations

import numpy as np

from xyzrender.types import Color

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


def get_color(atomic_number: int, overrides: dict[str, str] | None = None) -> Color:
    """Get element color by atomic number, with optional per-element overrides."""
    if overrides:
        from xyzgraph import DATA

        sym = DATA.n2s.get(atomic_number, "")
        if sym in overrides:
            return Color.from_hex(overrides[sym])
    if 0 < atomic_number < len(_CPK):
        return Color.from_int(_CPK[atomic_number])
    return Color.from_int(_DEFAULT_COLOR)


def get_gradient_colors(color: Color, strength: float = 1.0) -> tuple[Color, Color]:
    """Compute gradient pair from a base color: (lighter center, darker edge)."""
    return color.lighten(0.3), color.darken(0.2 * strength)


_FOG_NEAR = 1.0  # Å of depth before fog kicks in
_MAX_FOG = 0.70  # deepest atoms retain at least 30% of their color


def blend_fog(hex_color: str, fog_rgb: np.ndarray, strength: float) -> str:
    """Blend color toward fog using strength**2, capped so atoms stay visible."""
    s = min(strength**2, _MAX_FOG)
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)])
    blended = (1 - s) * rgb + s * fog_rgb
    r, g, b = np.clip(blended, 0, 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"
