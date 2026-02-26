"""Core types for xyzrender."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xyzrender.esp import ESPSurface
    from xyzrender.mo import MOContours


class BondStyle(Enum):
    """Visual bond style."""

    SOLID = "solid"
    DASHED = "dashed"  # TS bonds
    DOTTED = "dotted"  # NCI bonds


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

    def darken(self, factor: float) -> Color:
        """Multiply by (1-factor), clamped."""
        m = max(0.0, 1.0 - factor)
        return Color(int(self.r * m), int(self.g * m), int(self.b * m))

    def lighten(self, factor: float) -> Color:
        """Blend toward white."""
        return self.blend(Color(255, 255, 255), factor)

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


_NAMED_COLORS: dict[str, str] | None = None


def _load_named_colors() -> dict[str, str]:
    """Load CSS4 named colors from bundled JSON (cached on first call)."""
    global _NAMED_COLORS  # noqa: PLW0603
    if _NAMED_COLORS is None:
        import json
        from pathlib import Path

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


@dataclass
class RenderConfig:
    """Rendering settings."""

    canvas_size: int = 800
    padding: float = 20.0
    atom_scale: float = 1.0
    atom_stroke_width: float = 1.5
    atom_stroke_color: str = "#000000"
    bond_width: float = 5.0
    bond_color: str = "#333333"
    bond_gap: float = 0.6  # multi-bond spacing as fraction of bond_width
    gradient: bool = False
    gradient_strength: float = 1.4  # scales lighten/darken of gradient stops
    fog: bool = False
    fog_strength: float = 0.8
    hide_h: bool = False
    show_h_indices: list[int] = field(default_factory=list)
    bond_orders: bool = True
    ts_bonds: list[tuple[int, int]] = field(default_factory=list)  # 0-indexed pairs
    nci_bonds: list[tuple[int, int]] = field(default_factory=list)  # 0-indexed pairs
    vdw_indices: list[int] | None = None
    vdw_opacity: float = 0.5
    vdw_scale: float = 1.0
    vdw_gradient_strength: float = 1.0  # scales lighten/darken of VdW sphere gradient
    auto_orient: bool = False
    background: str = "#ffffff"
    dpi: int = 300
    fixed_span: float | None = None  # fixed viewport span (disables auto-fit)
    fixed_center: tuple[float, float] | None = None  # fixed XY center (disables auto-center)
    color_overrides: dict[str, str] | None = None  # element symbol → hex color
    # Surface rendering (MO / density / ESP share one opacity)
    mo_contours: MOContours | None = None
    dens_contours: MOContours | None = None
    esp_surface: ESPSurface | None = None
    surface_opacity: float = 1.0
