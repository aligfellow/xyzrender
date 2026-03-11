"""Core types for xyzrender."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xyzrender.annotations import Annotation
    from xyzrender.esp import ESPSurface
    from xyzrender.mo import SurfaceContours
    from xyzrender.nci import NCIContours


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
class VectorArrow:
    """A 3D vector to be drawn as an arrow in the rendered image.

    Parameters
    ----------
    vector:
        3-component array giving the direction and magnitude of the arrow (Å or
        any consistent unit — the length on screen scales with the molecule).
    origin:
        3D origin point of the arrow tail in the same coordinate frame as atom
        positions.  Set this after resolving ``"com"`` or atom-index origins.
    color:
        CSS hex color string (default ``'#444444'``).
    label:
        Optional text placed near the arrowhead.
    scale:
        Additional per-arrow length scale factor applied on top of any global
        ``vector_scale`` setting (default 1.0).
    host_atom:
        0-based index of the atom this arrow is centred on, or ``None`` when
        the origin was specified as ``"com"`` or explicit coordinates.  Used
        by the renderer to determine whether the arrowhead protrudes in front
        of the host atom without an expensive nearest-neighbour search.
    """

    vector: np.ndarray  # shape (3,)
    origin: np.ndarray  # shape (3,) — resolved Cartesian position
    color: str = "#444444"
    label: str = ""
    scale: float = 1.0
    anchor: str = "tail"  # "tail" (origin = arrow tail) or "center" (origin = arrow midpoint)
    host_atom: int | None = None  # 0-based atom index, or None for com/explicit origins


@dataclass
class CellData:
    """Periodic lattice data for crystal structure rendering.

    Parameters
    ----------
    lattice:
        3x3 array where each row is a lattice vector (a, b, c) in Ångströms.
    cell_origin:
        3-vector (Å) of the (0,0,0) cell corner in the current coordinate frame.
        Defaults to the origin; updated during GIF rotation so the box keeps
        pace with the atoms.
    """

    lattice: np.ndarray  # shape (3, 3), rows = a, b, c in Å
    cell_origin: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,) in Å


# ---------------------------------------------------------------------------
# Surface parameter defaults — named constants
# ---------------------------------------------------------------------------

_DEFAULT_MO_ISOVALUE: float = 0.05
_DEFAULT_MO_POS_COLOR: str = "steelblue"
_DEFAULT_MO_NEG_COLOR: str = "maroon"
_DEFAULT_MO_BLUR_SIGMA: float = 0.8  # Gaussian sigma in 2D grid cells
_DEFAULT_MO_UPSAMPLE_FACTOR: int = 3  # upsampling multiplier (80→240 grid)

_DEFAULT_DENS_ISOVALUE: float = 0.001
_DEFAULT_DENS_COLOR: str = "steelblue"

_DEFAULT_ESP_ISOVALUE: float = 0.001

_DEFAULT_NCI_ISOVALUE: float = 0.3
_DEFAULT_NCI_COLOR: str = "forestgreen"
_DEFAULT_NCI_COLOR_MODE: str = "avg"


# ---------------------------------------------------------------------------
# Per-surface parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MOParams:
    """Parameters for MO (molecular orbital) surface rendering.

    Parameters
    ----------
    isovalue:
        Isovalue at which to extract the MO surface.
    pos_color:
        Color for the positive-phase lobe (hex or CSS4 name).
    neg_color:
        Color for the negative-phase lobe (hex or CSS4 name).
    blur_sigma:
        Gaussian blur sigma in 2D grid-cell units applied before upsampling.
    upsample_factor:
        Integer upsampling multiplier applied to the 2D projection grid.
    flat:
        Render lobes as flat-filled shapes (no depth shading).
    """

    isovalue: float = _DEFAULT_MO_ISOVALUE
    pos_color: str = _DEFAULT_MO_POS_COLOR
    neg_color: str = _DEFAULT_MO_NEG_COLOR
    blur_sigma: float = _DEFAULT_MO_BLUR_SIGMA
    upsample_factor: int = _DEFAULT_MO_UPSAMPLE_FACTOR
    flat: bool = False


@dataclass
class DensParams:
    """Parameters for electron density surface rendering.

    Parameters
    ----------
    isovalue:
        Isovalue at which to extract the density isosurface.
    color:
        Fill color for the density contour (hex or CSS4 name).
    """

    isovalue: float = _DEFAULT_DENS_ISOVALUE
    color: str = _DEFAULT_DENS_COLOR


@dataclass
class ESPParams:
    """Parameters for electrostatic potential (ESP) surface rendering.

    Parameters
    ----------
    isovalue:
        Isovalue of the density isosurface onto which ESP is mapped.
    """

    isovalue: float = _DEFAULT_ESP_ISOVALUE


@dataclass
class NCIParams:
    """Parameters for NCI (non-covalent interaction) surface rendering.

    Parameters
    ----------
    isovalue:
        Reduced density gradient isovalue for the NCI flood-fill.
    color:
        Fallback fill color when ``color_mode`` is ``'uniform'`` (hex or CSS4 name).
    color_mode:
        How to assign colors to each NCI lobe:
        ``'avg'`` uses the average sign(lambda2)*rho value per lobe,
        ``'pixel'`` maps per-pixel values (raster PNG),
        ``'uniform'`` uses ``color`` for every lobe.
    dens_cutoff:
        Optional density magnitude cutoff; voxels with density magnitude (abs(rho)) above this are
        excluded (useful for non-NCIPLOT cubes where nuclear contributions
        are not pre-removed).
    """

    isovalue: float = _DEFAULT_NCI_ISOVALUE
    color: str = _DEFAULT_NCI_COLOR
    color_mode: str = _DEFAULT_NCI_COLOR_MODE
    dens_cutoff: float | None = None


@dataclass
class RenderConfig:
    """Rendering settings."""

    canvas_size: int = 800
    padding: float = 20.0
    atom_scale: float = 1.0
    atom_stroke_width: float = 1.5
    atom_stroke_color: str = "black"
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
    transparent: bool = False
    dpi: int = 300
    fixed_span: float | None = None  # fixed viewport span (disables auto-fit)
    fixed_center: tuple[float, float] | None = None  # fixed XY center (disables auto-center)
    color_overrides: dict[str, str] | None = None  # element symbol → hex color
    # Surface rendering (MO / density / ESP / NCI share one opacity)
    mo_contours: SurfaceContours | None = None
    dens_contours: SurfaceContours | None = None
    esp_surface: ESPSurface | None = None
    nci_contours: NCIContours | None = None
    surface_opacity: float = 1.0
    flat_mo: bool = False
    # Annotations and measurements
    annotations: list[Annotation] = field(default_factory=list)
    show_indices: bool = False
    idx_format: str = "sn"  # "sn" (C1) | "s" (C) | "n" (1) — 1-indexed numbers
    label_font_size: float = 11.0
    label_color: str = "#222222"
    label_offset: float = 0.5  # perpendicular label offset as a fraction of font size (bond: -, dihedral: +)
    # Atom property colormap (--cmap)
    atom_cmap: dict[int, float] | None = None
    cmap_range: tuple[float, float] | None = None
    cmap_unlabeled: str = "#ffffff"  # fill for atoms absent from cmap file
    # Surface parameter defaults (populated from preset by build_config)
    mo_isovalue: float = _DEFAULT_MO_ISOVALUE
    mo_pos_color: str = _DEFAULT_MO_POS_COLOR
    mo_neg_color: str = _DEFAULT_MO_NEG_COLOR
    mo_blur_sigma: float = _DEFAULT_MO_BLUR_SIGMA
    mo_upsample_factor: int = _DEFAULT_MO_UPSAMPLE_FACTOR
    dens_isovalue: float = _DEFAULT_DENS_ISOVALUE
    dens_color: str = _DEFAULT_DENS_COLOR
    nci_isovalue: float = _DEFAULT_NCI_ISOVALUE
    nci_color: str = _DEFAULT_NCI_COLOR
    nci_color_mode: str = _DEFAULT_NCI_COLOR_MODE
    # Overlay
    overlay_color: str = "mediumorchid"
    # Crystal / periodic structure
    cell_data: CellData | None = None
    show_cell: bool = True
    show_crystal_axes: bool = True
    cell_color: str = "#333333"
    cell_line_width: float = 2.0
    periodic_image_opacity: float = 0.5
    axis_colors: tuple[str, str, str] = (
        "firebrick",
        "forestgreen",
        "royalblue",
    )  # firebrick, forestgreen, royalblue
    axis_width_scale: float = 3.0  # multiplier on cell_line_width for axis stroke width
    # Arbitrary vector arrows (--vectors)
    vectors: list[VectorArrow] = field(default_factory=list)
    vector_scale: float = 1.0  # global length multiplier applied to all vectors
    vector_color: str = "firebrick"  # default arrow color (firebrick) when not specified per-arrow
