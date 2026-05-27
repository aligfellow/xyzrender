"""Core types for xyzrender."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import os

    from xyzrender.annotations import Annotation
    from xyzrender.contours import SurfaceContours
    from xyzrender.esp import ESPSurface
    from xyzrender.nci import NCIContours


class BondStyle(Enum):
    """Visual bond style."""

    SOLID = "solid"
    DASHED = "dashed"  # TS bonds
    DOTTED = "dot"  # NCI bonds


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
    draw_on_top: bool = False
    is_axis: bool = False  # True for crystallographic axis arrows (not affected by vector_scale)
    font_size: float | None = None
    width: float | None = None


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
_DEFAULT_NCI_MODE: str = "avg"


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
    color: str = "forestgreen"
    color_mode: str = "avg"
    dens_cutoff: float | None = None


@dataclass
class HighlightGroup:
    """A group of atoms to highlight with a specific color."""

    indices: list[int]  # 0-indexed atom indices
    color: str  # resolved hex color
    _index_set: set[int] = field(default_factory=set, repr=False, init=False)

    def __post_init__(self) -> None:
        """Pre-compute index set for O(1) membership checks."""
        self._index_set = set(self.indices)


@dataclass
class OverlayConfig:
    """Per-overlay style overrides.

    Scalar fields (``color``, ``atom_scale``, ``bond_width``, stroke / outline
    variants) are the CLI- and preset-friendly shortcuts.  All are absolute —
    same semantics as :class:`RenderConfig` — and ``None``/empty means inherit
    the primary.  For fields without a dedicated shortcut (``bond_gap``, fog,
    gradient strengths, ``skeletal_style``, ...), set ``config`` to a full
    :class:`RenderConfig`; it's attached as a :class:`StyleRegion` over the
    overlay's node IDs at merge time.  Scalar shortcuts still win over values
    inside ``config`` because the renderer applies per-node / per-edge override
    attributes on top of the style-region config.
    """

    color: str = "mediumorchid"
    opacity: float | None = None
    atom_scale: float | None = None
    atom_stroke_width: float | None = None
    atom_stroke_color: str | None = None  # "atom" literal = use per-element fill
    bond_width: float | None = None
    bond_color: str | None = None  # explicit; overrides the auto-darkened `color`
    bond_outline_width: float | None = None
    bond_outline_color: str | None = None
    unbond: list[str] = field(default_factory=list)  # overlay-only bond-hide rules
    bond: list[str] = field(default_factory=list)  # overlay-only bond force-show / add
    show: list[str] = field(default_factory=list)  # visibility filter (selector grammar)
    config: "RenderConfig | None" = None  # full escape hatch; see class docstring


@dataclass
class RenderConfig:
    """Rendering settings."""

    canvas_size: int = 800
    padding: float = 20.0
    atom_scale: float = 1.0
    radius_scale: list[tuple[str | list[int], float]] = field(
        default_factory=list
    )  # per-atom radius scale: [("N,M", 2.0), ([1,2], 3.0), ...]
    atom_stroke_width: float = 1.5
    atom_stroke_color: str = "black"  # "atom" = use per-atom element color
    atom_wash: float = 0.0  # blend atom fill toward white (0=none, 0.78=graph-style tint)
    atoms_above_bonds: bool = False  # draw atoms after all bonds for diagram aesthetic
    bond_width: float = 5.0
    bond_color: str = "#333333"
    ts_color: str | None = None  # dashed TS bonds; None -> use bond_color
    nci_color: str | None = None  # dotted NCI bonds; None -> use bond_color
    bond_gap: float = 0.6  # multi-bond spacing as fraction of bond_width
    bond_color_by_element: bool = False  # color bonds by endpoint atom colors
    bond_gradient: bool = False  # cylinder shading on bonds (perpendicular gradient for 3D tube look)
    bond_gradient_strength: float = 0.3  # lighten/darken strength for bond shading (cf. vdw_gradient_strength)
    bond_outline_color: str = "#000000"  # stroke colour drawn behind bonds
    bond_outline_width: float = 0.0  # absolute px; 0 = off, visible border = this value on each side
    gradient: bool = False
    atom_gradient_strength: float = 1.0  # atom gradient intensity
    hue_shift_factor: float = 0.2
    light_shift_factor: float = 0.2
    saturation_shift_factor: float = 0.2
    fog: bool = False
    fog_strength: float = 0.8
    hide_bonds: bool = False
    unbond: list[str] = field(default_factory=list)  # raw specs: ["M-L", "sbm", "1-3"]
    bond: list[str] = field(default_factory=list)  # raw index pairs: ["4-5"]
    haptic: bool = False  # replace pi-coordination bonds with single metal-to-centroid bond
    hide_h: bool = False
    show_h_indices: list[int] = field(default_factory=list)
    auto_hide_h: bool = True  # CLI default --hy=None hides C-H; set False in presets that want H always visible
    bond_orders: bool = True
    ts_bonds: list[tuple[int, int]] = field(default_factory=list)  # 0-indexed pairs
    nci_bonds: list[tuple[int, int]] = field(default_factory=list)  # 0-indexed pairs
    vdw_indices: list[int] | None = None
    vdw_opacity: float = 0.5
    vdw_scale: float = 1.0
    vdw_gradient_strength: float = 1.6  # strength for VdW sphere gradient darken
    vdw_interlocking: bool = False  # interlocked silhouettes for the --vdw overlay layer
    atom_interlocking: bool = False  # interlocked silhouettes for primary atom spheres (e.g. --config vdw)
    vdw_interlock_samples: int = (
        64  # perimeter samples per sphere on the silhouette great circle (shared by both paths)
    )
    vdw_outline_width: float | None = None  # overlay outline width; None inherits atom_stroke_width, 0 = no outline
    vdw_outline_color: str | None = None  # overlay outline colour; None inherits atom_stroke_color
    vdw_h_scale: float = 0.7  # hydrogen radius shrink on the --vdw overlay; independent of primary h_scale
    h_scale: float = 0.6  # hydrogen radius shrink for primary atom rendering (~0.6 ball-and-stick, ~0.8 CPK)
    auto_orient: bool = False
    # Kabsch/MCS alignment applied when merging an overlay or ensemble frames.
    # Disable with --no-align to keep each structure's raw coordinates.
    auto_align: bool = True
    background: str = "#ffffff"
    transparent: bool = False
    dpi: int = 300
    fixed_span: float | None = None  # fixed viewport span (disables auto-fit)
    fixed_center: tuple[float, float] | None = None  # fixed XY center (disables auto-center)
    color_overrides: dict[str, str] | None = None  # element symbol → hex color
    mol_color: str | None = None  # flat color for all atoms + bonds (overrides CPK; highlight paints on top)
    # Surface rendering (MO / density / ESP / NCI share one opacity)
    mo_contours: SurfaceContours | None = None
    dens_contours: SurfaceContours | None = None
    esp_surface: ESPSurface | None = None
    nci_contours: NCIContours | None = None
    surface_opacity: float = 1.0
    flat_mo: bool = False
    surface_style: str = "solid"  # "solid" | "mesh" | "contour" | "dot"
    # Annotations and measurements
    annotations: list[Annotation] = field(default_factory=list)
    show_indices: bool = False
    idx_format: str = "sn"  # "sn" (C1) | "s" (C) | "n" (1) — 1-indexed numbers
    label_font_size: float = 11.0
    label_color: str = "#222222"
    label_offset: float = 0.5  # perpendicular label offset as a fraction of font size (bond: -, dihedral: +)
    # Per-atom fill opacity override (0-indexed).  Affects the atom circle only;
    # adjacent bonds remain fully opaque so connectivity stays readable.
    atom_opacity: dict[int, float] = field(default_factory=dict)
    # Atom property colormap (--cmap)
    atom_cmap: dict[int, float] | None = None
    cmap_range: tuple[float, float] | None = None
    cmap_symm: bool = False  # symmetric range about 0: [-max(|v|), +max(|v|)]
    cmap_unlabeled: str = "#ffffff"  # fill for atoms absent from cmap file
    cmap_palette: str | None = None
    cbar: bool = False  # show a vertical colorbar on the right
    # Surface parameter defaults (populated from preset by build_config)
    mo_isovalue: float = _DEFAULT_MO_ISOVALUE
    mo_pos_color: str = _DEFAULT_MO_POS_COLOR
    mo_neg_color: str = _DEFAULT_MO_NEG_COLOR
    mo_blur_sigma: float = _DEFAULT_MO_BLUR_SIGMA
    mo_upsample_factor: int = _DEFAULT_MO_UPSAMPLE_FACTOR
    mo_outline_width: float = 0.0  # 0 = no outline (solid surface mode only)
    mo_outline_color: str = "#000000"
    dens_isovalue: float = _DEFAULT_DENS_ISOVALUE
    dens_color: str = _DEFAULT_DENS_COLOR
    nci_isovalue: float = _DEFAULT_NCI_ISOVALUE
    nci_mode: str = _DEFAULT_NCI_MODE
    # Highlight (atom group coloring)
    highlight_groups: list[HighlightGroup] = field(default_factory=list)
    highlight_colors: list[str] = field(
        default_factory=lambda: [
            "orchid",
            "mediumseagreen",
            "goldenrod",
            "coral",
            "mediumpurple",
            "hotpink",
        ]
    )
    # Depth of field
    dof: bool = False
    dof_strength: float = 3.0  # max blur stdDeviation in SVG units
    # Atom glow (blurred duplicate circle rendered under selected atoms)
    glow_indices: list[int] = field(default_factory=list)  # 0-indexed
    glow_strength: float = 5.0
    # Overlay (nested sub-config so new per-overlay knobs slot in without churn)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    # Ensemble
    ensemble_colors: list[str] | None = None  # resolved hex per conformer (None = CPK)
    # Skeletal formula line rendering
    skeletal_style: bool = False
    skeletal_label_color: str | None = None  # override all element labels (None = per-element CPK)
    # Crystal / periodic structure
    cell_data: CellData | None = None
    show_cell: bool = True
    cell_color: str = "#333333"
    cell_line_width: float = 2.0
    periodic_image_opacity: float = 0.5
    axis_colors: tuple[str, str, str] = (
        "firebrick",
        "forestgreen",
        "royalblue",
    )  # firebrick, forestgreen, royalblue
    axis_width_scale: float = 3.0  # multiplier on cell_line_width for axis stroke width
    # Arbitrary vector arrows (--vector)
    vectors: list[VectorArrow] = field(default_factory=list)
    vector_scale: float = 1.0  # global length multiplier applied to all vectors
    vector_color: str = "firebrick"  # default arrow color (firebrick) when not specified per-arrow
    # Convex hull facets (low-alpha plane behind molecule)
    show_convex_hull: bool = False
    hull_opacity: float = 0.2
    hull_colors: list[str] = field(
        default_factory=lambda: [
            "steelblue",
            "firebrick",
            "mediumseagreen",
            "mediumpurple",
            "darkgoldenrod",
            "cadetblue",
        ]
    )
    hull_atom_indices: list[int] | list[list[int]] | None = None
    # If None, hull uses all non-dummy atoms. If a flat list of ints, one subset (e.g. ring carbons).
    # If a list of lists, multiple subsets: each inner list is 0-based atom indices for one hull.
    # Auto-set True for "faces"/"pores"/"rings" modes — tells renderer to use
    # ordered polygon (triangle fan) instead of convex hull for each subset.
    hull_ordered: bool = False
    # Non-bond hull edges (1-skeleton) drawn as thin lines for better 3D perception.
    # Edge color is auto-derived as a darkened shade of the hull fill color.
    show_hull_edges: bool = True
    hull_edge_width_ratio: float = 0.4  # stroke width = bond_width * this
    pore_spheres: bool = False  # draw inscribed sphere at each pore centroid
    pore_sphere_opacity: float = 0.5
    pore_sphere_color: str = "#f0d060"  # warm yellow default
    # Pore geometry: node-ID lists per pore (for hull drawing).
    pore_node_ids: list[list[int]] | None = None
    # Pore sphere placement: true centroid + radius from coarse-grained detection.
    # Transformed by PCA rotation in the renderer for correct placement.
    pore_centroids: list[tuple[float, float, float]] | None = None
    pore_radii: list[float] | None = None
    # Style regions: render subsets of atoms with a different preset/config
    style_regions: list[StyleRegion] = field(default_factory=list)
    # Preset-defined regions: {"M": "flat"} or {"M": {"atom_scale": 4.0}} resolved at render time
    region_specs: dict[str, str | dict] | None = None


@dataclass
class StyleRegion:
    """A subset of atoms rendered with a different visual style.

    Only per-atom/bond fields are used (atom_scale, gradient, bond_width,
    etc.); global fields (canvas_size, background, fog, surfaces) are
    taken from the base config.
    """

    indices: list[int]  # 0-indexed atom indices
    config: RenderConfig  # resolved config for this region
    _index_set: set[int] = field(default_factory=set, repr=False, init=False)

    def __post_init__(self):
        """Pre-compute index set for O(1) membership checks."""
        self._index_set = set(self.indices)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class SVGResult:
    """Wraps a rendered SVG string with Jupyter display and file-save support."""

    def __init__(self, svg: str) -> None:
        self._svg = svg

    def __str__(self) -> str:
        """Return the raw SVG string."""
        return self._svg

    def _repr_svg_(self) -> str:
        """Return the SVG string for Jupyter inline display, scaled to max 500 px wide."""
        return re.sub(
            r'(<svg\b[^>]*?)\s+width="[^"]*"\s+height="[^"]*"',
            r'\1 width="500" height="auto"',
            self._svg,
            count=1,
        )

    def save(self, path: str | os.PathLike) -> None:
        """Write the SVG to *path* (must end with ``.svg``)."""
        Path(path).write_text(self._svg)


class GIFResult:
    """Wraps a rendered GIF path with Jupyter inline display support."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._bytes: bytes | None = None

    @property
    def path(self) -> Path:
        """Path to the GIF file on disk."""
        return self._path

    def __repr__(self) -> str:
        """Return a string representation of the GIFResult."""
        return f"GIFResult(path={self._path!r})"

    def __bytes__(self) -> bytes:
        """Return the raw GIF bytes."""
        if self._bytes is None:
            self._bytes = self._path.read_bytes()
        return self._bytes

    def save(self, path: str | os.PathLike) -> None:
        """Write the GIF to *path*."""
        data = bytes(self)
        Path(path).write_bytes(data)

    def _repr_html_(self) -> str:
        """Embed the GIF inline in Jupyter, capped to 500 px wide."""
        data = base64.b64encode(bytes(self)).decode("ascii")
        return f'<img src="data:image/gif;base64,{data}" width="500" style="height:auto"/>'
