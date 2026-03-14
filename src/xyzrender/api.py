"""High-level Python API for xyzrender.

Typical usage in a Jupyter notebook::

    from xyzrender import load, render, render_gif

    mol = load("mol.xyz")
    render(mol)  # displays inline in Jupyter
    render(mol, hy=True)  # show all hydrogens
    render(mol, atom_scale=1.5, bond_width=8)
    render(mol, mo=True, iso=0.05)  # MO surface (mol loaded from .cube)
    render(mol, nci="grad.cube")  # NCI surface

    # Short-form path string (loads with defaults):
    render("mol.xyz")

    # Reuse a style config:
    cfg = build_config("flat", atom_scale=1.5)
    render(mol1, config=cfg)
    render(mol2, config=cfg)

For GIFs use :func:`render_gif`::

    render_gif("mol.xyz", gif_rot="y")
    render_gif("trajectory.xyz", gif_trj=True)
    render_gif("ts.xyz", gif_ts=True)
"""

from __future__ import annotations

import base64
import copy
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import os

    import networkx as nx

    from xyzrender.cube import CubeData
    from xyzrender.types import CellData, RenderConfig, VectorArrow

from xyzrender.types import resolve_color

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core types
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


@dataclass
class EnsembleFrames:
    """Per-conformer data for an ensemble loaded with ``load(ensemble=True)``.

    Kept separate from ``Molecule.graph`` (which holds only the reference frame)
    so the graph always represents a single n_atoms structure regardless of
    ensemble size.  Consumers (render, render_gif) build the merged multi-
    conformer graph lazily from these arrays.

    Attributes
    ----------
    positions:
        Stacked conformer positions, shape ``(n_conformers, n_atoms, 3)``.
        All frames are RMSD-aligned onto the reference frame.
        Contiguous memory allows vectorised rotation across all conformers
        simultaneously (single matmul for GIF frames).
    colors:
        Resolved hex color string per conformer (``None`` = use CPK).
    opacities:
        Per-conformer opacity override (``None`` = fully opaque).
    conformer_graphs:
        Optional per-frame graphs for ``rebuild=True`` ensembles (topology
        can differ per frame).  ``None`` means all frames share the reference
        topology.
    reference_idx:
        Index into *positions* / *colors* / *opacities* that is the reference.
    """

    positions: np.ndarray  # shape (n_conformers, n_atoms, 3)
    colors: list[str | None]  # per-conformer hex color
    opacities: list[float | None]  # per-conformer opacity
    conformer_graphs: list[nx.Graph] | None = None
    reference_idx: int = 0


@dataclass
class Molecule:
    """Container for a loaded molecular structure.

    Obtain via :func:`load`.  Pass directly to :func:`render` or
    :func:`render_gif` to avoid re-parsing the file.

    For ensemble molecules (``load(ensemble=True)``), ``graph`` holds only the
    reference conformer (n_atoms nodes).  The full per-conformer data lives in
    ``ensemble``; the merged multi-conformer graph is built lazily at render time.
    """

    graph: nx.Graph
    cube_data: CubeData | None = None
    cell_data: CellData | None = None
    oriented: bool = False
    ensemble: EnsembleFrames | None = None

    def to_xyz(self, path: str | os.PathLike, title: str = "") -> None:
        """Write the molecule to an XYZ file.

        If the molecule carries ``cell_data`` (e.g. loaded with ``cell=True``
        or ``crystal=...``), the file is written in extXYZ format with a
        ``Lattice=`` header so it can be reloaded with ``load(..., cell=True)``.
        Ghost (periodic image) atoms are excluded.

        Parameters
        ----------
        path:
            Output path — should end with ``.xyz``.
        title:
            Comment line written as the second line of the file.
        """
        if not path or not str(path).strip():
            msg = "to_xyz: output path cannot be empty"
            raise ValueError(msg)
        if not str(path).lower().endswith(".xyz"):
            logger.warning("to_xyz: output path does not end with .xyz: %s", path)
        nodes = [(i, self.graph.nodes[i]) for i in self.graph.nodes() if self.graph.nodes[i].get("symbol", "") != "*"]

        lines: list[str] = [f"{len(nodes)}\n"]

        if self.cell_data is not None:
            lat = self.cell_data.lattice  # shape (3, 3), rows = a, b, c in Å
            flat = " ".join(f"{v:.10g}" for v in lat.ravel())
            header = f'Lattice="{flat}" Properties=species:S:1:pos:R:3'
            if title:
                header = f"{header} # {title}"
            lines.append(header + "\n")
        else:
            lines.append((title or "") + "\n")

        for _, data in nodes:
            sym = data["symbol"]
            x, y, z = data["position"]
            lines.append(f"{sym:<3} {x:15.8f} {y:15.8f} {z:15.8f}\n")

        Path(path).write_text("".join(lines))


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def load(
    molecule: str | os.PathLike,
    *,
    smiles: bool = False,
    charge: int = 0,
    multiplicity: int | None = None,
    kekule: bool = False,
    rebuild: bool = False,
    mol_frame: int = 0,
    ts_detect: bool = False,
    ts_frame: int = 0,
    nci_detect: bool = False,
    crystal: bool | str = False,
    cell: bool = False,
    quick: bool = False,
    # --- Ensemble (multi-frame trajectory) ---
    ensemble: bool = False,
    reference_frame: int = 0,
    max_frames: int | None = None,
    align_atoms: list[int] | None = None,
    ensemble_color: str | list[str] | None = None,
    ensemble_palette: str | None = None,
    ensemble_opacity: float | None = None,
    reference_mol: Molecule | None = None,
) -> Molecule:
    """Load a molecule from file (or SMILES string) and return a :class:`Molecule`.

    Parameters
    ----------
    molecule:
        Path to the input file, or a SMILES string when *smiles* is ``True``.
        Supported extensions: ``.xyz``, ``.cube``, ``.mol``, ``.sdf``,
        ``.mol2``, ``.pdb``, ``.smi``, ``.cif``, and any QM output
        supported by cclib.
    smiles:
        Treat *molecule* as a SMILES string and generate 3-D geometry.
    charge:
        Formal molecular charge (0 = read from file when available).
    multiplicity:
        Spin multiplicity (``None`` = read from file).
    kekule:
        Convert aromatic bonds to alternating single/double (Kekulé form).
    rebuild:
        Force xyzgraph distance-based bond detection even when the file
        provides explicit connectivity.  When used with ``ensemble=True``,
        each frame's graph is rebuilt independently (for trajectories where
        bonding changes between frames).
    mol_frame:
        Zero-based frame index for multi-record SDF files.
    ts_detect:
        Run graphRC transition-state detection (requires ``xyzrender[ts]``).
    ts_frame:
        Reference frame index for TS detection in multi-frame files.
    nci_detect:
        Detect non-covalent interactions with xyzgraph after loading.
        When used with ``ensemble=True``, NCI detection is run on
        each frame independently.
    crystal:
        Load as a periodic crystal structure via phonopy.  Pass ``True``
        to auto-detect the interface from the filename, or a string such
        as ``"vasp"`` or ``"qe"`` to specify explicitly.
    cell:
        Read the periodic cell box from an extXYZ ``Lattice=`` header and
        store it on the returned :class:`Molecule`.
    quick:
        Skip bond-order optimisation (``build_graph(quick=True)``).  Use
        when you know bond orders will be suppressed at render time (e.g.
        ``render(mol, bo=False)``).  CIF and PDB-with-cell always use
        ``quick=True`` automatically regardless of this flag.
    ensemble:
        Load as a multi-frame trajectory ensemble.  All frames are
        RMSD-aligned onto *reference_frame* and merged into a single graph.
    reference_frame:
        Index of the reference frame for ensemble alignment (default: 0).
    max_frames:
        Maximum number of frames to include (default: all).
    align_atoms:
        0-indexed atom indices for Kabsch alignment subset (min 3).
        When given, the rotation is computed from this subset only
        but applied to all atoms.
    ensemble_color:
        Single color string or list of hex/named colors for conformers.
    ensemble_palette:
        Named continuous colormap (``"viridis"``, ``"spectral"``,
        ``"coolwarm"``).  Overrides *ensemble_color*.
    ensemble_opacity:
        Opacity for non-reference conformer atoms (0-1).
    reference_mol:
        Optional pre-loaded (and possibly oriented) :class:`Molecule` for the
        reference frame.  When given, its graph and positions are used directly
        instead of loading the reference frame from *molecule*.  This lets
        interactive orientation be applied before ensemble alignment.

    Returns
    -------
    Molecule
    """
    # --- Ensemble: load multi-frame trajectory as merged molecule ---
    if ensemble:
        return _build_ensemble_molecule(
            molecule,
            reference_frame=reference_frame,
            max_frames=max_frames,
            align_atoms=align_atoms,
            conformer_colors=_resolve_ensemble_colors(
                ensemble_color=ensemble_color,
                ensemble_palette=None,
                n_conformers=None,
            ),
            ensemble_opacity=ensemble_opacity,
            ensemble_palette=ensemble_palette,
            charge=charge,
            multiplicity=multiplicity,
            kekule=kekule,
            rebuild=rebuild,
            quick=quick,
            nci_detect=nci_detect,
            reference_mol=reference_mol,
        )

    import xyzrender.parsers as fmt
    from xyzrender.readers import graph_from_moldata

    mol_path = Path(str(molecule))
    cube_data = None
    cell_data = None
    graph = None

    if smiles:
        # molecule is a SMILES string
        logger.info("Loading SMILES: %s", molecule)
        data = fmt.parse_smiles(str(molecule), kekule=kekule)
        graph = graph_from_moldata(
            data, charge=charge, multiplicity=multiplicity, kekule=kekule, rebuild=rebuild, quick=quick
        )

    elif crystal:
        interface_mode = _resolve_crystal_interface(mol_path, crystal)
        from xyzrender.crystal import load_crystal

        graph, cell_data = load_crystal(mol_path, interface_mode)

    elif mol_path.suffix.lower() == ".cube":
        from xyzrender.readers import load_cube

        graph, cube_data = load_cube(mol_path, charge=charge, multiplicity=multiplicity, kekule=kekule, quick=quick)

    elif ts_detect:
        from xyzrender.readers import load_ts_molecule

        graph, _frames = load_ts_molecule(
            mol_path,
            charge=charge,
            multiplicity=multiplicity,
            ts_frame=ts_frame,
            kekule=kekule,
        )

    else:
        from xyzrender.readers import load_molecule

        graph, cell_data = load_molecule(
            mol_path,
            frame=mol_frame,
            charge=charge,
            multiplicity=multiplicity,
            kekule=kekule,
            rebuild=rebuild,
            quick=quick,
        )

    # Auto-promote: any file that carried lattice data (extXYZ Lattice=, PDB CRYST1, CIF)
    # exposes it as cell_data so render() applies crystal display automatically.
    if cell_data is None and graph is not None and "lattice" in graph.graph:
        from xyzrender.types import CellData

        cell_data = CellData(
            lattice=np.array(graph.graph["lattice"], dtype=float),
            cell_origin=np.array(graph.graph.get("lattice_origin", np.zeros(3)), dtype=float),
        )
    elif cell and cell_data is None:
        logger.warning("load(..., cell=True): no Lattice= found in input file")

    if nci_detect:
        from xyzrender.readers import detect_nci

        graph = detect_nci(graph)

    return Molecule(graph=graph, cube_data=cube_data, cell_data=cell_data)


def orient(mol: Molecule) -> None:
    """Open molecule in v viewer to set orientation interactively.

    The user rotates the molecule and presses ``z`` to output coordinates,
    then ``q`` to quit.  Atom positions are written back to ``mol.graph``
    in-place.  Sets ``mol.oriented = True`` so subsequent :func:`render`
    calls skip PCA auto-orientation.

    For cube-file molecules the cube grid alignment is handled automatically
    at render time via Kabsch rotation from original cube atom positions to
    the updated graph positions.

    Parameters
    ----------
    mol:
        Molecule returned by :func:`load`.
    """
    from xyzrender.viewer import rotate_with_viewer

    rot, _c1, _c2 = rotate_with_viewer(mol.graph)
    if rot is None:
        logger.warning("orient(): no orientation received from viewer; mol.oriented not set")
        return

    # Cube grid alignment is handled automatically by resolve_orientation() at
    # render time via Kabsch rotation from original cube atoms → rotated graph.
    # Re-sync cell_data from the rotated graph lattice (rotate_with_viewer updates
    # graph.graph["lattice"] in-place but mol.cell_data was built before rotation).
    if mol.cell_data is not None and "lattice" in mol.graph.graph:
        mol.cell_data.lattice = np.array(mol.graph.graph["lattice"], dtype=float)
        mol.cell_data.cell_origin = np.array(mol.graph.graph.get("lattice_origin", [0, 0, 0]), dtype=float)
    mol.oriented = True


def measure(
    molecule: str | os.PathLike | Molecule,
    modes: list[str] | None = None,
) -> dict:
    """Return geometry measurements as a dict.

    Parameters
    ----------
    molecule:
        A :class:`Molecule` object or a file path (loaded with defaults).
    modes:
        Subset of ``["d", "a", "t"]`` for distances, angles, dihedrals.
        ``None`` (default) returns all three.

    Returns
    -------
    dict with keys ``"distances"``, ``"angles"``, ``"dihedrals"``.
    """
    if isinstance(molecule, Molecule):
        graph = molecule.graph
    else:
        graph = load(molecule).graph

    from xyzrender.measure import all_bond_angles, all_bond_lengths, all_dihedrals

    result: dict = {}
    active = set(modes) if modes is not None else {"d", "a", "t"}
    if "d" in active:
        result["distances"] = all_bond_lengths(graph)
    if "a" in active:
        result["angles"] = all_bond_angles(graph)
    if "t" in active:
        result["dihedrals"] = all_dihedrals(graph)
    return result


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render(
    molecule: str | os.PathLike | Molecule,
    *,
    config: str | RenderConfig = "default",
    # --- Style (only when config is a preset name or file path) ---
    canvas_size: int | None = None,
    atom_scale: float | None = None,
    bond_width: float | None = None,
    atom_stroke_width: float | None = None,
    bond_color: str | None = None,
    background: str | None = None,
    transparent: bool = False,
    gradient: bool | None = None,
    hue_shift_factor: float | None = None,
    light_shift_factor: float | None = None,
    saturation_shift_factor: float | None = None,
    fog: bool | None = None,
    fog_strength: float | None = None,
    label_font_size: float | None = None,
    vdw_opacity: float | None = None,
    vdw_scale: float | None = None,
    vdw_gradient_strength: float | None = None,
    # --- Display ---
    hy: bool | list[int] | None = None,
    no_hy: bool = False,
    bo: bool | None = None,
    orient: bool | None = None,
    # --- Crystal display (when mol has cell_data) ---
    no_cell: bool = False,
    axes: bool = True,
    axis: str | None = None,
    ghosts: bool | None = None,
    cell_color: str | None = None,
    cell_width: float | None = None,
    ghost_opacity: float | None = None,
    # --- Rendering overlays (1-indexed atom numbering) ---
    ts_bonds: list[tuple[int, int]] | None = None,
    nci_bonds: list[tuple[int, int]] | None = None,
    vdw: bool | list[int] | None = None,
    idx: bool | str = False,
    cmap: str | os.PathLike | dict[int, float] | None = None,
    cmap_range: tuple[float, float] | None = None,
    # --- Annotations ---
    labels: list[str] | None = None,
    label_file: str | None = None,
    # --- Vector arrows ---
    vector: str | Path | dict | list[VectorArrow] | None = None,
    vector_scale: float | None = None,
    vector_color: str | None = None,
    # --- Surface opacity ---
    opacity: float | None = None,
    # --- Surfaces ---
    mo: bool = False,
    dens: bool = False,
    esp: str | os.PathLike | None = None,
    nci: str | os.PathLike | None = None,
    iso: float | None = None,
    mo_pos_color: str | None = None,
    mo_neg_color: str | None = None,
    mo_blur: float | None = None,
    mo_upsample: int | None = None,
    flat_mo: bool = False,
    dens_color: str | None = None,
    nci_color: str | None = None,
    nci_coloring: str | None = None,
    nci_cutoff: float | None = None,
    # --- Convex hull ---
    hull: bool | str | list[int] | list[list[int]] | None = None,
    hull_color: str | list[str] | None = None,
    hull_opacity: float | None = None,
    hull_edge: bool | None = None,
    hull_edge_width_ratio: float | None = None,
    # --- Overlay ---
    overlay: str | os.PathLike | Molecule | None = None,
    overlay_color: str | None = None,
    # --- Alignment (overlay subset alignment) ---
    align_atoms: list[int] | None = None,
    # --- Output ---
    output: str | os.PathLike | None = None,
) -> SVGResult:
    """Render a molecule to SVG and return an :class:`SVGResult`.

    In a Jupyter cell the result displays inline automatically via
    ``_repr_svg_()``.  Pass *output* to save to disk at the same time.

    Parameters
    ----------
    molecule:
        A :class:`Molecule` from :func:`load`, or a file path (loaded with
        defaults).
    config:
        Config preset name (``"default"``, ``"flat"``, …), path to a JSON
        config file, or a pre-built :class:`~xyzrender.types.RenderConfig`
        from :func:`build_config`.  Style kwargs below are only applied when
        *config* is a string.
    orient:
        ``True`` / ``False`` to force / suppress PCA auto-orientation.
        ``None`` (default) enables auto-orientation, unless the molecule was
        manually oriented via :func:`orient`.
    ts_bonds, nci_bonds:
        Manual TS / NCI bond overlays as 1-indexed atom pairs.
    vdw:
        VdW sphere display.  ``True`` = all atoms; a list of 1-indexed atom
        indices = specific atoms; ``None`` = off (default).
    idx:
        Atom index labels.  ``True`` or ``"sn"`` (e.g. ``C1``); ``"s"``
        (element only); ``"n"`` (number only).
    cmap:
        Atom property colour map: either a ``{1-indexed atom: value}`` dict,
        or a path to a two-column text file (index value, same format as
        ``--cmap`` in the CLI).
    labels:
        Inline annotation spec strings (e.g. ``["1 2 d", "3 a", "1 NBO"]``).
    label_file:
        Path to an annotation file (same format as ``--label``).
    vectors:
        Vector arrows to overlay.  Pass a path/dict to a JSON file, or a list
        of :class:`xyzrender.types.VectorArrow` objects.  Each arrow is drawn
        as a shaft + filled arrowhead pointing from ``origin`` in the direction
        of ``vector``.  When the 2D projected length is shorter than the
        arrowhead size (i.e. the arrow points nearly along the viewing axis), a
        compact symbol is drawn instead: a filled dot (•) when the tip is closer
        to the viewer, or a cross (x) when it points away.  The label is
        suppressed in these cases and reappears automatically once the arrow is
        long enough to draw a proper arrowhead.
    mo, dens:
        Render MO lobes / density isosurface from a cube file loaded via
        :func:`load`.
    esp:
        Path to an ESP ``.cube`` file (density iso + ESP colour map).
    nci:
        Path to an NCI reduced-density-gradient ``.cube`` file.
    hull:
        ``True`` = hull over all heavy atoms; ``"rings"`` = one hull per
        aromatic ring (auto-detected from the molecular graph); a flat list
        of 1-indexed atom indices (one hull, e.g. ``[1,2,3,4,5,6]``); a list
        of lists (multiple hulls, e.g. ``[[1,2,3,4,5,6], [7,8,9]]``).
        ``None`` (default) = off.
    hull_color:
        A single color string for all hulls, or a list of colors for per-subset
        colouring (one per subset).  Hex or named color.
    hull_opacity:
        Fill opacity for all hull surfaces.
    hull_edge, hull_edge_width_ratio:
        Draw hull edges that are not bonds as thin lines.

    Returns
    -------
    SVGResult
        Wrapper around the SVG string.  Displays inline in Jupyter.
    """
    from xyzrender.config import build_config, build_surface_params, collect_surf_overrides
    from xyzrender.renderer import render_svg

    # --- Early parameter validation ---
    if transparent and background is not None:
        logger.warning("transparent and background are mutually exclusive; transparent takes precedence")
    if isinstance(idx, str) and idx not in {"sn", "s", "n"}:
        msg = f"idx: unknown format {idx!r} (valid: 'sn', 's', 'n')"
        raise ValueError(msg)

    # --- Load if path ---
    if isinstance(molecule, Molecule):
        mol = molecule
    else:
        mol = load(molecule)

    # Detect ensemble (mol.ensemble is populated by load(ensemble=True))
    _is_ensemble = mol.ensemble is not None
    if _is_ensemble:
        if overlay is not None:
            msg = "ensemble cannot be combined with overlay="
            raise ValueError(msg)
        if mo or dens or esp is not None or nci is not None:
            msg = "ensemble: surface rendering (mo/dens/esp/nci) is not supported"
            raise ValueError(msg)
        # Ensemble defaults: show all H, hide bond orders (unless explicitly set)
        if hy is None and not no_hy:
            hy = True
        if bo is None:
            bo = False

    # --- Orient resolution ---
    # orient=None: auto-orient, but skip if mol was manually oriented
    _orient: bool | None = orient
    if _orient is None and mol.oriented:
        _orient = False

    # --- Config resolution ---
    if not isinstance(config, str):
        # Pre-built RenderConfig — shallow copy so we don't mutate the caller's object
        cfg = copy.copy(config)
        cfg.vectors = list(cfg.vectors)
        if _orient is not None:
            cfg.auto_orient = _orient
        elif mol.oriented:
            cfg.auto_orient = False
    else:
        cfg = build_config(
            config,
            canvas_size=canvas_size,
            atom_scale=atom_scale,
            bond_width=bond_width,
            atom_stroke_width=atom_stroke_width,
            bond_color=bond_color,
            background=background,
            transparent=transparent,
            gradient=gradient,
            hue_shift_factor=hue_shift_factor,
            light_shift_factor=light_shift_factor,
            saturation_shift_factor=saturation_shift_factor,
            fog=fog,
            fog_strength=fog_strength,
            label_font_size=label_font_size,
            vdw_opacity=vdw_opacity,
            vdw_scale=vdw_scale,
            vdw_gradient_strength=vdw_gradient_strength,
            bo=bo,
            hy=hy,
            no_hy=no_hy,
            orient=_orient,
        )

    _apply_render_overlays(
        cfg,
        mol.graph,
        ts_bonds=ts_bonds,
        nci_bonds=nci_bonds,
        vdw=vdw,
        idx=idx,
        cmap=cmap,
        cmap_range=cmap_range,
        opacity=opacity,
    )

    # --- Convex hull (both config paths) ---
    from xyzrender.hull import apply_hull_to_config

    apply_hull_to_config(cfg, hull, hull_color, hull_opacity, hull_edge, hull_edge_width_ratio, mol.graph)

    # --- Never mutate mol — work on a render-time copy ---
    # resolve_orientation() (called by every compute_*_surface) writes PCA-rotated
    # positions back into the graph in-place and add_crystal_images() appends ghost
    # nodes.  Without a copy, a second render() of the same Molecule sees already-
    # oriented positions; the second PCA is ~identity so atom_centroid (original cube
    # frame) no longer matches target_centroid (≈ 0,0,0), misaligning the surface.
    rmol = Molecule(
        graph=copy.deepcopy(mol.graph),
        cube_data=mol.cube_data,  # read-only - no copy needed
        cell_data=copy.deepcopy(mol.cell_data) if mol.cell_data is not None else None,
        oriented=mol.oriented,
    )

    # --- Ensemble: build merged graph lazily (z_nudge=True for static renders) ---
    # mol.graph holds only the reference frame; conformer data lives in mol.ensemble.
    # We merge here so the renderer sees the full n_conformers x n_atoms graph, while
    # mol.graph stays clean for repeated render() calls.
    if _is_ensemble:
        from xyzrender.ensemble import merge_graphs as _ensemble_merge_graphs

        ens = mol.ensemble
        assert ens is not None  # narrowing: _is_ensemble = mol.ensemble is not None
        merged_graph = _ensemble_merge_graphs(
            rmol.graph,
            ens.positions,
            conformer_colors=ens.colors,
            conformer_graphs=ens.conformer_graphs,
            z_nudge=True,
        )
        # Apply per-conformer opacity stored in EnsembleFrames
        for nid in merged_graph.nodes():
            conf_idx = merged_graph.nodes[nid].get("molecule_index", 0)
            if conf_idx > 0:
                op = ens.opacities[conf_idx]
                if op is not None:
                    merged_graph.nodes[nid]["ensemble_opacity"] = op
        rmol = Molecule(
            graph=merged_graph,
            cube_data=rmol.cube_data,
            cell_data=rmol.cell_data,
            oriented=rmol.oriented,
        )

    # --- Vectors (user-supplied + crystal axes) ---
    _combine_vector_sources(
        cfg,
        rmol.graph,
        vector=vector,
        vector_scale=vector_scale,
        vector_color=vector_color,
        cell_data=rmol.cell_data,
        axes=axes,
    )

    # --- Cell / crystal config ---
    if rmol.cell_data is not None:
        _apply_cell_config(
            rmol,
            cfg,
            no_cell=no_cell,
            axis=axis,
            ghosts=ghosts,
            cell_color=cell_color,
            cell_width=cell_width,
            ghost_opacity=ghost_opacity,
            bo_explicit=bo,
        )
        if bo is None and not cfg.bond_orders:
            logger.warning("Periodic structure: bond orders disabled by default (pass bo=True to override)")
    elif "lattice" in mol.graph.graph:
        logger.info("Lattice found in graph; use load(..., cell=True) to draw the unit cell box")

    # --- Annotations ---
    if labels or label_file:
        from xyzrender.annotations import parse_annotations

        inline = [s.split() for s in labels] if labels else None
        cfg.annotations = parse_annotations(inline_specs=inline, file_path=label_file, graph=rmol.graph)

    # --- Early overlay validation (before ghost atoms are added to g1) ---
    if overlay is not None and mol.cell_data is not None:
        msg = "overlay= is mutually exclusive with crystal/cell display"
        raise ValueError(msg)
    if overlay is not None and (mo or dens or esp is not None or nci is not None):
        msg = "overlay= is mutually exclusive with surface rendering (mo/dens/esp/nci)"
        raise ValueError(msg)

    # --- Overlay alignment ---
    if overlay is not None:
        from xyzrender.overlay import align, merge_graphs
        from xyzrender.utils import pca_orient

        if isinstance(overlay, Molecule):
            overlay_mol = overlay
        else:
            # Inherit charge/multiplicity from the main molecule so bond-order
            # detection uses the correct electron count for charged species.
            _ov_charge = mol.graph.graph.get("total_charge", 0)
            _ov_mult = mol.graph.graph.get("multiplicity")
            overlay_mol = load(overlay, charge=_ov_charge, multiplicity=_ov_mult)
        g1 = rmol.graph
        g2 = copy.deepcopy(overlay_mol.graph)

        # PCA-orient g1 (the already-copied mol graph) to set the viewing frame
        if cfg.auto_orient and g1.number_of_nodes() > 1:
            nodes1 = list(g1.nodes())
            pos1 = np.array([g1.nodes[n]["position"] for n in nodes1], dtype=float)
            atom_mask = np.array([g1.nodes[n]["symbol"] != "*" for n in nodes1])
            fit_mask = atom_mask if not atom_mask.all() else None
            pos1_oriented = pca_orient(pos1, fit_mask=fit_mask)
            for k, nid in enumerate(nodes1):
                g1.nodes[nid]["position"] = tuple(float(v) for v in pos1_oriented[k])
        cfg.auto_orient = False

        if overlay_color is not None:
            cfg.overlay_color = resolve_color(overlay_color)
        # Convert 1-indexed align_atoms to 0-indexed for overlay
        _ov_align = [i - 1 for i in align_atoms] if align_atoms is not None else None
        aligned2 = align(g1, g2, align_atoms=_ov_align)
        rmol = Molecule(
            graph=merge_graphs(g1, g2, aligned2, overlay_color=cfg.overlay_color),
            cube_data=None,
            cell_data=None,
            oriented=True,
        )

    # --- Skeletal-style validation ---
    if cfg.skeletal_style:
        if mo or dens or esp is not None:
            msg = "skeletal_style is mutually exclusive with surface rendering (mo/dens/esp)"
            raise ValueError(msg)
        if vdw is not None:
            msg = "skeletal_style is mutually exclusive with vdw spheres"
            raise ValueError(msg)

    # --- Surface validation ---
    cube_data = rmol.cube_data
    _hull_active = cfg.show_convex_hull
    if _hull_active and (mo or dens or esp is not None or nci is not None):
        msg = "convex hull and surface rendering (mo/dens/esp/nci) are mutually exclusive"
        raise ValueError(msg)
    if vdw is not None and (mo or dens or esp is not None or nci is not None):
        msg = "vdw spheres and surface rendering (mo/dens/esp/nci) are mutually exclusive"
        raise ValueError(msg)
    n_surf = sum([mo, dens, esp is not None, nci is not None])
    if n_surf > 1:
        active = [n for n, v in [("mo", mo), ("dens", dens), ("esp", esp), ("nci", nci)] if v]
        msg = f"Surface flags are mutually exclusive: {', '.join(active)}"
        raise ValueError(msg)
    if mo and cube_data is None:
        msg = "mo=True requires a .cube file loaded via load()"
        raise ValueError(msg)
    if dens and cube_data is None:
        msg = "dens=True requires a .cube file loaded via load()"
        raise ValueError(msg)
    if esp is not None and cube_data is None:
        msg = "esp= requires a density .cube file loaded via load()"
        raise ValueError(msg)
    if nci is not None and cube_data is None:
        msg = "nci= requires a density .cube file loaded via load()"
        raise ValueError(msg)

    has_mo = bool(mo)
    has_dens = bool(dens)
    has_esp = esp is not None
    has_nci = nci is not None

    surf_overrides = collect_surf_overrides(
        iso=iso,
        mo_pos_color=mo_pos_color,
        mo_neg_color=mo_neg_color,
        mo_blur=mo_blur,
        mo_upsample=mo_upsample,
        flat_mo=flat_mo,
        dens_color=dens_color,
        nci_color=nci_color,
        nci_coloring=nci_coloring,
        nci_cutoff=nci_cutoff,
    )

    mo_params, dens_params, esp_params, nci_params = build_surface_params(
        cfg,
        surf_overrides,
        has_mo=has_mo,
        has_dens=has_dens,
        has_esp=has_esp,
        has_nci=has_nci,
    )

    from xyzrender.cube import parse_cube
    from xyzrender.surfaces import (
        compute_dens_surface,
        compute_esp_surface,
        compute_mo_surface,
        compute_nci_surface,
    )

    if mo_params is not None and cube_data is not None:
        compute_mo_surface(rmol.graph, cube_data, cfg, mo_params)

    if dens_params is not None and cube_data is not None:
        compute_dens_surface(rmol.graph, cube_data, cfg, dens_params)

    if esp_params is not None and esp is not None and cube_data is not None:
        esp_cube = parse_cube(str(esp))
        compute_esp_surface(rmol.graph, cube_data, esp_cube, cfg, esp_params)

    if nci_params is not None and nci is not None and cube_data is not None:
        nci_cube = parse_cube(str(nci))
        compute_nci_surface(rmol.graph, cube_data, nci_cube, cfg, nci_params)

    # --- Render ---
    svg = render_svg(rmol.graph, cfg)

    # --- Write output ---
    if output is not None:
        _write_output(svg, Path(output), cfg)

    return SVGResult(svg)


# ---------------------------------------------------------------------------
# render_gif
# ---------------------------------------------------------------------------


def render_gif(
    molecule: str | os.PathLike | Molecule,
    *,
    gif_rot: str | None = None,
    gif_trj: bool = False,
    gif_ts: bool = False,
    output: str | os.PathLike | None = None,
    gif_fps: int = 10,
    rot_frames: int = 120,
    ts_frame: int = 0,
    config: str | RenderConfig = "default",
    # --- Style (same as render(), only used when config is a string) ---
    canvas_size: int | None = None,
    atom_scale: float | None = None,
    bond_width: float | None = None,
    atom_stroke_width: float | None = None,
    bond_color: str | None = None,
    background: str | None = None,
    transparent: bool = False,
    gradient: bool | None = None,
    hue_shift_factor: float | None = None,
    light_shift_factor: float | None = None,
    saturation_shift_factor: float | None = None,
    fog: bool | None = None,
    fog_strength: float | None = None,
    label_font_size: float | None = None,
    vdw_opacity: float | None = None,
    vdw_scale: float | None = None,
    vdw_gradient_strength: float | None = None,
    hy: bool | list[int] | None = None,
    no_hy: bool = False,
    bo: bool | None = None,
    orient: bool | None = None,
    # --- Structural overlay (gif_rot only) ---
    overlay: str | os.PathLike | Molecule | None = None,
    overlay_color: str | None = None,
    # --- Orientation reference (gif_ts / gif_trj: graph after orient()) ---
    reference_graph: "nx.Graph | None" = None,
    # --- NCI detection (gif_ts / gif_trj / gif_rot) ---
    detect_nci: bool = False,
    # --- Vector arrows (gif_rot only) ---
    vector: str | Path | dict | list[VectorArrow] | None = None,
    vector_scale: float | None = None,
    vector_color: str | None = None,
    # --- Surfaces (gif_rot only) ---
    mo: bool = False,
    dens: bool = False,
    iso: float | None = None,
    mo_pos_color: str | None = None,
    mo_neg_color: str | None = None,
    mo_blur: float | None = None,
    mo_upsample: int | None = None,
    flat_mo: bool = False,
    dens_color: str | None = None,
    # --- Convex hull (gif_rot only) ---
    hull: bool | str | list[int] | list[list[int]] | None = None,
    hull_color: str | list[str] | None = None,
    hull_opacity: float | None = None,
    hull_edge: bool | None = None,
    hull_edge_width_ratio: float | None = None,
    # --- Crystal / cell (gif_rot only, when molecule has cell_data) ---
    no_cell: bool = False,
    axes: bool = True,
    axis: str | None = None,
    ghosts: bool | None = None,
    cell_color: str | None = None,
    cell_width: float | None = None,
    ghost_opacity: float | None = None,
) -> GIFResult:
    """Render a molecule to an animated GIF and return a :class:`GIFResult`.

    The result displays the GIF inline in Jupyter via ``_repr_html_``.
    Access the file path via ``result.path``.

    At least one of *gif_rot*, *gif_trj*, *gif_ts* must be set.

    Parameters
    ----------
    molecule:
        A :class:`Molecule` from :func:`load`, or a file path.  For
        *gif_ts* and *gif_trj* modes, a file path is required (the
        trajectory or vibration data is read directly from disk).
    gif_rot:
        Rotation axis: ``"x"``, ``"y"``, ``"z"``, diagonal (``"xy"``,
        …), or a 3-digit Miller index (``"111"``).
    gif_trj:
        Trajectory animation — *molecule* must be a multi-frame XYZ.
    gif_ts:
        Transition-state vibration animation (requires ``xyzrender[ts]``).
    output:
        Output ``.gif`` path.  Defaults to ``<stem>.gif`` beside *molecule*.
    gif_fps:
        Frames per second.
    rot_frames:
        Number of frames for a full rotation.
    ts_frame:
        Reference frame index for TS detection (0-indexed).
    config:
        Preset name, JSON path, or pre-built :class:`~xyzrender.types.RenderConfig`.

    Returns
    -------
    GIFResult
        Wrapper with path to the written GIF file.
    """
    from xyzrender.config import build_config
    from xyzrender.gif import (
        ROTATION_AXES,
        render_rotation_gif,
        render_trajectory_gif,
        render_vibration_gif,
        render_vibration_rotation_gif,
    )

    if not (gif_rot or gif_trj or gif_ts):
        msg = "render_gif: set gif_rot, gif_trj=True, or gif_ts=True"
        raise ValueError(msg)

    if gif_ts and gif_trj:
        msg = "render_gif: gif_ts and gif_trj are mutually exclusive"
        raise ValueError(msg)

    if (mo or dens) and (gif_ts or gif_trj):
        active_surf = "mo" if mo else "dens"
        active_gif = "gif_ts" if gif_ts else "gif_trj"
        msg = f"render_gif: {active_surf} surface is only supported with gif_rot, not {active_gif}"
        raise ValueError(msg)

    if overlay is not None and (gif_ts or gif_trj):
        msg = "render_gif: overlay= is only supported with gif_rot"
        raise ValueError(msg)

    if overlay is not None and (mo or dens):
        msg = "render_gif: overlay= is mutually exclusive with surface rendering (mo/dens)"
        raise ValueError(msg)

    # skeletal_style is a 2D line diagram — GIF rotation/animation is not meaningful
    _cd_flag = (isinstance(config, str) and config == "skeletal") or (
        not isinstance(config, str) and config.skeletal_style
    )
    if _cd_flag:
        msg = "render_gif: skeletal_style is not supported with GIF rendering"
        raise ValueError(msg)

    if gif_rot and gif_rot not in ROTATION_AXES:
        test = gif_rot.lstrip("-")
        if not (test.isdigit() and len(test) >= 3):
            msg = f"render_gif: invalid gif_rot {gif_rot!r} — use 'x', 'y', 'z', or 3-digit Miller index"
            raise ValueError(msg)

    if rot_frames != 120 and not gif_rot:
        logger.warning("rot_frames has no effect without gif_rot")

    # Resolve config
    _gif_graph = molecule.graph if isinstance(molecule, Molecule) else load(molecule).graph
    if not isinstance(config, str):
        cfg = copy.copy(config)
        cfg.vectors = list(cfg.vectors)
    else:
        cfg = build_config(
            config,
            canvas_size=canvas_size,
            atom_scale=atom_scale,
            bond_width=bond_width,
            atom_stroke_width=atom_stroke_width,
            bond_color=bond_color,
            background=background,
            transparent=transparent,
            gradient=gradient,
            hue_shift_factor=hue_shift_factor,
            light_shift_factor=light_shift_factor,
            saturation_shift_factor=saturation_shift_factor,
            fog=fog,
            fog_strength=fog_strength,
            label_font_size=label_font_size,
            vdw_opacity=vdw_opacity,
            vdw_scale=vdw_scale,
            vdw_gradient_strength=vdw_gradient_strength,
            bo=bo,
            hy=hy,
            no_hy=no_hy,
            orient=orient,
        )

    # --- Convex hull (both config paths) ---
    from xyzrender.hull import apply_hull_to_config

    apply_hull_to_config(cfg, hull, hull_color, hull_opacity, hull_edge, hull_edge_width_ratio, _gif_graph)

    # Surface / hull mutual exclusivity (also catches hull set on pre-built config)
    if cfg.show_convex_hull and (mo or dens):
        msg = "render_gif: convex hull and surface rendering (mo/dens) are mutually exclusive"
        raise ValueError(msg)

    # Resolve molecule → path and/or graph
    if isinstance(molecule, Molecule):
        if gif_ts or gif_trj:
            msg = (
                "render_gif: pass a file path (not a Molecule) for gif_ts / gif_trj modes — "
                "the trajectory is read from disk."
            )
            raise ValueError(msg)
        mol_path = None
        ref_graph = molecule.graph
    else:
        mol_path = Path(str(molecule))
        ref_graph = None

    # Resolve output path
    if output is not None:
        gif_path = Path(output)
    elif mol_path is not None:
        gif_path = mol_path.with_suffix(".gif")
    else:
        import tempfile

        _, tmp = tempfile.mkstemp(suffix=".gif")
        gif_path = Path(tmp)

    if gif_path.suffix.lower() != ".gif":
        msg = f"render_gif: output must have .gif extension, got {gif_path.suffix!r}"
        raise ValueError(msg)

    # --- Dispatch ---
    if gif_ts and gif_rot:
        render_vibration_rotation_gif(
            str(mol_path),
            cfg,
            str(gif_path),
            ts_frame=ts_frame,
            fps=gif_fps,
            axis=gif_rot,
            n_frames=rot_frames,
            reference_graph=reference_graph,
            detect_nci=detect_nci,
        )

    elif gif_ts:
        render_vibration_gif(
            str(mol_path),
            cfg,
            str(gif_path),
            ts_frame=ts_frame,
            fps=gif_fps,
            reference_graph=reference_graph,
            detect_nci=detect_nci,
        )

    elif gif_trj:
        from xyzrender.readers import load_molecule, load_trajectory_frames

        frames = load_trajectory_frames(str(mol_path))
        if len(frames) < 2:
            msg = "render_gif(gif_trj=True) requires a multi-frame XYZ file"
            raise ValueError(msg)
        _trj_ref = reference_graph
        if _trj_ref is None:
            graph, _ = load_molecule(str(mol_path))
            _trj_ref = graph
        render_trajectory_gif(
            frames,
            cfg,
            str(gif_path),
            fps=gif_fps,
            reference_graph=_trj_ref,
            detect_nci=detect_nci,
            axis=gif_rot,
        )

    else:
        # gif_rot only
        if ref_graph is None:
            from xyzrender.readers import load_molecule

            ref_graph, _ = load_molecule(str(mol_path))
        else:
            # Deep-copy so render_rotation_gif (which mutates positions in-place) doesn't
            # corrupt the caller's Molecule, and so _apply_cell_config can add ghost atoms.
            ref_graph = copy.deepcopy(ref_graph)

        # --- Ensemble: build scratch merged graph (z_nudge=False — meaningless for rotation) ---
        if isinstance(molecule, Molecule) and molecule.ensemble is not None:
            from xyzrender.ensemble import merge_graphs as _ensemble_merge_graphs

            ens = molecule.ensemble
            ref_graph = _ensemble_merge_graphs(
                ref_graph,
                ens.positions,
                conformer_colors=ens.colors,
                conformer_graphs=ens.conformer_graphs,
                z_nudge=False,
            )
            for nid in ref_graph.nodes():
                conf_idx = ref_graph.nodes[nid].get("molecule_index", 0)
                if conf_idx > 0:
                    op = ens.opacities[conf_idx]
                    if op is not None:
                        ref_graph.nodes[nid]["ensemble_opacity"] = op

        # --- Overlay alignment (gif_rot only) ---
        if overlay is not None:
            from xyzrender.overlay import align, merge_graphs

            if isinstance(overlay, Molecule):
                overlay_mol = overlay
            else:
                _ov_charge = ref_graph.graph.get("total_charge", 0)
                _ov_mult = ref_graph.graph.get("multiplicity")
                overlay_mol = load(overlay, charge=_ov_charge, multiplicity=_ov_mult)
            if overlay_color is not None:
                cfg.overlay_color = resolve_color(overlay_color)
            g2 = copy.deepcopy(overlay_mol.graph)
            aligned2 = align(ref_graph, g2)
            ref_graph = merge_graphs(ref_graph, g2, aligned2, overlay_color=cfg.overlay_color)

        # --- Vectors (user-supplied + crystal axes; gif_rot only) ---
        _cell_data_for_vecs = molecule.cell_data if isinstance(molecule, Molecule) else None
        _combine_vector_sources(
            cfg,
            ref_graph,
            vector=vector,
            vector_scale=vector_scale,
            vector_color=vector_color,
            cell_data=_cell_data_for_vecs,
            axes=axes,
        )

        cube_data = molecule.cube_data if isinstance(molecule, Molecule) else None

        # Apply crystal/cell config when the molecule carries cell_data
        if isinstance(molecule, Molecule) and molecule.cell_data is not None:
            _gif_mol = Molecule(
                graph=ref_graph,
                cube_data=None,
                cell_data=copy.deepcopy(molecule.cell_data),
                oriented=molecule.oriented,
            )
            _apply_cell_config(
                _gif_mol,
                cfg,
                no_cell=no_cell,
                axis=axis,
                ghosts=ghosts,
                cell_color=cell_color,
                cell_width=cell_width,
                ghost_opacity=ghost_opacity,
                bo_explicit=bo,
            )
            ref_graph = _gif_mol.graph
        # Build surface params when a cube is present
        mo_params = dens_params = None
        if cube_data is not None and (mo or dens):
            from xyzrender.config import build_surface_params, collect_surf_overrides

            surf_overrides = collect_surf_overrides(
                iso=iso,
                mo_pos_color=mo_pos_color,
                mo_neg_color=mo_neg_color,
                mo_blur=mo_blur,
                mo_upsample=mo_upsample,
                flat_mo=flat_mo,
                dens_color=dens_color,
            )
            mo_params, dens_params, _, _ = build_surface_params(
                cfg,
                surf_overrides,
                has_mo=mo,
                has_dens=dens,
                has_esp=False,
                has_nci=False,
            )
        render_rotation_gif(
            ref_graph,
            cfg,
            str(gif_path),
            n_frames=rot_frames,
            fps=gif_fps,
            axis=gif_rot or "y",
            mo_params=mo_params,
            mo_cube=cube_data if mo_params is not None else None,
            dens_params=dens_params,
            dens_cube=cube_data if dens_params is not None else None,
        )

    logger.info("GIF written to %s", gif_path)
    return GIFResult(gif_path)


# ---------------------------------------------------------------------------
# Ensemble overlay
# ---------------------------------------------------------------------------


def _resolve_ensemble_colors(
    *,
    ensemble_color: str | list[str] | None,
    ensemble_palette: str | None,
    n_conformers: int | None,
) -> list[str] | None:
    """Resolve ensemble colour specification to a list of hex strings (one per conformer).

    Returns ``None`` when no colouring is requested (CPK default).
    When *n_conformers* is ``None`` (not yet known), returns a sentinel
    that ``_build_ensemble_molecule`` will expand after counting frames.
    """
    if ensemble_palette is not None:
        from xyzrender.colors import sample_palette

        if n_conformers is None:
            # Return palette name as sentinel — _build_ensemble_molecule resolves later
            return None  # handled below in _build_ensemble_molecule
        return sample_palette(ensemble_palette, n_conformers)

    if ensemble_color is None:
        return None

    if isinstance(ensemble_color, str):
        # Single colour → all non-ref conformers get this colour
        hex_c = resolve_color(ensemble_color)
        if n_conformers is None:
            return [hex_c]  # single-element sentinel
        return [hex_c] * n_conformers

    # List of colours
    return [resolve_color(c) for c in ensemble_color]


def _build_ensemble_molecule(
    trajectory: str | os.PathLike,
    *,
    reference_frame: int = 0,
    max_frames: int | None = None,
    align_atoms: list[int] | None = None,
    conformer_colors: list[str] | None = None,
    ensemble_opacity: float | None = None,
    ensemble_palette: str | None = None,
    charge: int = 0,
    multiplicity: int | None = None,
    kekule: bool = False,
    rebuild: bool = False,
    quick: bool = False,
    nci_detect: bool = False,
    reference_mol: Molecule | None = None,
) -> Molecule:
    """Build a :class:`Molecule` representing an ensemble of conformers.

    Frames from *trajectory* are RMSD-aligned onto *reference_frame* using
    index-based pairing (atom *i* in each frame corresponds to atom *i* in
    the reference frame).

    When *rebuild* is ``True``, each frame's graph is built independently
    so that bonding can differ between conformers — analogous to ``--gif-trj``
    but rendered on one image.  NCI detection is run on each rebuilt frame too
    when *nci_detect* is also ``True``.

    When *reference_mol* is given, its graph (and positions) are used as the
    reference frame instead of loading from *trajectory*.  This lets
    interactive orientation be applied before ensemble alignment.
    """
    from xyzrender.ensemble import align as ensemble_align
    from xyzrender.readers import load_molecule, load_trajectory_frames

    traj_path = Path(str(trajectory))
    frames = load_trajectory_frames(traj_path)
    if len(frames) < 2:
        msg = "ensemble: trajectory must contain at least two frames"
        raise ValueError(msg)
    if not (0 <= reference_frame < len(frames)):
        msg = f"ensemble: reference_frame {reference_frame} out of range for {len(frames)} frames"
        raise ValueError(msg)

    # Optional frame cap: first max_frames frames, always including reference_frame.
    if max_frames is not None:
        if max_frames < 2:
            msg = "ensemble: max_frames must be at least 2 when set"
            raise ValueError(msg)
        max_frames = min(max_frames, len(frames))
        # Ensure the reference frame is included: if it lies beyond the window,
        # fall back to using frame 0 as the reference.
        if reference_frame >= max_frames:
            reference_frame = 0
        frames = frames[:max_frames]

    # Sanity-check that all frames share the same symbols and atom counts.
    ref_symbols = frames[reference_frame]["symbols"]
    for idx, fr in enumerate(frames):
        if fr["symbols"] != ref_symbols:
            msg = f"ensemble: frame {idx} atom symbols do not match reference frame"
            raise ValueError(msg)

    # Use pre-loaded reference Molecule when provided (e.g. after interactive orient),
    # otherwise load from the trajectory file.
    if reference_mol is not None:
        ref_graph = copy.deepcopy(reference_mol.graph)
        cell_data = copy.deepcopy(reference_mol.cell_data)
        oriented = reference_mol.oriented
    else:
        ref_graph, cell_data = load_molecule(
            traj_path,
            frame=reference_frame,
            charge=charge,
            multiplicity=multiplicity,
            kekule=kekule,
            rebuild=rebuild,
            quick=quick,
        )
        oriented = False

    # For ensemble overlays we ignore bond orders in the rendering.  Flatten any
    # existing bond_order values to 1 so everything is drawn as single bonds.
    for _i, _j, data in ref_graph.edges(data=True):
        if "bond_order" in data:
            data["bond_order"] = 1

    # When using a pre-oriented reference, update the reference frame's positions
    # in the trajectory data so alignment targets the oriented coordinates.
    # Only extract real atom positions (exclude NCI centroid dummy nodes with symbol="*").
    if reference_mol is not None:
        from xyzrender.overlay import _node_list

        real_nodes = [n for n in _node_list(ref_graph) if ref_graph.nodes[n].get("symbol") != "*"]
        frames[reference_frame]["positions"] = [list(ref_graph.nodes[n]["position"]) for n in real_nodes]

    aligned_positions = ensemble_align(frames, reference_frame=reference_frame, align_atoms=align_atoms)

    # NCI detection and per-frame graph building happen *after* alignment so that
    # centroid dummy nodes don't interfere with position array sizes.
    if nci_detect:
        from xyzrender.readers import detect_nci as _detect_nci

        if reference_mol is None:
            ref_graph = _detect_nci(ref_graph)

    conformer_graphs: list[nx.Graph] | None = None
    if rebuild:
        from xyzgraph import build_graph

        conformer_graphs = []
        for fi, frame in enumerate(frames):
            if fi == reference_frame:
                conformer_graphs.append(ref_graph)
                continue
            atoms = list(zip(frame["symbols"], [tuple(p) for p in frame["positions"]], strict=True))
            fg = build_graph(atoms, charge=charge, multiplicity=multiplicity, kekule=kekule, quick=quick)
            for _i, _j, d in fg.edges(data=True):
                if "bond_order" in d:
                    d["bond_order"] = 1
            if nci_detect:
                fg = _detect_nci(fg)
            conformer_graphs.append(fg)

    # Resolve palette colours now that we know n_conformers.
    # Default: None (CPK atom colours). Palette/explicit colour opt-in only.
    n_conf = len(frames)
    if ensemble_palette is not None:
        from xyzrender.colors import sample_palette

        conformer_colors = sample_palette(ensemble_palette, n_conf)
    elif conformer_colors is not None and len(conformer_colors) == 1:
        # Single-colour sentinel → expand to all conformers
        conformer_colors = conformer_colors * n_conf
    # else: conformer_colors is None → CPK default, leave as-is

    # Build per-conformer opacities list (None = fully opaque / use default)
    opacities: list[float | None] = [None] * n_conf
    if ensemble_opacity is not None:
        for i in range(n_conf):
            if i != reference_frame:
                opacities[i] = ensemble_opacity

    colors: list[str | None] = list(conformer_colors) if conformer_colors is not None else [None] * n_conf
    ens = EnsembleFrames(
        positions=np.stack(aligned_positions, axis=0),  # (n_conformers, n_atoms, 3)
        colors=colors,
        opacities=opacities,
        conformer_graphs=conformer_graphs,
        reference_idx=reference_frame,
    )

    return Molecule(graph=ref_graph, cube_data=None, cell_data=cell_data, oriented=oriented, ensemble=ens)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _apply_render_overlays(
    cfg: "RenderConfig",
    graph: "nx.Graph",
    *,
    ts_bonds: list[tuple[int, int]] | None = None,
    nci_bonds: list[tuple[int, int]] | None = None,
    vdw: bool | list[int] | None = None,
    idx: bool | str = False,
    cmap: str | os.PathLike | dict[int, float] | None = None,
    cmap_range: tuple[float, float] | None = None,
    opacity: float | None = None,
) -> None:
    """Apply render()-specific overlays to cfg (mutates in place).

    All atom indices in ts_bonds, nci_bonds, vdw are 1-indexed (user-facing).
    """
    if ts_bonds is not None:
        cfg.ts_bonds = [(a - 1, b - 1) for a, b in ts_bonds]
    if nci_bonds is not None:
        cfg.nci_bonds = [(a - 1, b - 1) for a, b in nci_bonds]
    if vdw is not None:
        cfg.vdw_indices = [i - 1 for i in vdw] if isinstance(vdw, list) else []
    if idx:
        cfg.show_indices = True
        cfg.idx_format = idx if isinstance(idx, str) else "sn"
    if cmap is not None:
        cfg.atom_cmap = _resolve_cmap(cmap, graph)
    if cmap_range is not None:
        cfg.cmap_range = cmap_range
    if opacity is not None:
        cfg.surface_opacity = opacity


def _resolve_cmap(
    cmap: str | os.PathLike | dict[int, float],
    graph: nx.Graph | None,
) -> dict[int, float]:
    """Resolve *cmap* to a 0-indexed ``{atom_idx: value}`` dict.

    Accepts either a ``{1-indexed atom: value}`` dict or a path to a
    two-column text file (same format as ``--cmap`` in the CLI).
    """
    if isinstance(cmap, dict):
        from typing import cast

        d = cast("dict[int, float]", cmap)
        return {k - 1: v for k, v in d.items()}
    # File path
    from xyzrender.annotations import load_cmap

    return load_cmap(str(cmap), graph)


def _combine_vector_sources(
    cfg: "RenderConfig",
    graph: "nx.Graph",
    *,
    vector=None,
    vector_scale: "float | None" = None,
    vector_color: "str | None" = None,
    cell_data: "CellData | None" = None,
    axes: bool = True,
) -> None:
    """Populate ``cfg.vectors`` from user-supplied vectors and crystal axis arrows.

    Must be called *before* :func:`_apply_cell_config` so that all vectors are
    already in ``cfg.vectors`` when :func:`orient_hkl_to_view` applies the HKL
    rotation to the whole list in one pass.
    """
    if vector_scale is not None:
        cfg.vector_scale = vector_scale
    if vector_color is not None:
        cfg.vector_color = resolve_color(vector_color)
    if vector is not None:
        if not isinstance(vector, list):
            from xyzrender.annotations import load_vectors

            _vec_src = vector if isinstance(vector, dict) else Path(vector)
            vector = load_vectors(_vec_src, graph, default_color=cfg.vector_color)
        cfg.vectors.extend(vector)
    if cell_data is not None and axes:
        from xyzrender.types import VectorArrow

        lat = cell_data.lattice
        orig3d = cell_data.cell_origin
        for vec, color, label in zip(lat, cfg.axis_colors, ("a", "b", "c"), strict=True):
            length = float(np.linalg.norm(vec))
            if length < 1e-6:
                continue
            frac = min(0.25, 2.0 / length)
            cfg.vectors.append(
                VectorArrow(
                    vector=vec * frac,
                    origin=orig3d,
                    color=color,
                    label=label,
                    scale=1.0,
                    draw_on_top=True,
                    is_axis=True,
                    font_size=cfg.label_font_size * 1.8,
                    width=cfg.bond_width * 1.1,
                )
            )


def _apply_cell_config(
    mol: Molecule,
    cfg: RenderConfig,
    *,
    no_cell: bool,
    axis: str | None,
    ghosts: bool | None,
    cell_color: str | None,
    cell_width: float | None,
    ghost_opacity: float | None,
    bo_explicit: bool | None,
) -> None:
    """Configure crystal/cell display options on *cfg* from *mol.cell_data*."""
    cell_data = mol.cell_data
    assert cell_data is not None  # caller guarantees this
    cfg.cell_data = cell_data
    cfg.show_cell = not no_cell
    # PCA auto-orient makes no sense for full periodic crystals (unless user overrides)
    if cfg.auto_orient:
        cfg.auto_orient = False

    if cell_color is not None:
        from xyzrender.types import resolve_color

        cfg.cell_color = resolve_color(cell_color)
    if cell_width is not None:
        cfg.cell_line_width = cell_width
    if ghost_opacity is not None:
        cfg.periodic_image_opacity = ghost_opacity

    # axis HKL: orient so [hkl] points along the viewing (+z) axis
    if axis is not None:
        from xyzrender.viewer import orient_hkl_to_view

        orient_hkl_to_view(mol.graph, cell_data, axis, cfg)
        cfg.auto_orient = False

    # Ghost (periodic image) atoms — default: on when cell_data is present
    _show_ghosts = ghosts if ghosts is not None else True
    if _show_ghosts:
        from xyzrender.crystal import add_crystal_images

        add_crystal_images(mol.graph, cell_data)

    # Default no-bo for periodic structures (bond orders are not PBC-aware)
    if bo_explicit is None:
        cfg.bond_orders = False


def _resolve_crystal_interface(path: Path, crystal: bool | str) -> str:
    """Resolve the phonopy interface mode from *crystal* param and file path."""
    if isinstance(crystal, str) and crystal in {"vasp", "qe"}:
        return crystal
    # auto-detect from filename
    stem = path.stem.upper()
    ext = path.suffix.lower()
    if ext == ".vasp" or stem in {"POSCAR", "CONTCAR"}:
        return "vasp"
    if ext == ".in":
        return "qe"
    msg = f"Cannot auto-detect crystal interface from {str(path)!r}. Specify explicitly: crystal='vasp' or crystal='qe'"
    raise ValueError(msg)


def _write_output(svg: str, output: Path, cfg: RenderConfig) -> None:
    """Write SVG to file, converting format based on extension."""
    ext = output.suffix.lower()
    if ext == ".svg":
        output.write_text(svg)
    elif ext == ".png":
        from xyzrender.export import svg_to_png

        svg_to_png(svg, str(output), size=cfg.canvas_size, dpi=getattr(cfg, "dpi", 300))
    elif ext == ".pdf":
        from xyzrender.export import svg_to_pdf

        svg_to_pdf(svg, str(output))
    else:
        msg = f"Unsupported output format: {ext!r} (use .svg, .png, or .pdf)"
        raise ValueError(msg)
