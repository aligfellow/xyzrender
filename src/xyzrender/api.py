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

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import os

    import networkx as nx

    from xyzrender.cube import CubeData
    from xyzrender.types import CellData, RenderConfig, VectorArrow

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
        import re

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

    @property
    def path(self) -> Path:
        """Path to the GIF file on disk."""
        return self._path

    def __repr__(self) -> str:
        """Return a string representation of the GIFResult."""
        return f"GIFResult(path={self._path!r})"

    def __bytes__(self) -> bytes:
        """Return the raw GIF bytes."""
        return self._path.read_bytes()

    def save(self, path: str | os.PathLike) -> None:
        """Write the GIF to *path*."""
        Path(path).write_bytes(self._path.read_bytes())

    def _repr_html_(self) -> str:
        """Embed the GIF inline in Jupyter, capped to 500 px wide."""
        import base64

        data = base64.b64encode(self._path.read_bytes()).decode("ascii")
        return f'<img src="data:image/gif;base64,{data}" width="500" style="height:auto"/>'


@dataclass
class Molecule:
    """Container for a loaded molecular structure.

    Obtain via :func:`load`.  Pass directly to :func:`render` or
    :func:`render_gif` to avoid re-parsing the file.
    """

    graph: nx.Graph
    cube_data: CubeData | None = None
    cell_data: CellData | None = None
    oriented: bool = False

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
        if path and not str(path).lower().endswith(".xyz"):
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
        provides explicit connectivity.
    mol_frame:
        Zero-based frame index for multi-record SDF files.
    ts_detect:
        Run graphRC transition-state detection (requires ``xyzrender[ts]``).
    ts_frame:
        Reference frame index for TS detection in multi-frame files.
    nci_detect:
        Detect non-covalent interactions with xyzgraph after loading.
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

    Returns
    -------
    Molecule
    """
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
    gradient_strength: float | None = None,
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
    vectors: str | Path | dict | list[VectorArrow] | None = None,
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
    hull: bool | list[int] | list[list[int]] | None = None,
    hull_color: str | list[str] | None = None,
    hull_opacity: float | None = None,
    hull_edge: bool | None = None,
    hull_edge_width_ratio: float | None = None,
    # --- Overlay ---
    overlay: str | os.PathLike | Molecule | None = None,
    overlay_color: str | None = None,
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
        ``True`` = hull over all heavy atoms; a flat list of 1-indexed atom
        indices (one hull, e.g. ``[1,2,3,4,5,6]``); a list of lists (multiple
        hulls, e.g. ``[[1,2,3,4,5,6], [7,8,9]]``).  ``None`` (default) = off.
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

    # --- Orient resolution ---
    # orient=None: auto-orient, but skip if mol was manually oriented
    _orient: bool | None = orient
    if _orient is None and mol.oriented:
        _orient = False

    # --- Config resolution ---
    if not isinstance(config, str):
        # Pre-built RenderConfig — shallow copy so we don't mutate the caller's object
        cfg = copy.copy(config)
        if _orient is not None:
            cfg.auto_orient = _orient
        elif mol.oriented:
            cfg.auto_orient = False
        # Apply render()-specific overlays on top of pre-built config
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
            cfg.atom_cmap = _resolve_cmap(cmap, mol.graph)
        if cmap_range is not None:
            cfg.cmap_range = cmap_range
        if opacity is not None:
            cfg.surface_opacity = opacity
        if hull is not None:
            if isinstance(hull, list):
                cfg.show_convex_hull = True
                # 1-indexed user input → 0-indexed internal
                from xyzrender.hull import hull_indices_to_0indexed

                cfg.hull_atom_indices = hull_indices_to_0indexed(hull)
            else:
                cfg.show_convex_hull = hull
        if hull_color is not None:
            if isinstance(hull_color, list):
                cfg.hull_colors = hull_color
            else:
                cfg.hull_colors = [hull_color]
        if hull_opacity is not None:
            cfg.hull_opacity = hull_opacity
        if hull_edge is not None:
            cfg.show_hull_edges = hull_edge
        if hull_edge_width_ratio is not None:
            cfg.hull_edge_width_ratio = hull_edge_width_ratio
    else:
        # Build from preset/file
        _ts_0 = [(a - 1, b - 1) for a, b in ts_bonds] if ts_bonds else None
        _nci_0 = [(a - 1, b - 1) for a, b in nci_bonds] if nci_bonds else None
        _vdw_0 = [] if vdw is True else ([i - 1 for i in vdw] if vdw else None)
        _show = bool(idx)
        _ifmt = idx if isinstance(idx, str) else "sn"
        _cmap_0 = _resolve_cmap(cmap, mol.graph) if cmap is not None else None
        _hull_flag: bool | None = True if isinstance(hull, list) else hull
        # 1-indexed user input → 0-indexed for build_config
        if isinstance(hull, list):
            from xyzrender.hull import hull_indices_to_0indexed

            _hull_idx: list[int] | list[list[int]] | None = hull_indices_to_0indexed(hull)
        else:
            _hull_idx = None

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
            gradient_strength=gradient_strength,
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
            opacity=opacity,
            ts_bonds=_ts_0,
            nci_bonds=_nci_0,
            vdw_indices=_vdw_0,
            show_indices=_show,  # RenderConfig field keeps its name
            idx_format=_ifmt,
            atom_cmap=_cmap_0,
            cmap_range=cmap_range,
            hull=_hull_flag,
            hull_opacity=hull_opacity,
            hull_colors=[hull_color] if isinstance(hull_color, str) else hull_color,
            hull_idx=_hull_idx,
            hull_edge=hull_edge,
            hull_edge_width_ratio=hull_edge_width_ratio,
        )

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

    # --- Cell / crystal config ---
    if rmol.cell_data is not None:
        _apply_cell_config(
            rmol,
            cfg,
            no_cell=no_cell,
            axes=axes,
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

    # --- Vector arrows ---
    if vector_scale is not None:
        cfg.vector_scale = vector_scale
    if vector_color is not None:
        from xyzrender.types import resolve_color

        cfg.vector_color = resolve_color(vector_color)
    if vectors is not None:
        if isinstance(vectors, list):
            cfg.vectors = vectors
        else:
            from xyzrender.annotations import load_vectors

            _vec_src = vectors if isinstance(vectors, dict) else Path(vectors)
            cfg.vectors = load_vectors(_vec_src, rmol.graph, default_color=cfg.vector_color)

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
            from xyzrender.types import resolve_color

            cfg.overlay_color = resolve_color(overlay_color)
        aligned2 = align(g1, g2)
        rmol = Molecule(
            graph=merge_graphs(g1, g2, aligned2, overlay_color=cfg.overlay_color),
            cube_data=None,
            cell_data=None,
            oriented=True,
        )

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
        from xyzrender.cube import parse_cube

        esp_cube = parse_cube(str(esp))
        compute_esp_surface(rmol.graph, cube_data, esp_cube, cfg, esp_params)

    if nci_params is not None and nci is not None and cube_data is not None:
        from xyzrender.cube import parse_cube

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
    gradient_strength: float | None = None,
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
    reference_graph=None,
    # --- NCI detection (gif_ts / gif_trj / gif_rot) ---
    detect_nci: bool = False,
    # --- Vector arrows (gif_rot only) ---
    vectors: str | Path | dict | list[VectorArrow] | None = None,
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
    hull: bool | list[int] | list[list[int]] | None = None,
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
    Path
        Path to the written GIF file.
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

    if gif_rot and gif_rot not in ROTATION_AXES:
        test = gif_rot.lstrip("-")
        if not (test.isdigit() and len(test) >= 3):
            msg = f"render_gif: invalid gif_rot {gif_rot!r} — use 'x', 'y', 'z', or 3-digit Miller index"
            raise ValueError(msg)

    if rot_frames != 120 and not gif_rot:
        logger.warning("rot_frames has no effect without gif_rot")

    # Resolve config
    if not isinstance(config, str):
        cfg = copy.copy(config)
        # Apply hull overrides to pre-built config
        if hull is not None:
            if isinstance(hull, list):
                cfg.show_convex_hull = True
                # 1-indexed user input → 0-indexed internal
                from xyzrender.hull import hull_indices_to_0indexed

                cfg.hull_atom_indices = hull_indices_to_0indexed(hull)
            else:
                cfg.show_convex_hull = hull
        if hull_color is not None:
            if isinstance(hull_color, list):
                cfg.hull_colors = hull_color
            else:
                cfg.hull_colors = [hull_color]
        if hull_opacity is not None:
            cfg.hull_opacity = hull_opacity
        if hull_edge is not None:
            cfg.show_hull_edges = hull_edge
        if hull_edge_width_ratio is not None:
            cfg.hull_edge_width_ratio = hull_edge_width_ratio
    else:
        _hull_flag: bool | None = True if isinstance(hull, list) else hull
        # 1-indexed user input → 0-indexed for build_config
        if isinstance(hull, list):
            from xyzrender.hull import hull_indices_to_0indexed

            _hull_idx: list[int] | list[list[int]] | None = hull_indices_to_0indexed(hull)
        else:
            _hull_idx = None
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
            gradient_strength=gradient_strength,
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
            hull=_hull_flag,
            hull_opacity=hull_opacity,
            hull_colors=[hull_color] if isinstance(hull_color, str) else hull_color,
            hull_idx=_hull_idx,
            hull_edge=hull_edge,
            hull_edge_width_ratio=hull_edge_width_ratio,
        )

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
                from xyzrender.types import resolve_color

                cfg.overlay_color = resolve_color(overlay_color)
            g2 = copy.deepcopy(overlay_mol.graph)
            aligned2 = align(ref_graph, g2)
            ref_graph = merge_graphs(ref_graph, g2, aligned2, overlay_color=cfg.overlay_color)

        # --- Vector arrows (gif_rot only; needs ref_graph for COM) ---
        if vector_scale is not None:
            cfg.vector_scale = vector_scale
        if vector_color is not None:
            from xyzrender.types import resolve_color

            cfg.vector_color = resolve_color(vector_color)
        if vectors is not None:
            if isinstance(vectors, list):
                cfg.vectors = vectors
            else:
                from xyzrender.annotations import load_vectors

                _vec_src = vectors if isinstance(vectors, dict) else Path(vectors)
                cfg.vectors = load_vectors(_vec_src, ref_graph, default_color=cfg.vector_color)

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
                axes=axes,
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
# Private helpers
# ---------------------------------------------------------------------------


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


def _apply_cell_config(
    mol: Molecule,
    cfg: RenderConfig,
    *,
    no_cell: bool,
    axes: bool,
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

        orient_hkl_to_view(mol.graph, cell_data, axis)
        cfg.auto_orient = False

    # Ghost (periodic image) atoms — default: on when cell_data is present
    _show_ghosts = ghosts if ghosts is not None else True
    if _show_ghosts:
        from xyzrender.crystal import add_crystal_images

        add_crystal_images(mol.graph, cell_data)

    # Crystal axes a/b/c as annotation vectors at the cell origin
    if axes:
        from xyzrender.types import VectorArrow

        lat = cell_data.lattice
        orig3d = cell_data.cell_origin
        for vec, color, label in zip(lat, cfg.axis_colors, ("a", "b", "c"), strict=True):
            length = float(np.linalg.norm(vec))
            if length < 1e-6:
                continue
            # Arrow spans 25% of the cell edge (max 2 Å) from the origin corner
            frac = min(0.25, 2.0 / length)
            cfg.vectors.append(
                VectorArrow(
                    vector=vec * frac,
                    origin=orig3d,
                    color=color,
                    label=label,
                    scale=1.0,
                    draw_on_top=True,
                    font_size=cfg.label_font_size * 1.8,
                    width=cfg.bond_width * 1.1,
                )
            )

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
