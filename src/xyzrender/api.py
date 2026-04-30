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
    from xyzrender.types import CellData, VectorArrow

from xyzrender.colors import resolve_color
from xyzrender.types import GIFResult, OverlayConfig, RenderConfig, SVGResult
from xyzrender.utils import parse_atom_indices

logger = logging.getLogger(__name__)


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
    cell: bool = False,
    quick: bool = False,
    bohr: bool | None = None,
    # --- Ensemble (multi-frame trajectory) ---
    ensemble: bool = False,
    reference_frame: int = 0,
    max_frames: int | None = None,
    align_atoms: str | list[int] | None = None,
    ensemble_color: str | list[str] | None = None,
    ensemble_opacity: float | None = None,
    auto_align: bool = True,
    reference_mol: Molecule | None = None,
) -> Molecule:
    """Load a molecule from file (or SMILES string) and return a :class:`Molecule`.

    Parameters
    ----------
    molecule:
        Path to the input file, or a SMILES string when *smiles* is ``True``.
        Supported extensions: ``.xyz``, ``.cube``, ``.cub``, ``.mol``, ``.sdf``,
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
        1-indexed atom indices for Kabsch alignment subset (min 3).
        When given, the rotation is computed from this subset only
        but applied to all atoms.
    ensemble_color:
        Conformer colour spec.  May be a palette name from
        :data:`xyzrender.colors.PALETTE_NAMES` (sampled across frames),
        a single hex/named colour (broadcast to every conformer), a
        comma-separated list, or an explicit list of colours.
    ensemble_opacity:
        Opacity for non-reference conformer atoms (0-1).
    auto_align:
        ``True`` (default) runs Kabsch alignment onto *reference_frame*.
        ``False`` keeps each frame's raw coordinates — useful when absolute
        geometry matters (e.g. IRC paths).
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
            ensemble_color=ensemble_color,
            ensemble_opacity=ensemble_opacity,
            auto_align=auto_align,
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
            data,
            charge=charge,
            multiplicity=multiplicity,
            kekule=kekule,
            rebuild=rebuild,
            quick=quick,
        )
    elif not Path(mol_path).is_file():
        raise FileNotFoundError(f"[Errno 2] No such file or directory: '{mol_path}'")

    elif mol_path.suffix.lower() in {".cube", ".cub"}:
        from xyzrender.readers import load_cube

        graph, cube_data = load_cube(
            mol_path,
            charge=charge,
            multiplicity=multiplicity,
            kekule=kekule,
            quick=quick,
        )

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
            bohr=bohr,
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


def orient(mol: Molecule, viewer: str = "vmol", also: list[Molecule] | None = None) -> None:
    """Open molecule in an interactive viewer to set orientation interactively.

    Atom positions are written back to ``mol.graph`` in-place.  Sets
    ``mol.oriented = True`` so subsequent :func:`render` calls skip PCA
    auto-orientation.

    For cube-file molecules the cube grid alignment is handled automatically
    at render time via Kabsch rotation from original cube atom positions to
    the updated graph positions.

    Parameters
    ----------
    mol:
        Molecule returned by :func:`load`.
    viewer:
        Viewer backend: ``"vmol"`` (default, requires vmol) or ``"ase"``
        (requires ase).  For vmol, rotate with the mouse/arrows then press
        ``z`` to confirm and ``q`` to quit.  For ASE GUI, rotate then close
        the window to confirm.
    also:
        Optional list of additional Molecules that should receive the same
        rigid rotation about *mol*'s centroid.  Use this for overlay structures
        you want to track *mol*'s orientation (matters when combined with
        ``auto_align=False`` — otherwise Kabsch realigns anyway).
    """
    from xyzrender.viewer import rotate_with_viewer

    rot, c1, c2 = rotate_with_viewer(mol.graph, backend=viewer)
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

    # Propagate the same rigid transform to any "also" molecules so their
    # geometry tracks *mol* (needed when auto_align is off — otherwise Kabsch
    # re-aligns them anyway and this step is a harmless no-op).
    for other in also or ():
        nodes = list(other.graph.nodes())
        if not nodes:
            continue
        p = np.array([other.graph.nodes[n]["position"] for n in nodes], dtype=float)
        q = (p - c1) @ rot.T + c2
        for i, n in enumerate(nodes):
            other.graph.nodes[n]["position"] = tuple(q[i])
        other.oriented = True


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


def _tile_supercell_indices(
    subsets: list[list[int]],
    supercell: tuple[int, int, int],
    n_base: int,
) -> list[list[int]]:
    """Replicate index subsets across supercell replicas."""
    sc_m, sc_n, sc_l = supercell
    if not subsets or (sc_m, sc_n, sc_l) == (1, 1, 1):
        return subsets

    ii, jj, kk = np.mgrid[0:sc_m, 0:sc_n, 0:sc_l]
    offsets = (ii * sc_n * sc_l + jj * sc_l + kk).ravel() * n_base

    lens = list(map(len, subsets))
    flat_subs = np.concatenate(subsets)
    tiled_flat = (flat_subs[None, :] + offsets[:, None]).ravel()
    split_idx = np.cumsum(np.tile(lens, len(offsets)))[:-1]
    return list(map(np.ndarray.tolist, np.split(tiled_flat, split_idx)))


def _tile_pore_centroids_radii(
    centroids: list[tuple[float, float, float]],
    radii: list[float] | None,
    supercell: tuple[int, int, int],
    lattice: np.ndarray,
) -> tuple[list[tuple[float, float, float]], list[float] | None]:
    """Tile pore centroids and radii across supercell replicas.

    Parameters
    ----------
    centroids : list[tuple[float, float, float]]
        Original pore centroids (unit cell).
    radii : list[float] | None
        Original pore radii (unit cell).
    supercell : tuple[int, int, int]
        Supercell repetition counts (m, n, l).
    lattice : np.ndarray
        Lattice vectors as (3, 3) array with rows = a, b, c.

    Returns
    -------
    tuple
        (tiled_centroids, tiled_radii) where tiled_radii is None if input was None.
    """
    m_sc, n_sc, l_sc = supercell
    if not centroids or (m_sc, n_sc, l_sc) == (1, 1, 1):
        return centroids, radii

    from xyzrender.pore import _tile_positions

    c_arr = np.array(centroids, dtype=float)
    shifts, tiled_arr = _tile_positions(
        c_arr,
        lattice,
        ((0, m_sc), (0, n_sc), (0, l_sc)),
    )

    tiled_c = [tuple(pt) for pt in tiled_arr.reshape(-1, 3)]
    tiled_r = radii * shifts.shape[0] if radii is not None else None

    return tiled_c, tiled_r


def render(
    molecule: str | os.PathLike | Molecule,
    *,
    config: str | RenderConfig = "default",
    # --- Style (only when config is a preset name or file path) ---
    canvas_size: int | None = None,
    atom_scale: float | None = None,
    radius_scale: list[tuple[str | list[int], float]] | None = None,
    bond_width: float | None = None,
    atom_stroke_width: float | None = None,
    bond_color: str | None = None,
    bond_outline_color: str | None = None,
    bond_outline_width: float | None = None,
    ts_color: str | None = None,
    nci_color: str | None = None,
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
    atom_gradient_strength: float | None = None,
    bond_gradient_strength: float | None = None,
    vdw_gradient_strength: float | None = None,
    # --- Display ---
    hide_bonds: bool = False,
    unbond: list[str] | None = None,
    bond: list[str] | None = None,
    haptic: bool = False,
    hy: bool | list[int] | None = None,
    no_hy: bool = False,
    bo: bool | None = None,
    orient: bool | None = None,
    ref: str | os.PathLike | None = None,
    # --- Crystal display (when mol has cell_data) ---
    no_cell: bool = False,
    axes: bool | None = None,
    axis: str | None = None,
    supercell: tuple[int, int, int] = (1, 1, 1),
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
    cmap_palette: str | None = None,
    cmap_symm: bool = False,
    cbar: bool = False,
    # Per-atom fill opacity.  Either a ``{1-indexed atom: value}`` dict or a
    # selector list ``[(sel, value), ...]`` with the same grammar as
    # ``radius_scale`` (strings like "1-5,8", "M", "het"; or bare 1-indexed
    # lists).  Affects the atom circle only; adjacent bonds stay opaque.
    atom_opacity: dict[int, float] | list[tuple[str | list[int], float]] | None = None,
    # --- Annotations ---
    labels: list[str] | None = None,
    label_file: str | None = None,
    stereo: bool | list[str] = False,
    stereo_style: str = "atom",
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
    nci_mode: str | None = None,
    nci_cutoff: float | None = None,
    surface_style: str | None = None,
    # --- Convex hull ---
    hull: bool | str | list[int] | list[list[int]] | None = None,
    hull_color: str | list[str] | None = None,
    hull_opacity: float | None = None,
    hull_edge: bool | None = None,
    hull_edge_width_ratio: float | None = None,
    hull_color_type: str = "type",
    # --- Pore / face detection ---
    pore: bool = False,
    ring_max_size: int = 100,
    ring_min_size: int = 3,
    face_planarity: float = 0.25,
    pore_color: str | None = None,
    pore_opacity: float | None = None,
    # --- Molecule color ---
    mol_color: str | None = None,
    # --- Highlight ---
    highlight: str | list[int] | list[list[int] | str] | list[tuple] | None = None,
    # --- Style regions ---
    regions: list[tuple[str | list[int], str | RenderConfig]] | None = None,
    # --- Bond coloring ---
    bond_color_by_element: bool | None = None,
    bond_gradient: bool | None = None,
    # --- Depth of field ---
    dof: bool = False,
    dof_strength: float | None = None,
    glow: str | list[int] | None = None,
    glow_strength: float | None = None,
    # --- Overlay ---
    overlay: str | os.PathLike | Molecule | None = None,
    overlay_color: str | None = None,
    overlay_config: "OverlayConfig | None" = None,
    # --- Alignment (overlay subset alignment) ---
    align_atoms: str | list[int] | None = None,
    auto_align: bool | None = None,
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
    ref:
        Path to an orientation reference XYZ file.  If the file exists,
        the molecule is Kabsch-aligned to it and PCA auto-orientation is
        disabled regardless of *orient*.  If the file does not exist,
        current (possibly PCA-oriented) positions are saved to it.
        Not supported for periodic structures (raises ``ValueError``).
    unbond:
        Bond display rules.  A list of spec strings that hide bonds:
        categories (``"M"``, ``"sbm"``, ``"L"``, ``"het"``), element
        pairs (``"M-L"``, ``"Fe-het"``), pi-coordination (``"pi"``,
        ``"M-pi"``), element symbols (``"Li"``), atom indices
        (``"2"``), or index pairs (``"1-3"``).  Specs are 1-indexed.
        NCI / TS overlay edges are never removed by rules.
    bond:
        Force-show or add bonds as 1-indexed index-pair strings
        (``["4-5"]``).  Overrides ``unbond`` — a bond listed here
        will not be removed even if it matches an unbond rule.
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
    atom_opacity:
        Per-atom fill opacity.  Accepts either a ``{1-indexed atom: value}``
        dict (use for per-atom levels) or a selector list
        ``[(selector, value), ...]`` with the same grammar as ``radius_scale``
        — strings (``"1-5,8"``, ``"M"``, ``"het"``) are resolved against the
        molecular graph; bare lists are treated as 1-indexed atom indices.
        Affects the atom circle only; adjacent bonds stay fully opaque.
        Composes with overlay / ensemble opacity via ``min``.
    cmap_palette:
        Shared scalar palette override for atom colormaps and ESP surfaces.
        Defaults to ``viridis`` for ``cmap=...`` and ``rainbow`` for ESP
        when not specified explicitly.
    labels:
        Inline annotation spec strings (e.g. ``["1 2 d", "3 a", "1 NBO"]``).
    label_file:
        Path to an annotation file (same format as ``--label``).
    stereo:
        ``True`` for all stereochemistry labels, or a list of classes to show
        (``"point"``, ``"ez"``, ``"axis"``, ``"plane"``, ``"helix"``).
    stereo_style:
        Placement for R/S labels: ``"atom"`` (centered on atom) or ``"label"`` (offset near atom).
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
        Path to an ESP ``.cube`` or ``.cub`` file (density iso + ESP colour map).
    nci:
        Path to an NCI reduced-density-gradient ``.cube`` or ``.cub`` file.
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
    overlay:
        Second structure to overlay (path, ``Molecule``, or ``None``).  The
        overlay is Kabsch-aligned onto the primary (MCS fallback for different
        atom counts).  Mutually exclusive with crystal display and surfaces.
    overlay_color:
        Shortcut for ``overlay_config.color``; wins when both are set.
    overlay_config:
        :class:`~xyzrender.types.OverlayConfig` carrying per-overlay style
        overrides (``color``, ``opacity``, ``atom_scale``, ``bond_width``,
        ``atom_stroke_*``, ``bond_outline_*``, ``unbond``, ``bond``, ``show``,
        ``config``).  All fields are absolute (same semantics as on
        :class:`RenderConfig`) and individually optional — unset fields inherit
        the primary config.

        Precedence when multiple entry points are used at once: the flat
        ``overlay_color`` / ``opacity`` kwargs on :func:`render` override
        matching fields on *overlay_config*, which overrides the preset's
        ``overlay`` block, which overrides the ``OverlayConfig`` defaults.
    auto_align:
        ``True`` (default) runs Kabsch/MCS to align the overlay onto the
        primary.  ``False`` keeps each structure's raw coordinates; the
        interactive viewer rotation via :func:`orient` still propagates.
    opacity:
        Transparency 0 to 1.  Applied to the overlay when ``overlay`` is
        given, to the ensemble when the molecule is an ensemble, else to the
        active surface.  The three modes are mutually exclusive.

    Returns
    -------
    SVGResult
        Wrapper around the SVG string.  Displays inline in Jupyter.
    """
    from xyzrender.config import build_config
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

    # Supercell requires lattice/cell_data
    if supercell != (1, 1, 1) and mol.cell_data is None:
        raise ValueError("supercell requires an input with a unit cell (lattice).")

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
        # Pre-built RenderConfig — shallow copy so we don't mutate the caller's object.
        # Also detach mutable containers and the nested OverlayConfig (which _apply_overlay
        # mutates) so later field writes can't leak back to the caller's config.
        cfg = copy.copy(config)
        cfg.vectors = list(cfg.vectors)
        cfg.annotations = list(cfg.annotations)
        cfg.overlay = copy.copy(cfg.overlay)
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
            bond_outline_color=bond_outline_color,
            bond_outline_width=bond_outline_width,
            ts_color=ts_color,
            nci_color=nci_color,
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
            atom_gradient_strength=atom_gradient_strength,
            bond_gradient_strength=bond_gradient_strength,
            vdw_gradient_strength=vdw_gradient_strength,
            bo=bo,
            hide_bonds=hide_bonds,
            unbond=unbond,
            bond=bond,
            hy=hy,
            no_hy=no_hy,
            orient=_orient,
        )

    if haptic:
        cfg.haptic = True

    # --opacity steering: when an overlay is active it applies to the overlay
    # molecule (set later via _apply_overlay); otherwise it's a surface setting.
    # The two paths are mutually exclusive, so a single flag covers both.
    _surface_opacity = None if overlay is not None else opacity
    _apply_render_overlays(
        cfg,
        mol.graph,
        ts_bonds=ts_bonds,
        nci_bonds=nci_bonds,
        vdw=vdw,
        idx=idx,
        cmap=cmap,
        cmap_range=cmap_range,
        cmap_palette=cmap_palette,
        cmap_symm=cmap_symm,
        cbar=cbar,
        opacity=_surface_opacity,
        atom_opacity=atom_opacity,
    )

    from xyzrender.colors import resolve_color

    # --- Molecule color ---
    if mol_color is not None:
        cfg.mol_color = resolve_color(mol_color)

    # --- Highlight ---
    _apply_highlight(cfg, highlight=highlight)

    # --- Style regions (user + preset-defined) ---
    _apply_style_regions(cfg, mol.graph, regions=regions)

    # --- Bond coloring ---
    if ts_color is not None:
        cfg.ts_color = resolve_color(ts_color)
    if nci_color is not None:
        cfg.nci_color = resolve_color(nci_color)
    if bond_color_by_element is not None:
        cfg.bond_color_by_element = bond_color_by_element
    if bond_gradient is not None:
        cfg.bond_gradient = bond_gradient

    # --- Depth of field ---
    if dof:
        cfg.dof = True
    if dof_strength is not None:
        cfg.dof_strength = dof_strength
    if glow_strength is not None:
        if glow_strength < 0:
            msg = "glow_strength must be >= 0"
            raise ValueError(msg)
        cfg.glow_strength = glow_strength
    if glow is not None:
        cfg.glow_indices = sorted(_resolve_glow_indices(glow, mol.graph))

    # --- Per-atom radius scale ---
    if radius_scale is not None:
        cfg.radius_scale = radius_scale

    # --- Hull faces / pores: detect on unit cell BEFORE supercell expansion ---
    # This avoids running expensive cycle detection on a larger supercell graph.
    # Indices are tiled across supercell replicas after expansion.
    _unit_cell_hull_indices: list[list[int]] | None = None
    _unit_cell_n_base: int | None = None
    _hull_is_str = isinstance(hull, str)
    if (_hull_is_str and hull in {"faces", "face", "pores", "pore"}) or pore:
        from xyzrender.hull import resolve_hull_faces, resolve_hull_pores

        if _hull_is_str and hull in {"faces", "face"}:
            _unit_cell_hull_indices = resolve_hull_faces(
                mol.graph,
                max_size=ring_max_size,
                min_size=ring_min_size,
                cell_data=mol.cell_data,
                face_planarity=face_planarity,
            )
        if hull in {"pores", "pore"} or pore:
            # resolve_hull_pores handles detection, node mapping, and storing
            # centroids/radii on cfg — single path for both --hull pore and --pore.
            _pore_indices = resolve_hull_pores(
                mol.graph,
                cfg,
                max_size=ring_max_size,
                min_size=ring_min_size,
                cell_data=mol.cell_data,
            )
            if hull in {"pores", "pore"}:
                _unit_cell_hull_indices = _pore_indices
        if _unit_cell_hull_indices or cfg.pore_node_ids:
            _unit_cell_n_base = mol.graph.number_of_nodes()

    # --- Surface style ---
    if surface_style is not None:
        cfg.surface_style = surface_style

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

    # --- Orientation reference ---
    if ref is not None:
        ref_path = Path(ref)
        if ref_path.is_file():
            if mol.oriented:
                logger.warning("ref overrides interactive orientation (ref file %s exists)", ref_path)
            _apply_ref_orientation(rmol, ref_path, cfg)
        else:
            _apply_and_save_ref(rmol, cfg, ref_path)

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
            conformer_opacities=ens.opacities,
            conformer_graphs=ens.conformer_graphs,
            z_nudge=True,
        )
        rmol = Molecule(
            graph=merged_graph,
            cube_data=rmol.cube_data,
            cell_data=rmol.cell_data,
            oriented=rmol.oriented,
        )

    # --- Vectors (user-supplied + crystal axes) ---
    # axes=None (default) → show axes unless no_cell is set.
    _show_axes = (not no_cell) if axes is None else axes
    _combine_vector_sources(
        cfg,
        rmol.graph,
        vector=vector,
        vector_scale=vector_scale,
        vector_color=vector_color,
        cell_data=rmol.cell_data,
        axes=_show_axes,
    )

    # --- Cell / crystal config ---
    if rmol.cell_data is not None:
        _apply_cell_config(
            rmol,
            cfg,
            no_cell=no_cell,
            axis=axis,
            supercell=supercell,
            ghosts=ghosts,
            cell_color=cell_color,
            cell_width=cell_width,
            ghost_opacity=ghost_opacity,
            bo_explicit=bo,
        )
    elif "lattice" in mol.graph.graph:
        logger.info("Lattice found in graph; use load(..., cell=True) to draw the unit cell box")

    # --- Annotations ---
    if labels or label_file:
        from xyzrender.annotations import parse_annotations

        inline = [s.split() for s in labels] if labels else None
        cfg.annotations = parse_annotations(inline_specs=inline, file_path=label_file, graph=rmol.graph)
    if stereo:
        from xyzrender.stereo import build_stereo_annotations

        _cls = set(stereo) if isinstance(stereo, list) else None
        cfg.annotations.extend(build_stereo_annotations(rmol.graph, rs_style=stereo_style, classes=_cls))

    # --- Overlay ---
    if overlay_config is not None:
        # Explicit OverlayConfig overrides preset defaults; flat kwargs below
        # (overlay_color, opacity) still win over matching fields on it.
        cfg.overlay = overlay_config
    if auto_align is not None:
        cfg.auto_align = auto_align
    if overlay is not None:
        rmol = _apply_overlay(
            mol,
            rmol,
            cfg,
            overlay,
            overlay_color=overlay_color,
            overlay_opacity=opacity,
            align_atoms=align_atoms,
            has_surfaces=mo or dens or esp is not None or nci is not None,
        )

    # --- Warn about ignored surface-specific params ---
    if not mo and (mo_pos_color or mo_neg_color or mo_blur is not None or mo_upsample is not None or flat_mo):
        logger.warning("MO-specific params ignored (mo not active)")
    if not dens and dens_color is not None:
        logger.warning("dens_color ignored (dens not active)")
    if nci is None and nci_mode is not None:
        logger.warning("nci_mode ignored (no NCI surface)")
    if hull is None and (hull_color is not None or hull_opacity is not None or hull_edge is not None):
        logger.warning("hull params ignored (hull not active)")

    # --- Surfaces ---
    _validate_and_compute_surfaces(
        rmol,
        cfg,
        mo=mo,
        dens=dens,
        esp=esp,
        nci=nci,
        vdw=vdw,
        iso=iso,
        mo_pos_color=mo_pos_color,
        mo_neg_color=mo_neg_color,
        mo_blur=mo_blur,
        mo_upsample=mo_upsample,
        flat_mo=flat_mo,
        dens_color=dens_color,
        nci_mode=nci_mode,
        nci_cutoff=nci_cutoff,
        surface_style=surface_style,
    )

    # --- Bond rules (unbond / bond / haptic) ---
    if cfg.unbond or cfg.bond or cfg.haptic:
        from xyzrender.bond_rules import apply_bond_rules

        apply_bond_rules(rmol.graph, cfg)

    # --- Convex hull ---
    from xyzrender.hull import apply_hull_to_config

    if _unit_cell_hull_indices and _hull_is_str and hull in {"faces", "face", "pores", "pore"}:
        # Tile unit-cell face/pore indices across supercell replicas.
        from xyzrender.hull import normalize_hull_subsets

        subsets = normalize_hull_subsets(_unit_cell_hull_indices)
        if supercell != (1, 1, 1) and _unit_cell_n_base is not None:
            subsets = _tile_supercell_indices(subsets, supercell, _unit_cell_n_base)
        # Apply via the standard path — hull_ordered, colours, opacity all handled.
        apply_hull_to_config(
            cfg,
            hull,
            hull_color,
            hull_opacity,
            hull_edge,
            hull_edge_width_ratio,
            rmol.graph,
            face_planarity=face_planarity,
            precomputed_indices=subsets,
            hull_color_type=hull_color_type,
        )
    elif hull is not None:
        apply_hull_to_config(
            cfg,
            hull,
            hull_color,
            hull_opacity,
            hull_edge,
            hull_edge_width_ratio,
            rmol.graph,
            face_planarity=face_planarity,
            hull_color_type=hull_color_type,
        )

    # --- Pore spheres ---
    if pore or cfg.pore_node_ids:
        if cfg.pore_node_ids:
            # Tile pore node IDs and centroids across supercell replicas.
            if supercell != (1, 1, 1) and _unit_cell_n_base is not None:
                cfg.pore_node_ids = _tile_supercell_indices(cfg.pore_node_ids, supercell, _unit_cell_n_base)
                if cfg.pore_centroids:
                    _lat = np.array(mol.cell_data.lattice) if mol.cell_data else None
                    if _lat is not None:
                        cfg.pore_centroids, cfg.pore_radii = _tile_pore_centroids_radii(
                            cfg.pore_centroids,
                            cfg.pore_radii,
                            supercell,
                            _lat,
                        )
            cfg.pore_spheres = True
        if pore_color is not None:
            cfg.pore_sphere_color = pore_color
        if pore_opacity is not None:
            cfg.pore_sphere_opacity = pore_opacity

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
    gif_bounce: float | tuple[float, str] | None = None,
    gif_trj: bool = False,
    gif_ts: bool = False,
    gif_diffuse: bool = False,
    # --- Diffuse params ---
    diffuse_frames: int = 60,
    diffuse_noise: float = 0.3,
    diffuse_bonds: str = "fade",
    diffuse_rot: int | None = None,
    diffuse_reverse: bool = True,
    anchor: str | list[int] | None = None,
    # --- Common ---
    output: str | os.PathLike | None = None,
    gif_fps: int = 10,
    rot_frames: int = 120,
    ts_frame: int = 0,
    config: str | RenderConfig = "default",
    # --- Style (same as render(), only used when config is a string) ---
    canvas_size: int | None = None,
    atom_scale: float | None = None,
    radius_scale: list[tuple[str | list[int], float]] | None = None,
    bond_width: float | None = None,
    atom_stroke_width: float | None = None,
    bond_color: str | None = None,
    bond_outline_color: str | None = None,
    bond_outline_width: float | None = None,
    ts_color: str | None = None,
    nci_color: str | None = None,
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
    atom_gradient_strength: float | None = None,
    bond_gradient_strength: float | None = None,
    vdw_gradient_strength: float | None = None,
    hide_bonds: bool = False,
    unbond: list[str] | None = None,
    bond: list[str] | None = None,
    haptic: bool = False,
    hy: bool | list[int] | None = None,
    no_hy: bool = False,
    bo: bool | None = None,
    orient: bool | None = None,
    ref: str | os.PathLike | None = None,
    # --- Molecule color ---
    mol_color: str | None = None,
    # --- Highlight ---
    highlight: str | list[int] | list[list[int] | str] | list[tuple] | None = None,
    # --- Style regions ---
    regions: list[tuple[str | list[int], str | RenderConfig]] | None = None,
    # --- Bond coloring ---
    bond_color_by_element: bool | None = None,
    bond_gradient: bool | None = None,
    # --- Depth of field ---
    dof: bool = False,
    dof_strength: float | None = None,
    glow: str | list[int] | None = None,
    glow_strength: float | None = None,
    # --- Structural overlay (gif_rot only) ---
    overlay: str | os.PathLike | Molecule | None = None,
    overlay_color: str | None = None,
    overlay_config: "OverlayConfig | None" = None,
    auto_align: bool | None = None,
    # Applies to the overlay molecule (gif_rot only); mutually exclusive with surfaces.
    opacity: float | None = None,
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
    surface_style: str | None = None,
    # --- Convex hull / pore (gif_rot only) ---
    hull: bool | str | list[int] | list[list[int]] | None = None,
    hull_color: str | list[str] | None = None,
    hull_opacity: float | None = None,
    hull_edge: bool | None = None,
    hull_edge_width_ratio: float | None = None,
    hull_color_type: str = "type",
    pore: bool = False,
    ring_max_size: int = 100,
    ring_min_size: int = 3,
    face_planarity: float = 0.25,
    pore_color: str | None = None,
    pore_opacity: float | None = None,
    # --- Crystal / cell (gif_rot only, when molecule has cell_data) ---
    no_cell: bool = False,
    axes: bool | None = None,
    axis: str | None = None,
    supercell: tuple[int, int, int] = (1, 1, 1),
    ghosts: bool | None = None,
    cell_color: str | None = None,
    cell_width: float | None = None,
    ghost_opacity: float | None = None,
) -> GIFResult:
    """Render a molecule to an animated GIF and return a :class:`GIFResult`.

    The result displays the GIF inline in Jupyter via ``_repr_html_``.
    Access the file path via ``result.path``.

    At least one of *gif_rot*, *gif_bounce*, *gif_trj*, *gif_ts*, or *gif_diffuse* must be set.

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
    gif_bounce:
        Bounce rotation GIF. Either an amplitude in degrees (axis defaults
        to ``"y"``) or a ``(degrees, axis)`` tuple — e.g. ``50`` or
        ``(50, "xy")``. *axis* uses the same vocabulary as *gif_rot*.
        Mutually exclusive with *gif_rot*.
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
        render_diffuse_gif,
        render_rotation_gif,
        render_trajectory_gif,
        render_vibration_gif,
        render_vibration_rotation_gif,
    )

    if isinstance(gif_bounce, tuple):
        bounce_deg, bounce_ax = gif_bounce
    else:
        bounce_deg, bounce_ax = gif_bounce, None

    if not (gif_rot or bounce_deg is not None or gif_trj or gif_ts or gif_diffuse):
        msg = "render_gif: set gif_rot, gif_bounce, gif_trj=True, gif_ts=True, or gif_diffuse=True"
        raise ValueError(msg)

    if gif_ts and gif_trj:
        msg = "render_gif: gif_ts and gif_trj are mutually exclusive"
        raise ValueError(msg)

    if gif_diffuse and (gif_ts or gif_trj):
        msg = "render_gif: gif_diffuse is mutually exclusive with gif_ts / gif_trj"
        raise ValueError(msg)

    if bounce_deg is not None:
        if bounce_deg <= 0:
            msg = "render_gif: gif_bounce must be > 0"
            raise ValueError(msg)
        if gif_ts or gif_trj or gif_diffuse:
            msg = "render_gif: gif_bounce is mutually exclusive with gif_ts / gif_trj / gif_diffuse"
            raise ValueError(msg)
        if gif_rot:
            msg = (
                "render_gif: gif_bounce and gif_rot are mutually exclusive — use gif_bounce=(deg, axis) to set the axis"
            )
            raise ValueError(msg)

    if (mo or dens) and (gif_ts or gif_trj or gif_diffuse):
        active_surf = "mo" if mo else "dens"
        active_gif = "gif_ts" if gif_ts else ("gif_trj" if gif_trj else "gif_diffuse")
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

    for _name, _ax in (("gif_rot", gif_rot), ("gif_bounce axis", bounce_ax)):
        if not _ax or _ax in ROTATION_AXES:
            continue
        test = _ax.lstrip("-")
        if not (test.isdigit() and len(test) >= 3):
            msg = f"render_gif: invalid {_name} {_ax!r} — use 'x', 'y', 'z', or 3-digit Miller index"
            raise ValueError(msg)

    if rot_frames != 120 and not gif_rot and bounce_deg is None:
        logger.warning("rot_frames has no effect without gif_rot")

    # Resolve config
    _gif_graph = molecule.graph if isinstance(molecule, Molecule) else load(molecule).graph
    if not isinstance(config, str):
        cfg = copy.copy(config)
        cfg.vectors = list(cfg.vectors)
        cfg.annotations = list(cfg.annotations)
        cfg.overlay = copy.copy(cfg.overlay)
    else:
        cfg = build_config(
            config,
            canvas_size=canvas_size,
            atom_scale=atom_scale,
            bond_width=bond_width,
            atom_stroke_width=atom_stroke_width,
            bond_color=bond_color,
            bond_outline_color=bond_outline_color,
            bond_outline_width=bond_outline_width,
            ts_color=ts_color,
            nci_color=nci_color,
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
            atom_gradient_strength=atom_gradient_strength,
            bond_gradient_strength=bond_gradient_strength,
            vdw_gradient_strength=vdw_gradient_strength,
            bo=bo,
            hide_bonds=hide_bonds,
            unbond=unbond,
            bond=bond,
            hy=hy,
            no_hy=no_hy,
            orient=orient,
        )

    if haptic:
        cfg.haptic = True

    from xyzrender.colors import resolve_color

    # --- Molecule color ---
    if mol_color is not None:
        cfg.mol_color = resolve_color(mol_color)

    # --- Highlight ---
    _apply_highlight(cfg, highlight=highlight)

    # --- Style regions (user + preset-defined) ---
    _apply_style_regions(cfg, _gif_graph, regions=regions)

    # --- Bond coloring ---
    if ts_color is not None:
        cfg.ts_color = resolve_color(ts_color)
    if nci_color is not None:
        cfg.nci_color = resolve_color(nci_color)
    if bond_color_by_element is not None:
        cfg.bond_color_by_element = bond_color_by_element
    if bond_gradient is not None:
        cfg.bond_gradient = bond_gradient

    # --- Depth of field ---
    if dof:
        cfg.dof = True
    if dof_strength is not None:
        cfg.dof_strength = dof_strength
    if glow_strength is not None:
        if glow_strength < 0:
            msg = "render_gif: glow_strength must be >= 0"
            raise ValueError(msg)
        cfg.glow_strength = glow_strength
    if glow is not None:
        cfg.glow_indices = sorted(_resolve_glow_indices(glow, _gif_graph))

    # --- Per-atom radius scale ---
    if radius_scale is not None:
        cfg.radius_scale = radius_scale

    # --- Convex hull / pore (detection + config) ---
    from xyzrender.hull import apply_hull_to_config

    _hull_is_str = isinstance(hull, str)

    # --- Face/pore detection (needs molecule loading) ---
    if (_hull_is_str and hull in {"faces", "face"}) or pore:
        from xyzrender.hull import normalize_hull_subsets, resolve_hull_faces, resolve_hull_pores

        _mol = molecule if isinstance(molecule, Molecule) else load(molecule)
        _cd = _mol.cell_data
        _n_base = _gif_graph.number_of_nodes()

        if _hull_is_str and hull in {"faces", "face"}:
            face_idx = resolve_hull_faces(
                _gif_graph,
                max_size=ring_max_size,
                min_size=ring_min_size,
                cell_data=_cd,
                face_planarity=face_planarity,
            )
            if face_idx:
                subsets = normalize_hull_subsets(face_idx)
                _color_graph = _gif_graph
                if supercell != (1, 1, 1):
                    subsets = _tile_supercell_indices(subsets, supercell, _n_base)
                    _color_graph = None
                apply_hull_to_config(
                    cfg,
                    hull,
                    hull_color,
                    hull_opacity,
                    hull_edge,
                    hull_edge_width_ratio,
                    _color_graph,
                    face_planarity=face_planarity,
                    precomputed_indices=subsets,
                    hull_color_type=hull_color_type,
                )
        if pore:
            resolve_hull_pores(
                _gif_graph,
                cfg,
                max_size=ring_max_size,
                min_size=ring_min_size,
                cell_data=_cd,
            )
            if cfg.pore_node_ids:
                if supercell != (1, 1, 1):
                    cfg.pore_node_ids = _tile_supercell_indices(cfg.pore_node_ids, supercell, _n_base)
                    if cfg.pore_centroids and _cd is not None:
                        _lat = np.array(_cd.lattice)
                        cfg.pore_centroids, cfg.pore_radii = _tile_pore_centroids_radii(
                            cfg.pore_centroids,
                            cfg.pore_radii,
                            supercell,
                            _lat,
                        )
                cfg.pore_spheres = True
            if pore_color is not None:
                cfg.pore_sphere_color = pore_color
            if pore_opacity is not None:
                cfg.pore_sphere_opacity = pore_opacity

    # --- Hull (independent of face/pore detection) ---
    if hull is not None and not cfg.show_convex_hull:
        apply_hull_to_config(
            cfg,
            hull,
            hull_color,
            hull_opacity,
            hull_edge,
            hull_edge_width_ratio,
            _gif_graph,
            hull_color_type=hull_color_type,
        )

    # --- Surface style ---
    if surface_style is not None:
        cfg.surface_style = surface_style

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

    elif gif_diffuse:
        if ref_graph is None:
            from xyzrender.readers import load_molecule

            ref_graph, _ = load_molecule(str(mol_path))
        else:
            ref_graph = copy.deepcopy(ref_graph)
        if cfg.unbond or cfg.bond or cfg.haptic:
            from xyzrender.bond_rules import apply_bond_rules

            apply_bond_rules(ref_graph, cfg)
        from xyzrender.diffuse import parse_anchor

        render_diffuse_gif(
            ref_graph,
            cfg,
            str(gif_path),
            n_frames=diffuse_frames,
            noise=diffuse_noise,
            bonds=diffuse_bonds,
            reverse=diffuse_reverse,
            fps=gif_fps,
            rotation_axis=gif_rot,
            rotation_degrees=float(diffuse_rot) if diffuse_rot else 360.0,
            anchor=parse_anchor(anchor),
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

        # --- Orientation reference (gif_rot only) ---
        if ref is not None:
            _ref_path = Path(ref)
            _ref_mol = Molecule(graph=ref_graph)
            if _ref_path.is_file():
                _apply_ref_orientation(_ref_mol, _ref_path, cfg)
            else:
                _apply_and_save_ref(_ref_mol, cfg, _ref_path)
            ref_graph = _ref_mol.graph

        # --- Ensemble: build scratch merged graph (z_nudge=False — meaningless for rotation) ---
        if isinstance(molecule, Molecule) and molecule.ensemble is not None:
            from xyzrender.ensemble import merge_graphs as _ensemble_merge_graphs

            ens = molecule.ensemble
            ref_graph = _ensemble_merge_graphs(
                ref_graph,
                ens.positions,
                conformer_colors=ens.colors,
                conformer_opacities=ens.opacities,
                conformer_graphs=ens.conformer_graphs,
                z_nudge=False,
            )

        # --- Overlay alignment (gif_rot only) ---
        if overlay_config is not None:
            cfg.overlay = overlay_config
        if auto_align is not None:
            cfg.auto_align = auto_align
        if overlay is not None:
            # Disable PCA-orient inside _apply_overlay — gif_rot already handled orientation above.
            _prev_auto = cfg.auto_orient
            cfg.auto_orient = False
            _ov_base = molecule if isinstance(molecule, Molecule) else Molecule(graph=ref_graph)
            _ov_rmol = _apply_overlay(
                _ov_base,
                Molecule(graph=ref_graph),
                cfg,
                overlay,
                overlay_color=overlay_color,
                overlay_opacity=opacity,
                align_atoms=None,
                has_surfaces=False,
            )
            ref_graph = _ov_rmol.graph
            cfg.auto_orient = _prev_auto

        # Bond rules run after overlay merge so haptic sees both molecules'
        # aromatic rings (merged.graph["aromatic_rings"] is the union).
        if cfg.unbond or cfg.bond or cfg.haptic:
            from xyzrender.bond_rules import apply_bond_rules

            apply_bond_rules(ref_graph, cfg)

        # --- Vectors (user-supplied + crystal axes; gif_rot only) ---
        _cell_data_for_vecs = molecule.cell_data if isinstance(molecule, Molecule) else None
        _gif_show_axes = (not no_cell) if axes is None else axes
        _combine_vector_sources(
            cfg,
            ref_graph,
            vector=vector,
            vector_scale=vector_scale,
            vector_color=vector_color,
            cell_data=_cell_data_for_vecs,
            axes=_gif_show_axes,
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
                supercell=supercell,
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
            axis=bounce_ax or gif_rot or "y",
            bounce_degrees=float(bounce_deg) if bounce_deg is not None else None,
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
    ensemble_color: str | list[str] | None,
    n_conformers: int,
) -> list[str] | None:
    """Resolve ensemble colour spec to one hex string per conformer.

    Accepts a palette name (sampled across *n_conformers*), a single hex /
    named colour (broadcast), a comma-separated list, or an explicit list.
    Returns ``None`` when no colouring is requested (CPK default).
    """
    from xyzrender.colors import PALETTES, sample_palette

    if ensemble_color is None:
        return None
    if isinstance(ensemble_color, list):
        return [resolve_color(c) for c in ensemble_color]
    if ensemble_color in PALETTES:
        return sample_palette(ensemble_color, n_conformers)
    parts = [c.strip() for c in ensemble_color.split(",")]
    if len(parts) > 1:
        return [resolve_color(c) for c in parts]
    return [resolve_color(ensemble_color)] * n_conformers


def _build_ensemble_molecule(
    trajectory: str | os.PathLike,
    *,
    reference_frame: int = 0,
    max_frames: int | None = None,
    align_atoms: str | list[int] | None = None,
    ensemble_color: str | list[str] | None = None,
    ensemble_opacity: float | None = None,
    auto_align: bool = True,
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

    if auto_align:
        _align_0 = parse_atom_indices(align_atoms) if align_atoms is not None else None
        aligned_positions = ensemble_align(frames, reference_frame=reference_frame, align_atoms=_align_0)
    else:
        # --no-align: keep each frame's raw coordinates; no Kabsch step.
        aligned_positions = [np.array(fr["positions"], dtype=float) for fr in frames]

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

    n_conf = len(frames)
    conformer_colors = _resolve_ensemble_colors(ensemble_color, n_conf)

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


def _apply_highlight(
    cfg: "RenderConfig",
    *,
    highlight: "str | list[int] | list[list[int] | str] | list[tuple] | None" = None,
) -> None:
    """Apply highlight atom coloring to *cfg* (mutates in place).

    Accepts multiple forms (all atom indices are 1-indexed):

    - ``str``: single group, auto-color — ``"1-5,8"``
    - ``list[int]``: single group, auto-color — ``[1, 2, 3, 4, 5]``
    - ``list[str | list[int]]``: multi-group, auto-color — ``["1-5", "10-15"]``
    - ``list[tuple]``: multi-group with colors — ``[("1-5", "blue"), ...]``
    """
    if highlight is None:
        return

    from xyzrender.colors import resolve_color
    from xyzrender.types import HighlightGroup

    palette = cfg.highlight_colors
    groups: list[HighlightGroup] = []

    from typing import cast

    # Normalise into list of (indices_spec, color_or_None)
    raw_groups: list[tuple[str | list[int], str | None]]

    if isinstance(highlight, str):
        # Single group from string: "1-5,8"
        raw_groups = [(highlight, None)]
    elif isinstance(highlight, list) and highlight:
        first = highlight[0]
        if isinstance(first, int):
            # Single group from list[int]: [1, 2, 3, 4, 5]
            raw_groups = [(cast("list[int]", highlight), None)]
        elif isinstance(first, str):
            # Multi-group from list[str]: ["1-5", "10-15"]
            raw_groups = [(cast("str", s), None) for s in highlight]
        elif isinstance(first, list):
            # Multi-group from list[list[int]]: [[1,2,3], [5,6,7,8]] — auto-color
            raw_groups = [(cast("list[int]", sub), None) for sub in highlight]
        elif isinstance(first, tuple):
            # Multi-group from list[tuple]: [("1-5", "blue"), ([1,2,3], "red"), ...]
            raw_groups = []
            for entry in highlight:
                if isinstance(entry, tuple):
                    atoms_spec = entry[0]
                    color_spec = entry[1] if len(entry) > 1 else None
                    raw_groups.append((atoms_spec, color_spec))
                else:
                    msg = f"highlight entry must be a tuple, got {type(entry)}"
                    raise TypeError(msg)
        else:
            msg = f"unexpected highlight element type: {type(first)}"
            raise TypeError(msg)
    else:
        return

    seen: set[int] = set()
    auto_idx = 0
    for atoms_spec, color_spec in raw_groups:
        indices = parse_atom_indices(atoms_spec)

        overlap = seen & set(indices)
        if overlap:
            examples = sorted(overlap)[:5]
            msg = f"atom(s) {', '.join(str(i + 1) for i in examples)} appear in multiple highlight groups (1-indexed)"
            raise ValueError(msg)
        seen.update(indices)

        if color_spec is not None:
            color = resolve_color(color_spec)
        else:
            color = resolve_color(palette[auto_idx % len(palette)])
            auto_idx += 1

        groups.append(HighlightGroup(indices=indices, color=color))

    cfg.highlight_groups = groups


def _apply_style_regions(
    cfg: "RenderConfig",
    graph: "nx.Graph",
    *,
    regions: "list[tuple[str | list[int], str | RenderConfig]] | None" = None,
) -> None:
    """Resolve atom specs and apply style-region overrides to *cfg*.

    Handles both user-defined regions (from ``regions=`` parameter) and
    preset-defined regions (from the JSON ``"regions"`` key on *cfg*).

    *atoms_spec* is a string (``"1-5"``, ``"M"``, ``"Pt"``) resolved via
    selectors, or a 1-indexed ``list[int]``.  *config_spec* is a preset
    name, a :class:`RenderConfig`, or (for preset regions) a dict of
    overrides merged on top of the parent config.

    User-defined regions are applied first; preset regions skip atoms
    already claimed.
    """
    import copy

    from xyzrender.config import build_region_config, load_config
    from xyzrender.selectors import resolve_atom_indices
    from xyzrender.types import StyleRegion

    seen: set[int] = set()

    # Preset regions first — so user regions can override with a warning
    preset_claimed: set[int] = set()
    for spec in cfg.region_specs or {}:
        preset_claimed.update(resolve_atom_indices(spec, graph))

    # --- User-defined regions ---
    for atoms_spec, config_spec in regions or []:
        if isinstance(atoms_spec, str):
            indices = sorted(resolve_atom_indices(atoms_spec, graph))
        else:
            indices = parse_atom_indices(atoms_spec)

        # Error on user-vs-user overlap
        overlap = seen & set(indices)
        if overlap:
            examples = sorted(overlap)[:5]
            msg = f"atom(s) {', '.join(str(i + 1) for i in examples)} appear in multiple style regions (1-indexed)"
            raise ValueError(msg)
        # Warn on user-vs-preset overlap (user wins)
        preset_overlap = preset_claimed & set(indices)
        if preset_overlap:
            logger.warning(
                "style region overrides preset region for atom(s) %s",
                ", ".join(str(i + 1) for i in sorted(preset_overlap)[:5]),
            )
            preset_claimed -= preset_overlap
        seen.update(indices)

        if isinstance(config_spec, str):
            rcfg = build_region_config(config_spec)
        elif isinstance(config_spec, RenderConfig):
            rcfg = copy.copy(config_spec)
        else:
            msg = f"region config must be a preset name (str) or RenderConfig, got {type(config_spec)}"
            raise TypeError(msg)

        rcfg.style_regions = []
        cfg.style_regions.append(StyleRegion(indices=indices, config=rcfg))

    # --- Preset-defined regions (from JSON "regions" key) ---
    _pending_specs = cfg.region_specs or {}
    cfg.region_specs = None  # clear so they aren't resolved again if called twice
    for spec, region_def in _pending_specs.items():
        indices = sorted(resolve_atom_indices(spec, graph))
        free = [i for i in indices if i not in seen]
        if not free:
            continue
        rcfg = copy.copy(cfg)
        rcfg.style_regions = []
        rcfg.region_specs = None
        if isinstance(region_def, str):
            overrides = load_config(region_def)
        else:
            overrides = region_def
        for k, v in overrides.items():
            if hasattr(rcfg, k):
                setattr(rcfg, k, v)
            else:
                logger.warning("preset region %r: unknown config key %r (ignored)", spec, k)
        cfg.style_regions.append(StyleRegion(indices=free, config=rcfg))
        seen.update(free)


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
    cmap_palette: str | None = None,
    cmap_symm: bool = False,
    cbar: bool = False,
    opacity: float | None = None,
    atom_opacity: dict[int, float] | list[tuple[str | list[int], float]] | None = None,
) -> None:
    """Apply render()-specific overlays to cfg (mutates in place).

    All atom indices in ts_bonds, nci_bonds, vdw, atom_opacity are 1-indexed
    (user-facing); they are converted to 0-indexed storage on *cfg*.
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
    if cmap_palette is not None:
        cfg.cmap_palette = cmap_palette
    if cmap_symm:
        cfg.cmap_symm = True
    if cbar:
        cfg.cbar = True
    if opacity is not None:
        cfg.surface_opacity = opacity
    if atom_opacity is not None:
        cfg.atom_opacity = _resolve_atom_opacity(atom_opacity, graph)


def _resolve_atom_opacity(
    spec: dict[int, float] | list[tuple[str | list[int], float]],
    graph: "nx.Graph",
) -> dict[int, float]:
    """Resolve *spec* to a 0-indexed ``{atom_idx: opacity}`` dict.

    Accepts either:
    - a ``{1-indexed atom: value}`` dict — converted to 0-indexed, OR
    - a selector list ``[(selector, value), ...]`` with the same grammar as
      ``radius_scale`` — strings (``"1-5,8"``, ``"M"``, ``"het"``) are resolved
      against *graph*, bare lists are treated as 1-indexed atom indices.
      Later specs overwrite earlier ones for overlapping atoms.
    """
    if isinstance(spec, dict):
        return {int(k) - 1: float(v) for k, v in spec.items()}

    from xyzrender.selectors import resolve_atom_indices
    from xyzrender.utils import parse_atom_indices

    out: dict[int, float] = {}
    for sel, val in spec:
        if isinstance(sel, str):
            indices = resolve_atom_indices(sel, graph)
        else:
            indices = set(parse_atom_indices(sel))  # 1-indexed list → 0-indexed
        fval = float(val)
        for idx in indices:
            out[idx] = fval
    return out


def _resolve_glow_indices(
    spec: str | list[int],
    graph: "nx.Graph",
) -> set[int]:
    """Resolve glow atom selection to 0-indexed atom indices."""
    if isinstance(spec, str):
        from xyzrender.selectors import resolve_atom_indices

        return set(resolve_atom_indices(spec, graph))
    from xyzrender.utils import parse_atom_indices

    return set(parse_atom_indices(spec))


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
    axes: bool | None = None,
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


def _apply_ref_orientation(rmol: Molecule, ref_path: Path, cfg: "RenderConfig") -> None:
    """Kabsch-align *rmol* onto a saved reference XYZ.  Force-disables auto_orient.

    Centroid-dummy nodes (``symbol == "*"``, e.g. π-centroids added by
    ``--nci-detect``) are co-rotated by the same rigid transform so they
    track the real atoms.
    """
    if rmol.cell_data is not None:
        msg = "--ref is not supported for periodic structures"
        raise ValueError(msg)

    ref_mol = load(ref_path, quick=True)

    # Reference XYZ is real-only (to_xyz strips *).  Mobile graph may contain
    # * dummies — load all nodes and fit on real atoms only, but apply the
    # transform to every node so dummies stay locked to the structure.
    ref_nodes = [n for n in ref_mol.graph.nodes() if ref_mol.graph.nodes[n]["symbol"] != "*"]
    all_nodes = list(rmol.graph.nodes())
    real_local = [k for k, n in enumerate(all_nodes) if rmol.graph.nodes[n]["symbol"] != "*"]
    mol_nodes = [all_nodes[k] for k in real_local]

    ref_pos = np.array([ref_mol.graph.nodes[n]["position"] for n in ref_nodes], dtype=float)
    all_pos = np.array([rmol.graph.nodes[n]["position"] for n in all_nodes], dtype=float)

    from xyzrender.utils import mcs_kabsch_align

    # Fast path: same atom count and element sequence between real-atom subsets.
    # mcs_kabsch_align fits on the matched subset and applies the transform to
    # all of *all_pos*, which is exactly what we need to co-rotate the dummies.
    if len(ref_nodes) == len(mol_nodes) and all(
        ref_mol.graph.nodes[r]["symbol"] == rmol.graph.nodes[m]["symbol"]
        for r, m in zip(ref_nodes, mol_nodes, strict=True)
    ):
        aligned = mcs_kabsch_align(ref_pos, all_pos, list(range(len(ref_nodes))), real_local)
    else:
        # Different molecules — MCS alignment
        from xyzrender.mcs import find_mcs_mapping

        mapping = find_mcs_mapping(ref_mol.graph, rmol.graph)
        if mapping is None:
            msg = (
                f"--ref: no common substructure (>= 3 atoms) between "
                f"reference ({len(ref_nodes)} atoms) and molecule ({len(mol_nodes)} atoms)"
            )
            raise ValueError(msg)
        g1_ids, g2_ids = mapping
        matched_frac = len(g1_ids) / min(len(ref_nodes), len(mol_nodes))
        if matched_frac < 0.25:
            logger.warning(
                "--ref: only %d/%d atoms matched (%.0f%%) — alignment may be poor",
                len(g1_ids),
                min(len(ref_nodes), len(mol_nodes)),
                matched_frac * 100,
            )
        g1_idx = [ref_nodes.index(n) for n in g1_ids]
        g2_idx = [all_nodes.index(n) for n in g2_ids]
        aligned = mcs_kabsch_align(ref_pos, all_pos, g1_idx, g2_idx)

    for k, nid in enumerate(all_nodes):
        rmol.graph.nodes[nid]["position"] = tuple(float(v) for v in aligned[k])

    # Reference IS the orientation — --orient ignored
    cfg.auto_orient = False


def _apply_and_save_ref(rmol: Molecule, cfg: "RenderConfig", ref_path: Path) -> None:
    """Orient graph positions (PCA or already done by -I), then dump to XYZ.

    Here we PCA graph nodes for the saved file to match the rendered view.
    With -I, auto_orient is already False — this is just a dump.
    """
    if rmol.cell_data is not None:
        msg = "--ref is not supported for periodic structures"
        raise ValueError(msg)

    if cfg.auto_orient and rmol.graph.number_of_nodes() > 1:
        from xyzrender.utils import pca_orient

        nodes = list(rmol.graph.nodes())
        pos = np.array([rmol.graph.nodes[n]["position"] for n in nodes], dtype=float)
        pos = pca_orient(pos)
        for k, nid in enumerate(nodes):
            rmol.graph.nodes[nid]["position"] = tuple(pos[k].tolist())

    cfg.auto_orient = False
    rmol.to_xyz(ref_path, title="xyzrender orientation reference")


def _apply_overlay(
    mol: Molecule,
    rmol: Molecule,
    cfg: "RenderConfig",
    overlay: "str | os.PathLike | Molecule",
    *,
    overlay_color: str | None,
    overlay_opacity: float | None,
    align_atoms: "str | list[int] | None",
    has_surfaces: bool,
) -> Molecule:
    """Load, align, and merge an overlay molecule onto *rmol*.

    Validates mutual exclusivity with crystal and surface modes, PCA-orients
    the main molecule, Kabsch-aligns the overlay, resolves any per-overlay
    style overrides onto ``cfg.overlay``, and returns a new :class:`Molecule`
    with the merged graph.
    """
    from xyzrender.colors import resolve_color
    from xyzrender.overlay import align, merge_graphs
    from xyzrender.utils import parse_atom_indices, pca_orient

    if mol.cell_data is not None:
        msg = "overlay= is mutually exclusive with crystal/cell display"
        raise ValueError(msg)
    if has_surfaces:
        msg = "overlay= is mutually exclusive with surface rendering (mo/dens/esp/nci)"
        raise ValueError(msg)

    if isinstance(overlay, Molecule):
        overlay_mol = overlay
    else:
        _ov_charge = mol.graph.graph.get("total_charge", 0)
        _ov_mult = mol.graph.graph.get("multiplicity")
        overlay_mol = load(overlay, charge=_ov_charge, multiplicity=_ov_mult)
    g1 = rmol.graph
    g2 = copy.deepcopy(overlay_mol.graph)

    # PCA-orient g1 (the already-copied mol graph) to set the viewing frame.
    # Capture the (rotation, centroid) applied so we can mirror it onto g2
    # below when auto_align is off — otherwise mol2 stays in the file frame
    # while mol1 is rotated/centred, and the two separate visually.
    _pca_rot: np.ndarray | None = None
    _pca_centroid: np.ndarray | None = None
    if cfg.auto_orient and g1.number_of_nodes() > 1:
        nodes1 = list(g1.nodes())
        pos1 = np.array([g1.nodes[n]["position"] for n in nodes1], dtype=float)
        atom_mask = np.array([g1.nodes[n]["symbol"] != "*" for n in nodes1])
        fit_mask = atom_mask if not atom_mask.all() else None
        _fit_pos = pos1[fit_mask] if fit_mask is not None else pos1
        _pca_centroid = _fit_pos.mean(axis=0)
        pos1_oriented, _pca_rot = pca_orient(pos1, fit_mask=fit_mask, return_matrix=True)
        for k, nid in enumerate(nodes1):
            g1.nodes[nid]["position"] = tuple(float(v) for v in pos1_oriented[k])
    cfg.auto_orient = False

    if overlay_color is not None:
        cfg.overlay.color = resolve_color(overlay_color)
    if overlay_opacity is not None:
        cfg.overlay.opacity = overlay_opacity

    # Overlay-only bond rules applied pre-merge so index specs refer to mol2's
    # own 1-indexed atoms (not the renumbered merged-graph IDs).
    if cfg.overlay.unbond or cfg.overlay.bond:
        from xyzrender.bond_rules import apply_bond_rules

        _ov_cfg = copy.copy(cfg)
        _ov_cfg.unbond = list(cfg.overlay.unbond)
        _ov_cfg.bond = list(cfg.overlay.bond)
        _ov_cfg.haptic = False  # haptic is global; runs post-merge on the full graph
        apply_bond_rules(g2, _ov_cfg)

    if cfg.auto_align:
        _ov_align = parse_atom_indices(align_atoms) if align_atoms is not None else None
        aligned2 = align(g1, g2, align_atoms=_ov_align)
    else:
        # Keep mol2's raw coordinates — but mirror whatever rigid transform mol1
        # received during PCA-orientation so the two stay co-registered when the
        # files were already aligned.
        aligned2 = np.array([g2.nodes[n]["position"] for n in g2.nodes()], dtype=float)
        if _pca_rot is not None:
            aligned2 = (aligned2 - _pca_centroid) @ _pca_rot.T

    # Visibility filter applied AFTER alignment (so Kabsch uses the full
    # scaffold) and AFTER the overlay unbond/bond rules (so index specs refer
    # to the original 1-indexed overlay atoms).  Dropping nodes also drops
    # their incident edges — bonds to hidden atoms are removed cleanly.
    if cfg.overlay.show:
        from xyzrender.selectors import resolve_atom_indices

        nodes_before = list(g2.nodes())
        keep_0idx = resolve_atom_indices(",".join(cfg.overlay.show), g2)
        keep_mask = [idx in keep_0idx for idx in range(len(nodes_before))]
        aligned2 = aligned2[keep_mask]
        drop = [nid for idx, nid in enumerate(nodes_before) if not keep_mask[idx]]
        g2.remove_nodes_from(drop)

    merged = merge_graphs(g1, g2, aligned2, cfg)

    # Full-config escape hatch: when cfg.overlay.config is set, attach it as a
    # StyleRegion over the mol2 node IDs so every per-atom/bond field the
    # renderer reads via _acfg takes effect on the overlay.  Scalar shortcuts
    # from OverlayConfig still win via their per-node / per-edge overrides.
    if cfg.overlay.config is not None:
        from xyzrender.types import StyleRegion

        mol2_nodes = [nid for nid, d in merged.nodes(data=True) if d.get("molecule_index", 0) == 1]
        cfg.style_regions = [*cfg.style_regions, StyleRegion(indices=mol2_nodes, config=cfg.overlay.config)]

    return Molecule(
        graph=merged,
        cube_data=None,
        cell_data=None,
        oriented=True,
    )


def _validate_and_compute_surfaces(
    rmol: Molecule,
    cfg: "RenderConfig",
    *,
    mo: bool,
    dens: bool,
    esp: "str | os.PathLike | None",
    nci: "str | os.PathLike | None",
    vdw: "bool | list[int] | None",
    iso: float | None,
    mo_pos_color: str | None,
    mo_neg_color: str | None,
    mo_blur: float | None,
    mo_upsample: int | None,
    flat_mo: bool,
    dens_color: str | None,
    nci_mode: str | None,
    nci_cutoff: float | None,
    surface_style: str | None,
) -> None:
    """Validate surface flag combinations and compute active surfaces.

    Checks mutual exclusivity (surfaces vs hull vs vdw vs skeletal), verifies
    cube data availability, builds surface params, and runs the compute
    functions that populate *cfg* with contour data.
    """
    from xyzrender.config import build_surface_params, collect_surf_overrides

    iso_was_explicit = iso is not None

    # --- Skeletal-style validation ---
    if cfg.skeletal_style:
        if mo or dens or esp is not None or nci is not None:
            msg = "skeletal_style is mutually exclusive with surface rendering (mo/dens/esp/nci)"
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
        msg = "mo=True requires a .cube or .cub file loaded via load()"
        raise ValueError(msg)
    if dens and cube_data is None:
        msg = "dens=True requires a .cube or .cub file loaded via load()"
        raise ValueError(msg)
    if esp is not None and cube_data is None:
        msg = "esp= requires a density .cube or .cub file loaded via load()"
        raise ValueError(msg)
    if nci is not None and cube_data is None:
        msg = "nci= requires a density .cube or .cub file loaded via load()"
        raise ValueError(msg)
    has_mo = bool(mo)
    has_dens = bool(dens)
    has_esp = esp is not None
    has_nci = nci is not None

    if not (has_mo or has_dens or has_esp or has_nci):
        return

    surf_overrides = collect_surf_overrides(
        iso=iso,
        mo_pos_color=mo_pos_color,
        mo_neg_color=mo_neg_color,
        mo_blur=mo_blur,
        mo_upsample=mo_upsample,
        flat_mo=flat_mo,
        dens_color=dens_color,
        nci_mode=nci_mode,
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
    from xyzrender.surfaces import compute_dens_surface, compute_esp_surface, compute_mo_surface, compute_nci_surface

    if mo_params is not None and cube_data is not None:
        compute_mo_surface(rmol.graph, cube_data, cfg, mo_params)

    if dens_params is not None and cube_data is not None:
        compute_dens_surface(rmol.graph, cube_data, cfg, dens_params)

    if esp_params is not None and esp is not None and cube_data is not None:
        if cfg.cmap_range is not None and cfg.cmap_symm:
            msg = "--cmap-range and --cmap-symm are mutually exclusive"
            raise ValueError(msg)
        if cfg.surface_style != "solid":
            logger.info("ESP uses raster rendering; --surface-style %s is ignored", cfg.surface_style)
        esp_cube = parse_cube(str(esp))
        compute_esp_surface(rmol.graph, cube_data, esp_cube, cfg, esp_params)

    if nci_params is not None and nci is not None and cube_data is not None:
        nci_cube = parse_cube(str(nci))
        compute_nci_surface(rmol.graph, cube_data, nci_cube, cfg, nci_params, iso_was_explicit=iso_was_explicit)


def _apply_cell_config(
    mol: Molecule,
    cfg: RenderConfig,
    *,
    no_cell: bool,
    axis: str | None,
    supercell: tuple[int, int, int] = (1, 1, 1),
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
        from xyzrender.colors import resolve_color

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

    # Supercell replication (must occur before adding ghost atoms)
    _supercell_lattice = None
    _n_base = None
    if supercell != (1, 1, 1):
        lat = getattr(cell_data, "lattice", None)
        if lat is None:
            raise ValueError("supercell requires an input with a unit cell (lattice).")
        lat = np.array(lat, dtype=float)
        if lat.shape != (3, 3) or np.allclose(lat, 0.0):
            raise ValueError("supercell requires a non-zero 3x3 lattice matrix.")
        from xyzrender.crystal import build_supercell

        _n_base = mol.graph.number_of_nodes()
        mol.graph = build_supercell(mol.graph, cell_data, supercell)
        # Scaled lattice for ghost generation (ghosts = periodic images of the
        # supercell, not the unit cell).  cell_data stays as unit cell for the
        # cell-box overlay.
        _supercell_lattice = np.vstack(
            [
                supercell[0] * lat[0],
                supercell[1] * lat[1],
                supercell[2] * lat[2],
            ]
        )

    # Ghost (periodic image) atoms — default: on when cell_data is present
    _show_ghosts = ghosts if ghosts is not None else True
    if _show_ghosts:
        from xyzrender.crystal import add_crystal_images
        from xyzrender.types import CellData as _CellData

        ghost_cd = (
            _CellData(lattice=_supercell_lattice, cell_origin=cell_data.cell_origin)
            if _supercell_lattice is not None
            else cell_data
        )
        add_crystal_images(
            mol.graph,
            ghost_cd,
            supercell_repeats=supercell if _n_base is not None else None,
            unit_cell_data=cell_data if _n_base is not None else None,
            n_base=_n_base,
        )

    # Bond orders are not meaningful for periodic structures (xyzgraph bond
    # order assignment assumes isolated molecules).
    if bo_explicit:
        logger.warning("Bond orders are not supported for periodic structures (--bo ignored)")
    cfg.bond_orders = False


def _write_output(svg: str, output: Path, cfg: RenderConfig) -> None:
    """Write SVG to file, converting format based on extension."""
    ext = output.suffix.lower()
    if ext == ".svg":
        output.write_text(svg)
    elif ext == ".png":
        from xyzrender.export import svg_to_png

        svg_to_png(svg, str(output), size=cfg.canvas_size, dpi=getattr(cfg, "dpi", 300))
    elif ext == ".pdf":
        if cfg.dof:
            logger.warning("PDF output uses cairosvg which does not support SVG filters — --dof blur will not appear")
        from xyzrender.export import svg_to_pdf

        svg_to_pdf(svg, str(output))
    elif ext in (".tiff", ".tif"):
        from xyzrender.export import svg_to_tiff

        svg_to_tiff(svg, str(output), size=cfg.canvas_size, dpi=getattr(cfg, "dpi", 300))
    else:
        msg = f"Unsupported output format: {ext!r} (use .svg, .png, .pdf, or .tiff)"
        raise ValueError(msg)
