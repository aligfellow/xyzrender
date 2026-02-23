"""Molecular input parsing."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from xyzgraph import DATA, build_graph, read_xyz_file

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    import networkx as nx

_Atoms: TypeAlias = list[tuple[str, tuple[float, float, float]]]


def load_molecule(
    path: str | Path,
    charge: int = 0,
    multiplicity: int | None = None,
    kekule: bool = False,
) -> nx.Graph:
    """Read molecular structure file and build graph.

    Supports .xyz natively; all other formats (ORCA .out, Gaussian .log,
    Q-Chem, etc.) are parsed via cclib.  Bond orders are always determined
    by xyzgraph.
    """
    p = str(path)
    logger.info("Loading %s", p)
    if p.endswith(".xyz"):
        logger.debug("Parsing as XYZ")
        graph = build_graph(read_xyz_file(p), charge=charge, multiplicity=multiplicity, kekule=kekule)
    else:
        logger.debug("Parsing as QM output via cclib")
        atoms, file_charge, file_mult = _parse_qm_output(p)
        c = charge if charge != 0 else file_charge
        m = multiplicity if multiplicity is not None else file_mult
        logger.debug("cclib: charge=%d, multiplicity=%s", c, m)
        graph = build_graph(atoms, charge=c, multiplicity=m, kekule=kekule)
    logger.info("Built graph: %d atoms, %d bonds", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def detect_nci(graph: nx.Graph) -> nx.Graph:
    """Detect non-covalent interactions and return decorated graph.

    Uses xyzgraph's NCI detection.  Returns a new graph with ``NCI=True``
    edges for each detected interaction.  Pi-system interactions use
    centroid dummy nodes (``symbol="*"``).
    """
    from xyzgraph import detect_ncis
    from xyzgraph.nci import build_nci_graph

    logger.info("Detecting NCI interactions")
    detect_ncis(graph)
    nci_graph = build_nci_graph(graph)
    n_nci = sum(1 for _, _, d in nci_graph.edges(data=True) if d.get("NCI"))
    logger.info("Detected %d NCI interactions", n_nci)
    return nci_graph


def load_ts_molecule(
    path: str | Path,
    charge: int = 0,
    multiplicity: int | None = None,
    mode: int = 0,
    ts_frame: int = 0,
    kekule: bool = False,
) -> tuple[nx.Graph, list[dict]]:
    """Load TS and detect forming/breaking bonds via graphRC.

    Accepts QM output files or multi-frame XYZ trajectories (e.g. IRC paths).
    Returns the TS graph (with ``TS=True`` edges) and the trajectory frames.
    """
    try:
        from graphrc import run_vib_analysis
    except ImportError:
        msg = "TS detection requires graphrc: pip install 'xyzrender[ts]'"
        raise ImportError(msg) from None

    logger.info("Running graphRC analysis on %s (ts_frame=%d)", path, ts_frame)
    results = run_vib_analysis(
        input_file=str(path),
        mode=mode,
        ts_frame=ts_frame,
        enable_graph=True,
        charge=charge,
        multiplicity=multiplicity,
        print_output=False,
    )

    graph = results["graph"]["ts_graph"]
    frames = results["trajectory"]["frames"]

    # Rebuild graph with Kekule bond orders if requested, copying TS attributes
    if kekule:
        ts_frame_data = frames[ts_frame]
        atoms = list(zip(ts_frame_data["symbols"], [tuple(p) for p in ts_frame_data["positions"]], strict=True))
        kekule_graph = build_graph(atoms, charge=charge, multiplicity=multiplicity, kekule=True)
        for i, j, d in graph.edges(data=True):
            if d.get("TS", False):
                if kekule_graph.has_edge(i, j):
                    kekule_graph[i][j].update({k: v for k, v in d.items() if k.startswith(("TS", "vib"))})
                else:
                    kekule_graph.add_edge(i, j, **{k: v for k, v in d.items() if k.startswith(("TS", "vib"))})
        graph = kekule_graph

    logger.info(
        "TS graph: %d atoms, %d bonds, %d frames", graph.number_of_nodes(), graph.number_of_edges(), len(frames)
    )
    return graph, frames


def rotate_with_viewer(graph: nx.Graph) -> None:
    """Open graph in v viewer for interactive rotation, update positions in-place.

    Writes a temp XYZ from current positions, launches v, and reads back
    the rotated coordinates.  All edge attributes (TS labels, bond orders, etc.)
    are preserved.
    """
    viewer = _find_viewer()
    logger.info("Opening viewer: %s", viewer)
    n = graph.number_of_nodes()
    atoms: _Atoms = [(graph.nodes[i]["symbol"], graph.nodes[i]["position"]) for i in range(n)]

    rotated_text = _run_viewer_with_atoms(viewer, atoms)

    if not rotated_text.strip():
        sys.exit("No output from viewer — press 'z' in v to output coordinates before closing.")

    rotated_atoms = _parse_auto(rotated_text)
    if not rotated_atoms or len(rotated_atoms) != n:
        sys.exit("Could not parse viewer output.")

    for i, (_sym, pos) in enumerate(rotated_atoms):
        graph.nodes[i]["position"] = pos


def _find_viewer() -> str:
    """Locate the v molecular viewer binary."""
    # Check PATH first (works if user has a symlink or v in PATH)
    v = shutil.which("v")
    if v:
        return v

    # Search ~/bin/ for v.* (e.g. v.2.2) — picks highest version
    import glob

    candidates = sorted(glob.glob(os.path.expanduser("~/bin/v.[0-9]*")))
    if candidates:
        return candidates[-1]

    sys.exit("Cannot find 'v' viewer — add it to your PATH or install in ~/bin/.")


def _run_viewer(viewer: str, xyz_path: str) -> str:
    """Launch v on an XYZ file and capture stdout."""
    result = subprocess.run([viewer, xyz_path], capture_output=True, text=True, check=False)
    return result.stdout


def _run_viewer_with_atoms(viewer: str, atoms: _Atoms) -> str:
    """Write atoms to temp XYZ, launch v, capture stdout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(f"{len(atoms)}\n\n")
        for sym, (x, y, z) in atoms:
            f.write(f"{sym}  {x: .6f}  {y: .6f}  {z: .6f}\n")
        tmp = f.name
    try:
        return _run_viewer(viewer, tmp)
    finally:
        os.unlink(tmp)


def apply_rotation(graph: nx.Graph, rx: float, ry: float, rz: float) -> None:
    """Rotate all atom positions in-place by Euler angles (degrees).

    Rotation is around the molecular centroid so the molecule stays centered.
    """
    nodes = list(graph.nodes())
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    # Rz @ Ry @ Rx
    rot = np.array(
        [
            [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
            [-sy, sx * cy, cx * cy],
        ]
    )
    positions = np.array([graph.nodes[n]["position"] for n in nodes])
    centroid = positions.mean(axis=0)
    rotated = (rot @ (positions - centroid).T).T + centroid
    for i, nid in enumerate(nodes):
        graph.nodes[nid]["position"] = tuple(rotated[i].tolist())


def apply_axis_angle_rotation(graph: nx.Graph, axis: np.ndarray, angle: float) -> None:
    """Rotate all atom positions in-place around an arbitrary axis (degrees).

    Uses Rodrigues' rotation formula for a clean rotation around a single
    axis vector. Rotation is around the molecular centroid.
    """
    nodes = list(graph.nodes())
    theta = np.radians(angle)
    k = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    k_cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot = c * np.eye(3) + s * k_cross + (1 - c) * np.outer(k, k)

    positions = np.array([graph.nodes[n]["position"] for n in nodes])
    centroid = positions.mean(axis=0)
    rotated = (rot @ (positions - centroid).T).T + centroid
    for i, nid in enumerate(nodes):
        graph.nodes[nid]["position"] = tuple(rotated[i].tolist())


def load_trajectory_frames(path: str | Path) -> list[dict]:
    """Load all frames from a multi-frame XYZ or QM output (cclib).

    Returns list of ``{"symbols": [...], "positions": [[x,y,z], ...]}``
    matching the graphRC frame format.
    """
    p = str(path)
    logger.info("Loading trajectory from %s", p)
    frames = _load_xyz_frames(p) if p.endswith(".xyz") else _load_qm_frames(p)
    logger.info("Loaded %d frames", len(frames))
    return frames


def load_stdin(charge: int = 0, multiplicity: int | None = None, kekule: bool = False) -> nx.Graph:
    """Read atoms from stdin — auto-detects XYZ and line-by-line formats."""
    return build_graph(_parse_auto(sys.stdin.read()), charge=charge, multiplicity=multiplicity, kekule=kekule)


def _parse_auto(text: str) -> list[tuple[str, tuple[float, float, float]]]:
    """Auto-detect format: standard XYZ or line-by-line (symbol/Z x y z)."""
    lines = text.strip().splitlines()
    if not lines:
        return []
    # Standard XYZ: first line is atom count
    try:
        n = int(lines[0].strip())
        if n > 0 and len(lines) >= n + 2:
            return _parse_xyz(text)
    except ValueError:
        pass
    # Line-by-line: "symbol x y z" or "Z x y z" (e.g. v pipe output)
    return _parse_lines(lines)


def _parse_xyz(text: str) -> list[tuple[str, tuple[float, float, float]]]:
    lines = text.strip().splitlines()
    n = int(lines[0])
    atoms = []
    for line in lines[2 : 2 + n]:
        s, x, y, z = line.split()[:4]
        atoms.append((s, (float(x), float(y), float(z))))
    return atoms


def _parse_lines(lines: list[str]) -> list[tuple[str, tuple[float, float, float]]]:
    """Parse line-by-line atom format: 'symbol x y z' or 'Z x y z'."""
    atoms = []
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except (ValueError, IndexError):
            continue
        # First field: element symbol or atomic number
        try:
            sym = DATA.n2s[int(parts[0])]
        except (ValueError, KeyError):
            sym = parts[0]
        atoms.append((sym, (x, y, z)))
    return atoms


def _parse_qm_output(path: str) -> tuple[_Atoms, int, int | None]:
    """Extract coordinates from any QM output file via cclib."""
    try:
        import cclib
    except ImportError:
        msg = "QM output parsing requires cclib"
        raise ImportError(msg) from None

    logging.getLogger("cclib").setLevel(logging.CRITICAL)
    parser = cclib.io.ccopen(path, loglevel=logging.CRITICAL)
    try:
        data = parser.parse()
    except Exception:
        # cclib may crash mid-parse but still have extracted coordinates
        logger.debug("cclib raised an error; using partial data")
        data = parser

    if not hasattr(data, "atomcoords") or not hasattr(data, "atomnos") or len(data.atomcoords) == 0:
        msg = f"No coordinates found in {path}"
        raise ValueError(msg)

    atoms: _Atoms = []
    for z, (x, y, zc) in zip(data.atomnos, data.atomcoords[-1], strict=True):
        atoms.append((DATA.n2s[int(z)], (float(x), float(y), float(zc))))

    return atoms, getattr(data, "charge", 0), getattr(data, "mult", None)


def _load_xyz_frames(path: str) -> list[dict]:
    """Read all frames from a multi-frame XYZ file."""
    from xyzgraph import count_frames_and_atoms

    n_frames, n_atoms = count_frames_and_atoms(path)
    logger.debug("XYZ file: %d frames, %d atoms per frame", n_frames, n_atoms)
    frames = []
    for i in range(n_frames):
        atoms = read_xyz_file(path, frame=i)
        frames.append(
            {
                "symbols": [a[0] for a in atoms],
                "positions": [list(a[1]) for a in atoms],
            }
        )
    return frames


def _load_qm_frames(path: str) -> list[dict]:
    """Extract all optimization steps from QM output via cclib."""
    try:
        import cclib
    except ImportError:
        msg = "QM output parsing requires cclib"
        raise ImportError(msg) from None

    logging.getLogger("cclib").setLevel(logging.CRITICAL)
    parser = cclib.io.ccopen(path, loglevel=logging.CRITICAL)
    try:
        data = parser.parse()
    except Exception:
        logger.debug("cclib raised an error; using partial data")
        data = parser
    symbols = [DATA.n2s[int(z)] for z in data.atomnos]
    coords = np.array(data.atomcoords)
    logger.debug("cclib trajectory: %d steps, %d atoms", len(coords), len(symbols))

    return [{"symbols": symbols, "positions": step.tolist()} for step in coords]
