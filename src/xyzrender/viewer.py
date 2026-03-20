"""Interactive v-viewer integration for xyzrender.

Provides :func:`rotate_with_viewer` which opens the molecule in the ``v``
viewer, lets the user rotate it interactively, then reads back the new
coordinates so subsequent rendering uses the chosen orientation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np

if TYPE_CHECKING:
    import networkx as nx
    from vmol import Vmol

    from xyzrender.config import RenderConfig
    from xyzrender.types import CellData

_Atoms: TypeAlias = list[tuple[str, tuple[float, float, float]]]


def rotate_with_viewer(
    graph: nx.Graph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """Open graph in v viewer for interactive rotation, update positions in-place.

    Writes a temp XYZ from current positions, launches v, and reads back
    the rotated coordinates.  All edge attributes (TS labels, bond orders, etc.)
    are preserved.  If the graph has a lattice, it is rotated by the same
    transformation and the cell origin is updated accordingly.

    Parameters
    ----------
    graph:
        Molecular graph whose node positions are updated in-place.

    Returns
    -------
    tuple of (rot, c1, c2) : (ndarray, ndarray, ndarray)
        Kabsch rotation matrix and centroid before/after rotation (in Å).
        Returns ``(None, None, None)`` if the user quit without pressing z.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        from vmol import vmol as viewer
    except ImportError:
        msg = "Interactive viewer requires vmol: `pip install xyzrender[v]` or pip install vmol"
        raise ImportError(msg) from None

    logger.info("Using viewer: %s", viewer)
    n = graph.number_of_nodes()
    symbols = [graph.nodes[i]["symbol"] for i in range(n)]
    orig_pos = np.array([graph.nodes[i]["position"] for i in range(n)], dtype=float)
    lattice = graph.graph.get("lattice")

    atoms: _Atoms = list(zip(symbols, [tuple(row) for row in orig_pos], strict=True))
    rotated_text = _run_viewer_with_atoms(viewer, atoms, lattice=lattice)

    if not rotated_text or not rotated_text.strip():
        logger.warning("No output from viewer.")
        return None, None, None

    # Extract rotation matrix from vmol 'u' output (rot:r00,r01,...,r22)
    # and strip rotation lines so _parse_auto sees clean XYZ.
    rot = None
    xyz_lines: list[str] = []
    for line in rotated_text.splitlines():
        if line.startswith("rot:"):
            vals = [float(v) for v in line[4:].split(",")]
            rot = np.array(vals, dtype=float).reshape(3, 3)
        elif line.startswith("rotation>"):
            continue  # skip verbose rotation lines
        else:
            xyz_lines.append(line)

    from xyzrender.readers import _parse_auto

    rotated_atoms = _parse_auto("\n".join(xyz_lines))
    if not rotated_atoms or len(rotated_atoms) != n:
        logger.warning("Could not parse viewer output.")
        return None, None, None

    new_pos = np.array([pos for _sym, pos in rotated_atoms], dtype=float)
    for i in range(n):
        graph.nodes[i]["position"] = tuple(new_pos[i])

    if rot is None:
        logger.warning("No rotation matrix from viewer.")
        return None, None, None

    c1 = orig_pos.mean(axis=0)
    c2 = new_pos.mean(axis=0)

    # Check if the viewer applied cell wrapping (atoms moved relative to
    # each other, not just rotated).
    rotated_orig = (rot @ (orig_pos - c1).T).T + c2
    rmsd = float(np.sqrt(np.mean(np.sum((new_pos - rotated_orig) ** 2, axis=1))))
    wrapped = rmsd > 0.1

    if lattice is not None:
        lat = np.array(lattice, dtype=float)
        origin = np.array(graph.graph.get("lattice_origin", np.zeros(3)), dtype=float)
        # Rotation matrix from vmol is exact — apply to lattice regardless
        # of wrapping (wrapping only affects atom positions, not the lattice).
        graph.graph["lattice"] = (rot @ lat.T).T
        graph.graph["lattice_origin"] = rot @ (origin - c1) + c2

    if wrapped:
        logger.info("Cell wrapping detected (RMSD=%.3f Å), rebuilding bonds", rmsd)
        from xyzgraph import build_graph as _build_graph

        new_graph = _build_graph(
            list(zip(symbols, [tuple(row) for row in new_pos], strict=True)),
            charge=0,
            multiplicity=None,
            kekule=False,
            quick=True,
        )
        graph.remove_edges_from(list(graph.edges()))
        graph.add_edges_from(new_graph.edges(data=True))

    return rot, c1, c2


def orient_hkl_to_view(graph: nx.Graph, cell_data: "CellData", axis_str: str, cfg: "RenderConfig") -> None:
    """Rotate *graph* and *cell_data* so that the [hkl] direction points along +z.

    Parameters
    ----------
    graph:
        Molecular graph whose node positions are updated in-place.
    cell_data:
        Crystal cell data whose lattice and origin are updated in-place.
    axis_str:
        3-digit Miller index string, optionally prefixed with ``-`` (e.g. ``'111'``, ``'-110'``).
    cfg:
        Render configuration object.

    Raises
    ------
    ValueError
        If *axis_str* is not a valid 3-digit Miller index or resolves to a zero vector.
    """
    hkl = axis_str.lstrip("-")
    if not (hkl.isdigit() and len(hkl) >= 3):
        msg = f"axis: expected a 3-digit Miller index string (e.g. '111'), got {axis_str!r}"
        raise ValueError(msg)
    h, k_idx, l_idx = int(hkl[0]), int(hkl[1]), int(hkl[2])
    v = h * cell_data.lattice[0] + k_idx * cell_data.lattice[1] + l_idx * cell_data.lattice[2]
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-10:
        msg = f"axis [{hkl}] has zero length (h={h}, k={k_idx}, l={l_idx})"
        raise ValueError(msg)
    v = v / v_norm
    z = np.array([0.0, 0.0, 1.0])
    cos_a = float(np.clip(np.dot(v, z), -1.0, 1.0))
    if abs(cos_a - 1.0) < 1e-9:
        rot_view: np.ndarray = np.eye(3)
    elif abs(cos_a + 1.0) < 1e-9:
        rot_view = np.diag([1.0, -1.0, -1.0])
    else:
        ax = np.cross(v, z)
        ax = ax / np.linalg.norm(ax)
        s_a = float(np.sqrt(max(0.0, 1.0 - cos_a**2)))
        ax_cross = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        rot_view = cos_a * np.eye(3) + s_a * ax_cross + (1 - cos_a) * np.outer(ax, ax)
    node_ids = list(graph.nodes())
    pos = np.array([graph.nodes[i]["position"] for i in node_ids], dtype=float)
    centroid = pos.mean(axis=0)
    pos_rot = (rot_view @ (pos - centroid).T).T + centroid
    for idx, nid in enumerate(node_ids):
        graph.nodes[nid]["position"] = tuple(pos_rot[idx].tolist())
    from xyzrender.utils import _apply_rot_to_vecs

    cell_data.lattice, cell_data.cell_origin = _apply_rot_to_vecs(
        rot_view, cell_data.lattice, cell_data.cell_origin, centroid
    )
    if hasattr(cfg, "vectors"):
        for vec in cfg.vectors:
            vec.vector, vec.origin = _apply_rot_to_vecs(rot_view, vec.vector, vec.origin, centroid)


def _run_viewer(viewer: Vmol, mol: dict, extra_args: list[str] | None = None) -> str:
    """Launch v on an input mol and capture stdout."""
    return viewer.capture(mols=mol, args=(extra_args or []))


def _run_viewer_with_atoms(viewer: Vmol, atoms: _Atoms, lattice: np.ndarray | None = None) -> str:
    """Launch v and capture stdout.

    If *lattice* is a diagonal (orthogonal) box, passes ``cell:b{a},{b},{c}``
    to v so the cell frame is shown in the viewer too.

    Parameters
    ----------
    viewer:
        v-viewer instance.
    atoms:
        List of ``(symbol, (x, y, z))`` tuples.
    lattice:
        Optional ``(3, 3)`` lattice matrix to pass as a cell argument.

    Returns
    -------
    str
        Captured stdout from the viewer.
    """
    q, r = zip(*atoms, strict=True)
    mol = {"q": q, "r": r, "name": "Rotate molecule with mouse / arrows and press q / Esc to confirm"}

    # print rotation matrix (u) then coordinates (z) before exiting
    extra: list[str] = ["exitcom:uz", "colors:cpk"]

    if lattice is not None:
        # v accepts the 3x3 matrix as 9 comma-separated values
        flat = lattice.flatten()
        extra.append("cell:" + ",".join(f"{v:.6f}" for v in flat))
    return _run_viewer(viewer, mol, extra)
