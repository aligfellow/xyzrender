"""Ensemble overlay: align and merge multiple conformers into one graph.

Frames from a multi-frame trajectory are RMSD-aligned onto a reference frame
using the shared Kabsch algorithm from :mod:`xyzrender.overlay`.  The merged
graph can optionally apply per-conformer colours and opacity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from xyzrender.merge import (
    _Z_NUDGE,
    merge_aromatic_rings,
    stamp_structure_edges,
    stamp_structure_nodes,
)
from xyzrender.overlay import _node_list
from xyzrender.utils import kabsch_align

if TYPE_CHECKING:
    import networkx as nx


def align(
    frames: list[dict],
    *,
    reference_frame: int = 0,
    align_atoms: list[int] | None = None,
) -> list[np.ndarray]:
    """Align all trajectory *frames* onto *reference_frame*.

    Parameters
    ----------
    frames:
        List of ``{"symbols": [...], "positions": [[x,y,z], ...]}`` dicts as
        returned by :func:`xyzrender.readers.load_trajectory_frames`.
    reference_frame:
        Index of the reference frame.  All other frames are RMSD-aligned
        onto this frame via the Kabsch algorithm.
    align_atoms:
        Optional 0-indexed atom indices to fit on (min 3).  When given, only
        these atoms contribute to the Kabsch fit; the rotation is applied to
        all atoms.

    Returns
    -------
    list of np.ndarray
        One array per frame with aligned 3-D positions, in the same order as
        *frames*.  The reference frame positions are returned unchanged.
    """
    if not frames:
        msg = "ensemble.align: no frames provided"
        raise ValueError(msg)
    if not (0 <= reference_frame < len(frames)):
        msg = f"ensemble.align: reference_frame {reference_frame} out of range for {len(frames)} frames"
        raise ValueError(msg)

    ref_pos = np.array(frames[reference_frame]["positions"], dtype=float)
    n_atoms = ref_pos.shape[0]

    aligned: list[np.ndarray] = []

    for idx, frame in enumerate(frames):
        pos = np.array(frame["positions"], dtype=float)
        if pos.shape != ref_pos.shape:
            msg = f"ensemble.align: frame {idx} has shape {pos.shape}, expected {ref_pos.shape} from reference frame"
            raise ValueError(msg)
        if idx == reference_frame:
            aligned.append(ref_pos.copy())
            continue
        aligned.append(kabsch_align(ref_pos, pos, align_atoms=align_atoms))

    assert len(aligned) == len(frames)
    assert all(a.shape == (n_atoms, 3) for a in aligned)
    return aligned


def merge_graphs(
    reference_graph: nx.Graph,
    aligned_positions: list[np.ndarray] | np.ndarray,  # list or (n_conformers, n_atoms, 3)
    *,
    conformer_colors: list[str | None] | None = None,
    conformer_opacities: list[float | None] | None = None,
    conformer_graphs: list[nx.Graph] | None = None,
    z_nudge: bool = True,
) -> nx.Graph:
    """Merge *reference_graph* with additional conformers into a single graph.

    Parameters
    ----------
    reference_graph:
        The graph for the reference conformer (frame 0).
    aligned_positions:
        One (N, 3) position array per frame (including reference).
    conformer_colors, conformer_opacities:
        Optional per-conformer overrides (one value per frame, including the
        reference).  The reference frame's values are ignored (uses CPK / full
        opacity).  Non-reference atoms get ``structure_color`` /
        ``structure_opacity`` node attributes and bonds get the matching
        ``bond_color_override`` edge attribute (30 % darkened).
    conformer_graphs:
        Optional per-frame graphs (one per frame, including reference).  When
        given, each conformer uses its own graph's edges instead of copying
        the reference frame's edges.  Useful for trajectories where bonding
        or NCI interactions differ between frames.
    """
    import networkx as nx

    if len(aligned_positions) == 0:
        msg = "ensemble.merge_graphs: aligned_positions must contain at least one frame"
        raise ValueError(msg)

    all_nodes = _node_list(reference_graph)
    # Separate real atoms from NCI centroid dummy nodes (symbol="*").
    real_nodes = [n for n in all_nodes if reference_graph.nodes[n].get("symbol") != "*"]
    centroid_nodes = [n for n in all_nodes if reference_graph.nodes[n].get("symbol") == "*"]
    n_real = len(real_nodes)
    n_frames = len(aligned_positions)

    if aligned_positions[0].shape[0] != n_real:
        msg = (
            "ensemble.merge_graphs: position array length does not match "
            f"real atom count in reference graph (got {aligned_positions[0].shape[0]}, expected {n_real})"
        )
        raise ValueError(msg)

    merged = nx.Graph()
    merged.graph.update(reference_graph.graph)

    # Reference conformer (index 0): keep original node IDs.
    ref_map = {nid: nid for nid in real_nodes}
    stamp_structure_nodes(merged, reference_graph, ref_map, aligned_positions[0], molecule_index=0)
    stamp_structure_edges(merged, reference_graph, ref_map, molecule_index=0)

    # NCI centroid dummy nodes (reference frame only) keep their existing positions.
    for nid in centroid_nodes:
        data = dict(reference_graph.nodes[nid])
        data["molecule_index"] = 0
        merged.add_node(nid, **data)

    # Additional conformers: copy node/edge attributes, renumbering node IDs.
    next_id = max(all_nodes) + 1 if all_nodes else 0
    for conf_idx in range(1, n_frames):
        pos = aligned_positions[conf_idx]
        if pos.shape[0] != n_real:
            msg = (
                "ensemble.merge_graphs: position array length does not match "
                f"real atom count in reference graph (got {pos.shape[0]}, expected {n_real})"
            )
            raise ValueError(msg)

        color = (
            conformer_colors[conf_idx] if conformer_colors is not None and conf_idx < len(conformer_colors) else None
        )
        opacity = (
            conformer_opacities[conf_idx]
            if conformer_opacities is not None and conf_idx < len(conformer_opacities)
            else None
        )
        frame_graph = conformer_graphs[conf_idx] if conformer_graphs is not None else reference_graph

        id_map = {old: next_id + i for i, old in enumerate(real_nodes)}
        # Slight z-offset so conformers don't z-fight; scaled by index so later
        # frames sit further back than earlier ones.
        z_offset = conf_idx * _Z_NUDGE if z_nudge else 0.0

        stamp_structure_nodes(
            merged,
            reference_graph,  # source of node attributes (symbols, etc.) — per-frame positions come from `pos`
            id_map,
            pos,
            molecule_index=conf_idx,
            color=color,
            opacity=opacity,
            z_offset=z_offset,
        )
        stamp_structure_edges(merged, frame_graph, id_map, molecule_index=conf_idx, color=color)
        merge_aromatic_rings(merged, frame_graph, id_map)

        next_id += n_real

    return merged
