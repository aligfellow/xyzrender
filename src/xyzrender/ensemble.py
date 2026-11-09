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
        Graph used for every conformer when *conformer_graphs* is not given.
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

    n_frames = len(aligned_positions)
    if conformer_graphs is not None and len(conformer_graphs) != n_frames:
        msg = (
            "ensemble.merge_graphs: conformer_graphs length does not match "
            f"aligned_positions ({len(conformer_graphs)} != {n_frames})"
        )
        raise ValueError(msg)

    first_graph = conformer_graphs[0] if conformer_graphs is not None else reference_graph
    first_nodes = _node_list(first_graph)
    first_real_nodes = [n for n in first_nodes if first_graph.nodes[n].get("symbol") != "*"]
    first_centroid_nodes = [n for n in first_nodes if first_graph.nodes[n].get("symbol") == "*"]
    n_real = len(first_real_nodes)

    if aligned_positions[0].shape[0] != n_real:
        msg = (
            "ensemble.merge_graphs: position array length does not match "
            f"real atom count in reference graph (got {aligned_positions[0].shape[0]}, expected {n_real})"
        )
        raise ValueError(msg)

    merged = nx.Graph()
    merged.graph.update(first_graph.graph)
    merged_centroids: list[int] = []
    merged_centroid_sites: dict[int, tuple[int, ...]] = {}

    def _prepare_centroids(
        source: nx.Graph,
        centroid_nodes: list,
        node_map: dict,
        real_nodes: list,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Return aligned dummy positions and translate NCI centroid metadata."""
        real_position_index = {node: idx for idx, node in enumerate(real_nodes)}
        source_centroids = set(source.graph.get("nci_centroid", []))
        source_sites = source.graph.get("nci_centroid_sites", {})
        centroid_positions = []
        for old_id in centroid_nodes:
            new_id = node_map[old_id]
            sites = tuple(source_sites.get(old_id, ()))
            if sites and all(site in real_position_index for site in sites):
                site_positions = np.array([positions[real_position_index[site]] for site in sites])
                centroid_position = site_positions.mean(axis=0)
            else:
                centroid_position = np.asarray(source.nodes[old_id]["position"], dtype=float)
            centroid_positions.append(centroid_position)
            if old_id in source_centroids:
                merged_centroids.append(new_id)
                if sites:
                    merged_centroid_sites[new_id] = tuple(node_map[site] for site in sites)
        return np.asarray(centroid_positions, dtype=float).reshape((-1, 3))

    # Reference conformer (index 0): keep original node IDs.
    ref_map = {nid: nid for nid in first_real_nodes}
    ref_centroid_map = {nid: nid for nid in first_centroid_nodes}
    ref_all_map = {**ref_map, **ref_centroid_map}
    ref_positions = np.vstack(
        (
            aligned_positions[0],
            _prepare_centroids(
                first_graph,
                first_centroid_nodes,
                ref_all_map,
                first_real_nodes,
                aligned_positions[0],
            ),
        )
    )
    stamp_structure_nodes(merged, first_graph, ref_all_map, ref_positions, molecule_index=0)
    stamp_structure_edges(merged, first_graph, ref_all_map, molecule_index=0)

    # Additional conformers: copy node/edge attributes, renumbering node IDs.
    next_id = max(first_nodes) + 1 if first_nodes else 0
    for conf_idx in range(1, n_frames):
        pos = aligned_positions[conf_idx]
        frame_graph = conformer_graphs[conf_idx] if conformer_graphs is not None else reference_graph
        frame_nodes = _node_list(frame_graph)
        frame_real_nodes = [n for n in frame_nodes if frame_graph.nodes[n].get("symbol") != "*"]
        frame_centroid_nodes = [n for n in frame_nodes if frame_graph.nodes[n].get("symbol") == "*"]
        if pos.shape[0] != len(frame_real_nodes):
            msg = (
                "ensemble.merge_graphs: position array length does not match "
                f"real atom count in conformer {conf_idx} (got {pos.shape[0]}, expected {len(frame_real_nodes)})"
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
        id_map = {old: next_id + i for i, old in enumerate(frame_real_nodes)}
        next_id += len(frame_real_nodes)
        centroid_map = {old: next_id + i for i, old in enumerate(frame_centroid_nodes)}
        next_id += len(frame_centroid_nodes)
        all_map = {**id_map, **centroid_map}
        # Slight z-offset so conformers don't z-fight; scaled by index so later
        # frames sit further back than earlier ones.
        z_offset = conf_idx * _Z_NUDGE if z_nudge else 0.0

        all_positions = np.vstack(
            (
                pos,
                _prepare_centroids(frame_graph, frame_centroid_nodes, all_map, frame_real_nodes, pos),
            )
        )
        stamp_structure_nodes(
            merged,
            frame_graph,
            all_map,
            all_positions,
            molecule_index=conf_idx,
            color=color,
            opacity=opacity,
            z_offset=z_offset,
        )
        stamp_structure_edges(merged, frame_graph, all_map, molecule_index=conf_idx, color=color)
        merge_aromatic_rings(merged, frame_graph, id_map)

    if merged_centroids:
        merged.graph["nci_centroid"] = merged_centroids
        merged.graph["nci_centroid_sites"] = merged_centroid_sites
    else:
        merged.graph.pop("nci_centroid", None)
        merged.graph.pop("nci_centroid_sites", None)

    return merged
