"""Ensemble overlay: align and render multi-frame XYZ (conformers) together.

Unlike :mod:`xyzrender.overlay`, ensemble rendering keeps the standard atom
CPK palette and does not apply any special overlay colors or bond overrides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

# Push later conformers slightly back in z to avoid z-fighting.
_Z_NUDGE: float = -1e-3


def _node_list(graph: nx.Graph) -> list:
    return list(graph.nodes())


def _positions(graph: nx.Graph) -> tuple[np.ndarray, list]:
    nodes = _node_list(graph)
    pos = np.array([graph.nodes[n]["position"] for n in nodes], dtype=float)
    return pos, nodes


def _kabsch_rotation(p_centered: np.ndarray, q_centered: np.ndarray) -> np.ndarray:
    """Kabsch rotation matrix rot s.t. q_centered @ rot.T ≈ p_centered."""
    h = q_centered.T @ p_centered
    u, _, vt = np.linalg.svd(h)
    det = np.linalg.det(vt.T @ u.T)
    d_mat = np.diag([1.0, 1.0, det])
    return vt.T @ d_mat @ u.T


def align(reference_graph: nx.Graph, mobile_graph: nx.Graph) -> np.ndarray:
    """Align *mobile_graph* onto *reference_graph* by index; return aligned positions.

    Atom pairing is index-based: atom *i* in reference corresponds to atom *i*
    in mobile. Both graphs must have the same number of atoms/nodes.
    """
    pos_ref, nodes_ref = _positions(reference_graph)
    pos_mob, _nodes_mob = _positions(mobile_graph)
    n1, n2 = len(nodes_ref), len(pos_mob)

    if n1 != n2:
        msg = f"ensemble: reference has {n1} atoms, mobile has {n2} — counts must match."
        raise ValueError(msg)

    c1 = pos_ref.mean(axis=0)
    c2 = pos_mob.mean(axis=0)
    rot = _kabsch_rotation(pos_ref - c1, pos_mob - c2)
    return (pos_mob - c2) @ rot.T + c1


def merge_graphs(reference_graph: nx.Graph, aligned_positions: list[np.ndarray]) -> nx.Graph:
    """Merge *reference_graph* with additional conformers into a single graph.

    The returned graph contains:
    - the reference conformer nodes/bonds unchanged
    - one additional copy of all nodes/bonds per conformer, with only positions changed

    No overlay-specific styling attributes are added; the renderer will use
    the normal CPK palette based on element symbols.
    """
    import networkx as nx

    if not aligned_positions:
        return reference_graph

    nodes = _node_list(reference_graph)
    n_atoms = len(nodes)
    if n_atoms == 0:
        return reference_graph

    merged = nx.Graph()
    merged.graph.update(reference_graph.graph)

    # Reference nodes/bonds as-is
    for nid in nodes:
        merged.add_node(nid, **dict(reference_graph.nodes[nid]))
    for i, j, d in reference_graph.edges(data=True):
        merged.add_edge(i, j, **dict(d))

    next_id = n_atoms
    for conf_index, pos in enumerate(aligned_positions, start=1):
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (n_atoms, 3):
            msg = f"ensemble: expected aligned positions shape {(n_atoms, 3)}, got {pos.shape!r} for conformer {conf_index}"
            raise ValueError(msg)

        id_map = {old: next_id + k for k, old in enumerate(nodes)}

        for k, old_id in enumerate(nodes):
            data = dict(reference_graph.nodes[old_id])
            x, y, z = pos[k]
            data["position"] = (float(x), float(y), float(z) + conf_index * _Z_NUDGE)
            merged.add_node(id_map[old_id], **data)

        for i, j, d in reference_graph.edges(data=True):
            merged.add_edge(id_map[i], id_map[j], **dict(d))

        next_id += n_atoms

    # Preserve aromatic rings metadata from the reference conformer.
    if "aromatic_rings" in reference_graph.graph:
        merged.graph["aromatic_rings"] = [set(r) for r in reference_graph.graph["aromatic_rings"]]

    return merged

