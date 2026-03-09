"""Molecule overlay: RMSD-minimising structural alignment and combined rendering.

Two molecules are aligned via the Kabsch algorithm so that mol2 is superimposed
onto mol1 in its coordinate frame.  The merged graph is rendered with the overlay
color (default: mediumorchid); mol1 atoms use the standard CPK palette and are
always on top when depths are equal (drawn last in SVG order).

Atom pairing is index-based: atom *i* in mol1 corresponds to atom *i* in mol2.
Both molecules must have the same number of atoms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from xyzrender.types import Color

if TYPE_CHECKING:
    import networkx as nx

# Push overlay atom z-positions back by this tiny amount (Å) so that mol1
# atoms are always rendered on top when depths coincide (SVG: last = front).
_Z_NUDGE: float = -1e-3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _node_list(graph: nx.Graph) -> list:
    return list(graph.nodes())


def _positions(graph: nx.Graph) -> tuple[np.ndarray, list]:
    nodes = _node_list(graph)
    pos = np.array([graph.nodes[n]["position"] for n in nodes], dtype=float)
    return pos, nodes


def _kabsch_rotation(p_centered: np.ndarray, q_centered: np.ndarray) -> np.ndarray:
    """Kabsch rotation matrix rot s.t. q_centered @ rot.T ≈ p_centered.

    Both arrays must already be mean-centred.  Ensures det(rot) = +1 (proper
    rotation, no reflection).
    """
    h = q_centered.T @ p_centered
    u, _, vt = np.linalg.svd(h)
    det = np.linalg.det(vt.T @ u.T)
    d_mat = np.diag([1.0, 1.0, det])
    return vt.T @ d_mat @ u.T


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def align(mol1_graph: nx.Graph, mol2_graph: nx.Graph) -> np.ndarray:
    """Align mol2 onto mol1 by index; return aligned positions for mol2 nodes.

    Atom *i* in mol1 is paired with atom *i* in mol2 — both molecules must
    have the same number of atoms.

    Parameters
    ----------
    mol1_graph, mol2_graph:
        NetworkX graphs.  This function does not mutate them.

    Returns
    -------
    np.ndarray, shape (n2, 3)
        Aligned 3-D positions for mol2 nodes in their original graph order.
    """
    pos1, nodes1 = _positions(mol1_graph)
    pos2, _nodes2 = _positions(mol2_graph)
    n1, n2 = len(nodes1), len(pos2)

    if n1 != n2:
        msg = f"overlay: mol1 has {n1} atoms, mol2 has {n2} — counts must match."
        raise ValueError(msg)

    c1 = pos1.mean(axis=0)
    c2 = pos2.mean(axis=0)
    rot = _kabsch_rotation(pos1 - c1, pos2 - c2)
    return (pos2 - c2) @ rot.T + c1


def merge_graphs(
    mol1_graph: nx.Graph,
    mol2_graph: nx.Graph,
    aligned_pos2: np.ndarray,
    overlay_color: str = "mediumorchid",
) -> nx.Graph:
    """Build a merged NetworkX graph containing both molecules.

    mol1 nodes keep their original integer IDs (0 … n1-1).
    mol2 nodes are renumbered to n1 … n1+n2-1 (consecutive).

    Node attributes added:
    - ``molecule_index``: 0 for mol1, 1 for mol2.
    - ``overlay``: ``True`` for mol2 atoms (renderer uses this for magenta).

    Edge attributes added:
    - ``molecule_index``: 0 or 1.
    - ``bond_color_override``: hex colour for mol2 bonds (30% darker than overlay_color).

    mol2 z-positions are nudged back by ``_Z_NUDGE`` Å so mol1 atoms render
    on top when projected depths coincide.
    """
    import networkx as nx

    n1 = mol1_graph.number_of_nodes()
    merged = nx.Graph()
    merged.graph.update(mol1_graph.graph)

    # mol1 atoms + bonds
    for nid in _node_list(mol1_graph):
        data = dict(mol1_graph.nodes[nid])
        data["molecule_index"] = 0
        merged.add_node(nid, **data)

    for i, j, d in mol1_graph.edges(data=True):
        merged.add_edge(i, j, **dict(d), molecule_index=0)

    # mol2 atoms + bonds
    node_ids2 = _node_list(mol2_graph)
    id_map = {old: n1 + k for k, old in enumerate(node_ids2)}

    for k, old_id in enumerate(node_ids2):
        data = dict(mol2_graph.nodes[old_id])
        data["molecule_index"] = 1
        data["overlay"] = True
        x, y, z = aligned_pos2[k]
        data["position"] = (float(x), float(y), float(z) + _Z_NUDGE)
        merged.add_node(id_map[old_id], **data)

    bond_color = Color.from_str(overlay_color).darken(0.30).hex
    for i, j, d in mol2_graph.edges(data=True):
        merged.add_edge(id_map[i], id_map[j], **dict(d), molecule_index=1, bond_color_override=bond_color)

    # Keep aromatic rings from mol1 only (mol2 ring node IDs are offset)
    if "aromatic_rings" in mol1_graph.graph:
        merged.graph["aromatic_rings"] = [set(r) for r in mol1_graph.graph["aromatic_rings"]]

    return merged
