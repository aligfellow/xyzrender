"""Post-alignment merge helpers for overlay and ensemble paths.

Shared by :mod:`xyzrender.overlay` (one extra structure) and
:mod:`xyzrender.ensemble` (N extra structures).

Both paths produce one merged graph: reference atoms keep their IDs, extras
are renumbered, and each extra's nodes / edges are stamped with the same
per-structure attribute set the renderer consumes — ``molecule_index``,
``structure_color``, ``structure_opacity``, ``structure_atom_scale``, stroke /
outline overrides, and (per edge) ``bond_color_override`` / width / outline
overrides.  See each helper's docstring for the exact contract.

``merged.graph["aromatic_rings"]`` is the union across every structure, with
IDs translated, so downstream consumers like :mod:`xyzrender.bond_rules` see
all rings (needed for haptic detection on overlays).

Alignment (Kabsch / MCS) stays in the overlay and ensemble modules because
the two paths genuinely differ; this module only handles the post-alignment
merge bookkeeping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xyzrender.colors import Color, bond_color_from_atom

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np


# Per-structure z-offset step (Å).  Applied with a different multiplier by
# overlay (constant) and ensemble (conformer index) but shared as a single
# constant so the two paths agree on the magnitude.
_Z_NUDGE: float = -1e-3


def stamp_structure_nodes(
    merged: nx.Graph,
    source: nx.Graph,
    id_map: dict,
    aligned_positions: np.ndarray,
    *,
    molecule_index: int,
    color: str | None = None,
    opacity: float | None = None,
    atom_scale: float | None = None,
    stroke_width: float | None = None,
    stroke_color: str | None = None,
    z_offset: float = 0.0,
) -> None:
    """Copy *source* nodes into *merged* with per-structure attributes.

    *id_map* maps each source node ID to its renumbered ID in *merged*.
    *aligned_positions* is an ``(n, 3)`` array indexed in ``id_map`` key order.
    The ``z_offset`` is added to each node's z coordinate to avoid z-fighting.
    The stroke and scale values are absolute: when set they replace (not modify)
    the primary ``RenderConfig`` equivalents for this structure's atoms.
    """
    for k, old_id in enumerate(id_map):
        data = dict(source.nodes[old_id])
        data["molecule_index"] = molecule_index
        x, y, z = aligned_positions[k]
        data["position"] = (float(x), float(y), float(z) + z_offset)
        if color is not None:
            data["structure_color"] = color
        if opacity is not None:
            data["structure_opacity"] = opacity
        if atom_scale is not None:
            data["structure_atom_scale"] = atom_scale
        if stroke_width is not None:
            data["structure_atom_stroke_width"] = stroke_width
        if stroke_color is not None:
            data["structure_atom_stroke_color"] = stroke_color
        merged.add_node(id_map[old_id], **data)


def stamp_structure_edges(
    merged: nx.Graph,
    source: nx.Graph,
    id_map: dict,
    *,
    molecule_index: int,
    color: str | None = None,
    bond_color: str | None = None,
    bond_width: float | None = None,
    outline_width: float | None = None,
    outline_color: str | None = None,
) -> None:
    """Copy *source* edges into *merged* (renumbered via *id_map*).

    *bond_color* (when given) is stamped as ``bond_color_override`` verbatim.
    Otherwise, when *color* (the atom colour) is given, each copied edge gets
    ``bond_color_override`` set to a 30 %-darkened hex so bonds render darker
    than atoms of the same hue.  This lets callers decouple atom and bond
    colour on a structure without removing the ergonomic default.

    The width / outline kwargs stamp absolute per-edge overrides
    (``bond_width_override``, ``bond_outline_width_override``,
    ``bond_outline_color_override``) that the renderer applies on top of the
    base / style-region config — an explicit overlay value wins.  Edges whose
    endpoints are not both in *id_map* are skipped (useful for per-frame
    ensemble graphs that contain out-of-range references).
    """
    extras: dict = {"molecule_index": molecule_index}
    if bond_color is not None:
        extras["bond_color_override"] = bond_color
    elif color is not None:
        extras["bond_color_override"] = bond_color_from_atom(Color.from_str(color))
    if bond_width is not None:
        extras["bond_width_override"] = bond_width
    if outline_width is not None:
        extras["bond_outline_width_override"] = outline_width
    if outline_color is not None:
        extras["bond_outline_color_override"] = outline_color

    for i, j, d in source.edges(data=True):
        if i in id_map and j in id_map:
            merged.add_edge(id_map[i], id_map[j], **dict(d), **extras)


def merge_aromatic_rings(merged: nx.Graph, source: nx.Graph, id_map: dict) -> None:
    """Append *source*'s aromatic rings to *merged*, translated through *id_map*.

    Call this once per extra structure (the reference's rings are copied by
    ``merged.graph.update(reference.graph)`` at the top of each merge).  A ring
    is skipped when any of its atoms is absent from *id_map* — this only
    happens for per-frame ensemble graphs whose topology differs.
    """
    rings: list = source.graph.get("aromatic_rings", [])
    if not rings:
        return
    existing = merged.graph.setdefault("aromatic_rings", [])
    for ring in rings:
        translated = {id_map[a] for a in ring if a in id_map}
        if len(translated) == len(ring):
            existing.append(translated)
