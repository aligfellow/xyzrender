"""Stereochemistry labeling wrapper (R/S and E/Z) using xyzgraph."""

from __future__ import annotations

from xyzgraph.stereo import annotate_stereo

from xyzrender.annotations import Annotation, AtomValueLabel, BondLabel

_EDGE_KEYS = {
    "stereo_ez": {"E", "Z"},
    "stereo_axial": {"Rₐ", "Sₐ"},
    "stereo_planar": {"Rₚ", "Sₚ"},
}


def build_stereo_annotations(
    graph,
    *,
    rs_style: str = "label",
) -> list[Annotation]:
    """Generate stereochemistry labels from a molecular graph."""
    if rs_style not in {"label", "atom"}:
        raise ValueError("rs_style must be 'label' or 'atom'")

    annotate_stereo(graph)

    annotations: list[Annotation] = []

    for idx, data in graph.nodes(data=True):
        label = data.get("stereo_rs") or data.get("stereo")
        if label in {"R", "S"}:
            annotations.append(AtomValueLabel(idx, label, on_atom=(rs_style == "atom")))

    for i, j, data in graph.edges(data=True):
        for key, valid in _EDGE_KEYS.items():
            label = data.get(key)
            if label in valid:
                annotations.append(BondLabel(i, j, label))

    for axis in graph.graph.get("stereo_axes", []):
        try:
            i = int(axis["i"])
            j = int(axis["j"])
            label = str(axis["label"])
        except Exception:
            continue
        annotations.append(BondLabel(i, j, label))

    return annotations
