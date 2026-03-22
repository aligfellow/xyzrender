"""Stereochemistry labeling wrapper using xyzgraph."""

from __future__ import annotations

from xyzgraph.stereo import annotate_stereo

from xyzrender.annotations import Annotation, AtomValueLabel, BondLabel, CentroidLabel


def build_stereo_annotations(
    graph,
    *,
    rs_style: str = "label",
) -> list[Annotation]:
    """Generate stereochemistry labels from a molecular graph."""
    if rs_style not in {"label", "atom"}:
        raise ValueError("rs_style must be 'label' or 'atom'")

    summary = annotate_stereo(graph)

    annotations: list[Annotation] = []

    for entry in summary["point"]:
        if entry["label"] in {"R", "S"}:
            annotations.append(AtomValueLabel(entry["atom"], entry["label"], on_atom=(rs_style == "atom")))

    for entry in summary["ez"]:
        i, j = entry["bond"]
        annotations.append(BondLabel(i, j, entry["label"]))

    for key in ("axial", "helical"):
        for entry in summary[key]:
            i, j = entry["atoms"]
            annotations.append(BondLabel(i, j, entry["label"]))

    for entry in summary["planar"]:
        annotations.append(CentroidLabel(tuple(entry["ring"]), entry["label"]))

    return annotations
