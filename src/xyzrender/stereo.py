"""Stereochemistry labeling wrapper using xyzgraph."""

from __future__ import annotations

from xyzgraph.stereo import annotate_stereo

from xyzrender.annotations import Annotation, AtomValueLabel, BondLabel


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

    for idx, label in summary["point"].items():
        if label in {"R", "S"}:
            annotations.append(AtomValueLabel(idx, label, on_atom=(rs_style == "atom")))

    for key in ("ez", "axial", "planar", "helical"):
        for (i, j), label in summary[key].items():
            annotations.append(BondLabel(i, j, label))

    return annotations
