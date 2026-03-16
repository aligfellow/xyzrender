"""Stereochemistry labeling wrapper (R/S and E/Z) using xyzgraph."""

from __future__ import annotations

import logging

from xyzgraph.stereo import annotate_stereo

from xyzrender.annotations import AtomValueLabel, BondLabel, Annotation

logger = logging.getLogger(__name__)


def build_stereo_annotations(
    graph,
    *,
    include_rs: bool = True,
    include_ez: bool = True,
    rs_style: str = "label",
) -> list[Annotation]:
    """Generate stereochemistry labels from a molecular graph."""
    if rs_style not in {"label", "atom"}:
        raise ValueError("rs_style must be 'label' or 'atom'")

    if graph.graph.get("stereo_hint", False):
        logger.warning(
            "Input contains stereochemistry annotations; --stereo uses 3D geometry only and may disagree."
        )

    annotations: list[Annotation] = []

    annotate_stereo(graph)

    if include_rs:
        for idx, data in graph.nodes(data=True):
            label = data.get("stereo_rs") or data.get("stereo")
            if label in {"R", "S"}:
                annotations.append(AtomValueLabel(idx, label, on_atom=(rs_style == "atom")))
    if include_ez:
        for i, j, data in graph.edges(data=True):
            label = data.get("stereo_ez") or data.get("stereo")
            if label in {"E", "Z"}:
                annotations.append(BondLabel(i, j, label))
        for i, j, data in graph.edges(data=True):
            label = data.get("stereo_axial")
            if label in {"Rₐ", "Sₐ"}:
                annotations.append(BondLabel(i, j, label))
        for i, j, data in graph.edges(data=True):
            label = data.get("stereo_planar")
            if label in {"Rₚ", "Sₚ"}:
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
