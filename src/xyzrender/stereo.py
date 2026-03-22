"""Stereochemistry labeling wrapper using xyzgraph."""

from __future__ import annotations

from xyzgraph.stereo import annotate_stereo

from xyzrender.annotations import Annotation, AtomValueLabel, BondLabel, CentroidLabel

STEREO_CLASSES = frozenset({"point", "ez", "axis", "plane", "helix"})

# User-facing name → xyzgraph StereoSummary key
_CLASS_KEY = {
    "point": "point",
    "ez": "ez",
    "axis": "axial",
    "plane": "planar",
    "helix": "helical",
}


def build_stereo_annotations(
    graph,
    *,
    rs_style: str = "label",
    classes: set[str] | None = None,
) -> list[Annotation]:
    """Generate stereochemistry labels from a molecular graph.

    Parameters
    ----------
    classes:
        Subset of :data:`STEREO_CLASSES` to include.  ``None`` means all.
    """
    if rs_style not in {"label", "atom"}:
        raise ValueError("rs_style must be 'label' or 'atom'")
    if classes is not None:
        unknown = classes - STEREO_CLASSES
        if unknown:
            raise ValueError(
                f"Unknown stereo classes: {', '.join(sorted(unknown))}. Valid: {', '.join(sorted(STEREO_CLASSES))}"
            )

    # Resolve user-facing names to internal keys
    keys = {_CLASS_KEY[c] for c in classes} if classes is not None else None

    summary = annotate_stereo(graph)

    annotations: list[Annotation] = []

    if keys is None or "point" in keys:
        for entry in summary["point"]:
            if entry["label"] in {"R", "S"}:
                annotations.append(AtomValueLabel(entry["atom"], entry["label"], on_atom=(rs_style == "atom")))

    if keys is None or "ez" in keys:
        for entry in summary["ez"]:
            i, j = entry["bond"]
            annotations.append(BondLabel(i, j, entry["label"]))

    for key in ("axial", "helical"):
        if keys is None or key in keys:
            for entry in summary[key]:
                i, j = entry["atoms"]
                annotations.append(BondLabel(i, j, entry["label"]))

    if keys is None or "planar" in keys:
        for entry in summary["planar"]:
            annotations.append(CentroidLabel(tuple(entry["ring"]), entry["label"]))

    return annotations
