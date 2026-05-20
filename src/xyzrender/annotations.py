"""Annotation dataclasses and parser for xyzrender.

All user-facing atom indices are 1-indexed. Internally everything converts to 0-indexed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xyzrender.measure import _pos

logger = logging.getLogger(__name__)
_ORIGINAL_INDEX_ATTR = "_xyzrender_original_index"


def _fmt(val: float, spec: str) -> str:
    """Format a float, replacing ASCII hyphen-minus with the Unicode minus sign (U+2212)."""
    return format(val, spec).replace("-", "\u2212")


@dataclass(frozen=True)
class AtomValueLabel:
    """Custom text or value label drawn near an atom."""

    index: int  # 0-indexed
    text: str
    on_atom: bool = False  # True = centered on atom, False = offset label


@dataclass(frozen=True)
class BondLabel:
    """Distance or custom text label at bond/contact midpoint."""

    i: int  # 0-indexed
    j: int  # 0-indexed
    text: str


@dataclass(frozen=True)
class AngleLabel:
    """Angle arc at atom j (center) with value text."""

    i: int  # 0-indexed
    j: int  # 0-indexed, vertex
    k: int  # 0-indexed
    text: str


@dataclass(frozen=True)
class DihedralLabel:
    """Dihedral backbone path i→j→k→m with value text."""

    i: int  # 0-indexed
    j: int
    k: int
    m: int
    text: str


@dataclass(frozen=True)
class CentroidLabel:
    """Text label placed at the centroid of a set of atoms."""

    atoms: tuple[int, ...]  # 0-indexed
    text: str


Annotation = AtomValueLabel | BondLabel | AngleLabel | DihedralLabel | CentroidLabel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _original_index_map(graph) -> dict[int, int] | None:
    if not any(_ORIGINAL_INDEX_ATTR in data for _, data in graph.nodes(data=True)):
        return None
    return {int(data.get(_ORIGINAL_INDEX_ATTR, nid)): nid for nid, data in graph.nodes(data=True)}


def _check_atom(idx_1based: int, graph) -> int:
    """Resolve a 1-indexed user atom index to the current graph node ID."""
    mapping = _original_index_map(graph)
    if mapping is not None:
        original = idx_1based - 1
        if original not in mapping:
            raise ValueError(f"Atom index {idx_1based} not found in filtered render; it may have been excluded")
        return mapping[original]

    i = idx_1based - 1
    if i not in graph.nodes():
        n = graph.number_of_nodes()
        raise ValueError(f"Atom index {idx_1based} not found in molecule ({n} atoms, valid range 1-{n})")
    return i


def _warn_no_bond(i: int, j: int) -> None:
    """Warn once that an explicit i-j pair has no graph edge."""
    logger.warning("no bond between atoms %d and %d - placing label at midpoint", i + 1, j + 1)


def _parse_spec(tokens: list[str], graph) -> list[Annotation]:
    """Parse one annotation spec (one -l invocation or one file line).

    All indices in ``tokens`` are 1-indexed. Returns a list of Annotation objects
    (may expand to multiple for wildcard specs like ``i d``).
    Raises ValueError on invalid input.
    """
    from xyzrender.measure import bond_angle, bond_length, dihedral_angle

    if not tokens:
        return []

    n = len(tokens)
    t_last = tokens[-1].lower()

    try:
        raw_indices = [int(t) for t in tokens[:-1]]
    except ValueError as err:
        raise ValueError(f"Expected integer atom indices before command/label, got: {' '.join(tokens[:-1])!r}") from err

    atoms = [_check_atom(raw_i, graph) for raw_i in raw_indices]

    if n == 2:
        i0 = atoms[0]

        if t_last == "d":
            # 1 d → distances to all bonded neighbours
            result = []
            for j in sorted(graph.neighbors(i0)):
                d = bond_length(_pos(graph, i0), _pos(graph, j))
                result.append(BondLabel(i0, j, f"{_fmt(d, '.2f')}Å"))
            return result

        elif t_last == "a":
            # 1 a → all angles where atom 1 is the center
            nbrs = list(graph.neighbors(i0))
            result = []
            for a_idx, ni in enumerate(nbrs):
                for nk in nbrs[a_idx + 1 :]:
                    theta = bond_angle(_pos(graph, ni), _pos(graph, i0), _pos(graph, nk))
                    result.append(AngleLabel(ni, i0, nk, f"{_fmt(theta, '.1f')}°"))
            return result

        else:
            # 1 value → custom atom label (preserve original case)
            return [AtomValueLabel(i0, tokens[-1])]

    if n == 3:
        i0, i1 = atoms

        if t_last == "a":
            raise ValueError(
                f"Angle requires 3 atom indices before 'a' (got 2). "
                f"Did you mean: {raw_indices[0]} <center> {raw_indices[1]} a ?"
            )

        if not graph.has_edge(i0, i1):
            _warn_no_bond(i0, i1)

        if t_last == "d":
            # 1 2 d → distance label on bond/contact 1-2
            d = bond_length(_pos(graph, i0), _pos(graph, i1))
            return [BondLabel(i0, i1, f"{_fmt(d, '.2f')}Å")]
        else:
            # 1 2 value → custom bond label (preserve original case)
            return [BondLabel(i0, i1, tokens[-1])]

    if n == 4:
        if t_last == "a":
            # 1 2 3 a → angle label, 2 is middle
            i0, i1, i2 = atoms
            theta = bond_angle(_pos(graph, i0), _pos(graph, i1), _pos(graph, i2))
            return [AngleLabel(i0, i1, i2, f"{_fmt(theta, '.1f')}°")]

        raise ValueError(f"4-token spec must end with 'a' (angle). Got: {' '.join(tokens)!r}")

    if n == 5:
        if t_last in ("t", "tor", "dih"):
            # 1 2 3 4 t → dihedral label
            i0, i1, i2, i3 = atoms
            phi = dihedral_angle(_pos(graph, i0), _pos(graph, i1), _pos(graph, i2), _pos(graph, i3))
            return [DihedralLabel(i0, i1, i2, i3, f"{_fmt(phi, '.1f')}°")]

        raise ValueError(f"5-token spec must end with 't'/'tor'/'dih'. Got: {' '.join(tokens)!r}")

    raise ValueError(f"Cannot parse annotation spec: {' '.join(tokens)!r}")


def _tokenize(line: str) -> list[str]:
    """Split line on whitespace and/or commas."""
    return line.replace(",", " ").split()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_annotations(
    inline_specs: list[list[str]] | None,
    file_path: str | None,
    graph,
) -> list[Annotation]:
    """Parse inline specs and/or annotation file into a flat list of Annotation objects.

    Indices in specs are 1-indexed (user-facing); stored annotations are 0-indexed.
    Hard errors on invalid atom indices. Warns (continues) for missing bond edges.
    """
    all_specs: list[list[str]] = []

    if inline_specs:
        all_specs.extend(inline_specs)

    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {file_path}")
        with path.open() as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = _tokenize(line)
                if not tokens:
                    continue
                # Skip lines whose first token is not an integer (CSV headers, etc.)
                try:
                    int(tokens[0])
                except ValueError:
                    continue
                all_specs.append(tokens)

    result: list[Annotation] = []
    for tokens in all_specs:
        try:
            result.extend(_parse_spec(tokens, graph))
        except ValueError as e:
            raise ValueError(f"annotation '{' '.join(tokens)}': {e}") from e

    return result


def load_cmap(file_path: str, graph) -> dict[int, float]:
    """Load atom property colormap from a strict idx-value file.

    Format (1-indexed atom indices, strict):
        1  +0.512
        2  -0.234

    Blank lines and lines starting with ``#`` are skipped. Non-integer first
    tokens are silently skipped (handles CSV headers). Any other malformed line
    is a hard error.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Colormap file not found: {file_path}")

    result: dict[int, float] = {}

    with path.open() as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = _tokenize(line)
            if not tokens:
                continue

            # Skip non-integer first tokens (CSV headers)
            try:
                raw_idx = int(tokens[0])
            except ValueError:
                continue

            if len(tokens) < 2:
                raise ValueError(f"cmap line {lineno}: missing value for atom {raw_idx}")

            try:
                val = float(tokens[1])
            except ValueError:
                raise ValueError(f"cmap line {lineno}: cannot parse value {tokens[1]!r} as float") from None

            try:
                idx = _check_atom(raw_idx, graph)
            except ValueError as exc:
                raise ValueError(f"cmap line {lineno}: atom index {raw_idx} not found in molecule") from exc

            result[idx] = val

    return result


# ---------------------------------------------------------------------------
# Vector arrows
# ---------------------------------------------------------------------------


def load_vectors(
    path: str | Path | dict | list,
    graph,
    default_color: str = "firebrick",
) -> list:
    """Load vector arrows from a JSON file and resolve their origins.

    Each entry in the JSON array may have:

    ``anchor``
        Controls how ``origin`` is interpreted.  Can be set at the top level
        (applies to all arrows in the file) or per-entry (overrides the
        file-level value for that arrow):

        * ``"tail"`` (default) — ``origin`` is the arrow tail (start point)
        * ``"center"`` — ``origin`` is the midpoint; the arrow is drawn from
          ``origin - scaled_vec/2`` to ``origin + scaled_vec/2``

        Useful for force/dipole vectors where you want the atom coordinate to
        sit at the center of the arrow rather than the base.

    ``origin``
        How to place the arrow tail:

        * ``"com"`` — centroid (mean position) of all atoms in *graph*
        * Integer (1-based atom index) — position of that atom
        * ``[x, y, z]`` — explicit Cartesian coordinates (Å)

        Defaults to ``"com"`` when omitted.

    ``vector``
        3-component list giving direction and magnitude (Å or any consistent
        unit).  **Required.**

    ``color``
        CSS hex (``"#ff0000"``) or named color (``"red"``).  Default
        ``"#444444"``.

    ``label``
        Optional text placed near the arrowhead.  Default ``""`` (no label).

    ``scale``
        Per-arrow length scale factor multiplied on top of the global
        ``--vector-scale``.  Default ``1.0``.

    Returns a list of :class:`~xyzrender.types.VectorArrow` objects.

    Example JSON
    ------------
    ::

        [
            {"origin": "com", "vector": [1.2, 0.0, 0.5], "color": "#e63030", "label": "μ"},
            {"origin": 3, "vector": [0.0, 0.8, 0.0], "color": "steelblue"},
            {"origin": [0, 0, 0], "vector": [0.5, 0.5, 0.5]},
        ]
    """
    import json
    import logging

    from xyzrender.colors import resolve_color
    from xyzrender.types import VectorArrow

    _log = logging.getLogger(__name__)

    if isinstance(path, (dict, list)):
        raw = path
    else:
        with Path(path).open() as fh:
            raw = json.load(fh)

    # Accept either a bare array or an object with optional top-level "anchor" key.
    anchor = "tail"
    if isinstance(raw, dict):
        anchor_raw = raw.get("anchor", "tail")
        if anchor_raw not in ("tail", "center"):
            msg = f"Vector file {path!r}: 'anchor' must be 'tail' or 'center', got {anchor_raw!r}"
            raise ValueError(msg)
        anchor = anchor_raw
        raw = raw.get("vectors", [])
        if not isinstance(raw, list):
            msg = f"Vector file {path!r}: 'vectors' must be a JSON array"
            raise ValueError(msg)
    elif not isinstance(raw, list):
        msg = f"Vector file {path!r}: expected a JSON array or an object with a 'vectors' key at the top level"
        raise ValueError(msg)

    node_ids = list(graph.nodes())
    real_ids = [i for i in node_ids if graph.nodes[i].get("symbol", graph.nodes[i].get("element", "")) != "*"]
    if not real_ids:
        real_ids = node_ids
    positions = np.array([graph.nodes[i]["position"] for i in real_ids], dtype=float)
    centroid = positions.mean(axis=0)
    _log.debug("load_vectors COM: %s (from %d real atoms)", centroid, len(real_ids))

    arrows: list[VectorArrow] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            msg = f"Vector file {path!r}: entry {idx} must be a JSON object"
            raise ValueError(msg)

        # --- vector (required) ---
        if "vector" not in entry:
            msg = f"Vector file {path!r}: entry {idx} is missing required key 'vector'"
            raise ValueError(msg)
        vec_raw = entry["vector"]
        if not (isinstance(vec_raw, list) and len(vec_raw) == 3):
            msg = f"Vector file {path!r}: entry {idx} 'vector' must be a list of 3 numbers"
            raise ValueError(msg)
        try:
            vec = np.array([float(v) for v in vec_raw])
        except (TypeError, ValueError) as exc:
            msg = f"Vector file {path!r}: entry {idx} 'vector' contains non-numeric value"
            raise ValueError(msg) from exc

        # --- origin (optional, default "com") ---
        origin_raw = entry.get("origin", "com")
        atom_idx: int | None = None
        if origin_raw == "com":
            origin = centroid.copy()
        elif isinstance(origin_raw, int):
            try:
                atom_idx = _check_atom(origin_raw, graph)
            except ValueError as exc:
                msg = (
                    f"Vector file {path!r}: entry {idx} 'origin' atom index {origin_raw} "
                    f"is out of range or not present in the rendered molecule ({exc})"
                )
                raise ValueError(msg) from exc
            origin = np.array(graph.nodes[atom_idx]["position"], dtype=float)
        elif isinstance(origin_raw, list) and len(origin_raw) == 3:
            try:
                origin = np.array([float(v) for v in origin_raw])
            except (TypeError, ValueError) as exc:
                msg = f"Vector file {path!r}: entry {idx} 'origin' contains non-numeric value"
                raise ValueError(msg) from exc
        else:
            msg = (
                f"Vector file {path!r}: entry {idx} 'origin' must be 'com', a 1-based atom "
                f"index (integer), or a list of 3 coordinates"
            )
            raise ValueError(msg)

        # --- optional fields ---
        try:
            color = resolve_color(entry.get("color", default_color))
        except ValueError as exc:
            msg = f"Vector file {path!r}: entry {idx} invalid color: {exc}"
            raise ValueError(msg) from exc

        label = str(entry.get("label", ""))
        try:
            per_scale = float(entry.get("scale", 1.0))
        except (TypeError, ValueError) as exc:
            msg = f"Vector file {path!r}: entry {idx} 'scale' must be a number"
            raise ValueError(msg) from exc

        # Per-entry anchor overrides the file-level default
        entry_anchor = entry.get("anchor", anchor)
        if entry_anchor not in ("tail", "center"):
            msg = f"Vector file {path!r}: entry {idx} 'anchor' must be 'tail' or 'center', got {entry_anchor!r}"
            raise ValueError(msg)

        arrows.append(
            VectorArrow(
                vector=vec,
                origin=origin,
                color=color,
                label=label,
                scale=per_scale,
                anchor=entry_anchor,
                host_atom=atom_idx,
            )
        )

    _log.info("Loaded %d vector arrows from %s", len(arrows), path)
    return arrows
