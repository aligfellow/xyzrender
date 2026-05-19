"""Element-category resolver for atom selection.

Provides a mini-language for selecting atoms by element symbol or
category (``M``, ``sbm``, ``L``, ``het``).  Used by :mod:`bond_rules`
and available for future reuse in style regions, highlights, etc.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from xyzgraph import DATA

if TYPE_CHECKING:
    import networkx as nx

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

_STATIC_CATEGORIES: dict[str, frozenset[str]] = {
    "M": frozenset(DATA.metals),
    "sbm": frozenset(DATA.sblock_metals),
    "L": frozenset(s for s in DATA.s2n if s not in DATA.metals),
    "het": frozenset(s for s in DATA.s2n if s not in DATA.metals and s not in ("C", "H")),
}

# Element symbols recognised by xyzgraph (1- or 2-char, title-case).
_ALL_SYMBOLS: frozenset[str] = frozenset(DATA.s2n.keys())


def normalize_token(token: str) -> str:
    """Normalise a category/element token and validate it.

    Categories (``M``, ``sbm``, ``het``, ``pi``, ``L``) are matched
    case-insensitively.  Element symbols are title-cased (``fe`` → ``Fe``).

    Raises :class:`ValueError` if *token* is not a recognised category
    or element symbol.
    """
    low = token.lower()
    # Check categories first (case-insensitive)
    for cat in ("sbm", "het", "pi"):
        if low == cat:
            return cat
    if low == "m":
        return "M"
    if low == "l":
        return "L"
    # Try as element symbol (title-case)
    title = token.capitalize()
    if title in _ALL_SYMBOLS:
        return title
    raise ValueError(
        f"Unknown category or element symbol {token!r}. "
        f"Valid categories: M, sbm, L, het, pi. "
        f"Or use an element symbol (Fe, Li, O, …)."
    )


def resolve_element_set(token: str) -> frozenset[str]:
    """Resolve a category or element token to a set of element symbols.

    Parameters
    ----------
    token:
        One of:

        * ``"M"`` - all metals (``DATA.metals``)
        * ``"sbm"`` - s-block metals (``DATA.sblock_metals``)
        * ``"L"`` - non-metals (complement of metals present in *all known symbols*)
        * ``"het"`` - heteroatoms (not C, not H, not metal)
        * An element symbol (``"Fe"``, ``"Li"``, …)

    Returns
    -------
    frozenset[str]
        Set of matching element symbols.

    Raises
    ------
    ValueError
        If *token* is not a recognised category or element symbol.
    """
    norm = normalize_token(token)
    if norm in _STATIC_CATEGORIES:
        return _STATIC_CATEGORIES[norm]
    if norm in _ALL_SYMBOLS:
        return frozenset({norm})
    # "pi" passes normalize_token but has no element set (topological rule
    # handled separately in bond_rules.py)
    raise ValueError(
        f"{token!r} cannot be resolved to an element set. "
        f"Valid categories: M, sbm, L, het. Or use an element symbol (Fe, Li, O, …)."
    )


def resolve_atom_indices(spec: str, graph: nx.Graph) -> set[int]:
    """Resolve a spec string to a set of 0-indexed atom indices in *graph*.

    Accepts comma-separated tokens where each token is a category/element
    (``"M"``, ``"Fe"``), a numeric index (``"8"``), or a numeric range
    (``"1-5"``).  Numeric specs are 1-indexed (converted to 0-indexed).

    Examples: ``"1,2-4,M,N"`` → union of {0}, {1,2,3}, metals, nitrogens.

    Parameters
    ----------
    spec:
        Comma-separated categories, element symbols, or 1-indexed numeric ranges.
    graph:
        The molecular graph to resolve against.

    Returns
    -------
    set[int]
        0-indexed atom indices.
    """
    # Comma-separated multi-spec: split, resolve each, union.
    if "," in spec:
        result: set[int] = set()
        for part in spec.split(","):
            stripped = part.strip()
            if stripped:
                result |= resolve_atom_indices(stripped, graph)
        return result
    # "all" / "*": every atom in the graph (symbol != "*" excludes NCI centroid
    # dummy nodes, which aren't real atoms).
    if spec.strip() in {"all", "*"}:
        return {nid for nid, data in graph.nodes(data=True) if data.get("symbol", "") != "*"}
    # Numeric range?  Check BEFORE normalize_token (which rejects digits).
    # Filtered render graphs may carry the original 0-indexed atom number so
    # user-facing selectors can still refer to the input file after relabeling.
    if re.fullmatch(r"\d+(-\d+)?", spec.strip()):
        stripped = spec.strip()
        if "-" in stripped:
            a, b = stripped.split("-")
            wanted = set(range(int(a) - 1, int(b)))
        else:
            wanted = {int(stripped) - 1}
        if any("_xyzrender_original_index" in data for _, data in graph.nodes(data=True)):
            return {nid for nid, data in graph.nodes(data=True) if data.get("_xyzrender_original_index", nid) in wanted}
        return wanted
    # Category / element
    norm = normalize_token(spec)
    symbols = resolve_element_set(norm)
    return {nid for nid, data in graph.nodes(data=True) if data.get("symbol") in symbols}
