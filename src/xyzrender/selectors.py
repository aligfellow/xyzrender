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

# Element symbols recognised by xyzgraph (1- or 2-char, title-case).
_ALL_SYMBOLS: frozenset[str] = frozenset(DATA.s2n.keys())

_STATIC_CATEGORIES: dict[str, frozenset[str]] = {
    "M": frozenset(DATA.metals),
    "sbm": frozenset(DATA.sblock_metals),
    "L": frozenset(s for s in DATA.s2n if s not in DATA.metals),
    "het": frozenset(s for s in DATA.s2n if s not in DATA.metals and s not in ("C", "H")),
    # Main-group buckets keyed by IUPAC group.  Overlaps with M (e.g. Al ∈ triel
    # ∩ M) and with each other (As ∈ pnic; metalloid status is implicit) are
    # intentional — set unions in the caller resolve membership.
    "hal": frozenset({"F", "Cl", "Br", "I", "At"}),
    "pnic": frozenset({"N", "P", "As", "Sb", "Bi"}),
    "chal": frozenset({"O", "S", "Se", "Te", "Po"}),
    "noble": frozenset({"He", "Ne", "Ar", "Kr", "Xe", "Rn"}),
    "triel": frozenset({"B", "Al", "Ga", "In", "Tl"}),
    "tetrel": frozenset({"C", "Si", "Ge", "Sn", "Pb"}),
}

# Topological categories — handled by callers, no associated element set.
_TOPOLOGICAL_CATEGORIES: frozenset[str] = frozenset({"pi"})

_VALID_TOKENS_HELP = "Valid categories: " + ", ".join(list(_STATIC_CATEGORIES) + sorted(_TOPOLOGICAL_CATEGORIES))


def normalize_token(token: str) -> str:
    """Normalise a category/element token and validate it.

    Categories are matched case-insensitively against the keys of
    :data:`_STATIC_CATEGORIES` plus the topological ``pi`` rule.  Element
    symbols are title-cased (``fe`` → ``Fe``).

    Raises :class:`ValueError` if *token* is not a recognised category
    or element symbol.
    """
    low = token.lower()
    for cat_key in _STATIC_CATEGORIES:
        if low == cat_key.lower():
            return cat_key
    if low in _TOPOLOGICAL_CATEGORIES:
        return low
    title = token.capitalize()
    if title in _ALL_SYMBOLS:
        return title
    raise ValueError(
        f"Unknown category or element symbol {token!r}. {_VALID_TOKENS_HELP}. Or use an element symbol (Fe, Li, O, …)."
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
        * ``"hal"`` - halogens (group 17: F, Cl, Br, I, At)
        * ``"pnic"`` - pnictogens (group 15: N, P, As, Sb, Bi)
        * ``"chal"`` - chalcogens (group 16: O, S, Se, Te, Po)
        * ``"noble"`` - noble gases (group 18: He, Ne, Ar, Kr, Xe, Rn)
        * ``"triel"`` - triels / group 13: B, Al, Ga, In, Tl
        * ``"tetrel"`` - tetrels / group 14: C, Si, Ge, Sn, Pb
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
    # Topological categories (e.g. "pi") pass normalize_token but have no
    # element set — they're handled separately in bond_rules.py.
    raise ValueError(
        f"{token!r} cannot be resolved to an element set. "
        f"{_VALID_TOKENS_HELP}. Or use an element symbol (Fe, Li, O, …)."
    )


def resolve_atom_indices(spec: str, graph: nx.Graph) -> set[int]:
    """Resolve a spec string to a set of 0-indexed atom indices in *graph*.

    Accepts comma-separated tokens where each token is a category/element
    (``"M"``, ``"Fe"``), an explicit chain selector (``"chain:A"``), a
    numeric index (``"8"``), or a numeric range (``"1-5"``).  Numeric specs
    are 1-indexed (converted to 0-indexed).

    The ``L`` and ``het`` category tokens are **graph-context-aware**: when the
    graph has metals they resolve to *atoms bonded to a metal* (the chemistry
    meaning of "ligand" / "ligand-heteroatom"), not the literal element
    complement.  Without metals in the graph they fall back to the full
    element set.  Specific element tokens (``Fe``, ``P``, ``Cl``, …) and
    group tokens (``hal``, ``pnic``, ``chal``, ``noble``, ``triel``,
    ``tetrel``) are always literal element-set matches.

    Examples: ``"1,2-4,M,N"`` → union of {0}, {1,2,3}, metals, nitrogens.
    ``"hal"`` → every halogen atom; ``"chal,pnic"`` → all O/S/Se/Te/Po and
    N/P/As/Sb/Bi atoms.

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
    stripped_spec = spec.strip()
    if stripped_spec in {"all", "*"}:
        return {nid for nid, data in graph.nodes(data=True) if data.get("symbol", "") != "*"}
    if stripped_spec.lower().startswith("chain:"):
        chain_id = stripped_spec.split(":", 1)[1]
        if not chain_id:
            raise ValueError("Chain selector requires an ID, for example 'chain:A'")
        return {nid for nid, data in graph.nodes(data=True) if data.get("chain_id") == chain_id}
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
    matched = {nid for nid, data in graph.nodes(data=True) if data.get("symbol") in symbols}
    # Chemistry-aware narrowing: L and het mean "ligand" and "ligand-heteroatom"
    # respectively — atoms bonded to a metal.  When the graph has no metals
    # this is a no-op (full element-set fallback).
    if norm in {"L", "het"}:
        metals = {nid for nid, data in graph.nodes(data=True) if data.get("symbol") in DATA.metals}
        if metals:
            matched = {n for n in matched if any(nb in metals for nb in graph.neighbors(n))}
    return matched
