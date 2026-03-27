"""Bond display rules — composable element-category filters.

Resolves ``--unbond`` / ``--bond`` specs against a molecular graph and
drops or adds edges accordingly.  Operates on the render-time *copy*
of the graph so the original :class:`~xyzrender.api.Molecule` is never
mutated.  Reads but never mutates the :class:`~xyzrender.types.RenderConfig`.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from xyzrender.selectors import normalize_token, resolve_element_set

if TYPE_CHECKING:
    import networkx as nx

    from xyzrender.types import RenderConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_bond_rules(graph: nx.Graph, cfg: RenderConfig) -> None:
    """Mutate *graph* (render-time copy) by dropping/adding edges.

    1. Parse ``cfg.unbond`` specs → collect edges to **remove**.
    2. Parse ``cfg.bond`` specs → index pairs to **keep / add**.
    3. Subtract bond pairs from the remove set (overrides).
    4. Drop collected edges from *graph*.
    5. Add bond pairs as new edges if not already present.
    """
    # -- Pre-build node symbol lookup (one traversal) ----------------------
    sym: dict[int, str] = {nid: data.get("symbol", "") for nid, data in graph.nodes(data=True)}

    # -- Classify specs ----------------------------------------------------
    pair_specs: list[tuple[int, int]] = []
    atom_idx_specs: list[int] = []  # standalone atom indices (0-indexed)
    standalone_specs: list[frozenset[str]] = []
    between_specs: list[tuple[frozenset[str], frozenset[str]]] = []
    pi_specs: list[frozenset[str] | None] = []

    for spec in cfg.unbond:
        pair = _parse_index_pair(spec)
        if pair is not None:
            pair_specs.append(pair)
            continue
        if "-" not in spec:
            stripped = spec.strip()
            # Standalone atom index: remove all bonds from that atom
            if stripped.isdigit() and int(stripped) >= 1:
                idx = int(stripped) - 1
                atom_idx_specs.append(idx)
            else:
                norm = normalize_token(stripped)
                if norm == "pi":
                    # Standalone pi: all eta-coordination bonds from any external atom
                    pi_specs.append(None)
                else:
                    standalone_specs.append(resolve_element_set(norm))
            continue
        parts = spec.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid unbond spec {spec!r}: expected a category/element (e.g. 'M', 'Li'), "
                f"a pair 'X-Y' (e.g. 'M-L', 'Fe-het'), or an index pair (e.g. '1-3')"
            )
        left = normalize_token(parts[0])
        right = normalize_token(parts[1])
        if left == "pi" or right == "pi":
            other = right if left == "pi" else left
            pi_specs.append(resolve_element_set(other))
        else:
            between_specs.append((resolve_element_set(left), resolve_element_set(right)))

    # -- Collect edges to remove (single edge traversal) -------------------
    remove: set[tuple[int, int]] = set()

    # Index pairs (skip NCI/TS overlay edges)
    for i, j in pair_specs:
        if i == j:
            logger.warning("unbond %d-%d: ignoring self-loop", i + 1, j + 1)
        elif i not in graph or j not in graph:
            logger.warning("unbond %d-%d: index out of range (molecule has %d atoms)", i + 1, j + 1, len(sym))
        elif graph.has_edge(i, j):
            d = graph.edges[i, j]
            if d.get("NCI", False) or d.get("TS", False):
                logger.warning("unbond %d-%d: skipping NCI/TS overlay edge", i + 1, j + 1)
            else:
                remove.add((i, j))
        else:
            logger.warning("unbond %d-%d: no bond exists between these atoms", i + 1, j + 1)

    # Standalone atom indices: remove all covalent bonds from specific atoms
    for idx in atom_idx_specs:
        if idx not in graph:
            logger.warning("unbond %d: index out of range (molecule has %d atoms)", idx + 1, len(sym))
        else:
            for nbr in list(graph.neighbors(idx)):
                d = graph.edges[idx, nbr]
                if not d.get("NCI", False) and not d.get("TS", False):
                    remove.add((idx, nbr))

    # Element-based rules: one pass over all edges
    # Skip NCI / TS overlay edges — these are structural annotations, not bonds.
    if standalone_specs or between_specs:
        for i, j, d in graph.edges(data=True):
            if d.get("NCI", False) or d.get("TS", False):
                continue
            si, sj = sym[i], sym[j]
            matched = any(si in syms or sj in syms for syms in standalone_specs)
            if not matched:
                matched = any((si in ls and sj in rs) or (si in rs and sj in ls) for ls, rs in between_specs)
            if matched:
                remove.add((i, j))

    # Pi-coordination (needs ring topology — separate pass over candidate nodes)
    for source_syms in pi_specs:
        remove |= _find_pi_edges(graph, sym, source_syms)

    # -- Parse and validate bond additions upfront (before any mutation) ----
    add: set[tuple[int, int]] = set()
    for spec in cfg.bond:
        pair = _parse_index_pair(spec)
        if pair is None:
            raise ValueError(f"--bond only accepts index pairs (e.g. '1-3'), got {spec!r}")
        i, j = pair
        if i == j:
            raise ValueError(f"--bond {i + 1}-{j + 1}: cannot bond an atom to itself")
        if i not in graph or j not in graph:
            raise ValueError(
                f"--bond {i + 1}-{j + 1}: atom index out of range (molecule has {graph.number_of_nodes()} atoms)"
            )
        add.add(pair)

    # -- Subtract overrides ------------------------------------------------
    if add:
        add_canonical = {_canonical(i, j) for i, j in add}
        remove = {(i, j) for i, j in remove if _canonical(i, j) not in add_canonical}

    # -- Apply removals ----------------------------------------------------
    if remove:
        graph.remove_edges_from(list(remove))
        logger.info("unbond: removed %d edge(s)", len(remove))

    # -- Apply additions ---------------------------------------------------
    for i, j in add:
        if not graph.has_edge(i, j):
            graph.add_edge(i, j, bond_order=1.0)
            logger.info("bond: added edge %d-%d", i + 1, j + 1)


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------

_INDEX_PAIR_RE = re.compile(r"^([1-9]\d*)-([1-9]\d*)$")


def _parse_index_pair(spec: str) -> tuple[int, int] | None:
    """Try to parse ``spec`` as a 1-indexed atom pair → 0-indexed tuple.

    Rejects zero-indexed (``0-1``) and self-referencing (``3-3``) pairs.
    """
    m = _INDEX_PAIR_RE.match(spec.strip())
    if m is None:
        return None
    return int(m.group(1)) - 1, int(m.group(2)) - 1


def _canonical(i: int, j: int) -> tuple[int, int]:
    return (min(i, j), max(i, j))


# ---------------------------------------------------------------------------
# Pi-coordination detection
# ---------------------------------------------------------------------------


def _find_pi_edges(
    graph: nx.Graph,
    sym: dict[int, str],
    source_symbols: frozenset[str] | None,
) -> set[tuple[int, int]]:
    """Find eta-coordination edges (≥2 bonds from an external atom to an aromatic ring).

    Only considers atoms that are NOT themselves members of the ring —
    ring-internal bonds (e.g. C-C within benzene) are never matched.

    *source_symbols* filters which atoms to check.  ``None`` = all atoms.
    """
    rings: list = graph.graph.get("aromatic_rings", [])
    if not rings:
        return set()

    ring_sets: list[set[int]] = [set(r) for r in rings]

    edges: set[tuple[int, int]] = set()
    for nid, s in sym.items():
        if source_symbols is not None and s not in source_symbols:
            continue
        neighbours = set(graph.neighbors(nid))
        for ring in ring_sets:
            if nid in ring:
                continue
            bonded_to_ring = neighbours & ring
            if len(bonded_to_ring) >= 2:
                for ring_atom in bonded_to_ring:
                    edges.add((nid, ring_atom))
    return edges
