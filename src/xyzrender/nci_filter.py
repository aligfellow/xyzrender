"""Selection helpers for xyzgraph non-covalent interaction types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xyzgraph.nci import NCI_TYPES

if TYPE_CHECKING:
    from xyzgraph.nci import NCIData


NCI_GROUPS: dict[str, frozenset[str]] = {
    "hb": frozenset({"hbond", "hbond_bifurcated", "hb_pi"}),
    "pi": frozenset(nci_type for nci_type in NCI_TYPES if nci_type.startswith("pi_") or nci_type.endswith("_pi")),
    "ion": frozenset({"cation_pi", "anion_pi", "cation_lp", "ionic", "salt_bridge"}),
}


def resolve_nci_types(selection: bool | str | list[str]) -> frozenset[str] | None:
    """Resolve exact xyzgraph NCI types and group aliases.

    ``True`` or ``"all"`` selects every type and is represented by ``None``.
    """
    if selection is True:
        return None
    if selection is False:
        return frozenset()

    values = [selection] if isinstance(selection, str) else selection
    names = [part.strip().lower() for value in values for part in value.split(",") if part.strip()]
    if not names:
        msg = "NCI type selection cannot be empty"
        raise ValueError(msg)
    if "all" in names:
        return None

    valid_types = set(NCI_TYPES)
    resolved: set[str] = set()
    for name in names:
        if name in NCI_GROUPS:
            resolved.update(NCI_GROUPS[name])
        elif name in valid_types:
            resolved.add(name)
        else:
            valid = ", ".join((*NCI_GROUPS, *NCI_TYPES))
            msg = f"Unknown NCI type or group {name!r}; valid values: {valid}"
            raise ValueError(msg)
    return frozenset(resolved)


def filter_ncis(ncis: list[NCIData], selection: bool | str | list[str]) -> list[NCIData]:
    """Return interactions selected by exact type or group alias."""
    selected = resolve_nci_types(selection)
    if selected is None:
        return ncis
    return [nci for nci in ncis if nci.type in selected]
