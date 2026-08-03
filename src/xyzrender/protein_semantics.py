"""Adapter between per-format protein metadata and ``xyzgraph.protein``.

Two jobs, and deliberately no more:

1. Canonicalise per-format metadata (PDB annotations from the parser, or
   MOL2 / extXYZ read here) into the annotation rows xyzgraph consumes.
2. Convert xyzgraph's semantics dataclasses into xyzrender's own types.

Secondary-structure assignment, residue grouping and ligand/water/ion
partitioning all live in ``xyzgraph.protein``.  This module used to carry a
parallel implementation of all three as a compatibility fallback for older
xyzgraph builds; that fallback labelled every residue coil, and because the
xyzgraph import was wrapped in a bare ``except`` it silently became the live
path whenever the dependency was missing -- which is what made draft PR #96
render proteins with no helices, sheets or arrowheads.
"""

from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xyzrender.types import ChainData, ProteinConfidence, ProteinSemantics, ResidueData

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx

    from xyzrender.parsers import MolData
    from xyzrender.types import ProteinData

logger = logging.getLogger(__name__)

_WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"})
_ION_RESNAMES: frozenset[str] = frozenset(
    {"NA", "K", "CA", "MG", "ZN", "CL", "FE", "CU", "MN", "CO", "NI", "SO4", "PO4"}
)


def from_protein_data(data: "ProteinData", *, provenance: str = "pdb") -> ProteinSemantics:
    """Promote legacy :class:`ProteinData` into :class:`ProteinSemantics`."""
    return ProteinSemantics(
        chains=data.chains,
        hetatm_indices=set(data.hetatm_indices),
        backbone_indices=set(data.backbone_indices),
        sidechain_indices=set(data.sidechain_indices),
        helix_spans=list(data.helix_spans),
        sheet_spans=list(data.sheet_spans),
        ligand_indices=set(data.ligand_indices),
        water_indices=set(data.water_indices),
        ion_indices=set(data.ion_indices),
        confidence_tier=ProteinConfidence.FULL_RIBBON,
        confidence_reasons=[f"semantic metadata extracted from {provenance}"],
        provenance=[provenance],
        trace_chains={},
        het_chains={cid: set(idxs) for cid, idxs in data.het_chains.items()},
    )


def _map_confidence_tier(value: Any) -> ProteinConfidence:
    raw = value.value if hasattr(value, "value") else str(value)
    raw = str(raw).lower().strip()
    if raw == ProteinConfidence.FULL_RIBBON.value:
        return ProteinConfidence.FULL_RIBBON
    if raw == ProteinConfidence.TRACE_ONLY.value:
        return ProteinConfidence.TRACE_ONLY
    if raw == ProteinConfidence.INSUFFICIENT.value:
        return ProteinConfidence.INSUFFICIENT
    raise ValueError(f"unknown protein confidence tier {raw!r}")


def _parse_spans(value: Any, *, field_name: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    try:
        entries = list(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence of (chain, start, end) spans") from exc
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError(f"{field_name} entries must be (chain, start, end) triples")
        chain, start_raw, end_raw = entry
        try:
            start, end = int(start_raw), int(end_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} start/end values must be integers") from exc
        if start > end:
            raise ValueError(f"{field_name} start must not exceed end")
        spans.append((str(chain), start, end))
    return spans


def _getter(obj: Any) -> Callable[..., Any]:
    """Return a ``get(key, default)`` accessor working on dicts or attribute objects.

    xyzgraph may hand back either dataclasses or the plain-dict payload form,
    so every field read goes through one of these.
    """
    if isinstance(obj, dict):
        return lambda key, default=None: obj.get(key, default)
    return lambda key, default=None: getattr(obj, key, default)


def _to_xyzrender_semantics(xg_sem: Any) -> ProteinSemantics:
    """Convert xyzgraph protein semantics payload/dataclasses into xyzrender types."""
    chains_in = getattr(xg_sem, "chains", None)
    if chains_in is None and isinstance(xg_sem, dict):
        chains_in = xg_sem.get("chains", {})
    chains: dict[str, ChainData] = {}

    if isinstance(chains_in, dict):
        for cid, chain_obj in chains_in.items():
            residues_obj = getattr(chain_obj, "residues", None)
            if residues_obj is None and isinstance(chain_obj, dict):
                residues_obj = chain_obj.get("residues", [])
            residues: list[ResidueData] = []
            if residues_obj is None:
                residues_obj = []
            for r in residues_obj:
                get = _getter(r)
                residues.append(
                    ResidueData(
                        res_name=str(get("res_name", "UNK")),
                        res_seq=int(get("res_seq", 0)),
                        chain_id=str(get("chain_id", cid)),
                        atom_indices=[int(i) for i in get("atom_indices", [])],
                        ca_index=(None if get("ca_index", None) is None else int(get("ca_index"))),
                        c_index=(None if get("c_index", None) is None else int(get("c_index"))),
                        o_index=(None if get("o_index", None) is None else int(get("o_index"))),
                        n_index=(None if get("n_index", None) is None else int(get("n_index"))),
                        ss_type=str(get("ss_type", "C")),
                        i_code=str(get("i_code", "") or ""),
                        b_factor=float(get("b_factor", 0.0) or 0.0),
                    )
                )
            chain_id = str(
                getattr(chain_obj, "chain_id", None)
                or (chain_obj.get("chain_id") if isinstance(chain_obj, dict) else cid)
                or cid
            )
            chains[str(cid)] = ChainData(chain_id=chain_id, residues=residues)

    get_top = _getter(xg_sem)
    return ProteinSemantics(
        chains=chains,
        hetatm_indices={int(i) for i in get_top("hetatm_indices", set())},
        backbone_indices={int(i) for i in get_top("backbone_indices", set())},
        sidechain_indices={int(i) for i in get_top("sidechain_indices", set())},
        helix_spans=_parse_spans(get_top("helix_spans", []), field_name="helix_spans"),
        sheet_spans=_parse_spans(get_top("sheet_spans", []), field_name="sheet_spans"),
        ligand_indices={int(i) for i in get_top("ligand_indices", set())},
        water_indices={int(i) for i in get_top("water_indices", set())},
        ion_indices={int(i) for i in get_top("ion_indices", set())},
        confidence_tier=_map_confidence_tier(get_top("confidence_tier", ProteinConfidence.INSUFFICIENT.value)),
        confidence_reasons=[str(r) for r in get_top("confidence_reasons", [])],
        provenance=[str(p) for p in get_top("provenance", [])],
        trace_chains={str(cid): [int(i) for i in idxs] for cid, idxs in dict(get_top("trace_chains", {})).items()},
        het_chains={str(cid): {int(i) for i in idxs} for cid, idxs in dict(get_top("het_chains", {})).items()},
    )


def _parse_extxyz_annotation_rows(path: Path) -> list[dict[str, object]] | None:
    """Parse canonical protein annotation keys from extXYZ Properties, if present."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        return None
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        return None
    props = _parse_extxyz_properties(lines[1])
    if props is None:
        return None
    required = ("atom_name", "res_name", "res_seq", "chain_id", "ss_type")
    if any(k not in props for k in required):
        return None

    rows: list[dict[str, object]] = []
    for ln in lines[2 : 2 + n_atoms]:
        parts = ln.split()
        if len(parts) < 4:
            return None

        def _get(name: str, _parts: list[str] = parts) -> str:
            offset, count = props[name]
            if offset >= len(_parts):
                return ""
            return " ".join(_parts[offset : offset + count]).strip()

        res_seq_raw = _get("res_seq")
        try:
            res_seq = int(res_seq_raw)
        except ValueError:
            return None
        rows.append(
            {
                "record_type": "ATOM",
                "atom_name": _get("atom_name"),
                "res_name": _get("res_name"),
                "res_seq": res_seq,
                "chain_id": _get("chain_id"),
                "ss_type": _get("ss_type"),
            }
        )
    return rows or None


def _parse_mol2_annotation_rows(path: Path) -> list[dict[str, object]] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    in_atom = False
    rows: list[dict[str, object]] = []
    for ln in lines:
        s = ln.strip()
        up = s.upper()
        if up.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if up.startswith("@<TRIPOS>") and in_atom:
            break
        if not in_atom or not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 6:
            continue
        atom_name = parts[1]
        if len(parts) > 6:
            try:
                subst_id = int(parts[6])
            except ValueError:
                subst_id = 0
        else:
            subst_id = 0
        subst_name = parts[7] if len(parts) > 7 else "RES"
        m_chain = re.search(r"[:_]([A-Za-z0-9])$", subst_name)
        chain_id = m_chain.group(1) if m_chain else "A"
        m_res = re.match(r"([A-Za-z]{3})", subst_name)
        res_name = m_res.group(1).upper() if m_res else "RES"
        m_seq = re.search(r"(-?\d+)", subst_name)
        res_seq = int(m_seq.group(1)) if m_seq else subst_id
        rows.append(
            {
                "record_type": "ATOM",
                "atom_name": atom_name,
                "res_name": res_name,
                "res_seq": res_seq,
                "chain_id": chain_id,
                "ss_type": "C",
            }
        )
    return rows or None


def _canonical_annotations(
    *,
    moldata: "MolData | None",
    source_path: str | Path | None,
    format_hint: str | None,
) -> list[dict[str, object]] | None:
    if moldata is not None:
        rows = getattr(moldata, "atom_annotations", None)
        if rows:
            return rows
    path = Path(source_path) if source_path is not None else None
    fmt = (format_hint or (path.suffix.lower() if path is not None else "")).lower()
    if path is not None and fmt == ".mol2":
        return _parse_mol2_annotation_rows(path)
    if path is not None and fmt == ".xyz":
        return _parse_extxyz_annotation_rows(path)
    return None


def xyzgraph_protein_available() -> bool:
    """Report whether the installed xyzgraph provides the protein semantics API.

    Protein rendering hard-depends on ``xyzgraph.protein``; older releases do
    not ship it.  Tests use this to skip rather than fail, and callers can use
    it to give a better message than the raised :class:`ImportError`.
    """
    try:
        mod = importlib.import_module("xyzgraph.protein")
    except ImportError:
        return False
    return all(hasattr(mod, n) for n in ("annotate_protein_semantics", "protein_semantics_from_dict"))


def _extract_from_xyzgraph(
    graph: "nx.Graph",
    *,
    moldata: "MolData | None",
    source_path: str | Path | None,
    protein_requested: bool,
    format_hint: str | None,
) -> ProteinSemantics | None:
    # Deliberately loud when the user asked for a protein render.  This is the
    # ONLY real secondary-structure source; the remaining fallbacks label every
    # residue coil.  Swallowing the import error silently degrades every protein
    # render to a coil-only trace with no helices, sheets or arrowheads -- a
    # failure that is invisible in the output and was the single largest cause of
    # bad renders in draft PR #96.  Semantics extraction also runs speculatively
    # on every load, though, so when nothing protein-specific was requested a
    # missing module just means "no semantics", not an error.
    if not xyzgraph_protein_available():
        if not protein_requested:
            return None
        msg = (
            "protein rendering requires a build of xyzgraph that provides "
            "`xyzgraph.protein` with annotate_protein_semantics / "
            "protein_semantics_from_dict (secondary-structure assignment). "
            "The installed xyzgraph does not. Install/upgrade xyzgraph, then retry."
        )
        raise ImportError(msg)
    protein_mod = importlib.import_module("xyzgraph.protein")
    annotate_protein_semantics = protein_mod.annotate_protein_semantics
    protein_semantics_from_dict = protein_mod.protein_semantics_from_dict

    annotations = _canonical_annotations(moldata=moldata, source_path=source_path, format_hint=format_hint)
    report = annotate_protein_semantics(
        graph,
        atom_annotations=annotations,
        format_hint=format_hint,
        protein_requested=protein_requested,
    )
    if report is None:
        return None

    payload = graph.graph.get("protein_semantics")
    if isinstance(payload, dict):
        return _to_xyzrender_semantics(protein_semantics_from_dict(payload))
    return _to_xyzrender_semantics(report.semantics)


def _parse_extxyz_properties(comment: str) -> dict[str, tuple[int, int]] | None:
    m = re.search(r"Properties=([^\s]+)", comment)
    if m is None:
        return None
    spec = m.group(1)
    toks = spec.split(":")
    if len(toks) < 3 or len(toks) % 3:
        return None
    props: dict[str, tuple[int, int]] = {}
    col = 0
    i = 0
    while i + 2 < len(toks):
        name = toks[i]
        try:
            count = int(toks[i + 2])
        except ValueError:
            return None
        if count <= 0:
            return None
        props[name] = (col, count)
        col += count
        i += 3
    return props


def extract_protein_semantics(
    graph: "nx.Graph",
    *,
    moldata: "MolData | None" = None,
    source_path: str | Path | None = None,
    protein_requested: bool = False,
    format_hint: str | None = None,
) -> ProteinSemantics | None:
    """Extract protein semantics for *graph*, delegating to xyzgraph.

    This module is a thin adapter.  Secondary-structure assignment, residue
    grouping and ligand/water/ion partitioning all live in ``xyzgraph.protein``
    so that one implementation serves every consumer; what remains here is
    canonicalising per-format metadata into annotation rows and converting
    xyzgraph's dataclasses into xyzrender's.

    Returns ``None`` when the structure is not protein-like.  When protein
    rendering was explicitly requested but the signal is too weak, an
    ``INSUFFICIENT`` result is returned instead so the caller can say so.
    """
    sem = _extract_from_xyzgraph(
        graph,
        moldata=moldata,
        source_path=source_path,
        protein_requested=protein_requested,
        format_hint=format_hint,
    )
    if sem is not None:
        return sem
    if not protein_requested:
        return None
    return ProteinSemantics(
        chains={},
        hetatm_indices=set(),
        backbone_indices=set(),
        sidechain_indices=set(),
        helix_spans=[],
        sheet_spans=[],
        confidence_tier=ProteinConfidence.INSUFFICIENT,
        confidence_reasons=["insufficient metadata and weak graph-only signal"],
        provenance=["none"],
    )
