"""Parsers for common molecular file formats.

Python parsers for MOL/SDF, MOL2 and PDB require no additional
dependencies.  SMILES parsing requires rdkit (``pip install 'xyzrender[smi]'``).
CIF parsing requires ase (``pip install 'xyzrender[cif]'``).

All parsers return a :class:`MolData` instance which carries:

- ``atoms`` — list of ``(symbol, (x, y, z))`` tuples in Ångström
- ``bonds`` — list of ``(i, j, bond_order)`` tuples (0-indexed) or ``None``
  when the format carries no connectivity
- ``name`` — molecule name/title (may be empty)
- ``charge`` — formal charge parsed from the file (0 when unavailable)
- ``pbc_cell`` — ``(3, 3)`` float array of row lattice vectors (Å) or ``None``
- ``atom_annotations`` — optional canonical per-atom metadata used for
  protein semantics extraction
"""

from __future__ import annotations

import bisect
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from xyzrender.types import ProteinData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protein HETATM classification sets
# ---------------------------------------------------------------------------

_WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"})
_ION_RESNAMES: frozenset[str] = frozenset(
    {"NA", "K", "CA", "MG", "ZN", "CL", "FE", "CU", "MN", "CO", "NI", "SO4", "PO4"}
)

# ---------------------------------------------------------------------------
# Common data container
# ---------------------------------------------------------------------------


@dataclass
class MolData:
    """Intermediate representation returned by all format parsers.

    Parameters
    ----------
    atoms:
        List of ``(element_symbol, (x, y, z))`` in Ångström.
    bonds:
        List of ``(atom_i, atom_j, bond_order)`` with 0-based indices, or
        ``None`` when the format does not contain connectivity information.
    name:
        Molecule/structure name or title (empty string when unavailable).
    charge:
        Total formal charge (0 when unavailable).
    pbc_cell:
        ``(3, 3)`` float array whose rows are the lattice vectors **a**, **b**,
        **c** in Ångström, or ``None`` for non-periodic structures.
    atom_annotations:
        Optional canonical per-atom metadata rows used for downstream protein
        semantics extraction.
    """

    atoms: list[tuple[str, tuple[float, float, float]]]
    bonds: list[tuple[int, int, float]] | None
    name: str = ""
    charge: int = 0
    pbc_cell: np.ndarray | None = field(default=None, repr=False)
    protein_data: "ProteinData | None" = field(default=None, repr=False)
    atom_annotations: list[dict[str, object]] | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# MOL / SDF  (MDL V2000 and V3000)
# ---------------------------------------------------------------------------

# MDL V2000 charge table (M  CHG / formal charge code in atom block)
_V2000_ATOM_CHARGE: dict[int, int] = {
    0: 0,
    1: 3,
    2: 2,
    3: 1,
    4: 0,  # 4 = doublet radical, treated as 0
    5: -1,
    6: -2,
    7: -3,
}

# MDL V2000 bond-type → bond order
_V2000_BOND_ORDER: dict[int, float] = {
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 1.5,  # 4 = aromatic
    5: 1.0,
    6: 1.0,
    7: 1.0,
    8: 0.0,  # 5-8 = query/any, use 1.0 / 0.0
}


def _parse_mol_block(lines: list[str]) -> MolData:
    """Parse a single MOL block (list of lines, no trailing $$$$).

    Handles both V2000 (fixed-width counts line) and V3000 (M  V30 records).
    """
    if not lines:
        msg = "Empty MOL block"
        raise ValueError(msg)

    name = lines[0].strip() if lines else ""

    # Detect V3000 by presence of "M  V30 BEGIN CTAB"
    is_v3000 = any("M  V30" in ln for ln in lines)

    if is_v3000:
        return _parse_mol_v3000(lines, name)
    return _parse_mol_v2000(lines, name)


def _parse_mol_v2000(lines: list[str], name: str) -> MolData:
    """Parse a V2000 MOL block."""
    # Find the counts line by scanning for the V2000 tag — more robust than
    # assuming fixed index 3, since writers (e.g. rdkit SDWriter) may omit the
    # blank molecule-name line.
    counts_idx = next(
        (i for i, ln in enumerate(lines) if ln.rstrip().endswith("V2000")),
        None,
    )
    if counts_idx is None:
        msg = "V2000 counts line not found"
        raise ValueError(msg)

    counts = lines[counts_idx]
    try:
        n_atoms = int(counts[0:3])
        n_bonds = int(counts[3:6])
    except (ValueError, IndexError) as exc:
        msg = f"Cannot parse V2000 counts line: {counts!r}"
        raise ValueError(msg) from exc

    # Atom block immediately follows the counts line
    atom_start = counts_idx + 1
    bond_start = atom_start + n_atoms

    if len(lines) < bond_start + n_bonds:
        msg = "MOL file truncated"
        raise ValueError(msg)

    atoms: list[tuple[str, tuple[float, float, float]]] = []
    atom_charges: dict[int, int] = {}  # index → charge (from M  CHG)

    for i, ln in enumerate(lines[atom_start : atom_start + n_atoms]):
        parts = ln.split()
        if len(parts) < 4:
            msg = f"Short atom line {i}: {ln!r}"
            raise ValueError(msg)
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError as exc:
            msg = f"Non-numeric coordinates in atom line {i}: {ln!r}"
            raise ValueError(msg) from exc
        sym = parts[3].capitalize()
        # Charge code at column 9 (space-separated field index 5+2 = field[5])
        chg_code = 0
        if len(parts) > 5:
            try:
                chg_code = int(parts[5])
            except ValueError:
                pass
        atom_charges[i] = _V2000_ATOM_CHARGE.get(chg_code, 0)
        atoms.append((sym, (x, y, z)))

    bonds: list[tuple[int, int, float]] = []
    for ln in lines[bond_start : bond_start + n_bonds]:
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            a1, a2, btype = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2])
        except ValueError:
            continue
        bonds.append((a1, a2, _V2000_BOND_ORDER.get(btype, 1.0)))

    # Override charges from M  CHG lines (more reliable than atom block codes)
    for ln in lines:
        if ln.startswith("M  CHG"):
            parts = ln.split()
            # Format: M  CHG  n  a1  c1  a2  c2 ...
            try:
                n = int(parts[2])
                for k in range(n):
                    idx = int(parts[3 + 2 * k]) - 1
                    chg = int(parts[4 + 2 * k])
                    atom_charges[idx] = chg
            except (IndexError, ValueError):
                pass

    total_charge = sum(atom_charges.values())
    return MolData(atoms=atoms, bonds=bonds, name=name, charge=total_charge)


def _parse_mol_v3000(lines: list[str], name: str) -> MolData:
    """Parse a V3000 MOL block (M  V30 records)."""
    in_atom = False
    in_bond = False
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    bonds: list[tuple[int, int, float]] = []
    total_charge = 0

    for ln in lines:
        s = ln.strip()

        if "M  V30 BEGIN ATOM" in s:
            in_atom = True
            in_bond = False
            continue
        if "M  V30 END ATOM" in s:
            in_atom = False
            continue
        if "M  V30 BEGIN BOND" in s:
            in_atom = False
            in_bond = True
            continue
        if "M  V30 END BOND" in s:
            in_bond = False
            continue

        if not s.startswith("M  V30"):
            continue
        content = s[len("M  V30") :].strip()

        if in_atom:
            # Format: index symbol x y z map [CHG=n ...]
            parts = content.split()
            if len(parts) < 5:
                continue
            try:
                sym = parts[1].capitalize()
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            # Parse CHG= keyword if present
            chg = 0
            for p in parts[5:]:
                if p.upper().startswith("CHG="):
                    try:
                        chg = int(p.split("=", 1)[1])
                    except ValueError:
                        pass
            total_charge += chg
            atoms.append((sym, (x, y, z)))

        elif in_bond:
            # Format: index type atom1 atom2 [stereo ...]
            parts = content.split()
            if len(parts) < 4:
                continue
            try:
                btype = int(parts[1])
                a1 = int(parts[2]) - 1
                a2 = int(parts[3]) - 1
            except ValueError:
                continue
            bonds.append((a1, a2, _V2000_BOND_ORDER.get(btype, 1.0)))

    return MolData(atoms=atoms, bonds=bonds, name=name, charge=total_charge)


def parse_mol(path: str | Path) -> MolData:
    """Parse a MDL MOL file (V2000 or V3000).

    Parameters
    ----------
    path:
        Path to the ``.mol`` file.

    Returns
    -------
    MolData
        Parsed structure with atom coordinates and bond connectivity.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return _parse_mol_block(lines)


def parse_sdf(path: str | Path, frame: int = 0) -> MolData:
    """Parse one molecule from a multi-record SDF file.

    Parameters
    ----------
    path:
        Path to the ``.sdf`` file.
    frame:
        Zero-based index of the molecule record to read (default: 0).

    Returns
    -------
    MolData
        Parsed structure for the requested record.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    # Split on $$$$ record separator
    records = re.split(r"^\$\$\$\$$", text, flags=re.MULTILINE)
    # Filter out empty trailing records
    records = [r for r in records if r.strip()]
    if frame >= len(records):
        msg = f"SDF frame {frame} requested but file has only {len(records)} record(s)"
        raise IndexError(msg)
    return _parse_mol_block(records[frame].splitlines())


# ---------------------------------------------------------------------------
# Tripos MOL2
# ---------------------------------------------------------------------------

# Tripos bond type → bond order
_MOL2_BOND_ORDER: dict[str, float] = {
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "ar": 1.5,
    "am": 1.0,  # aromatic, amide
    "un": 1.0,
    "nc": 0.0,  # unknown, not connected
    "du": 1.0,  # dummy
}


def parse_mol2(path: str | Path) -> MolData:
    """Parse a Tripos MOL2 file.

    Only the first molecule (``@<TRIPOS>MOLECULE`` block) is read.

    Parameters
    ----------
    path:
        Path to the ``.mol2`` file.

    Returns
    -------
    MolData
        Parsed structure with atom coordinates and bond connectivity.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Only parse the first MOLECULE block
    mol_start = next(
        (i for i, ln in enumerate(lines) if ln.strip().upper() == "@<TRIPOS>MOLECULE"),
        None,
    )
    if mol_start is None:
        msg = "No @<TRIPOS>MOLECULE section found"
        raise ValueError(msg)

    # Find section boundaries within the first molecule
    section_indices: dict[str, int] = {}
    for i, ln in enumerate(lines[mol_start:], start=mol_start):
        stripped = ln.strip().upper()
        if stripped.startswith("@<TRIPOS>"):
            tag = stripped[len("@<TRIPOS>") :]
            section_indices[tag] = i
            # Stop at the start of a second MOLECULE block
            if tag == "MOLECULE" and i != mol_start:
                break

    # Name is the line immediately after @<TRIPOS>MOLECULE
    name = lines[mol_start + 1].strip() if len(lines) > mol_start + 1 else ""

    atoms: list[tuple[str, tuple[float, float, float]]] = []
    bonds: list[tuple[int, int, float]] = []
    atom_annotations: list[dict[str, object]] = []

    # --- ATOM section ---
    if "ATOM" in section_indices:
        idx = section_indices["ATOM"] + 1
        while idx < len(lines):
            ln = lines[idx].strip()
            if ln.startswith("@<TRIPOS>"):
                break
            if ln and not ln.startswith("#"):
                parts = ln.split()
                if len(parts) >= 5:
                    try:
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    except ValueError:
                        idx += 1
                        continue
                    # atom_type field (index 5) may be "C.ar", "N.am", etc.
                    raw_type = parts[5] if len(parts) > 5 else parts[1]
                    sym = raw_type.split(".")[0].capitalize()
                    atoms.append((sym, (x, y, z)))
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
                    atom_annotations.append(
                        {
                            "record_type": "ATOM",
                            "atom_name": atom_name,
                            "res_name": res_name,
                            "res_seq": res_seq,
                            "chain_id": chain_id,
                            "ss_type": "C",
                        }
                    )
            idx += 1

    # --- BOND section ---
    if "BOND" in section_indices:
        idx = section_indices["BOND"] + 1
        while idx < len(lines):
            ln = lines[idx].strip()
            if ln.startswith("@<TRIPOS>"):
                break
            if ln and not ln.startswith("#"):
                parts = ln.split()
                if len(parts) >= 4:
                    try:
                        a1 = int(parts[1]) - 1
                        a2 = int(parts[2]) - 1
                    except ValueError:
                        idx += 1
                        continue
                    btype = parts[3].lower()
                    bonds.append((a1, a2, _MOL2_BOND_ORDER.get(btype, 1.0)))
            idx += 1

    return MolData(atoms=atoms, bonds=bonds or None, name=name, atom_annotations=atom_annotations or None)


# ---------------------------------------------------------------------------
# PDB
# ---------------------------------------------------------------------------


def _is_placeholder_cryst1(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> bool:
    """Whether a CRYST1 record is the ``1 1 1 90 90 90 P 1`` no-cell sentinel.

    The PDB spec mandates a CRYST1 record, so non-crystallographic entries
    (cryo-EM, NMR, predicted models) carry a unit cell of 1 A.  Taking it
    literally draws a sub-Angstrom cell box and axis triad over the structure.
    """
    return (a, b, c) == (1.0, 1.0, 1.0) and (alpha, beta, gamma) == (90.0, 90.0, 90.0)


def _abc_angles_to_cell(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Convert unit-cell parameters to a (3, 3) row-vector matrix.

    Uses the standard crystallographic convention where **a** is along x,
    **b** is in the xy-plane, and **c** is defined by the remaining angles.

    Parameters
    ----------
    a, b, c:
        Lattice vector lengths in Ångström.
    alpha, beta, gamma:
        Inter-axial angles in degrees (alpha between b/c, beta between a/c, gamma between a/b).

    Returns
    -------
    numpy.ndarray
        Shape ``(3, 3)`` float array; rows are **a**, **b**, **c** vectors.
    """
    parameters = np.array([a, b, c, alpha, beta, gamma], dtype=float)
    if (
        not np.isfinite(parameters).all()
        or min(a, b, c) <= 0
        or not all(0 < angle < 180 for angle in (alpha, beta, gamma))
    ):
        msg = "invalid CRYST1 geometry: lengths must be positive and angles must lie between 0 and 180 degrees"
        raise ValueError(msg)

    ar, br, gr = np.radians(alpha), np.radians(beta), np.radians(gamma)
    ca, cb, cg = np.cos(ar), np.cos(br), np.cos(gr)
    sg = np.sin(gr)
    volume_factor = 1.0 + 2.0 * ca * cb * cg - ca**2 - cb**2 - cg**2
    if abs(sg) <= 1e-12 or volume_factor <= 1e-12:
        msg = "invalid CRYST1 geometry: unit cell is degenerate"
        raise ValueError(msg)

    ax = a
    bx = b * cg
    by = b * sg
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz_sq = c**2 - cx**2 - cy**2
    if cz_sq <= 1e-12:
        msg = "invalid CRYST1 geometry: unit cell volume is zero"
        raise ValueError(msg)
    cz = float(np.sqrt(cz_sq))

    cell = np.array([[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]], dtype=float)
    if not np.isfinite(cell).all():
        msg = "invalid CRYST1 geometry: non-finite lattice vectors"
        raise ValueError(msg)
    return cell


_BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})

# altLoc values (col 17) that are kept.  Blank means "no alternate conformer";
# "A" is the conventional first/highest-occupancy alternate.  Keeping every
# atoms at the same residue key, and the residue's CA would silently become
# whichever came last.
_KEPT_ALTLOCS = frozenset({"", "A"})


class _AtomRecord(NamedTuple):
    """Per-atom PDB metadata carried from the record loop to residue assembly."""

    record: str  # "ATOM" or "HETATM"
    atom_name: str
    res_name: str
    chain_id: str
    res_seq: int
    i_code: str  # insertion code (col 27); part of the residue identity
    b_factor: float  # temperature factor / pLDDT (cols 61-66)


def _residue_key(chain_id: str, res_seq: int, i_code: str, res_name: str) -> tuple[str, int, str, str]:
    """Identity of a residue.

    The insertion code is part of this on purpose.  Without it, an inserted
    residue sharing a ``res_name`` with its neighbour (antibody CDRs are full
    of these) collides on the key and the two residues *merge*, with
    ``ca_index`` overwritten by whichever CA was seen last — a silent jump in
    the backbone trace.
    """
    return chain_id, int(res_seq), i_code, res_name


def _build_span_index(spans: list[tuple[str, int, int]]) -> dict[str, list[tuple[int, int]]]:
    """Group HELIX/SHEET spans by chain, sorted by start for bisect lookup."""
    out: dict[str, list[tuple[int, int]]] = {}
    for chain, start, end in spans:
        out.setdefault(chain, []).append((start, end))
    for runs in out.values():
        runs.sort()
    return out


def _span_contains(index: dict[str, list[tuple[int, int]]], chain_id: str, res_seq: int) -> bool:
    """Whether *res_seq* falls in any span for *chain_id*.

    Binary search rather than the previous linear scan: this used to be run
    once per residue *and* once per atom, so a 8.6k-atom structure with 93
    spans did ~800k pure-Python range checks.
    """
    runs = index.get(chain_id)
    if not runs:
        return False
    lo = bisect.bisect_right(runs, (res_seq, _INF_SEQ)) - 1
    return lo >= 0 and runs[lo][0] <= res_seq <= runs[lo][1]


_INF_SEQ = 1 << 62


def parse_pdb(path: str | Path) -> MolData:
    """Parse a PDB file.

    Reads ``ATOM``/``HETATM`` records for coordinates, ``CONECT`` records for
    connectivity, and the ``CRYST1`` record for the unit cell (if present).
    Also extracts protein chain/residue/secondary-structure metadata when
    present (HELIX/SHEET records), returning a :class:`ProteinData` on the
    result.

    When ``CONECT`` records are absent (e.g. protein backbone only) ``bonds``
    is ``None`` and xyzgraph distance-based detection will be used instead.

    Parameters
    ----------
    path:
        Path to the ``.pdb`` file.

    Returns
    -------
    MolData
        Parsed structure.  ``pbc_cell`` is a ``(3, 3)`` array when a
        ``CRYST1`` record is present, otherwise ``None``.  ``protein_data``
        is populated when the file contains ATOM records with chain/residue
        information.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # serial → 0-based index mapping
    serial_to_idx: dict[int, int] = {}
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    pbc_cell: np.ndarray | None = None
    name = ""

    # CONECT entries: serial → set of connected serials
    conect: dict[int, set[int]] = {}

    # Protein metadata: per-atom record info for later assembly
    atom_meta: list[_AtomRecord] = []

    # Secondary structure spans
    helix_spans: list[tuple[str, int, int]] = []
    sheet_spans: list[tuple[str, int, int]] = []
    modified_polymer_residues: set[tuple[str, int, str, str]] = set()

    has_chain_info = False  # set True when we see chain IDs
    # Multi-model files (NMR ensembles) hold the same molecule many times over.
    # Without this guard every model is concatenated into one atom list and
    # ribboned on top of itself, and because res_seq restarts per model the
    # backbone is also shredded into fragments.  Take the first model, as
    # PyMOL and ChimeraX do.
    past_first_model = False
    n_models = 0

    for ln in lines:
        rec = ln[:6].strip().upper()

        if rec == "MODEL":
            n_models += 1
            continue
        if rec == "ENDMDL":
            past_first_model = True
            continue

        if rec in ("ATOM", "HETATM"):
            if past_first_model:
                continue
            # Alternate conformers share a residue key; keeping both would let
            # the residue's CA silently become whichever was parsed last.
            altloc = ln[16].strip() if len(ln) > 16 else ""
            if altloc not in _KEPT_ALTLOCS:
                continue
            # PDB fixed-column format
            try:
                serial = int(ln[6:11])
                x = float(ln[30:38])
                y = float(ln[38:46])
                z = float(ln[46:54])
            except (ValueError, IndexError):
                continue
            # Element: cols 77-78 (preferred) else infer from atom name
            elem = ln[76:78].strip() if len(ln) > 76 else ""
            if not elem:
                # Atom name in cols 12-16; strip digits and spaces
                aname = ln[12:16].strip() if len(ln) > 15 else ""
                elem = re.sub(r"[^A-Za-z]", "", aname)[:2]
            sym = elem.capitalize()
            idx = len(atoms)
            serial_to_idx[serial] = idx
            atoms.append((sym, (x, y, z)))

            # Protein metadata columns
            atom_name = ln[12:16].strip() if len(ln) > 15 else ""
            res_name = ln[17:20].strip() if len(ln) > 19 else ""
            chain_id = ln[21].strip() if len(ln) > 21 else ""
            try:
                res_seq = int(ln[22:26])
            except (ValueError, IndexError):
                res_seq = 0
            i_code = ln[26].strip() if len(ln) > 26 else ""
            try:
                b_factor = float(ln[60:66])
            except (ValueError, IndexError):
                b_factor = 0.0
            atom_meta.append(_AtomRecord(rec, atom_name, res_name, chain_id, res_seq, i_code, b_factor))
            if chain_id:
                has_chain_info = True

        elif rec == "CONECT":
            # CONECT lines: serial followed by up to 4 bonded serials (cols 7-10, 11-15, ...)
            try:
                origin = int(ln[6:11])
            except (ValueError, IndexError):
                continue
            if origin not in conect:
                conect[origin] = set()
            for start in (11, 16, 21, 26):
                seg = ln[start : start + 5].strip()
                if seg:
                    try:
                        conect[origin].add(int(seg))
                    except ValueError:
                        pass

        elif rec == "CRYST1":
            # CRYST1   a      b      c    alpha  beta   gamma sGroup Z
            try:
                a = float(ln[6:15])
                b = float(ln[15:24])
                c = float(ln[24:33])
                alpha = float(ln[33:40])
                beta = float(ln[40:47])
                gamma = float(ln[47:54])
            except (ValueError, IndexError) as exc:
                msg = "invalid CRYST1 record: expected numeric lengths and angles"
                raise ValueError(msg) from exc
            if not _is_placeholder_cryst1(a, b, c, alpha, beta, gamma):
                pbc_cell = _abc_angles_to_cell(a, b, c, alpha, beta, gamma)

        elif rec == "HELIX":
            # HELIX  seqNum helixID initResName initChainID initSeqNum ...
            # cols: chain=19, start=21-24, end=33-36 (all 1-indexed)
            try:
                chain = ln[19].strip() if len(ln) > 19 else ""
                start = int(ln[21:25])
                end = int(ln[33:37])
                if chain:
                    helix_spans.append((chain, start, end))
            except (ValueError, IndexError):
                pass

        elif rec == "SHEET":
            # SHEET  strand sheetID numStrands initResName initChainID initSeqNum ...
            # cols: chain=21, start=22-25, end=33-36
            try:
                chain = ln[21].strip() if len(ln) > 21 else ""
                start = int(ln[22:26])
                end = int(ln[33:37])
                if chain:
                    sheet_spans.append((chain, start, end))
            except (ValueError, IndexError):
                pass

        elif rec == "MODRES":
            # MODRES identifies a HETATM residue that belongs to the polymer,
            # such as selenomethionine (MSE).  Preserve arbitrary hetero
            # ligands as HETATM; only the exact chain/residue named here is
            # promoted into the protein residue sequence.
            try:
                res_name = ln[12:15].strip()
                chain_id = ln[16].strip() if len(ln) > 16 else ""
                res_seq = int(ln[18:22])
                i_code = ln[22].strip() if len(ln) > 22 else ""
            except (ValueError, IndexError):
                continue
            if res_name:
                modified_polymer_residues.add(_residue_key(chain_id, res_seq, i_code, res_name))

        elif rec in ("COMPND", "HEADER"):
            if not name:
                name = ln[10:].strip()

    if modified_polymer_residues:
        atom_meta = [
            atom._replace(record="ATOM")
            if atom.record == "HETATM"
            and _residue_key(atom.chain_id, atom.res_seq, atom.i_code, atom.res_name) in modified_polymer_residues
            else atom
            for atom in atom_meta
        ]

    # Build bond list from CONECT data (deduplicate by storing only i < j)
    bonds: list[tuple[int, int, float]] | None = None
    if conect:
        seen: set[tuple[int, int]] = set()
        bond_list: list[tuple[int, int, float]] = []
        for origin, partners in conect.items():
            i = serial_to_idx.get(origin)
            if i is None:
                continue
            for partner in partners:
                j = serial_to_idx.get(partner)
                if j is None:
                    continue
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    bond_list.append((key[0], key[1], 1.0))
        bonds = bond_list or None

    # Build ProteinData when chain info is present
    protein_data: ProteinData | None = None
    if has_chain_info and atom_meta:
        protein_data = _build_protein_data(atom_meta, helix_spans, sheet_spans)
    helix_index = _build_span_index(helix_spans)
    sheet_index = _build_span_index(sheet_spans)

    def _ss_type(chain_id: str, res_seq: int, rec: str) -> str:
        if rec != "ATOM":
            return "C"
        if _span_contains(helix_index, chain_id, res_seq):
            return "H"
        if _span_contains(sheet_index, chain_id, res_seq):
            return "E"
        return "C"

    atom_annotations: list[dict[str, object]] = [
        {
            "record_type": a.record,
            "atom_name": a.atom_name,
            "res_name": a.res_name or "UNK",
            "res_seq": a.res_seq,
            "chain_id": a.chain_id or "A",
            "i_code": a.i_code,
            "b_factor": a.b_factor,
            "ss_type": _ss_type(a.chain_id, a.res_seq, a.record),
        }
        for a in atom_meta
    ]

    if n_models > 1:
        logger.info("PDB contains %d models; rendering model 1", n_models)

    return MolData(
        atoms=atoms,
        bonds=bonds,
        name=name,
        pbc_cell=pbc_cell,
        protein_data=protein_data,
        atom_annotations=atom_annotations or None,
    )


def _build_protein_data(
    atom_meta: list[_AtomRecord],
    helix_spans: list[tuple[str, int, int]],
    sheet_spans: list[tuple[str, int, int]],
) -> "ProteinData":
    """Assemble :class:`ProteinData` from per-atom metadata."""
    from xyzrender.types import ChainData, ProteinData, ResidueData

    helix_index = _build_span_index(helix_spans)
    sheet_index = _build_span_index(sheet_spans)

    def _ss_type(chain_id: str, res_seq: int) -> str:
        if _span_contains(helix_index, chain_id, res_seq):
            return "H"
        if _span_contains(sheet_index, chain_id, res_seq):
            return "E"
        return "C"

    # Accumulate residues per chain, grouping consecutive (chain_id, res_seq, res_name)
    chains: dict[str, list[ResidueData]] = {}
    hetatm_indices: set[int] = set()
    ligand_indices: set[int] = set()
    water_indices: set[int] = set()
    ion_indices: set[int] = set()
    backbone_indices: set[int] = set()
    sidechain_indices: set[int] = set()

    # Group atoms into residues: key = (chain_id, res_seq, res_name)
    residue_atoms: dict[tuple[str, int, str, str], list[tuple[int, str, float]]] = {}
    residue_order: list[tuple[str, int, str, str]] = []
    # Heteroatoms never join a chain's residue list; keep them addressable so
    # excluding a chain can drop its ligands too.
    het_chains: dict[str, set[int]] = {}

    for idx, a in enumerate(atom_meta):
        if a.record == "HETATM":
            hetatm_indices.add(idx)
            het_chains.setdefault(a.chain_id or "A", set()).add(idx)
            if a.res_name in _WATER_RESNAMES:
                water_indices.add(idx)
            elif a.res_name in _ION_RESNAMES:
                ion_indices.add(idx)
            else:
                ligand_indices.add(idx)
            continue

        key = _residue_key(a.chain_id, a.res_seq, a.i_code, a.res_name)
        if key not in residue_atoms:
            residue_atoms[key] = []
            residue_order.append(key)
        residue_atoms[key].append((idx, a.atom_name, a.b_factor))

    # Build ResidueData objects
    for key in residue_order:
        chain_id, res_seq, i_code, res_name = key
        entries = residue_atoms[key]
        all_indices = [i for i, _, _ in entries]
        ca_index: int | None = None
        c_index: int | None = None
        o_index: int | None = None
        n_index: int | None = None

        ca_b_factor: float | None = None
        for i, atom_name, b_fac in entries:
            aname_upper = atom_name.upper()
            if aname_upper == "CA":
                ca_b_factor = b_fac
            if aname_upper == "CA":
                ca_index = i
                backbone_indices.add(i)
            elif aname_upper == "C":
                c_index = i
                backbone_indices.add(i)
            elif aname_upper in ("O", "OXT"):
                if aname_upper == "O" and o_index is None:
                    o_index = i
                backbone_indices.add(i)
            elif aname_upper == "N":
                n_index = i
                backbone_indices.add(i)
            else:
                sidechain_indices.add(i)

        ss = _ss_type(chain_id, res_seq)
        # CA B-factor is the per-residue value (it is what pLDDT is reported
        # on); fall back to the residue mean when the CA is missing.
        if ca_b_factor is None:
            b_vals = [b for _, _, b in entries]
            ca_b_factor = sum(b_vals) / len(b_vals) if b_vals else 0.0
        res = ResidueData(
            res_name=res_name,
            res_seq=res_seq,
            chain_id=chain_id,
            atom_indices=all_indices,
            ca_index=ca_index,
            c_index=c_index,
            o_index=o_index,
            n_index=n_index,
            ss_type=ss,
            i_code=i_code,
            b_factor=ca_b_factor,
        )
        chains.setdefault(chain_id, []).append(res)

    chain_data = {cid: ChainData(chain_id=cid, residues=residues) for cid, residues in chains.items()}
    return ProteinData(
        chains=chain_data,
        hetatm_indices=hetatm_indices,
        backbone_indices=backbone_indices,
        sidechain_indices=sidechain_indices,
        helix_spans=helix_spans,
        sheet_spans=sheet_spans,
        ligand_indices=ligand_indices,
        water_indices=water_indices,
        ion_indices=ion_indices,
        het_chains=het_chains,
    )


# ---------------------------------------------------------------------------
# Extension dispatcher
# ---------------------------------------------------------------------------


def parse(path: str | Path, frame: int = 0) -> MolData:
    """Parse a molecular file, dispatching on extension (.mol, .sdf, .mol2, .pdb)."""
    p = str(path)
    if p.endswith(".mol"):
        return parse_mol(path)
    if p.endswith(".sdf"):
        return parse_sdf(path, frame=frame)
    if p.endswith(".mol2"):
        return parse_mol2(path)
    if p.endswith(".pdb"):
        return parse_pdb(path)
    msg = f"Unsupported format for parsers.parse: {p!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# SMILES  (requires rdkit)
# ---------------------------------------------------------------------------


def parse_smiles(smiles: str, kekule: bool = False) -> MolData:
    """Embed a SMILES string into 3-D via rdkit (ETKDGv3 + MMFF94).

    Requires ``pip install 'xyzrender[smi]'``.  Bonds are read directly from
    the rdkit graph; kekule=True converts aromatic bonds to alternating 1/2.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        msg = "SMILES parsing requires rdkit: pip install 'xyzrender[smi]'"
        raise ImportError(msg) from None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        msg = f"rdkit could not parse SMILES: {smiles!r}"
        raise ValueError(msg)

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()  # ty: ignore[unresolved-attribute]
    params.randomSeed = 42
    cids = AllChem.EmbedMultipleConfs(mol, 10, params)  # ty: ignore[unresolved-attribute]
    if not cids:
        msg = f"rdkit failed to embed SMILES {smiles!r} in 3D"
        raise ValueError(msg)
    res = AllChem.MMFFOptimizeMoleculeConfs(mol)  # ty: ignore[unresolved-attribute]
    best_i = min(
        (i for i, (rc, _) in enumerate(res) if rc == 0),
        key=lambda i: res[i][1],
        default=0,
    )
    conf = mol.GetConformer(cids[best_i])

    if kekule:
        Chem.Kekulize(mol)

    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append((atom.GetSymbol(), (pos.x, pos.y, pos.z)))

    bonds: list[tuple[int, int, float]] = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()) for b in mol.GetBonds()
    ]

    return MolData(atoms=atoms, bonds=bonds, name=smiles)


# ---------------------------------------------------------------------------
# rdkit MolObject
# ---------------------------------------------------------------------------


def parse_molobject(mol, *, conf_id: int = -1, kekule: bool = False, name: str | None = None) -> MolData:
    """Convert an RDKit Mol with a conformer into xyzrender MolData.

    The RDKit mol must already have 3D coordinates unless you add an embedding
    fallback before calling this; ensemble=True expects multiple conformers creating a
    MolData ensemble.  ``kekule=True`` converts aromatic bonds to alternating
    single/double, matching :func:`parse_smiles`.
    """
    try:
        from rdkit import Chem
    except ImportError:
        msg = "MolObject parsing requires rdkit: pip install 'xyzrender[smi]'"
        raise ImportError(msg) from None

    if mol.GetNumConformers() == 0:
        msg = "rdkit MolObject has no conformers; embed it first or create Molecule via smiles: load(smiles)"
        raise ValueError(msg)

    mol = Chem.Mol(mol)

    if kekule:
        Chem.Kekulize(mol)

    conf = mol.GetConformer(conf_id)

    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(
            (
                atom.GetSymbol(),
                (float(pos.x), float(pos.y), float(pos.z)),
            )
        )
    bonds: list[tuple[int, int, float]] = [
        (
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            float(bond.GetBondTypeAsDouble()),
        )
        for bond in mol.GetBonds()
    ]
    charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    if not name:
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
    return MolData(
        atoms=atoms,
        bonds=bonds,
        name=name,
        charge=charge,
    )


# ---------------------------------------------------------------------------
# CIF  (requires ase)
# ---------------------------------------------------------------------------


def _reject_mmcif(path: str | Path) -> None:
    """Raise on mmCIF, which ase's CIF reader cannot parse.

    ase reads small-molecule CIF, whose tags are underscore-joined
    (``_atom_site_Cartn_x``); mmCIF uses a dotted form (``_atom_site.Cartn_x``)
    that every ase block rejects, leaving the caller a bare StopIteration.
    """
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            token = ln.strip()
            if token.startswith("_atom_site."):
                msg = f"{path}: mmCIF is not supported yet (small-molecule CIF only)"
                raise ValueError(msg)
            if token.startswith("_atom_site_"):
                return


def parse_cif(path: str | Path) -> MolData:
    """Parse a small-molecule CIF via ase.  Requires ``pip install 'xyzrender[cif]'``.

    bonds is None (ase does not store bonds); pbc_cell holds the lattice matrix.
    """
    try:
        import ase
        import ase.io
    except ImportError:
        msg = "CIF parsing requires ase: pip install 'xyzrender[cif]'"
        raise ImportError(msg) from None

    _reject_mmcif(path)
    structure = ase.io.read(str(path), format="cif", store_tags=True)

    assert isinstance(structure, ase.Atoms), f"Expected Atoms from CIF, got {type(structure)}"

    symbols: list[str] = list(structure.get_chemical_symbols())
    positions = structure.get_positions()
    cell = np.array(structure.get_cell())

    atoms: list[tuple[str, tuple[float, float, float]]] = [
        (sym, (float(x), float(y), float(z))) for sym, (x, y, z) in zip(symbols, positions, strict=True)
    ]

    return MolData(atoms=atoms, bonds=None, pbc_cell=cell, name=str(path))


# ---------------------------------------------------------------------------
# SHELXL  (requires shelxfile)
# ---------------------------------------------------------------------------


def parse_shelxl(path: str | Path) -> MolData:
    """Parse a SHELXL .res or .ins file via shelxfile. Requires ``pip install 'xyzrender[shelxl]'``.

    bonds is None (we don't extract them from SHELXL currently, relying on distance detection);
    pbc_cell holds the lattice matrix.
    """
    try:
        from shelxfile import Shelxfile
    except ImportError:
        msg = "SHELXL parsing requires shelxfile: pip install 'xyzrender[shelxl]'"
        raise ImportError(msg) from None

    shx = Shelxfile()
    shx.read_file(str(path))

    cell = getattr(shx, "cell", None)
    pbc_cell = None
    if cell is not None:
        pbc_cell = _abc_angles_to_cell(cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)

    atoms: list[tuple[str, tuple[float, float, float]]] = []
    source_atoms = getattr(getattr(shx, "atoms", None), "all_atoms", ())
    symmetry_ops = tuple(getattr(shx, "symmcards", ())) or (None,)
    for atom in source_atoms:
        for symm in symmetry_ops:
            sym = (
                atom.element.capitalize()
                if hasattr(atom, "element") and atom.element
                else atom.name.rstrip("0123456789").capitalize()
            )
            if pbc_cell is not None and hasattr(atom, "frac_coords"):
                frac = np.asarray(atom.frac_coords, dtype=float)
                if symm is not None:
                    frac = np.asarray(symm.matrix, dtype=float) @ frac + np.asarray(symm.trans, dtype=float)
                x, y, z = np.mod(frac, 1.0) @ pbc_cell
            elif hasattr(atom, "cart_coords") and atom.cart_coords is not None:
                x, y, z = atom.cart_coords
            elif hasattr(atom, "xc") and hasattr(atom, "yc") and hasattr(atom, "zc"):
                x, y, z = atom.xc, atom.yc, atom.zc
            else:
                x, y, z = 0.0, 0.0, 0.0
            atoms.append((sym, (float(x), float(y), float(z))))

    return MolData(atoms=atoms, bonds=None, pbc_cell=pbc_cell, name=str(path))
