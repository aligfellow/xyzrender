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
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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
    """

    atoms: list[tuple[str, tuple[float, float, float]]]
    bonds: list[tuple[int, int, float]] | None
    name: str = ""
    charge: int = 0
    pbc_cell: np.ndarray | None = field(default=None, repr=False)


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

    return MolData(atoms=atoms, bonds=bonds or None, name=name)


# ---------------------------------------------------------------------------
# PDB
# ---------------------------------------------------------------------------


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
    ar, br, gr = np.radians(alpha), np.radians(beta), np.radians(gamma)
    ca, cb, cg = np.cos(ar), np.cos(br), np.cos(gr)
    sg = np.sin(gr)

    ax = a
    bx = b * cg
    by = b * sg
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz_sq = c**2 - cx**2 - cy**2
    cz = float(np.sqrt(max(cz_sq, 0.0)))

    return np.array([[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]], dtype=float)


def parse_pdb(path: str | Path) -> MolData:
    """Parse a PDB file.

    Reads ``ATOM``/``HETATM`` records for coordinates, ``CONECT`` records for
    connectivity, and the ``CRYST1`` record for the unit cell (if present).

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
        ``CRYST1`` record is present, otherwise ``None``.
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

    for ln in lines:
        rec = ln[:6].strip().upper()

        if rec in ("ATOM", "HETATM"):
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
                pbc_cell = _abc_angles_to_cell(a, b, c, alpha, beta, gamma)
            except (ValueError, IndexError):
                pass

        elif rec in ("COMPND", "HEADER"):
            if not name:
                name = ln[10:].strip()

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

    return MolData(atoms=atoms, bonds=bonds, name=name, pbc_cell=pbc_cell)


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
    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = 42
    cids = AllChem.EmbedMultipleConfs(mol, 10, params)  # type: ignore[attr-defined]
    if not cids:
        msg = f"rdkit failed to embed SMILES {smiles!r} in 3D"
        raise ValueError(msg)
    res = AllChem.MMFFOptimizeMoleculeConfs(mol)  # type: ignore[attr-defined]
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
# CIF  (requires ase)
# ---------------------------------------------------------------------------


def parse_cif(path: str | Path) -> MolData:
    """Parse a CIF file via ase.  Requires ``pip install 'xyzrender[cif]'``.

    bonds is None (ase does not store bonds); pbc_cell holds the lattice matrix.
    """
    try:
        import ase
        import ase.io
    except ImportError:
        msg = "CIF parsing requires ase: pip install 'xyzrender[cif]'"
        raise ImportError(msg) from None

    structure = ase.io.read(str(path), format="cif")
    assert isinstance(structure, ase.Atoms), f"Expected Atoms from CIF, got {type(structure)}"

    symbols: list[str] = list(structure.get_chemical_symbols())
    positions = structure.get_positions()
    cell = np.array(structure.get_cell())

    atoms: list[tuple[str, tuple[float, float, float]]] = [
        (sym, (float(x), float(y), float(z))) for sym, (x, y, z) in zip(symbols, positions, strict=True)
    ]
    return MolData(atoms=atoms, bonds=None, pbc_cell=cell, name=str(path))
