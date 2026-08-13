"""Parsers for common molecular file formats.

Python parsers for MOL/SDF, MOL2, PDB and CJSON require no additional
dependencies.  SMILES parsing requires rdkit (``pip install 'xyzrender[smi]'``).
CIF parsing requires ase (``pip install 'xyzrender[cif]'``).

All parsers return a :class:`MolData` instance which carries:

- ``atoms`` — list of ``(symbol, (x, y, z))`` tuples in Ångström
- ``bonds`` — list of ``(i, j, bond_order)`` tuples (0-indexed) or ``None``
  when the format carries no connectivity
- ``name`` — molecule name/title (may be empty)
- ``charge`` — formal charge parsed from the file (0 when unavailable)
- ``pbc_cell`` — ``(3, 3)`` float array of row lattice vectors (Å) or ``None``
- ``colors`` — per-atom ``#rrggbb`` overrides carried by the file, or ``None``
- ``camera`` — saved viewer camera (:class:`CameraData`), or ``None``
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common data container
# ---------------------------------------------------------------------------


@dataclass
class CameraData:
    """Viewer camera saved alongside a structure (currently CJSON only).

    Parameters
    ----------
    rotation:
        ``(3, 3)`` world → eye rotation, re-orthonormalised.  Applying it to
        atom positions reproduces the on-screen orientation: xyzrender projects
        orthographically with +x right, +y up and larger z nearer the viewer,
        which is OpenGL eye-space convention.
    perspective:
        ``True`` when the saved projection matrix is a perspective frustum.
        Informational only — xyzrender always renders orthographically, so the
        foreshortening is not reproduced.
    """

    rotation: np.ndarray = field(repr=False)
    perspective: bool = False


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
    colors:
        Per-atom ``#rrggbb`` colours stored in the file (one per atom), or
        ``None`` when the format carries no explicit colouring.
    camera:
        Saved viewer camera, or ``None`` when the file carries none.
    """

    atoms: list[tuple[str, tuple[float, float, float]]]
    bonds: list[tuple[int, int, float]] | None
    name: str = ""
    charge: int = 0
    pbc_cell: np.ndarray | None = field(default=None, repr=False)
    colors: list[str] | None = field(default=None, repr=False)
    camera: CameraData | None = field(default=None, repr=False)


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
# Chemical JSON (CJSON) — Avogadro's native format
# ---------------------------------------------------------------------------


def _cjson_matrix4(value: object) -> np.ndarray | None:
    """Coerce a CJSON 4x4 matrix to a ``(4, 4)`` array, or ``None``.

    Avogadro writes these as four nested rows; a flat 16-element array is
    accepted too since other CJSON producers use that form.
    """
    if not isinstance(value, list):
        return None
    try:
        matrix = np.array(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    return matrix


def _cjson_camera(properties: dict) -> CameraData | None:
    """Build :class:`CameraData` from ``properties.modelView`` / ``projection``.

    Avogadro stores the GL camera on save (``modelView`` maps world → eye).
    Only the rotation is usable here: the translation is redundant because the
    renderer re-centres the molecule, and the projection cannot be reproduced
    by an orthographic renderer.
    """
    model_view = _cjson_matrix4(properties.get("modelView"))
    if model_view is None:
        return None

    # Re-orthonormalise (polar decomposition) so any scale or numerical drift
    # in the saved matrix cannot stretch bond lengths.
    u, _s, vt = np.linalg.svd(model_view[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        # A mirrored camera would invert stereochemistry — refuse it rather
        # than silently rendering the enantiomer.
        logger.warning("CJSON modelView is not a proper rotation (det < 0) — ignoring saved camera")
        return None

    projection = _cjson_matrix4(properties.get("projection"))
    # Orthographic projections keep the homogeneous row as (0, 0, 0, 1);
    # a perspective frustum puts -1 in the w row.
    perspective = projection is not None and not np.isclose(projection[3, 3], 1.0)
    return CameraData(rotation=rotation, perspective=perspective)


def _cjson_unit_cell(root: dict) -> np.ndarray | None:
    """Extract a ``(3, 3)`` row-vector lattice from a CJSON ``unitCell`` block."""
    cell = root.get("unitCell")
    if not isinstance(cell, dict):
        cell = root.get("unit cell")
    if not isinstance(cell, dict):
        return None

    # Cell vectors are preferred over a/b/c parameters (matches cjsonformat.cpp).
    vectors = cell.get("cellVectors")
    if isinstance(vectors, list) and len(vectors) == 9:
        try:
            matrix = np.array(vectors, dtype=float).reshape(3, 3)
        except (TypeError, ValueError):
            matrix = None
        if matrix is not None and np.isfinite(matrix).all():
            if abs(float(np.linalg.det(matrix))) > 1e-8:
                return matrix
            logger.warning("CJSON cellVectors are not linearly independent — falling back to a/b/c")

    try:
        a, b, c = float(cell["a"]), float(cell["b"]), float(cell["c"])
        alpha, beta, gamma = float(cell["alpha"]), float(cell["beta"]), float(cell["gamma"])
    except (KeyError, TypeError, ValueError):
        return None
    return _abc_angles_to_cell(a, b, c, alpha, beta, gamma)


def _cjson_coords(coords: dict, n_atoms: int, frame: int, pbc_cell: np.ndarray | None) -> np.ndarray:
    """Return ``(n_atoms, 3)`` Cartesian coordinates in Ångström.

    Cartesian ``3d`` coordinates are preferred; Avogadro always writes them,
    even for periodic structures.  ``3dFractional`` is used only when they are
    absent, and then requires a unit cell to convert.
    """
    sets = coords.get("3dSets")
    if isinstance(sets, list) and sets:
        if frame >= len(sets):
            msg = f"CJSON frame {frame} requested but file has only {len(sets)} coordinate set(s)"
            raise IndexError(msg)
        flat = sets[frame]
    elif frame != 0:
        msg = f"CJSON frame {frame} requested but file has no '3dSets' coordinate sets"
        raise IndexError(msg)
    else:
        flat = coords.get("3d")

    if isinstance(flat, list) and len(flat) == 3 * n_atoms:
        return np.array(flat, dtype=float).reshape(n_atoms, 3)

    fractional = coords.get("3dFractional")
    if not isinstance(fractional, list):
        fractional = coords.get("3d fractional")
    if isinstance(fractional, list) and len(fractional) == 3 * n_atoms:
        if pbc_cell is None:
            msg = "CJSON has fractional coordinates but no unit cell to convert them"
            raise ValueError(msg)
        # Rows of pbc_cell are a, b, c — so r = fa*a + fb*b + fc*c.
        return np.array(fractional, dtype=float).reshape(n_atoms, 3) @ pbc_cell

    msg = f"CJSON has no usable coordinates for {n_atoms} atoms"
    raise ValueError(msg)


def _cjson_bonds(root: dict, remap: dict[int, int]) -> list[tuple[int, int, float]] | None:
    """Extract bonds from the CJSON ``bonds`` block, or ``None`` when absent.

    *remap* maps a CJSON atom index to its index in the parsed atom list; atoms
    missing from it (dummies) have been dropped, so their bonds go with them.
    """
    bonds_obj = root.get("bonds")
    if not isinstance(bonds_obj, dict):
        return None
    connections = bonds_obj.get("connections")
    index = connections.get("index") if isinstance(connections, dict) else None
    if not isinstance(index, list):
        return None
    orders = bonds_obj.get("order")

    bonds: list[tuple[int, int, float]] = []
    for k in range(len(index) // 2):
        try:
            i, j = remap.get(int(index[2 * k]), -1), remap.get(int(index[2 * k + 1]), -1)
        except (TypeError, ValueError):
            continue
        # Drop self-bonds and out-of-range indices, as cjsonformat.cpp does.
        if i == j or i < 0 or j < 0:
            continue
        order = 1.0
        if isinstance(orders, list) and k < len(orders):
            try:
                order = float(orders[k])
            except (TypeError, ValueError):
                order = 1.0
        bonds.append((i, j, order))
    return bonds or None


def _cjson_colors(atoms_obj: dict, n_atoms: int) -> list[str] | None:
    """Extract per-atom ``#rrggbb`` colours from ``atoms.colors`` (flat 3N, 0-255)."""
    from xyzrender.colors import Color

    raw = atoms_obj.get("colors")
    if not isinstance(raw, list):
        return None
    if len(raw) != 3 * n_atoms:
        logger.warning("CJSON 'atoms.colors' has %d values, expected %d — ignoring", len(raw), 3 * n_atoms)
        return None
    try:
        channels = [min(255, max(0, round(float(v)))) for v in raw]
    except (TypeError, ValueError):
        logger.warning("CJSON 'atoms.colors' is not numeric — ignoring")
        return None
    return [Color(*channels[3 * i : 3 * i + 3]).hex for i in range(n_atoms)]


def parse_cjson(path: str | Path, frame: int = 0) -> MolData:
    """Parse a Chemical JSON (CJSON) file — Avogadro's native format.

    Reads atoms, explicit bond connectivity and bond orders, per-atom colour
    overrides, the unit cell, and the viewer camera saved by Avogadro.  Dummy
    atoms (atomic number 0) are ignored, along with any bonds to them.

    Parameters
    ----------
    path:
        Path to the ``.cjson`` file.
    frame:
        Zero-based index into ``atoms.coords.3dSets`` for multi-conformer
        files (default: 0, which uses the primary ``3d`` coordinates).

    Returns
    -------
    MolData
        Parsed structure.  ``pbc_cell`` is set when the file has a ``unitCell``
        block, ``colors`` when it has ``atoms.colors``, and ``camera`` when
        Avogadro saved a ``modelView`` matrix in ``properties``.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        root = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"{path} is not valid JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(root, dict):
        msg = f"{path} is not a CJSON object"
        raise ValueError(msg)
    return parse_cjson_dict(root, frame=frame)


def parse_cjson_dict(root: dict, frame: int = 0) -> MolData:
    """Parse an already-decoded CJSON object.  See :func:`parse_cjson`."""
    from xyzgraph import DATA

    version = root.get("chemicalJson", root.get("chemical json"))
    if version is None:
        msg = "Not a CJSON file: no 'chemicalJson' key"
        raise ValueError(msg)
    if version not in (0, 1):
        logger.warning("CJSON version %r is newer than 1 — parsing anyway", version)

    atoms_obj = root.get("atoms")
    if not isinstance(atoms_obj, dict):
        msg = "CJSON has no 'atoms' object"
        raise ValueError(msg)
    elements = atoms_obj.get("elements")
    numbers = elements.get("number") if isinstance(elements, dict) else None
    if not isinstance(numbers, list):
        msg = "CJSON has no 'atoms.elements.number' array"
        raise ValueError(msg)

    # Dummy atoms (atomic number 0, or anything outside the element table) are
    # dropped: they have no radius or colour to render.  `keep` maps CJSON atom
    # index → index in the parsed list, so bonds and colours can follow.
    symbols: list[str] = []
    keep: dict[int, int] = {}
    for raw_index, z in enumerate(numbers):
        sym = DATA.n2s.get(int(z)) if isinstance(z, int) or (isinstance(z, float) and z.is_integer()) else None
        if sym is None:
            continue
        keep[raw_index] = len(symbols)
        symbols.append(sym)
    n_raw = len(numbers)
    if len(symbols) < n_raw:
        logger.info("Ignored %d dummy atom(s) in CJSON", n_raw - len(symbols))

    pbc_cell = _cjson_unit_cell(root)
    coords = atoms_obj.get("coords")
    if not isinstance(coords, dict):
        msg = "CJSON has no 'atoms.coords' object"
        raise ValueError(msg)
    # Coordinates and colours are sized to the full atom list, so parse them at
    # full width and then take the kept rows.
    positions = _cjson_coords(coords, n_raw, frame, pbc_cell)[list(keep)]
    colors = _cjson_colors(atoms_obj, n_raw)
    if colors is not None:
        colors = [colors[i] for i in keep]

    properties = root.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    try:
        charge = int(properties.get("totalCharge", 0))
    except (TypeError, ValueError):
        charge = 0

    return MolData(
        atoms=[(sym, (float(x), float(y), float(z))) for sym, (x, y, z) in zip(symbols, positions, strict=True)],
        bonds=_cjson_bonds(root, keep),
        name=str(root.get("name", "")),
        charge=charge,
        pbc_cell=pbc_cell,
        colors=colors,
        camera=_cjson_camera(properties),
    )


# ---------------------------------------------------------------------------
# Extension dispatcher
# ---------------------------------------------------------------------------


def parse(path: str | Path, frame: int = 0) -> MolData:
    """Parse a molecular file, dispatching on extension (.mol, .sdf, .mol2, .pdb, .cjson)."""
    p = str(path)
    if p.endswith(".mol"):
        return parse_mol(path)
    if p.endswith(".sdf"):
        return parse_sdf(path, frame=frame)
    if p.endswith(".mol2"):
        return parse_mol2(path)
    if p.endswith(".pdb"):
        return parse_pdb(path)
    if p.endswith(".cjson"):
        return parse_cjson(path, frame=frame)
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

    atoms: list[tuple[str, tuple[float, float, float]]] = []
    if hasattr(shx, "pack"):
        for atom in shx.pack():
            sym = (
                atom.element.capitalize()
                if hasattr(atom, "element") and atom.element
                else atom.name.rstrip("0123456789").capitalize()
            )
            if hasattr(atom, "cart_coords") and atom.cart_coords is not None:
                x, y, z = atom.cart_coords
            elif hasattr(atom, "xc") and hasattr(atom, "yc") and hasattr(atom, "zc"):
                x, y, z = atom.xc, atom.yc, atom.zc
            else:
                x, y, z = 0.0, 0.0, 0.0
            atoms.append((sym, (float(x), float(y), float(z))))

    pbc_cell = None
    cell = getattr(shx, "cell", None)
    if cell is not None:
        pbc_cell = _abc_angles_to_cell(cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)

    return MolData(atoms=atoms, bonds=None, pbc_cell=pbc_cell, name=str(path))
