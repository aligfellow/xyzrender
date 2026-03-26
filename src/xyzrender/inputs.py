"""QM input file parsers.

Generic coordinate checker for molecular codes (Gaussian, ORCA, NWChem,
Q-Chem, Psi4, MOPAC, Molpro, GAMESS, Turbomole, CP2K, etc.) plus dedicated
parsers for periodic formats (VASP POSCAR, QE pw.in, SIESTA FDF, ABINIT).
"""

from __future__ import annotations

import logging
import re
from typing import TypeAlias

import numpy as np
from xyzgraph import DATA

logger = logging.getLogger(__name__)

_Atoms: TypeAlias = list[tuple[str, tuple[float, float, float]]]

BOHR_TO_ANG = 0.529177210903

# Known element symbols (lowercase) for fast lookup
_SYMBOLS = {s.lower() for s in DATA.s2n}


# ---------------------------------------------------------------------------
# Generic coordinate extraction
# ---------------------------------------------------------------------------


def _is_element(token: str) -> str | None:
    """Return canonical element symbol if *token* is a valid element, else None."""
    # Try as symbol
    low = token.lower()
    if low in _SYMBOLS:
        return token[0].upper() + token[1:].lower() if len(token) > 1 else token.upper()
    # Try as atomic number
    try:
        z = int(token)
        if z in DATA.n2s:
            return DATA.n2s[z]
    except ValueError:
        pass
    return None


def _try_parse_coord_line(line: str) -> tuple[str, tuple[float, float, float]] | None:
    """Try to parse a single coordinate line in various formats.

    Supported patterns (tried in order):
      SYMBOL  X  Y  Z          (standard - Gaussian, ORCA, NWChem, …)
      Z       X  Y  Z          (atomic number)
      SYMBOL  ZNUC  X  Y  Z    (GAMESS $DATA - 5+ fields)
      X  Y  Z  SYMBOL          (Turbomole - caller handles unit conversion)
    """
    parts = line.split()
    if len(parts) < 4:
        return None

    sym = _is_element(parts[0])

    if sym is not None:
        # GAMESS: SYMBOL ZNUC X Y Z (try 5-field first to avoid wrong coords)
        if len(parts) >= 5:
            try:
                float(parts[1])  # Znuc
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                return sym, (x, y, z)
            except (ValueError, IndexError):
                pass

        # Standard: SYMBOL/Z X Y Z
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            return sym, (x, y, z)
        except (ValueError, IndexError):
            pass

    # Turbomole: X Y Z SYMBOL
    tail_sym = _is_element(parts[-1])
    if tail_sym is not None:
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            return tail_sym, (x, y, z)
        except (ValueError, IndexError):
            pass

    return None


def _detect_bohr(lines: list[str]) -> bool:
    """Check whether the input file explicitly uses Bohr units."""
    for line in lines:
        low = line.strip().lower()
        # Turbomole: $coord without "angs"
        if low == "$coord":
            return True
        if low.startswith("$coord") and "angs" in low:
            return False
        # NWChem: geometry units bohr / geometry units au
        if low.startswith("geometry") and ("bohr" in low or "units au" in low):
            return True
        # Dalton: ATOMBASIS / INTGRL with Bohr default
        if low in ("atombasis", "intgrl"):
            return True
    return False


def _get_coords(lines: list[str]) -> tuple[_Atoms, int, bool]:
    """Find the longest contiguous block of coordinate lines.

    Returns ``(atoms, start_index, is_bohr)`` where *start_index* is the
    line index of the first coordinate line and *is_bohr* is ``True`` when
    the file explicitly declares Bohr units.

    Raises :class:`ValueError` if no coordinates are found.
    """
    best_atoms: _Atoms = []
    best_start = 0
    current_atoms: _Atoms = []
    current_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if len(current_atoms) > len(best_atoms):
                best_atoms = current_atoms
                best_start = current_start
            current_atoms = []
            continue

        result = _try_parse_coord_line(stripped)
        if result is not None:
            if not current_atoms:
                current_start = i
            current_atoms.append(result)
        else:
            if len(current_atoms) > len(best_atoms):
                best_atoms = current_atoms
                best_start = current_start
            current_atoms = []

    # Final block
    if len(current_atoms) > len(best_atoms):
        best_atoms = current_atoms
        best_start = current_start

    if not best_atoms:
        raise ValueError("No coordinate block found")

    return best_atoms, best_start, _detect_bohr(lines)


# ---------------------------------------------------------------------------
# Charge / multiplicity extraction
# ---------------------------------------------------------------------------

# NWChem multiplicity words
_MULT_WORDS = {
    "singlet": 1,
    "doublet": 2,
    "triplet": 3,
    "quartet": 4,
    "quintet": 5,
    "sextet": 6,
    "septet": 7,
    "octet": 8,
}


def _get_charge_mult(lines: list[str], coord_start: int) -> tuple[int, int | None]:
    """Extract charge and multiplicity from input file lines.

    Tries format-specific patterns in order, returns first match.
    Falls back to ``(0, 1)`` if nothing is found.
    """
    text_lower = [line.strip().lower() for line in lines]

    # 1. ORCA:  * xyz CHARGE MULT  or  * xyzfile CHARGE MULT
    for line in text_lower:
        if line.startswith("*") and ("xyz" in line):
            parts = line.split()
            if len(parts) >= 4 and parts[1] in ("xyz", "xyzfile"):
                try:
                    return int(parts[2]), int(parts[3])
                except (ValueError, IndexError):
                    pass

    # 2. Q-Chem:  $molecule → next non-blank line is CHARGE MULT
    for i, line in enumerate(text_lower):
        if line == "$molecule":
            for j in range(i + 1, len(text_lower)):
                if text_lower[j] and not text_lower[j].startswith("$"):
                    parts = text_lower[j].split()
                    if len(parts) >= 2:
                        try:
                            return int(parts[0]), int(parts[1])
                        except ValueError:
                            pass
                    break

    # 3. NWChem:  charge INT  and optionally  mult INT / scf; singlet; end
    nw_charge: int | None = None
    nw_mult: int | None = None
    for line in text_lower:
        if line.startswith("charge") and not line.startswith("charge_"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    nw_charge = int(parts[1])
                except ValueError:
                    pass
        if line.startswith("mult"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    nw_mult = int(parts[1])
                except ValueError:
                    pass
        for word, m in _MULT_WORDS.items():
            if word in line and ("scf" in line or "dft" in line):
                nw_mult = m
    if nw_charge is not None:
        return nw_charge, nw_mult if nw_mult is not None else 1

    # 4. Psi4:  molecule {  → next non-blank line is CHARGE MULT
    for i, line in enumerate(text_lower):
        if "molecule" in line and "{" in line:
            for j in range(i + 1, len(text_lower)):
                stripped = text_lower[j]
                if stripped and stripped != "}":
                    parts = stripped.split()
                    if len(parts) == 2:
                        try:
                            return int(parts[0]), int(parts[1])
                        except ValueError:
                            pass
                    break

    # 5. General / Gaussian:  line with exactly two integers just before coord block
    #    Walk backwards from coord_start to find it
    for i in range(coord_start - 1, max(coord_start - 5, -1), -1):
        if i < 0:
            break
        stripped = lines[i].strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass
        break  # Only check the first non-blank line before coords

    return 0, 1


# ---------------------------------------------------------------------------
# Public: generic QM input parser
# ---------------------------------------------------------------------------


def _find_external_xyz(lines: list[str], base_dir: str) -> str | None:
    """Search for an external ``.xyz`` file reference in the input lines.

    Looks for patterns like ``* xyzfile 0 1 molecule.xyz`` (ORCA),
    ``COORD_FILE_NAME structure.xyz`` (CP2K), or any ``.xyz`` token.
    """
    import os

    def _resolve(candidate: str) -> str | None:
        full = os.path.join(base_dir, candidate) if not os.path.isabs(candidate) else candidate
        return full if os.path.isfile(full) else None

    for line in lines:
        parts = line.split()
        # ORCA: * xyzfile charge mult filename.xyz
        if len(parts) >= 5 and parts[0] == "*" and parts[1].lower() == "xyzfile":
            hit = _resolve(parts[4])
            if hit:
                return hit
        # CP2K: COORD_FILE_NAME filename.xyz
        if len(parts) >= 2 and parts[0].upper() == "COORD_FILE_NAME":
            hit = _resolve(parts[1])
            if hit:
                return hit

    # Generic fallback: any .xyz token
    for line in lines:
        for token in line.split():
            if token.lower().endswith(".xyz") and token[0] not in "#!":
                hit = _resolve(token)
                if hit:
                    return hit

    return None


def parse_qm_input(path: str, *, bohr: bool | None = None) -> tuple[_Atoms, int, int | None]:
    """Parse any QM input file to extract atoms, charge, and multiplicity.

    Parameters
    ----------
    path:
        Path to the input file.
    bohr:
        ``True`` = force Bohr→Angstrom conversion.
        ``None`` (default) = auto-detect from file content.
    """
    import os

    with open(path) as f:
        lines = f.readlines()

    charge, mult = 0, 1
    is_bohr = False
    try:
        atoms, coord_start, is_bohr = _get_coords(lines)
        charge, mult = _get_charge_mult(lines, coord_start)
    except ValueError:
        base_dir = os.path.dirname(os.path.abspath(path))
        ext_path = _find_external_xyz(lines, base_dir)
        if ext_path is not None:
            logger.warning("No inline coordinates in %s - reading from %s", path, ext_path)
            with open(ext_path) as f:
                ext_lines = f.readlines()
            atoms, coord_start, is_bohr = _get_coords(ext_lines)
            charge, mult = _get_charge_mult(lines, 0)
        else:
            raise ValueError(f"No coordinate block found in {path}") from None

    convert = bohr if bohr is not None else is_bohr
    if convert:
        logger.info("Converting Bohr → Angstrom for %s%s", path, " (auto-detected)" if bohr is None else "")
        atoms = [(sym, (x * BOHR_TO_ANG, y * BOHR_TO_ANG, z * BOHR_TO_ANG)) for sym, (x, y, z) in atoms]

    logger.info("Parsed %d atoms from %s (charge=%d, mult=%s)", len(atoms), path, charge, mult)
    return atoms, charge, mult


# ---------------------------------------------------------------------------
# QE input detection
# ---------------------------------------------------------------------------

_QE_MARKERS = {"&system", "&control", "cell_parameters", "atomic_positions"}


def is_qe_input(path: str) -> bool:
    """Detect whether a ``.in`` file is Quantum ESPRESSO (vs Q-Chem etc.)."""
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            first_token = line.strip().lower().split()[0] if line.strip() else ""
            if first_token in _QE_MARKERS:
                return True
    return False


def is_cp2k_input(path: str) -> bool:
    """Detect whether a file is a CP2K input (has ``&FORCE_EVAL`` or ``&CELL``)."""
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 80:
                break
            stripped = line.strip().lower()
            if stripped in ("&force_eval", "&cell", "&coord"):
                return True
    return False


def parse_cp2k_cell(path: str) -> np.ndarray | None:
    """Extract the unit cell from a CP2K input file.

    Supports ``ABC a b c`` (orthorhombic) and explicit ``A``/``B``/``C``
    vector definitions inside the ``&CELL`` block.

    Returns a ``(3, 3)`` ndarray in Angstrom, or ``None`` if no cell found.
    """
    with open(path) as f:
        lines = f.readlines()

    in_cell = False
    lattice = np.zeros((3, 3))
    has_cell = False
    vec_map = {"a": 0, "b": 1, "c": 2}

    for line in lines:
        stripped = line.strip()
        low = stripped.lower()

        if low == "&cell":
            in_cell = True
            continue
        if low in ("&end cell", "&end"):
            if in_cell:
                in_cell = False
            continue

        if not in_cell:
            continue

        parts = stripped.split()
        if not parts:
            continue

        keyword = parts[0].lower()

        # ABC a b c  →  orthorhombic diagonal cell
        if keyword == "abc" and len(parts) >= 4:
            lattice[0, 0] = float(parts[1])
            lattice[1, 1] = float(parts[2])
            lattice[2, 2] = float(parts[3])
            has_cell = True

        # A ax ay az  /  B bx by bz  /  C cx cy cz
        elif keyword in vec_map and len(parts) >= 4:
            idx = vec_map[keyword]
            lattice[idx] = [float(parts[1]), float(parts[2]), float(parts[3])]
            has_cell = True

    return lattice if has_cell else None


# ---------------------------------------------------------------------------
# VASP POSCAR parser
# ---------------------------------------------------------------------------


def parse_poscar(path: str) -> tuple[_Atoms, np.ndarray]:
    """Parse a VASP POSCAR/CONTCAR file.

    Returns ``(atoms, lattice)`` where lattice is a ``(3, 3)`` ndarray in
    Angstrom (rows = a, b, c).
    """
    with open(path) as f:
        raw_lines = f.readlines()

    # Line 0: comment
    # Line 1: scale factor
    scale = float(raw_lines[1].strip())

    # Lines 2-4: lattice vectors
    lattice = np.array([[float(x) for x in raw_lines[i].split()] for i in range(2, 5)]) * scale

    # Line 5: element symbols
    symbols_line = raw_lines[5].split()
    if all(tok.isdigit() for tok in symbols_line):
        msg = (
            f"Old-format POSCAR without element names detected in {path}. Add element symbols on line 6 (e.g. 'Si O')."
        )
        raise ValueError(msg)

    # Line 6: atom counts
    counts = [int(x) for x in raw_lines[6].split()]

    # Expand symbols: ["N", "C"] + [1, 62] → ["N", "C", "C", ...]
    all_symbols: list[str] = []
    for sym, n in zip(symbols_line, counts, strict=True):
        all_symbols.extend([sym] * n)

    # Line 7+: optional "Selective dynamics", then "Direct" or "Cartesian"
    idx = 7
    if raw_lines[idx].strip().lower().startswith("s"):  # Selective dynamics
        idx += 1
    coord_type = raw_lines[idx].strip().lower()
    idx += 1

    # Read coordinates
    n_atoms = sum(counts)
    positions = np.array([[float(x) for x in raw_lines[idx + i].split()[:3]] for i in range(n_atoms)])

    if coord_type.startswith("d"):  # Direct = fractional
        positions = positions @ lattice
    else:  # Cartesian
        positions *= scale

    atoms: _Atoms = [
        (sym, (float(pos[0]), float(pos[1]), float(pos[2]))) for sym, pos in zip(all_symbols, positions, strict=True)
    ]
    logger.info("POSCAR: %d atoms, lattice diag=%s", n_atoms, np.diag(lattice).round(3))
    return atoms, lattice


# ---------------------------------------------------------------------------
# QE pw.in parser
# ---------------------------------------------------------------------------


def parse_qe_input(path: str) -> tuple[_Atoms, np.ndarray, int]:
    """Parse a Quantum ESPRESSO pw.in file.

    Returns ``(atoms, lattice, charge)`` where lattice is a ``(3, 3)``
    ndarray in Angstrom and charge is the total charge.
    """
    with open(path) as f:
        text = f.read()
    lines = text.splitlines()

    # --- Parse &SYSTEM namelist ---
    ibrav = 0
    tot_charge = 0
    system_match = re.search(r"&SYSTEM(.*?)(/|&END)", text, re.IGNORECASE | re.DOTALL)
    if system_match:
        block = system_match.group(1)
        m = re.search(r"ibrav\s*=\s*(\d+)", block, re.IGNORECASE)
        if m:
            ibrav = int(m.group(1))
        m = re.search(r"tot_charge\s*=\s*([-\d.]+)", block, re.IGNORECASE)
        if m:
            tot_charge = int(float(m.group(1)))

    if ibrav != 0:
        msg = f"Only ibrav=0 (explicit CELL_PARAMETERS) is supported, got ibrav={ibrav} in {path}"
        raise ValueError(msg)

    # --- Parse CELL_PARAMETERS ---
    lattice = np.zeros((3, 3))
    cell_unit = "bohr"  # default
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("cell_parameters"):
            # Extract unit keyword
            m = re.search(r"\{?\s*(bohr|angstrom)\s*\}?", line, re.IGNORECASE)
            if m:
                cell_unit = m.group(1).lower()
            for j in range(3):
                lattice[j] = [float(x) for x in lines[i + 1 + j].split()[:3]]
            break

    if cell_unit == "bohr":
        lattice *= BOHR_TO_ANG

    # --- Parse ATOMIC_SPECIES (label → element mapping) ---
    species_map: dict[str, str] = {}
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("atomic_species"):
            for j in range(i + 1, len(lines)):
                parts = lines[j].split()
                if (
                    not parts
                    or parts[0].startswith("&")
                    or parts[0].upper()
                    in (
                        "ATOMIC_POSITIONS",
                        "K_POINTS",
                        "CELL_PARAMETERS",
                        "CONSTRAINTS",
                        "OCCUPATIONS",
                        "ATOMIC_FORCES",
                    )
                ):
                    break
                # label  mass  pseudopotential
                if len(parts) >= 2:
                    label = parts[0]
                    # Element is typically the first 1-2 chars of the label
                    elem = _is_element(label)
                    if elem is None:
                        elem = _is_element(label[:2])
                    if elem is None:
                        elem = _is_element(label[:1])
                    if elem is not None:
                        species_map[label] = elem
            break

    # --- Parse ATOMIC_POSITIONS ---
    atoms: _Atoms = []
    pos_unit = "crystal"  # default
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("atomic_positions"):
            m = re.search(r"\{?\s*(crystal|angstrom|bohr|crystal_sg|alat)\s*\}?", line, re.IGNORECASE)
            if m:
                pos_unit = m.group(1).lower()
            for j in range(i + 1, len(lines)):
                parts = lines[j].split()
                if (
                    not parts
                    or parts[0].startswith("&")
                    or parts[0].upper()
                    in (
                        "ATOMIC_SPECIES",
                        "K_POINTS",
                        "CELL_PARAMETERS",
                        "CONSTRAINTS",
                        "OCCUPATIONS",
                        "ATOMIC_FORCES",
                    )
                ):
                    break
                if len(parts) >= 4:
                    label = parts[0]
                    sym = species_map.get(label) or _is_element(label) or label
                    pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    if pos_unit == "crystal":
                        pos = pos @ lattice  # fractional → Cartesian Å
                    elif pos_unit == "bohr":
                        pos *= BOHR_TO_ANG
                    atoms.append((sym, (float(pos[0]), float(pos[1]), float(pos[2]))))
            break

    if not atoms:
        raise ValueError(f"No ATOMIC_POSITIONS found in {path}")

    logger.info("QE: %d atoms, charge=%d, lattice diag=%s", len(atoms), tot_charge, np.diag(lattice).round(3))
    return atoms, lattice, tot_charge


# ---------------------------------------------------------------------------
# SIESTA FDF parser
# ---------------------------------------------------------------------------


def _parse_fdf_block(lines: list[str], block_name: str) -> list[list[str]]:
    """Extract rows from a SIESTA ``%block ... %endblock`` section."""
    rows: list[list[str]] = []
    in_block = False
    target = block_name.lower()
    for line in lines:
        low = line.strip().lower()
        if low.startswith("%block") and target in low:
            in_block = True
            continue
        if low.startswith("%endblock") and target in low:
            break
        if in_block:
            stripped = line.split("#")[0].strip()  # strip comments
            if stripped:
                rows.append(stripped.split())
    return rows


def _fdf_value(lines: list[str], key: str, default: str = "") -> str:
    """Find a scalar FDF key-value (case-insensitive, ignores comments)."""
    target = key.lower()
    for line in lines:
        stripped = line.split("#")[0].strip()
        parts = stripped.split()
        if len(parts) >= 2 and parts[0].lower() == target:
            return parts[1]
    return default


def is_siesta_input(path: str) -> bool:
    """Detect a SIESTA FDF file."""
    with open(path) as f:
        text = f.read(4000).lower()
    return "chemicalspecieslabel" in text or "atomiccoordinatesandatomicspecies" in text


def parse_siesta_fdf(path: str) -> tuple[_Atoms, np.ndarray]:
    """Parse a SIESTA ``.fdf`` input file.

    Returns ``(atoms, lattice)`` where lattice is ``(3, 3)`` in Angstrom.
    """
    with open(path) as f:
        lines = f.readlines()

    # Species mapping: index → element symbol
    species_map: dict[int, str] = {}
    for row in _parse_fdf_block(lines, "chemicalspecieslabel"):
        if len(row) >= 3:
            idx, z = int(row[0]), int(row[1])
            sym = _is_element(str(z)) or _is_element(row[2]) or row[2]
            species_map[idx] = sym

    # Lattice constant - SIESTA default unit is Bohr
    lat_const = 1.0
    for line in lines:
        low = line.split("#")[0].strip().lower()
        if low.startswith("latticeconstant"):
            parts = low.split()
            if len(parts) >= 2:
                lat_const = float(parts[1])
            # Unit: Ang keeps as-is, Bohr (or no unit) → convert
            if "ang" in low:
                pass
            else:
                lat_const *= BOHR_TO_ANG
            break

    # Lattice vectors
    lattice = np.eye(3) * lat_const
    lat_rows = _parse_fdf_block(lines, "latticevectors")
    if len(lat_rows) >= 3:
        lattice = np.array([[float(x) for x in row[:3]] for row in lat_rows[:3]]) * lat_const

    # Coordinate format
    coord_fmt = _fdf_value(lines, "atomiccoordinatesformat", "bohr").lower()

    # Atomic positions
    atoms: _Atoms = []
    for row in _parse_fdf_block(lines, "atomiccoordinatesandatomicspecies"):
        if len(row) >= 4:
            pos = np.array([float(row[0]), float(row[1]), float(row[2])])
            sp_idx = int(row[3])
            sym = species_map.get(sp_idx, f"X{sp_idx}")

            if "frac" in coord_fmt or "scaled" in coord_fmt:
                pos = pos @ lattice
            elif "bohr" in coord_fmt:
                pos *= BOHR_TO_ANG
            # "ang" or "notscaledcartesianang" → already Angstrom

            atoms.append((sym, (float(pos[0]), float(pos[1]), float(pos[2]))))

    if not atoms:
        raise ValueError(f"No atomic coordinates found in {path}")

    logger.info("SIESTA: %d atoms, lattice diag=%s", len(atoms), np.diag(lattice).round(3))
    return atoms, lattice


# ---------------------------------------------------------------------------
# ABINIT parser
# ---------------------------------------------------------------------------


def _abinit_read_values(lines: list[str], key: str, count: int) -> list[float]:
    """Read *count* float values after an ABINIT keyword.

    Values may be on the same line or continuation lines. Supports the
    ``N*val`` repeat syntax (e.g. ``3*5.0``).
    """
    vals: list[float] = []
    found = False
    for line in lines:
        stripped = line.split("#")[0].split("!")[0].strip()
        if not found:
            parts = stripped.split()
            if parts and parts[0].lower() == key.lower():
                tokens = parts[1:]
                found = True
            else:
                continue
        else:
            tokens = stripped.split()

        for tok in tokens:
            if "*" in tok:
                n_str, v_str = tok.split("*", 1)
                vals.extend([float(v_str)] * int(n_str))
            else:
                try:
                    vals.append(float(tok))
                except ValueError:
                    break
            if len(vals) >= count:
                return vals[:count]
    return vals[:count]


def _abinit_read_ints(lines: list[str], key: str, count: int) -> list[int]:
    """Like :func:`_abinit_read_values` but for integers."""
    return [int(v) for v in _abinit_read_values(lines, key, count)]


def is_abinit_input(path: str) -> bool:
    """Detect an ABINIT input file."""
    with open(path) as f:
        text = f.read(4000).lower()
    return ("ntypat" in text or "znucl" in text) and ("natom" in text or "xred" in text or "xcart" in text)


def parse_abinit_input(path: str) -> tuple[_Atoms, np.ndarray]:
    """Parse an ABINIT input file.

    Returns ``(atoms, lattice)`` where lattice is ``(3, 3)`` in Angstrom.
    """
    with open(path) as f:
        lines = f.readlines()

    natom = int(_abinit_read_values(lines, "natom", 1)[0])
    ntypat = int(_abinit_read_values(lines, "ntypat", 1)[0])
    znucl = _abinit_read_values(lines, "znucl", ntypat)
    typat = _abinit_read_ints(lines, "typat", natom)

    # Map each atom to an element symbol
    symbols = [DATA.n2s[int(znucl[t - 1])] for t in typat]

    # Unit cell: acell (default 1 1 1 Bohr) * rprim (default identity)
    acell = _abinit_read_values(lines, "acell", 3) or [1.0, 1.0, 1.0]
    rprim_vals = _abinit_read_values(lines, "rprim", 9)
    if len(rprim_vals) == 9:
        rprim = np.array(rprim_vals).reshape(3, 3)
    else:
        rprim = np.eye(3)

    # lattice[i] = acell[i] * rprim[i], in Bohr → convert to Angstrom
    lattice = rprim * np.array(acell)[:, None] * BOHR_TO_ANG

    # Coordinates: try xangst, xcart, xred in priority order
    coords = np.zeros((natom, 3))
    coord_type = None
    for key in ("xangst", "xcart", "xred"):
        vals = _abinit_read_values(lines, key, natom * 3)
        if len(vals) == natom * 3:
            coords = np.array(vals).reshape(natom, 3)
            coord_type = key
            break

    if coord_type is None:
        raise ValueError(f"No coordinates (xred/xcart/xangst) found in {path}")

    if coord_type == "xred":
        coords = coords @ lattice  # fractional → Cartesian Å
    elif coord_type == "xcart":
        coords *= BOHR_TO_ANG  # Bohr → Å

    atoms: _Atoms = [
        (sym, (float(pos[0]), float(pos[1]), float(pos[2]))) for sym, pos in zip(symbols, coords, strict=True)
    ]

    logger.info("ABINIT: %d atoms, lattice diag=%s", len(atoms), np.diag(lattice).round(3))
    return atoms, lattice
