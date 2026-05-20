"""Tests for xyzrender.formats and io loader functions.

All fixtures use checked-in example files — no rdkit or ase generation needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths to checked-in test files
# ---------------------------------------------------------------------------

_STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"
_INPUTS = Path(__file__).parent.parent / "examples" / "inputs"

_CAFFEINE_SDF = _STRUCTURES / "caffeine_sdf.sdf"
_MULTI_SDF = _STRUCTURES / "multi_mol.sdf"
_WATER_MOL2 = _STRUCTURES / "water_mol2.mol2"
_WATER_PDB = _STRUCTURES / "water.pdb"
_WATER_PDB_CRYST = _STRUCTURES / "water_cryst.pdb"
_ALA_PDB = _STRUCTURES / "ala_phe_ala.pdb"
_CIF_FILE = _STRUCTURES / "caffeine_cif.cif"

_CAFFEINE_ATOMS = 24  # C8N4O2 + 10H


# ---------------------------------------------------------------------------
# parse_mol (SDF V2000 is the same block format as .mol)
# ---------------------------------------------------------------------------


class TestParseMol:
    def test_atom_count(self):
        from xyzrender.parsers import parse_mol

        d = parse_mol(_CAFFEINE_SDF)
        assert len(d.atoms) == _CAFFEINE_ATOMS

    def test_element_symbols(self):
        from xyzrender.parsers import parse_mol

        d = parse_mol(_CAFFEINE_SDF)
        symbols = {sym for sym, _ in d.atoms}
        assert {"C", "N", "O", "H"} == symbols

    def test_bonds_present(self):
        from xyzrender.parsers import parse_mol

        d = parse_mol(_CAFFEINE_SDF)
        assert d.bonds is not None
        assert len(d.bonds) > 0

    def test_no_pbc_cell(self):
        from xyzrender.parsers import parse_mol

        d = parse_mol(_CAFFEINE_SDF)
        assert d.pbc_cell is None


# ---------------------------------------------------------------------------
# parse_sdf
# ---------------------------------------------------------------------------


class TestParseSdf:
    def test_atom_count(self):
        from xyzrender.parsers import parse_sdf

        d = parse_sdf(_CAFFEINE_SDF, frame=0)
        assert len(d.atoms) == _CAFFEINE_ATOMS

    def test_bonds_present(self):
        from xyzrender.parsers import parse_sdf

        d = parse_sdf(_CAFFEINE_SDF, frame=0)
        assert d.bonds is not None
        assert len(d.bonds) > 0

    def test_frame_out_of_range(self):
        from xyzrender.parsers import parse_sdf

        with pytest.raises(IndexError):
            parse_sdf(_CAFFEINE_SDF, frame=99)

    def test_multi_frame0(self):
        from xyzrender.parsers import parse_sdf

        d = parse_sdf(_MULTI_SDF, frame=0)
        assert len(d.atoms) == _CAFFEINE_ATOMS

    def test_multi_frame1(self):
        from xyzrender.parsers import parse_sdf

        d = parse_sdf(_MULTI_SDF, frame=1)
        assert len(d.atoms) == 3  # water

    def test_multi_frame_selects_different_molecules(self):
        from xyzrender.parsers import parse_sdf

        d0 = parse_sdf(_MULTI_SDF, frame=0)
        d1 = parse_sdf(_MULTI_SDF, frame=1)
        assert len(d0.atoms) != len(d1.atoms)


# ---------------------------------------------------------------------------
# parse_mol2
# ---------------------------------------------------------------------------


class TestParseMol2:
    def test_atom_count(self):
        from xyzrender.parsers import parse_mol2

        d = parse_mol2(_WATER_MOL2)
        assert len(d.atoms) == 3

    def test_element_symbols(self):
        from xyzrender.parsers import parse_mol2

        d = parse_mol2(_WATER_MOL2)
        symbols = {sym for sym, _ in d.atoms}
        assert symbols == {"O", "H"}

    def test_bonds_present(self):
        from xyzrender.parsers import parse_mol2

        d = parse_mol2(_WATER_MOL2)
        assert d.bonds is not None
        assert len(d.bonds) == 2


# ---------------------------------------------------------------------------
# parse_pdb
# ---------------------------------------------------------------------------


class TestParsePdb:
    def test_atom_count(self):
        from xyzrender.parsers import parse_pdb

        d = parse_pdb(_WATER_PDB)
        assert len(d.atoms) == 3

    def test_element_symbols(self):
        from xyzrender.parsers import parse_pdb

        d = parse_pdb(_WATER_PDB)
        symbols = {sym for sym, _ in d.atoms}
        assert symbols == {"O", "H"}

    def test_no_cryst1(self):
        from xyzrender.parsers import parse_pdb

        d = parse_pdb(_WATER_PDB)
        assert d.pbc_cell is None

    def test_cryst1_parsed(self):
        from xyzrender.parsers import parse_pdb

        d = parse_pdb(_WATER_PDB_CRYST)
        assert d.pbc_cell is not None
        assert d.pbc_cell.shape == (3, 3)

    def test_cryst1_orthorhombic(self):
        from xyzrender.parsers import parse_pdb

        d = parse_pdb(_WATER_PDB_CRYST)
        assert d.pbc_cell is not None
        # Cubic cell -> diagonal matrix with all 10 A
        diag = np.diag(d.pbc_cell)
        np.testing.assert_allclose(diag, [10.0, 10.0, 10.0], atol=1e-2)


# ---------------------------------------------------------------------------
# io loaders -- graph structure
# ---------------------------------------------------------------------------


class TestLoaders:
    def test_load_sdf_nodes(self):
        from xyzrender.readers import load_molecule

        g, _ = load_molecule(_CAFFEINE_SDF)
        assert g.number_of_nodes() == _CAFFEINE_ATOMS

    def test_load_sdf_edges(self):
        from xyzrender.readers import load_molecule

        g, _ = load_molecule(_CAFFEINE_SDF)
        assert g.number_of_edges() > 0

    def test_load_sdf_rebuild(self):
        from xyzrender.readers import load_molecule

        g, _ = load_molecule(_CAFFEINE_SDF, rebuild=True)
        assert g.number_of_nodes() == _CAFFEINE_ATOMS

    def test_load_mol2_nodes(self):
        from xyzrender.readers import load_molecule

        g, _ = load_molecule(_WATER_MOL2)
        assert g.number_of_nodes() == 3

    def test_load_pdb_no_crystal(self):
        from xyzrender.readers import load_molecule

        g, crystal = load_molecule(_WATER_PDB)
        assert g.number_of_nodes() == 3
        assert crystal is None

    def test_load_pdb_with_crystal(self):
        from xyzrender.readers import load_molecule
        from xyzrender.types import CellData

        g, crystal = load_molecule(_WATER_PDB_CRYST)
        assert g.number_of_nodes() == 3
        assert isinstance(crystal, CellData)
        assert crystal.lattice.shape == (3, 3)
        # Cubic 10 A cell
        np.testing.assert_allclose(np.diag(crystal.lattice), [10.0, 10.0, 10.0], atol=1e-2)

    def test_node_attributes(self):
        from xyzrender.readers import load_molecule

        g, _ = load_molecule(_CAFFEINE_SDF)
        for i in g.nodes:
            assert "symbol" in g.nodes[i]
            assert "position" in g.nodes[i]
            assert len(g.nodes[i]["position"]) == 3

    def test_edge_attributes(self):
        from xyzrender.readers import load_molecule

        g, _ = load_molecule(_CAFFEINE_SDF)
        for _, _, d in g.edges(data=True):
            assert "bond_order" in d
            assert d["bond_order"] > 0


# ---------------------------------------------------------------------------
# parse_smiles
# ---------------------------------------------------------------------------

pytest.importorskip("rdkit", reason="rdkit required for SMILES tests")


class TestParseSmiles:
    def test_atom_count(self):
        from xyzrender.parsers import parse_smiles

        d = parse_smiles("O")  # water
        assert len(d.atoms) == 3  # O + 2H

    def test_element_symbols(self):
        from xyzrender.parsers import parse_smiles

        d = parse_smiles("O")
        symbols = {sym for sym, _ in d.atoms}
        assert symbols == {"O", "H"}

    def test_bonds_present(self):
        from xyzrender.parsers import parse_smiles

        d = parse_smiles("O")
        assert d.bonds is not None
        assert len(d.bonds) == 2

    def test_3d_coords(self):
        from xyzrender.parsers import parse_smiles

        d = parse_smiles("O")
        for _, pos in d.atoms:
            assert len(pos) == 3
            assert all(isinstance(v, float) for v in pos)

    def test_no_pbc_cell(self):
        from xyzrender.parsers import parse_smiles

        d = parse_smiles("O")
        assert d.pbc_cell is None

    def test_benzene_heavy_atoms(self):
        from xyzrender.parsers import parse_smiles

        d = parse_smiles("c1ccccc1")  # benzene, no explicit H in SMILES
        # AddHs gives 12 atoms total (6C + 6H)
        assert len(d.atoms) == 12


# ---------------------------------------------------------------------------
# parse_cif / load_molecule(.cif) -- uses examples/structures/caffeine_cif.cif
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning:ase")
class TestParseCif:
    def test_atoms_present(self):
        from xyzrender.parsers import parse_cif

        d = parse_cif(_CIF_FILE)
        assert len(d.atoms) > 0

    def test_has_pbc_cell(self):
        from xyzrender.parsers import parse_cif

        d = parse_cif(_CIF_FILE)
        assert d.pbc_cell is not None
        assert d.pbc_cell.shape == (3, 3)

    def test_load_molecule_cif_graph(self):
        from xyzrender.readers import load_molecule
        from xyzrender.types import CellData

        g, crystal = load_molecule(_CIF_FILE)
        assert g.number_of_nodes() > 0
        assert isinstance(crystal, CellData)
        assert crystal.lattice.shape == (3, 3)


# ---------------------------------------------------------------------------
# QM input file parsers (inputs.py)
# ---------------------------------------------------------------------------


class TestQmInputs:
    """Test generic coordinate / charge-mult parsing for QM input files."""

    @pytest.mark.parametrize("ext", ["com", "inp", "nw", "psi4", "qcin"])
    def test_parse_qm_input_caffeine(self, ext):
        from xyzrender.inputs import parse_qm_input

        path = _INPUTS / f"caffeine.{ext}"
        if not path.exists():
            pytest.skip(f"Missing test file: {path}")
        atoms, charge, mult = parse_qm_input(str(path))
        assert len(atoms) == _CAFFEINE_ATOMS
        assert charge == 0
        assert mult == 1

    @pytest.mark.parametrize("ext", ["com", "inp", "nw"])
    def test_load_molecule_qm_input(self, ext):
        from xyzrender.readers import load_molecule

        path = _INPUTS / f"caffeine.{ext}"
        if not path.exists():
            pytest.skip(f"Missing test file: {path}")
        g, crystal = load_molecule(path)
        assert g.number_of_nodes() == _CAFFEINE_ATOMS
        assert crystal is None

    def test_get_coords_no_match(self, tmp_path):
        from xyzrender.inputs import parse_qm_input

        path = tmp_path / "empty.inp"
        path.write_text("! some route line\n%maxcore 500\nend\n")
        with pytest.raises(ValueError, match="No coordinate block found"):
            parse_qm_input(str(path))

    def test_charge_mult_orca(self, tmp_path):
        from xyzrender.inputs import parse_qm_input

        path = tmp_path / "test.inp"
        path.write_text("! HF\n* xyz -2 3\nH 0 0 0\nH 0 0 1\n*\n")
        atoms, charge, mult = parse_qm_input(str(path))
        assert len(atoms) == 2
        assert charge == -2
        assert mult == 3

    def test_charge_mult_qchem(self, tmp_path):
        from xyzrender.inputs import parse_qm_input

        path = tmp_path / "test.qcin"
        path.write_text("$molecule\n1 2\nH 0 0 0\nH 0 0 1\n$end\n")
        _, charge, mult = parse_qm_input(str(path))
        assert charge == 1
        assert mult == 2

    def test_charge_mult_gaussian(self, tmp_path):
        from xyzrender.inputs import parse_qm_input

        path = tmp_path / "test.com"
        path.write_text("#p HF/STO-3G\n\nTitle\n\n-1 4\nH 0 0 0\nH 0 0 1\n\n")
        _, charge, mult = parse_qm_input(str(path))
        assert charge == -1
        assert mult == 4

    def test_charge_mult_nwchem(self, tmp_path):
        from xyzrender.inputs import parse_qm_input

        path = tmp_path / "test.nw"
        path.write_text("charge 2\ngeometry\nH 0 0 0\nH 0 0 1\nend\n")
        _, charge, _ = parse_qm_input(str(path))
        assert charge == 2

    def test_charge_mult_psi4(self, tmp_path):
        from xyzrender.inputs import parse_qm_input

        path = tmp_path / "test.psi4"
        path.write_text("molecule {\n3 2\nH 0 0 0\nH 0 0 1\n}\n")
        _, charge, mult = parse_qm_input(str(path))
        assert charge == 3
        assert mult == 2


def _fake_cclib(monkeypatch, n_frames: int, parser_name: str) -> None:
    """Install a fake cclib in sys.modules.  Avoids importing real cclib
    (~0.4s) just to monkeypatch it."""
    import sys
    import types

    data = types.SimpleNamespace(
        atomnos=np.array([1, 1, 1]),
        atomcoords=np.zeros((n_frames, 3, 3)),
    )
    parser = type(parser_name, (), {"parse": lambda self: data})()
    fake_io = types.SimpleNamespace(ccopen=lambda path, loglevel=None: parser)
    monkeypatch.setitem(sys.modules, "cclib", types.SimpleNamespace(io=fake_io, __version__="mock"))
    monkeypatch.setitem(sys.modules, "cclib.io", fake_io)


def test_trajectory_diagnostic_logs_frame_count(caplog, monkeypatch):
    """_load_qm_frames logs cclib's frame count + parser, so users can tell
    upstream cclib issues from xyzrender issues."""
    import logging

    from xyzrender.readers import load_trajectory_frames

    _fake_cclib(monkeypatch, n_frames=5, parser_name="Gaussian")
    with caplog.at_level(logging.INFO, logger="xyzrender.readers"):
        load_trajectory_frames("dummy.log")
    msgs = [r.getMessage() for r in caplog.records]
    assert any("parsed 5 frame(s)" in m for m in msgs)
    assert any("parser=Gaussian" in m for m in msgs)


def test_trajectory_diagnostic_warns_on_single_frame(caplog, monkeypatch):
    """When cclib returns ≤1 frame for what's expected to be a trajectory,
    warn so users check cclib version or file format."""
    import logging

    from xyzrender.readers import load_trajectory_frames

    _fake_cclib(monkeypatch, n_frames=1, parser_name="ORCA")
    with caplog.at_level(logging.INFO, logger="xyzrender.readers"):
        load_trajectory_frames("dummy.out")
    msgs = [r.getMessage() for r in caplog.records]
    assert any("may not contain the expected multistep data" in m for m in msgs)


class TestQeSniff:
    """Test QE vs Q-Chem disambiguation for .in files."""

    def test_qe_detected(self):
        from xyzrender.inputs import is_qe_input

        assert is_qe_input(str(_STRUCTURES / "NV63.in")) is True

    def test_non_qe_not_detected(self, tmp_path):
        from xyzrender.inputs import is_qe_input

        path = tmp_path / "qchem.in"
        path.write_text("$molecule\n0 1\nH 0 0 0\n$end\n")
        assert is_qe_input(str(path)) is False

    def test_qe_loads_as_crystal_in_load_molecule(self):
        from xyzrender.readers import load_molecule
        from xyzrender.types import CellData

        g, crystal = load_molecule(_STRUCTURES / "NV63.in")
        assert g.number_of_nodes() == 63
        assert isinstance(crystal, CellData)


class TestPoscar:
    """Test VASP POSCAR parser."""

    def test_parse_poscar(self):
        from xyzrender.inputs import parse_poscar

        atoms, lattice = parse_poscar(str(_STRUCTURES / "NV63.vasp"))
        assert len(atoms) == 63
        assert lattice.shape == (3, 3)
        np.testing.assert_allclose(np.diag(lattice), [7.14, 7.14, 7.14], atol=0.01)

    def test_load_crystal_vasp(self):
        from xyzrender.crystal import load_crystal
        from xyzrender.types import CellData

        g, crystal = load_crystal(_STRUCTURES / "NV63.vasp", "vasp")
        assert g.number_of_nodes() == 63
        assert isinstance(crystal, CellData)
        np.testing.assert_allclose(np.diag(crystal.lattice), [7.14, 7.14, 7.14], atol=0.01)


class TestQeInput:
    """Test QE pw.in parser."""

    def test_parse_qe_input(self):
        from xyzrender.inputs import parse_qe_input

        atoms, lattice, charge = parse_qe_input(str(_STRUCTURES / "NV63.in"))
        assert len(atoms) == 63
        assert charge == -1
        assert lattice.shape == (3, 3)
        np.testing.assert_allclose(np.diag(lattice), [7.14, 7.14, 7.14], atol=0.01)

    def test_load_crystal_qe(self):
        from xyzrender.crystal import load_crystal
        from xyzrender.types import CellData

        g, crystal = load_crystal(_STRUCTURES / "NV63.in", "qe")
        assert g.number_of_nodes() == 63
        assert isinstance(crystal, CellData)
        np.testing.assert_allclose(np.diag(crystal.lattice), [7.14, 7.14, 7.14], atol=0.01)


class TestExtxyzChargeMult:
    """Test charge/mult parsing from extXYZ comment lines."""

    def test_charge_from_comment(self):
        from xyzrender.readers import _parse_extxyz_charge_mult

        c, m = _parse_extxyz_charge_mult("charge=2 mult=3")
        assert c == 2
        assert m == 3

    def test_aliases(self):
        from xyzrender.readers import _parse_extxyz_charge_mult

        c, m = _parse_extxyz_charge_mult("crg=-1 m=2")
        assert c == -1
        assert m == 2

    def test_no_charge_mult(self):
        from xyzrender.readers import _parse_extxyz_charge_mult

        c, m = _parse_extxyz_charge_mult('Lattice="1 0 0 0 1 0 0 0 1"')
        assert c is None
        assert m is None


class TestPeriodicInputsConsistent:
    """All periodic caffeine inputs should parse to 24 atoms with the same box."""

    @pytest.mark.parametrize(
        ("fmt", "path"),
        [
            ("vasp", _INPUTS / "caffeine.vasp"),
            ("qe", _INPUTS / "caffeine_qe.in"),
            ("siesta", _INPUTS / "caffeine.fdf"),
            ("siesta_bohr", _INPUTS / "caffeine_bohr.fdf"),
            ("abinit", _INPUTS / "caffeine.abi"),
        ],
    )
    def test_consistent_atom_count(self, fmt, path):
        from xyzrender.readers import load_molecule

        if not path.exists():
            pytest.skip(f"Missing: {path}")
        g, crystal = load_molecule(path)
        assert g.number_of_nodes() == _CAFFEINE_ATOMS
        assert crystal is not None
        assert crystal.lattice.shape == (3, 3)

    @pytest.mark.parametrize(
        ("fmt", "path"),
        [
            ("vasp", _INPUTS / "caffeine.vasp"),
            ("qe", _INPUTS / "caffeine_qe.in"),
            ("siesta", _INPUTS / "caffeine.fdf"),
            ("siesta_bohr", _INPUTS / "caffeine_bohr.fdf"),
            ("abinit", _INPUTS / "caffeine.abi"),
        ],
    )
    def test_consistent_lattice(self, fmt, path):
        from xyzrender.readers import load_molecule

        if not path.exists():
            pytest.skip(f"Missing: {path}")
        _, crystal = load_molecule(path)
        assert crystal is not None
        np.testing.assert_allclose(np.diag(crystal.lattice), [16.89, 16.47, 15.44], atol=0.1)
