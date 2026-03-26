"""Tests for xyzrender.formats and io loader functions.

Mol/SDF fixtures are generated with rdkit from caffeine SMILES (real output,
not hand-crafted strings).  MOL2 uses the checked-in example file.  PDB
fixtures are generated with ase.  All tests are skipped when the required
library is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit", reason="rdkit required for mol/sdf fixture generation")
pytest.importorskip("ase", reason="ase required for pdb fixture generation")

from pathlib import Path

import ase
import ase.io
import numpy as np
from ase.build import molecule
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter

# ---------------------------------------------------------------------------
# Caffeine: rdkit-generated mol/sdf fixtures
# ---------------------------------------------------------------------------

_CAFFEINE_SMI = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
_CAFFEINE_HEAVY = 14  # C8N4O2
_CAFFEINE_ATOMS = 24  # with explicit H


def _caffeine_3d() -> Chem.Mol:
    mol = Chem.MolFromSmiles(_CAFFEINE_SMI)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)  # type: ignore[attr-defined]
    AllChem.MMFFOptimizeMolecule(mol)  # type: ignore[attr-defined]
    return mol


@pytest.fixture(scope="module")
def caffeine_mol(tmp_path_factory):
    path = tmp_path_factory.mktemp("mol") / "caffeine.mol"
    Chem.MolToMolFile(_caffeine_3d(), str(path))
    return path


@pytest.fixture(scope="module")
def caffeine_sdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("sdf") / "caffeine.sdf"
    w = SDWriter(str(path))
    w.write(_caffeine_3d())
    w.close()
    return path


_WATER_ATOMS = 3  # O + 2H with explicit H


@pytest.fixture(scope="module")
def multi_sdf(tmp_path_factory):
    """Two-record SDF: caffeine (frame 0) then water (frame 1)."""
    water = Chem.MolFromSmiles("O")
    water = Chem.AddHs(water)
    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = 0
    AllChem.EmbedMolecule(water, params)  # type: ignore[attr-defined]

    path = tmp_path_factory.mktemp("multi_sdf") / "multi.sdf"
    w = SDWriter(str(path))
    w.write(_caffeine_3d())
    w.write(water)
    w.close()
    return path


# ---------------------------------------------------------------------------
# Water MOL2 — checked-in example file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def water_mol2():
    return Path(__file__).parent.parent / "examples" / "structures" / "water_mol2.mol2"


# ---------------------------------------------------------------------------
# Water PDB — ase-generated
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def water() -> ase.Atoms:
    return molecule("H2O")


@pytest.fixture(scope="module")
def water_pdb(water, tmp_path_factory):
    path = tmp_path_factory.mktemp("pdb") / "water.pdb"
    ase.io.write(str(path), water, format="proteindatabank")
    return path


@pytest.fixture(scope="module")
def water_pdb_cryst(water, tmp_path_factory):
    """Water molecule in a PDB with a CRYST1 unit cell."""
    from ase.cell import Cell

    w = water.copy()
    w.set_cell(Cell.fromcellpar([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]))
    w.set_pbc(True)
    path = tmp_path_factory.mktemp("pdb_cryst") / "water_cryst.pdb"
    ase.io.write(str(path), w, format="proteindatabank")
    return path


# ---------------------------------------------------------------------------
# parse_mol
# ---------------------------------------------------------------------------


class TestParseMol:
    def test_atom_count(self, caffeine_mol):
        from xyzrender.parsers import parse_mol

        d = parse_mol(caffeine_mol)
        assert len(d.atoms) == _CAFFEINE_ATOMS

    def test_element_symbols(self, caffeine_mol):
        from xyzrender.parsers import parse_mol

        d = parse_mol(caffeine_mol)
        symbols = {sym for sym, _ in d.atoms}
        assert {"C", "N", "O", "H"} == symbols

    def test_bonds_present(self, caffeine_mol):
        from xyzrender.parsers import parse_mol

        d = parse_mol(caffeine_mol)
        assert d.bonds is not None
        assert len(d.bonds) > 0

    def test_no_pbc_cell(self, caffeine_mol):
        from xyzrender.parsers import parse_mol

        d = parse_mol(caffeine_mol)
        assert d.pbc_cell is None


# ---------------------------------------------------------------------------
# parse_sdf
# ---------------------------------------------------------------------------


class TestParseSdf:
    def test_atom_count(self, caffeine_sdf):
        from xyzrender.parsers import parse_sdf

        d = parse_sdf(caffeine_sdf, frame=0)
        assert len(d.atoms) == _CAFFEINE_ATOMS

    def test_bonds_present(self, caffeine_sdf):
        from xyzrender.parsers import parse_sdf

        d = parse_sdf(caffeine_sdf, frame=0)
        assert d.bonds is not None
        assert len(d.bonds) > 0

    def test_frame_out_of_range(self, caffeine_sdf):
        from xyzrender.parsers import parse_sdf

        with pytest.raises(IndexError):
            parse_sdf(caffeine_sdf, frame=99)

    def test_multi_frame0(self, multi_sdf):
        from xyzrender.parsers import parse_sdf

        d = parse_sdf(multi_sdf, frame=0)
        assert len(d.atoms) == _CAFFEINE_ATOMS

    def test_multi_frame1(self, multi_sdf):
        from xyzrender.parsers import parse_sdf

        d = parse_sdf(multi_sdf, frame=1)
        assert len(d.atoms) == _WATER_ATOMS

    def test_multi_frame_selects_different_molecules(self, multi_sdf):
        from xyzrender.parsers import parse_sdf

        d0 = parse_sdf(multi_sdf, frame=0)
        d1 = parse_sdf(multi_sdf, frame=1)
        assert len(d0.atoms) != len(d1.atoms)


# ---------------------------------------------------------------------------
# parse_mol2
# ---------------------------------------------------------------------------


class TestParseMol2:
    def test_atom_count(self, water_mol2):
        from xyzrender.parsers import parse_mol2

        d = parse_mol2(water_mol2)
        assert len(d.atoms) == 3

    def test_element_symbols(self, water_mol2):
        from xyzrender.parsers import parse_mol2

        d = parse_mol2(water_mol2)
        symbols = {sym for sym, _ in d.atoms}
        assert symbols == {"O", "H"}

    def test_bonds_present(self, water_mol2):
        from xyzrender.parsers import parse_mol2

        d = parse_mol2(water_mol2)
        assert d.bonds is not None
        assert len(d.bonds) == 2


# ---------------------------------------------------------------------------
# parse_pdb
# ---------------------------------------------------------------------------


class TestParsePdb:
    def test_atom_count(self, water_pdb):
        from xyzrender.parsers import parse_pdb

        d = parse_pdb(water_pdb)
        assert len(d.atoms) == 3

    def test_element_symbols(self, water_pdb):
        from xyzrender.parsers import parse_pdb

        d = parse_pdb(water_pdb)
        symbols = {sym for sym, _ in d.atoms}
        assert symbols == {"O", "H"}

    def test_no_cryst1(self, water_pdb):
        from xyzrender.parsers import parse_pdb

        d = parse_pdb(water_pdb)
        assert d.pbc_cell is None

    def test_cryst1_parsed(self, water_pdb_cryst):
        from xyzrender.parsers import parse_pdb

        d = parse_pdb(water_pdb_cryst)
        assert d.pbc_cell is not None
        assert d.pbc_cell.shape == (3, 3)

    def test_cryst1_orthorhombic(self, water_pdb_cryst):
        from xyzrender.parsers import parse_pdb

        d = parse_pdb(water_pdb_cryst)
        assert d.pbc_cell is not None
        # Cubic cell → diagonal matrix with all 10 Å
        diag = np.diag(d.pbc_cell)
        np.testing.assert_allclose(diag, [10.0, 10.0, 10.0], atol=1e-2)


# ---------------------------------------------------------------------------
# io loaders — graph structure
# ---------------------------------------------------------------------------


class TestLoaders:
    def test_load_mol_nodes(self, caffeine_mol):
        from xyzrender.readers import load_molecule

        g, _ = load_molecule(caffeine_mol)
        assert g.number_of_nodes() == _CAFFEINE_ATOMS

    def test_load_mol_edges(self, caffeine_mol):
        from xyzrender.readers import load_molecule

        g, _ = load_molecule(caffeine_mol)
        assert g.number_of_edges() > 0

    def test_load_mol_rebuild(self, caffeine_mol):
        from xyzrender.readers import load_molecule

        g, _ = load_molecule(caffeine_mol, rebuild=True)
        assert g.number_of_nodes() == _CAFFEINE_ATOMS

    def test_load_sdf_nodes(self, caffeine_sdf):
        from xyzrender.readers import load_molecule

        g, _ = load_molecule(caffeine_sdf)
        assert g.number_of_nodes() == _CAFFEINE_ATOMS

    def test_load_mol2_nodes(self, water_mol2):
        from xyzrender.readers import load_molecule

        g, _ = load_molecule(water_mol2)
        assert g.number_of_nodes() == 3

    def test_load_pdb_no_crystal(self, water_pdb):
        from xyzrender.readers import load_molecule

        g, crystal = load_molecule(water_pdb)
        assert g.number_of_nodes() == 3
        assert crystal is None

    def test_load_pdb_with_crystal(self, water_pdb_cryst):
        from xyzrender.readers import load_molecule
        from xyzrender.types import CellData

        g, crystal = load_molecule(water_pdb_cryst)
        assert g.number_of_nodes() == 3
        assert isinstance(crystal, CellData)
        assert crystal.lattice.shape == (3, 3)
        # Cubic 10 Å cell — diagonal should be ~10 after round-trip through CellData
        np.testing.assert_allclose(np.diag(crystal.lattice), [10.0, 10.0, 10.0], atol=1e-2)

    def test_node_attributes(self, caffeine_mol):
        from xyzrender.readers import load_molecule

        g, _ = load_molecule(caffeine_mol)
        for i in g.nodes:
            assert "symbol" in g.nodes[i]
            assert "position" in g.nodes[i]
            assert len(g.nodes[i]["position"]) == 3

    def test_edge_attributes(self, caffeine_mol):
        from xyzrender.readers import load_molecule

        g, _ = load_molecule(caffeine_mol)
        for _, _, d in g.edges(data=True):
            assert "bond_order" in d
            assert d["bond_order"] > 0


# ---------------------------------------------------------------------------
# parse_smiles
# ---------------------------------------------------------------------------


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
# parse_cif / load_molecule(.cif) — uses examples/structures/caffeine_cif.cif
# ---------------------------------------------------------------------------

_CIF_FILE = Path(__file__).parent.parent / "examples" / "structures" / "caffeine_cif.cif"


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

_STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"
_INPUTS = Path(__file__).parent.parent / "examples" / "inputs"
_CAFFEINE_ATOMS = 24


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
