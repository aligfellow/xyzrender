"""Tests for xyzrender.formats and io loader functions.

All fixtures use checked-in example files — no rdkit or ase generation needed.
"""

from __future__ import annotations

import importlib.util
import re
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
_SHELXL_FILE = _STRUCTURES / "roy.res"
_CORONENE_CJSON = _STRUCTURES / "coronene_colors.cjson"
_SILICON_CJSON = _STRUCTURES / "silicon.cjson"

_CAFFEINE_ATOMS = 24  # C8N4O2 + 10H
_CORONENE_ATOMS = 36  # C24H12


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
# parse_cjson
# ---------------------------------------------------------------------------


# 90 degrees about z — used to check that a camera rotation is actually applied
_ROT_Z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _minimal_cjson(**extra):
    """Two-carbon CJSON skeleton; *extra* keys are merged into the root."""
    root = {
        "chemicalJson": 1,
        "atoms": {
            "coords": {"3d": [0.0, 0.0, 0.0, 1.5, 0.0, 0.0]},
            "elements": {"number": [6, 6]},
        },
        "bonds": {"connections": {"index": [0, 1]}, "order": [1]},
    }
    root.update(extra)
    return root


def _model_view(rotation: np.ndarray) -> list:
    """A 4x4 modelView matrix carrying *rotation*, as CJSON nested rows."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    return matrix.tolist()


def _dummy_atom_cjson():
    """CJSON with a dummy atom (O, dummy, N) sitting between two real ones."""
    root = _minimal_cjson()
    root["atoms"]["elements"]["number"] = [8, 0, 7]
    root["atoms"]["coords"]["3d"] = [0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 1.2, 0.0, 0.0]
    return root


class TestParseCjson:
    def test_atom_count(self):
        from xyzrender.parsers import parse_cjson

        d = parse_cjson(_CORONENE_CJSON)
        assert len(d.atoms) == _CORONENE_ATOMS

    def test_element_symbols(self):
        from xyzrender.parsers import parse_cjson

        d = parse_cjson(_CORONENE_CJSON)
        assert {sym for sym, _ in d.atoms} == {"C", "H"}

    def test_bonds_from_file(self):
        from xyzrender.parsers import parse_cjson

        d = parse_cjson(_CORONENE_CJSON)
        assert d.bonds is not None
        assert len(d.bonds) == 42
        # Coronene is aromatic — the file carries alternating Kekulé orders
        assert {order for _, _, order in d.bonds} == {1.0, 2.0}

    def test_name_and_no_cell(self):
        from xyzrender.parsers import parse_cjson

        d = parse_cjson(_CORONENE_CJSON)
        assert d.name == "coronene"
        assert d.pbc_cell is None

    def test_colors(self):
        from xyzrender.parsers import parse_cjson

        d = parse_cjson(_CORONENE_CJSON)
        assert d.colors is not None
        assert len(d.colors) == _CORONENE_ATOMS
        assert d.colors[0] == "#30123b"  # first entry is 48, 18, 59
        assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in d.colors)

    def test_camera(self):
        from xyzrender.parsers import parse_cjson

        d = parse_cjson(_CORONENE_CJSON)
        assert d.camera is not None
        # This file was saved from the default (unrotated) Avogadro view
        np.testing.assert_allclose(d.camera.rotation, np.eye(3), atol=1e-6)
        assert d.camera.perspective is True

    def test_camera_rotation_orthonormalised(self):
        from xyzrender.parsers import parse_cjson_dict

        # A uniform scale on top of the rotation must be divided out
        d = parse_cjson_dict(_minimal_cjson(properties={"modelView": _model_view(_ROT_Z90 * 2.0)}))
        assert d.camera is not None
        np.testing.assert_allclose(d.camera.rotation, _ROT_Z90, atol=1e-9)

    def test_camera_accepts_flat_matrix(self):
        from xyzrender.parsers import parse_cjson_dict

        # Nested rows are what Avogadro writes; a flat 16 is also valid JSON-wise
        flat = [v for row in _model_view(_ROT_Z90) for v in row]
        d = parse_cjson_dict(_minimal_cjson(properties={"modelView": flat}))
        assert d.camera is not None
        np.testing.assert_allclose(d.camera.rotation, _ROT_Z90, atol=1e-9)

    def test_legacy_space_separated_keys(self):
        from xyzrender.parsers import parse_cjson_dict

        # cjsonformat.cpp accepts both spellings for these three keys
        root = _minimal_cjson()
        root["chemical json"] = root.pop("chemicalJson")
        root["atoms"]["coords"] = {"3d fractional": [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]}
        root["unit cell"] = {"cellVectors": [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0]}
        d = parse_cjson_dict(root)
        assert d.pbc_cell is not None
        np.testing.assert_allclose(d.atoms[1][1], [2.0, 2.0, 2.0], atol=1e-9)

    def test_mirrored_camera_rejected(self):
        from xyzrender.parsers import parse_cjson_dict

        # A reflection would render the enantiomer — it must be dropped
        model_view = np.eye(4)
        model_view[0, 0] = -1.0
        d = parse_cjson_dict(_minimal_cjson(properties={"modelView": model_view.tolist()}))
        assert d.camera is None

    def test_orthographic_projection_flag(self):
        from xyzrender.parsers import parse_cjson_dict

        d = parse_cjson_dict(
            _minimal_cjson(properties={"modelView": np.eye(4).tolist(), "projection": np.eye(4).tolist()})
        )
        assert d.camera is not None
        assert d.camera.perspective is False

    def test_no_camera_without_modelview(self):
        from xyzrender.parsers import parse_cjson_dict

        assert parse_cjson_dict(_minimal_cjson()).camera is None

    def test_total_charge(self):
        from xyzrender.parsers import parse_cjson_dict

        d = parse_cjson_dict(_minimal_cjson(properties={"totalCharge": -2}))
        assert d.charge == -2

    def test_bond_order_defaults_to_single(self):
        from xyzrender.parsers import parse_cjson_dict

        root = _minimal_cjson()
        del root["bonds"]["order"]
        assert parse_cjson_dict(root).bonds == [(0, 1, 1.0)]

    def test_self_and_out_of_range_bonds_dropped(self):
        from xyzrender.parsers import parse_cjson_dict

        root = _minimal_cjson()
        root["bonds"] = {"connections": {"index": [0, 0, 0, 7, 0, 1]}, "order": [1, 1, 2]}
        assert parse_cjson_dict(root).bonds == [(0, 1, 2.0)]

    def test_mismatched_colors_ignored(self):
        from xyzrender.parsers import parse_cjson_dict

        root = _minimal_cjson()
        root["atoms"]["colors"] = [255, 0, 0]  # 1 colour for 2 atoms
        assert parse_cjson_dict(root).colors is None

    def test_unit_cell_from_vectors(self):
        from xyzrender.parsers import parse_cjson_dict

        root = _minimal_cjson(unitCell={"cellVectors": [4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0]})
        d = parse_cjson_dict(root)
        assert d.pbc_cell is not None
        np.testing.assert_allclose(np.diag(d.pbc_cell), [4.0, 5.0, 6.0])

    def test_cell_vectors_preferred_over_parameters(self):
        from xyzrender.parsers import parse_cjson_dict

        root = _minimal_cjson(
            unitCell={
                "cellVectors": [4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0],
                "a": 1.0,
                "b": 1.0,
                "c": 1.0,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0,
            }
        )
        d = parse_cjson_dict(root)
        assert d.pbc_cell is not None
        np.testing.assert_allclose(np.diag(d.pbc_cell), [4.0, 5.0, 6.0])

    def test_fractional_coords_converted(self):
        from xyzrender.parsers import parse_cjson

        # silicon.cjson has only 3dFractional plus a/b/c/angles
        d = parse_cjson(_SILICON_CJSON)
        assert d.pbc_cell is not None
        assert len(d.atoms) == 2
        # (0.25, 0.25, 0.25) fractional = a quarter of (a + b + c)
        expected = 0.25 * d.pbc_cell.sum(axis=0)
        np.testing.assert_allclose(d.atoms[0][1], expected, atol=1e-9)

    def test_fractional_without_cell_raises(self):
        from xyzrender.parsers import parse_cjson_dict

        root = _minimal_cjson()
        root["atoms"]["coords"] = {"3dFractional": [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]}
        with pytest.raises(ValueError, match="no unit cell"):
            parse_cjson_dict(root)

    def test_coordinate_sets_frame(self):
        from xyzrender.parsers import parse_cjson_dict

        root = _minimal_cjson()
        root["atoms"]["coords"]["3dSets"] = [
            [0.0, 0.0, 0.0, 1.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.5, 0.0],
        ]
        assert parse_cjson_dict(root, frame=1).atoms[1][1] == (0.0, 2.5, 0.0)

    def test_out_of_range_frame_raises(self):
        from xyzrender.parsers import parse_cjson_dict

        with pytest.raises(IndexError):
            parse_cjson_dict(_minimal_cjson(), frame=3)

    def test_missing_version_key_raises(self):
        from xyzrender.parsers import parse_cjson_dict

        root = _minimal_cjson()
        del root["chemicalJson"]
        with pytest.raises(ValueError, match="Not a CJSON file"):
            parse_cjson_dict(root)

    def test_dummy_atoms_ignored(self):
        from xyzrender.parsers import parse_cjson_dict

        root = _dummy_atom_cjson()
        root["bonds"] = {"connections": {"index": [0, 2]}, "order": [3]}
        d = parse_cjson_dict(root)
        assert [sym for sym, _ in d.atoms] == ["O", "N"]
        assert d.atoms[1][1] == (1.2, 0.0, 0.0)

    def test_dummy_atoms_remap_bond_indices(self):
        from xyzrender.parsers import parse_cjson_dict

        root = _dummy_atom_cjson()
        # Bond 0-2 survives (renumbered 0-1); bonds touching the dummy do not
        root["bonds"] = {"connections": {"index": [0, 1, 0, 2, 1, 2]}, "order": [1, 3, 1]}
        assert parse_cjson_dict(root).bonds == [(0, 1, 3.0)]

    def test_dummy_atoms_drop_matching_colors(self):
        from xyzrender.parsers import parse_cjson_dict

        root = _dummy_atom_cjson()
        root["atoms"]["colors"] = [255, 0, 0, 0, 255, 0, 0, 0, 255]
        assert parse_cjson_dict(root).colors == ["#ff0000", "#0000ff"]

    def test_dispatcher(self):
        from xyzrender.parsers import parse

        assert len(parse(_CORONENE_CJSON).atoms) == _CORONENE_ATOMS


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

    def test_load_cjson_graph(self):
        from xyzrender.readers import load_molecule

        g, crystal = load_molecule(_CORONENE_CJSON)
        assert g.number_of_nodes() == _CORONENE_ATOMS
        assert g.number_of_edges() == 42
        assert crystal is None

    def test_load_cjson_stamps_colors(self):
        from xyzrender.readers import load_molecule

        g, _ = load_molecule(_CORONENE_CJSON)
        assert g.nodes[0]["file_color"] == "#30123b"
        assert all("file_color" in g.nodes[i] for i in g.nodes)

    def test_load_cjson_crystal(self):
        from xyzrender.readers import load_molecule
        from xyzrender.types import CellData

        g, crystal = load_molecule(_SILICON_CJSON)
        assert g.number_of_nodes() == 2
        assert isinstance(crystal, CellData)
        np.testing.assert_allclose(crystal.lattice[0], [3.84, 0.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# CJSON camera -> orientation
# ---------------------------------------------------------------------------


class TestCjsonCamera:
    def test_camera_marks_molecule_oriented(self):
        import xyzrender as xr

        mol = xr.load(_CORONENE_CJSON)
        assert mol.oriented is True

    def test_camera_can_be_disabled(self):
        import xyzrender as xr

        mol = xr.load(_CORONENE_CJSON, camera=False)
        assert mol.oriented is False

    def test_camera_rotation_applied(self, tmp_path):
        import json

        import xyzrender as xr

        raw = json.loads(_CORONENE_CJSON.read_text())
        # 90 degrees about z on top of the file's identity modelView
        raw["properties"]["modelView"] = _model_view(_ROT_Z90)
        rotated_file = tmp_path / "rotated.cjson"
        rotated_file.write_text(json.dumps(raw))

        plain = xr.load(_CORONENE_CJSON)
        turned = xr.load(rotated_file)

        plain_pos = np.array([plain.graph.nodes[i]["position"] for i in plain.graph.nodes])
        turned_pos = np.array([turned.graph.nodes[i]["position"] for i in turned.graph.nodes])
        np.testing.assert_allclose(turned_pos, plain_pos @ _ROT_Z90.T, atol=1e-9)

    def test_camera_centres_molecule(self):
        import xyzrender as xr

        mol = xr.load(_CORONENE_CJSON)
        pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes])
        np.testing.assert_allclose(pos.mean(axis=0), np.zeros(3), atol=1e-9)

    def test_camera_rotates_lattice_with_atoms(self, tmp_path):
        import json

        import xyzrender as xr

        raw = json.loads(_SILICON_CJSON.read_text())
        raw["properties"] = {"modelView": _model_view(_ROT_Z90)}
        turned_file = tmp_path / "turned.cjson"
        turned_file.write_text(json.dumps(raw))

        plain = xr.load(_SILICON_CJSON)
        turned = xr.load(turned_file)
        assert plain.cell_data is not None
        assert turned.cell_data is not None
        np.testing.assert_allclose(turned.cell_data.lattice, plain.cell_data.lattice @ _ROT_Z90.T, atol=1e-9)

    def test_file_colors_reach_the_svg(self):
        import xyzrender as xr

        mol = xr.load(_CORONENE_CJSON)
        svg = str(xr.render(mol, gradient=False, fog=False, output=None))
        assert "#30123b" in svg.lower()

    def test_mol_color_overrides_file_colors(self):
        import xyzrender as xr

        mol = xr.load(_CORONENE_CJSON)
        svg = str(xr.render(mol, mol_color="blue", gradient=False, fog=False, output=None))
        assert "#30123b" not in svg.lower()

    def test_cmap_overrides_file_colors(self):
        import xyzrender as xr

        mol = xr.load(_CORONENE_CJSON)
        values = {i: float(i) for i in range(1, mol.graph.number_of_nodes() + 1)}
        svg = str(xr.render(mol, cmap=values, cmap_palette="viridis", gradient=False, fog=False, output=None))
        assert "#30123b" not in svg.lower()
        assert "#440154" in svg.lower()


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
# parse_shelxl / load_molecule(.res/.ins) -- uses examples/structures/roy.res
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("shelxfile") is None,
    reason="shelxfile required for SHELXL tests",
)
class TestParseShelxl:
    # roy.res is the ROY polymorph: UNIT C48 H36 N12 O8 S4 → 108 atoms per cell
    def test_atoms_present(self):
        from collections import Counter

        from xyzrender.parsers import parse_shelxl

        d = parse_shelxl(_SHELXL_FILE)
        assert len(d.atoms) == 108
        assert Counter(sym for sym, _ in d.atoms) == {"C": 48, "H": 36, "N": 12, "O": 8, "S": 4}

    def test_has_pbc_cell(self):
        from xyzrender.parsers import parse_shelxl

        d = parse_shelxl(_SHELXL_FILE)
        assert d.pbc_cell is not None
        assert d.pbc_cell.shape == (3, 3)
        # CELL 3.9453 18.685 16.3948 → row norms recover the a/b/c lengths
        lengths = np.linalg.norm(d.pbc_cell, axis=1)
        np.testing.assert_allclose(lengths, [3.9453, 18.685, 16.3948], atol=1e-3)

    def test_load_molecule_shelxl_graph(self):
        from xyzrender.readers import load_molecule
        from xyzrender.types import CellData

        g, crystal = load_molecule(_SHELXL_FILE)
        assert g.number_of_nodes() == 108
        assert isinstance(crystal, CellData)
        assert crystal.lattice.shape == (3, 3)
        # lattice must be published on the graph so the interactive viewer and
        # rotation code keep the cell box aligned with the atoms.
        assert "lattice" in g.graph
        assert "lattice_origin" in g.graph


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


_UNIFORM_XYZ = """2
frame 0
H 0.0 0.0 0.0
H 0.0 0.0 0.74
2
frame 1
H 0.0 0.0 0.0
H 0.0 0.0 0.80
"""

_NON_UNIFORM_XYZ = """2
frame 0
H 0.0 0.0 0.0
H 0.0 0.0 0.74
3
frame 1
H 0.0 0.0 0.0
H 0.0 0.0 0.80
H 0.0 0.0 1.60
"""


def test_load_trajectory_frames_uniform_atom_counts(tmp_path):
    """A fixed-atom-count trajectory loads normally with no error."""
    from xyzrender.readers import load_trajectory_frames

    p = tmp_path / "uniform.xyz"
    p.write_text(_UNIFORM_XYZ)

    frames = load_trajectory_frames(str(p))
    assert len(frames) == 2
    assert [len(f["symbols"]) for f in frames] == [2, 2]
    assert frames[1]["positions"][1] == [0.0, 0.0, 0.80]


def test_load_trajectory_frames_non_uniform_raises(tmp_path):
    """A trajectory whose frames have differing atom counts is rejected by
    default with a clear, early ValueError."""
    from xyzrender.readers import load_trajectory_frames

    p = tmp_path / "non_uniform.xyz"
    p.write_text(_NON_UNIFORM_XYZ)

    message = (
        "Trajectory has non-uniform atom counts: first frame has 2 atoms, "
        "frame 1 has 3. If this is intentional, use --var-atoms"
    )
    with pytest.raises(ValueError, match=re.escape(message)) as exc_info:
        load_trajectory_frames(str(p))
    assert str(exc_info.value) == message


def test_load_trajectory_frames_non_uniform_allowed_with_flag(tmp_path):
    """var_atoms=True opts into loading a non-uniform trajectory
    instead of raising."""
    from xyzrender.readers import load_trajectory_frames

    p = tmp_path / "non_uniform.xyz"
    p.write_text(_NON_UNIFORM_XYZ)

    frames = load_trajectory_frames(str(p), var_atoms=True)
    assert [len(f["symbols"]) for f in frames] == [2, 3]


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
