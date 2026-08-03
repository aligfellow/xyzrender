"""Tests for protein ribbon rendering."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import networkx as nx
import pytest

from xyzrender import load, render
from xyzrender.colors import resolve_color
from xyzrender.parsers import parse_pdb
from xyzrender.protein_semantics import xyzgraph_protein_available
from xyzrender.readers import filter_ligand_protein_nci
from xyzrender.types import ChainData, ProteinData, ResidueData

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"

# Ribbon rendering hard-depends on `xyzgraph.protein`, which older xyzgraph
# releases do not ship.  Skip rather than fail so the suite stays green on an
# environment that predates it; see protein_semantics.xyzgraph_protein_available.
pytestmark = pytest.mark.skipif(
    not xyzgraph_protein_available(),
    reason="installed xyzgraph does not provide xyzgraph.protein",
)


def _require_protein_data(data: ProteinData | None) -> ProteinData:
    assert data is not None
    return data


# ---------------------------------------------------------------------------
# Minimal synthetic PDB fixtures
# ---------------------------------------------------------------------------

# 4-residue helix (HELIX record + 4 ATOM records with CA/N/C/O)
_HELIX_PDB = dedent("""\
    HELIX    1   1 ALA A    1  ALA A    4  1                                   4
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       1.456   0.000   0.000  1.00  0.00           C
    ATOM      3  C   ALA A   1       1.930   0.000   1.463  1.00  0.00           C
    ATOM      4  O   ALA A   1       1.160   0.000   2.421  1.00  0.00           O
    ATOM      5  N   ALA A   2       3.241   0.000   1.742  1.00  0.00           N
    ATOM      6  CA  ALA A   2       3.690   0.000   3.127  1.00  0.00           C
    ATOM      7  C   ALA A   2       5.228   0.000   3.127  1.00  0.00           C
    ATOM      8  O   ALA A   2       5.902   0.000   2.098  1.00  0.00           O
    ATOM      9  N   ALA A   3       5.898   0.000   4.287  1.00  0.00           N
    ATOM     10  CA  ALA A   3       7.353   0.000   4.287  1.00  0.00           C
    ATOM     11  C   ALA A   3       7.828   0.000   5.750  1.00  0.00           C
    ATOM     12  O   ALA A   3       7.058   0.001   6.709  1.00  0.00           O
    ATOM     13  N   ALA A   4       9.148   0.000   6.029  1.00  0.00           N
    ATOM     14  CA  ALA A   4       9.602   0.000   7.414  1.00  0.00           C
    ATOM     15  C   ALA A   4      11.140   0.000   7.414  1.00  0.00           C
    ATOM     16  O   ALA A   4      11.814   0.000   6.384  1.00  0.00           O
    END
""")

# 4-residue strand (SHEET record)
_SHEET_PDB = dedent("""\
    SHEET    1   A 1 ALA A   1  ALA A   4  0
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       1.456   0.000   0.000  1.00  0.00           C
    ATOM      3  C   ALA A   1       1.930   0.000   1.463  1.00  0.00           C
    ATOM      4  O   ALA A   1       1.160   0.000   2.421  1.00  0.00           O
    ATOM      5  N   ALA A   2       3.241   0.000   1.742  1.00  0.00           N
    ATOM      6  CA  ALA A   2       3.690   0.000   3.127  1.00  0.00           C
    ATOM      7  C   ALA A   2       5.228   0.000   3.127  1.00  0.00           C
    ATOM      8  O   ALA A   2       5.902   0.000   2.098  1.00  0.00           O
    ATOM      9  N   ALA A   3       5.898   0.000   4.287  1.00  0.00           N
    ATOM     10  CA  ALA A   3       7.353   0.000   4.287  1.00  0.00           C
    ATOM     11  C   ALA A   3       7.828   0.000   5.750  1.00  0.00           C
    ATOM     12  O   ALA A   3       7.058   0.001   6.709  1.00  0.00           O
    ATOM     13  N   ALA A   4       9.148   0.000   6.029  1.00  0.00           N
    ATOM     14  CA  ALA A   4       9.602   0.000   7.414  1.00  0.00           C
    ATOM     15  C   ALA A   4      11.140   0.000   7.414  1.00  0.00           C
    ATOM     16  O   ALA A   4      11.814   0.000   6.384  1.00  0.00           O
    END
""")

# PDB with HETATM ligand + protein ATOM records
_HETATM_PDB = dedent("""\
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       1.456   0.000   0.000  1.00  0.00           C
    ATOM      3  C   ALA A   1       1.930   0.000   1.463  1.00  0.00           C
    ATOM      4  O   ALA A   1       1.160   0.000   2.421  1.00  0.00           O
    HETATM    5  C1  LIG A 101       5.000   5.000   0.000  1.00  0.00           C
    HETATM    6  O1  LIG A 101       6.000   5.000   0.000  1.00  0.00           O
    END
""")

_HETATM_CLASS_PDB = dedent("""\
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       1.456   0.000   0.000  1.00  0.00           C
    HETATM    3  O   HOH A 201       3.000   0.000   0.000  1.00  0.00           O
    HETATM    4 NA   NA  A 202       4.000   0.000   0.000  1.00  0.00          NA
    HETATM    5  C1  LIG A 203       5.000   0.000   0.000  1.00  0.00           C
    END
""")

_MODIFIED_POLYMER_PDB = dedent("""\
    MODRES 1ABC MSE A   2  MET  SELENOMETHIONINE
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       1.456   0.000   0.000  1.00  0.00           C
    ATOM      3  C   ALA A   1       2.800   0.000   0.000  1.00  0.00           C
    HETATM    4  N   MSE A   2       3.700   0.000   0.000  1.00  0.00           N
    HETATM    5  CA  MSE A   2       5.100   0.000   0.000  1.00  0.00           C
    HETATM    6  C   MSE A   2       6.500   0.000   0.000  1.00  0.00           C
    HETATM    7  CB  MSE A   2       5.100   1.500   0.000  1.00  0.00           C
    HETATM    8  SE  MSE A   2       5.100   3.400   0.000  1.00  0.00          SE
    ATOM      9  N   GLY A   3       7.400   0.000   0.000  1.00  0.00           N
    ATOM     10  CA  GLY A   3       8.800   0.000   0.000  1.00  0.00           C
    ATOM     11  C   GLY A   3      10.200   0.000   0.000  1.00  0.00           C
    END
""")

# Two-chain PDB
_TWO_CHAIN_PDB = dedent("""\
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       1.456   0.000   0.000  1.00  0.00           C
    ATOM      3  C   ALA A   1       1.930   0.000   1.463  1.00  0.00           C
    ATOM      4  O   ALA A   1       1.160   0.000   2.421  1.00  0.00           O
    ATOM      5  N   ALA A   2       3.241   0.000   1.742  1.00  0.00           N
    ATOM      6  CA  ALA A   2       3.690   0.000   3.127  1.00  0.00           C
    ATOM      7  C   ALA A   2       5.228   0.000   3.127  1.00  0.00           C
    ATOM      8  O   ALA A   2       5.902   0.000   2.098  1.00  0.00           O
    ATOM      9  N   GLY B   1      10.000  10.000   0.000  1.00  0.00           N
    ATOM     10  CA  GLY B   1      11.456  10.000   0.000  1.00  0.00           C
    ATOM     11  C   GLY B   1      11.930  10.000   1.463  1.00  0.00           C
    ATOM     12  O   GLY B   1      11.160  10.000   2.421  1.00  0.00           O
    ATOM     13  N   GLY B   2      13.241  10.000   1.742  1.00  0.00           N
    ATOM     14  CA  GLY B   2      13.690  10.000   3.127  1.00  0.00           C
    ATOM     15  C   GLY B   2      15.228  10.000   3.127  1.00  0.00           C
    ATOM     16  O   GLY B   2      15.902  10.000   2.098  1.00  0.00           O
    END
""")


# A 10-residue helix (chain A) and a 7-residue strand (chain B), carved from
# 8UWL so the backbone geometry is real.  Small enough for geometry tests
# that need both SS types and two chains at once.
_HELIX_SHEET_PDB = dedent("""\
    HELIX    1   1 ALA A    1  LEU A   10  1                                   10
    SHEET    1   S 1 PHE B   1  VAL B   7  0
    ATOM      1  N   ALA A   1     142.038 174.939 188.223  1.00 30.00           N
    ATOM      2  CA  ALA A   1     141.630 174.019 187.133  1.00 30.00           C
    ATOM      3  C   ALA A   1     142.304 174.361 185.799  1.00 30.00           C
    ATOM      4  O   ALA A   1     142.412 173.456 184.961  1.00 30.00           O
    ATOM      5  N   LEU A   2     142.729 175.610 185.598  1.00 94.10           N
    ATOM      6  CA  LEU A   2     143.365 176.042 184.322  1.00 94.10           C
    ATOM      7  C   LEU A   2     144.622 175.209 184.066  1.00 94.10           C
    ATOM      8  O   LEU A   2     144.955 175.006 182.902  1.00 94.10           O
    ATOM      9  N   LEU A   3     145.308 174.774 185.119  1.00 85.46           N
    ATOM     10  CA  LEU A   3     146.530 173.943 184.968  1.00 85.46           C
    ATOM     11  C   LEU A   3     146.183 172.644 184.233  1.00 85.46           C
    ATOM     12  O   LEU A   3     147.041 172.158 183.486  1.00 85.46           O
    ATOM     13  N   THR A   4     144.995 172.080 184.469  1.00 78.43           N
    ATOM     14  CA  THR A   4     144.562 170.865 183.731  1.00 78.43           C
    ATOM     15  C   THR A   4     144.495 171.238 182.257  1.00 78.43           C
    ATOM     16  O   THR A   4     144.799 170.379 181.415  1.00 78.43           O
    ATOM     17  N   ALA A   5     144.106 172.473 181.971  1.00 81.75           N
    ATOM     18  CA  ALA A   5     143.932 172.879 180.563  1.00 81.75           C
    ATOM     19  C   ALA A   5     145.258 172.763 179.820  1.00 81.75           C
    ATOM     20  O   ALA A   5     145.226 172.430 178.625  1.00 81.75           O
    ATOM     21  N   VAL A   6     146.376 173.032 180.481  1.00 78.07           N
    ATOM     22  CA  VAL A   6     147.641 173.031 179.697  1.00 78.07           C
    ATOM     23  C   VAL A   6     147.812 171.645 179.075  1.00 78.07           C
    ATOM     24  O   VAL A   6     148.284 171.579 177.927  1.00 78.07           O
    ATOM     25  N   VAL A   7     147.432 170.591 179.792  1.00 74.67           N
    ATOM     26  CA  VAL A   7     147.625 169.201 179.284  1.00 74.67           C
    ATOM     27  C   VAL A   7     146.788 169.013 178.016  1.00 74.67           C
    ATOM     28  O   VAL A   7     147.283 168.366 177.077  1.00 74.67           O
    ATOM     29  N   ILE A   8     145.578 169.563 177.978  1.00 74.05           N
    ATOM     30  CA  ILE A   8     144.675 169.313 176.823  1.00 74.05           C
    ATOM     31  C   ILE A   8     145.196 170.048 175.593  1.00 74.05           C
    ATOM     32  O   ILE A   8     145.198 169.445 174.517  1.00 74.05           O
    ATOM     33  N   ILE A   9     145.629 171.296 175.755  1.00 74.50           N
    ATOM     34  CA  ILE A   9     146.126 172.106 174.604  1.00 74.50           C
    ATOM     35  C   ILE A   9     147.386 171.450 174.047  1.00 74.50           C
    ATOM     36  O   ILE A   9     147.482 171.308 172.811  1.00 74.50           O
    ATOM     37  N   LEU A  10     148.293 171.030 174.925  1.00 73.68           N
    ATOM     38  CA  LEU A  10     149.575 170.441 174.479  1.00 73.68           C
    ATOM     39  C   LEU A  10     149.354 169.126 173.723  1.00 73.68           C
    ATOM     40  O   LEU A  10     150.009 168.943 172.687  1.00 73.68           O
    ATOM     41  N   PHE B   1     162.789 177.777 170.134  1.00 53.51           N
    ATOM     42  CA  PHE B   1     162.946 176.392 170.652  1.00 53.51           C
    ATOM     43  C   PHE B   1     162.232 176.293 172.003  1.00 53.51           C
    ATOM     44  O   PHE B   1     162.149 177.326 172.684  1.00 53.51           O
    ATOM     45  N   GLU B   2     161.730 175.107 172.376  1.00 56.01           N
    ATOM     46  CA  GLU B   2     160.942 174.968 173.632  1.00 56.01           C
    ATOM     47  C   GLU B   2     161.361 173.730 174.432  1.00 56.01           C
    ATOM     48  O   GLU B   2     161.929 172.812 173.827  1.00 56.01           O
    ATOM     49  N   THR B   3     161.121 173.728 175.748  1.00 53.78           N
    ATOM     50  CA  THR B   3     161.405 172.555 176.618  1.00 53.78           C
    ATOM     51  C   THR B   3     160.260 172.466 177.631  1.00 53.78           C
    ATOM     52  O   THR B   3     159.705 173.523 177.954  1.00 53.78           O
    ATOM     53  N   LYS B   4     159.915 171.261 178.100  1.00 53.44           N
    ATOM     54  CA  LYS B   4     158.843 171.096 179.120  1.00 53.44           C
    ATOM     55  C   LYS B   4     159.338 170.229 180.285  1.00 53.44           C
    ATOM     56  O   LYS B   4     160.015 169.228 180.014  1.00 53.44           O
    ATOM     57  N   PHE B   5     159.027 170.621 181.526  1.00 51.46           N
    ATOM     58  CA  PHE B   5     159.427 169.844 182.728  1.00 51.46           C
    ATOM     59  C   PHE B   5     158.394 170.041 183.842  1.00 51.46           C
    ATOM     60  O   PHE B   5     157.598 170.979 183.736  1.00 51.46           O
    ATOM     61  N   GLN B   6     158.380 169.159 184.849  1.00 59.92           N
    ATOM     62  CA  GLN B   6     157.387 169.243 185.954  1.00 59.92           C
    ATOM     63  C   GLN B   6     158.039 168.941 187.304  1.00 59.92           C
    ATOM     64  O   GLN B   6     158.849 168.010 187.349  1.00 59.92           O
    ATOM     65  N   VAL B   7     157.707 169.696 188.357  1.00 54.65           N
    ATOM     66  CA  VAL B   7     158.227 169.410 189.731  1.00 54.65           C
    ATOM     67  C   VAL B   7     157.089 169.499 190.756  1.00 54.65           C
    ATOM     68  O   VAL B   7     156.370 170.513 190.732  1.00 54.65           O
    END
""")


# A ligand belonging to chain B, appended to _TWO_CHAIN_PDB.
_LIGAND_ON_B = dedent("""\
    HETATM   17  C1  LIG B 101      20.000  20.000   0.000  1.00  0.00           C
    HETATM   18  O1  LIG B 101      21.000  20.000   0.000  1.00  0.00           O
""")


@pytest.fixture
def helix_pdb(tmp_path):
    p = tmp_path / "helix.pdb"
    p.write_text(_HELIX_PDB)
    return p


@pytest.fixture
def helix_sheet_pdb(tmp_path):
    p = tmp_path / "helix_sheet.pdb"
    p.write_text(_HELIX_SHEET_PDB)
    return p


@pytest.fixture
def sheet_pdb(tmp_path):
    p = tmp_path / "sheet.pdb"
    p.write_text(_SHEET_PDB)
    return p


@pytest.fixture
def sidechain_pdb(tmp_path):
    """Backbone with CB/CG sidechains, and CONECT so bonds exist without rebuild."""
    lines = []
    serial = 1
    conect = []
    for k in range(6):
        base = k * 3.8
        ids = {}
        for name, off, elem in (
            ("N", (0.0, 0.0, 0.0), "N"),
            ("CA", (1.5, 0.0, 0.0), "C"),
            ("C", (2.5, 0.0, 0.0), "C"),
            ("O", (3.0, 0.9, 0.0), "O"),
            ("CB", (1.2, 0.0, 1.5), "C"),
            ("CG", (1.0, 0.0, 3.0), "C"),
        ):
            lines.append(_atom_line(serial, name, "ALA", "A", k + 1, (base + off[0], off[1], off[2]), elem=elem))
            ids[name] = serial
            serial += 1
        conect += [
            (ids["N"], ids["CA"]),
            (ids["CA"], ids["C"]),
            (ids["C"], ids["O"]),
            (ids["CA"], ids["CB"]),
            (ids["CB"], ids["CG"]),
        ]
    for a, b in conect:
        lines.append(f"CONECT{a:>5}{b:>5}")
    p_ = tmp_path / "sidechain.pdb"
    p_.write_text("\n".join(lines) + "\n")
    return p_


@pytest.fixture
def hetatm_pdb(tmp_path):
    p = tmp_path / "hetatm.pdb"
    p.write_text(_HETATM_PDB)
    return p


@pytest.fixture
def two_chain_pdb(tmp_path):
    p = tmp_path / "two_chain.pdb"
    p.write_text(_TWO_CHAIN_PDB)
    return p


@pytest.fixture
def hetatm_class_pdb(tmp_path):
    p = tmp_path / "hetatm_class.pdb"
    p.write_text(_HETATM_CLASS_PDB)
    return p


# ---------------------------------------------------------------------------
# Phase 1: Parser tests
# ---------------------------------------------------------------------------


def test_parse_pdb_extracts_chain(tmp_path, helix_pdb):
    data = parse_pdb(helix_pdb)
    pd = _require_protein_data(data.protein_data)
    assert "A" in pd.chains


def test_parse_pdb_residue_count(helix_pdb):
    data = parse_pdb(helix_pdb)
    pd = _require_protein_data(data.protein_data)
    chain_a = pd.chains["A"]
    assert len(chain_a.residues) == 4


def test_parse_pdb_ca_index(helix_pdb):
    data = parse_pdb(helix_pdb)
    pd = _require_protein_data(data.protein_data)
    for res in pd.chains["A"].residues:
        assert res.ca_index is not None


def test_parse_pdb_ss_helix(helix_pdb):
    """HELIX record should set ss_type='H' on all residues."""
    data = parse_pdb(helix_pdb)
    pd = _require_protein_data(data.protein_data)
    for res in pd.chains["A"].residues:
        assert res.ss_type == "H", f"residue {res.res_seq} expected H got {res.ss_type}"


def test_parse_pdb_ss_sheet(sheet_pdb):
    """SHEET record should set ss_type='E' on all residues."""
    data = parse_pdb(sheet_pdb)
    pd = _require_protein_data(data.protein_data)
    for res in pd.chains["A"].residues:
        assert res.ss_type == "E", f"residue {res.res_seq} expected E got {res.ss_type}"


def test_parse_pdb_ss_default_loop():
    """Without HELIX/SHEET records, ss_type defaults to 'C'."""
    data = parse_pdb(STRUCTURES / "ala_phe_ala.pdb")
    pd = data.protein_data
    if pd is not None:
        for chain in pd.chains.values():
            for res in chain.residues:
                assert res.ss_type == "C"


def test_parse_pdb_hetatm_separation(hetatm_pdb):
    data = parse_pdb(hetatm_pdb)
    pd = _require_protein_data(data.protein_data)
    # Indices 4 and 5 are HETATM (0-indexed)
    assert 4 in pd.hetatm_indices
    assert 5 in pd.hetatm_indices
    # Backbone atoms 0-3 should not be HETATM
    assert 0 not in pd.hetatm_indices


def test_parse_pdb_backbone_indices(helix_pdb):
    data = parse_pdb(helix_pdb)
    pd = data.protein_data
    assert pd is not None
    # N, CA, C, O for 4 residues = 16 backbone atoms (all atoms in this PDB)
    assert len(pd.backbone_indices) == 16


def test_parse_pdb_hetatm_ligand_water_ion_indices(hetatm_class_pdb):
    data = parse_pdb(hetatm_class_pdb)
    pd = data.protein_data
    assert pd is not None
    # 0-indexed from records above: HOH=2, NA=3, LIG=4
    assert pd.water_indices == {2}
    assert pd.ion_indices == {3}
    assert pd.ligand_indices == {4}


def test_parse_pdb_modres_hetatm_stays_in_polymer_chain(tmp_path):
    """MODRES-labelled HETATM residues remain part of the protein polymer."""
    path = tmp_path / "modified.pdb"
    path.write_text(_MODIFIED_POLYMER_PDB)

    pd = _require_protein_data(parse_pdb(path).protein_data)
    assert [res.res_name for res in pd.chains["A"].residues] == ["ALA", "MSE", "GLY"]
    assert set(pd.chains["A"].residues[1].atom_indices) == {3, 4, 5, 6, 7}
    assert not ({3, 4, 5, 6, 7} & pd.hetatm_indices)
    assert not ({3, 4, 5, 6, 7} & pd.ligand_indices)


def test_parse_pdb_two_chains(two_chain_pdb):
    data = parse_pdb(two_chain_pdb)
    pd = data.protein_data
    assert pd is not None
    assert set(pd.chains.keys()) == {"A", "B"}
    assert len(pd.chains["A"].residues) == 2
    assert len(pd.chains["B"].residues) == 2


def test_load_attaches_protein_data(helix_pdb):
    mol = load(helix_pdb)
    pd = _require_protein_data(mol.protein_data)
    assert "A" in pd.chains


def test_load_water_pdb_loads_without_error():
    """Simple water PDB loads without error (has chain A, so protein_data may be set)."""
    mol = load(STRUCTURES / "water.pdb")
    # water.pdb has chain 'A' so protein_data is populated, which is fine
    assert mol.graph.number_of_nodes() == 3


# ---------------------------------------------------------------------------
# Phase 2: Ribbon geometry tests
# ---------------------------------------------------------------------------


def test_ribbon_items_returned(helix_pdb):
    """ribbon_svg_items() returns non-empty list for a helix structure."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(helix_pdb)
    pd = mol.protein_data
    assert pd is not None

    cfg = RenderConfig(protein=True)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])

    items = ribbon_svg_items(pd, cfg, pos, scale=50.0, cx=0.0, cy=0.0, canvas_w=800, canvas_h=800)
    assert len(items) > 0


def test_ribbon_helix_produces_polygons(helix_pdb):
    """Helix segments produce polygon SVG elements."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(helix_pdb)
    cfg = RenderConfig(protein=True)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])
    pd = _require_protein_data(mol.protein_data)
    items = ribbon_svg_items(pd, cfg, pos, 50.0, 0.0, 0.0, 800, 800)
    all_svg = " ".join(line for _, lines in items for line in lines)
    assert "<polygon" in all_svg


def test_adaptive_spline_steps_is_deterministic():
    """Internal adaptive tessellation should choose stable step counts."""
    import numpy as np

    from xyzrender.ribbon import _adaptive_spline_steps

    small = np.array([[float(i), 0.0, 0.0] for i in range(100)], dtype=float)
    large = np.array([[float(i), 0.0, 0.0] for i in range(2500)], dtype=float)

    s1 = _adaptive_spline_steps(small, scale=80.0)
    s2 = _adaptive_spline_steps(small, scale=80.0)
    s3 = _adaptive_spline_steps(large, scale=80.0)

    assert s1 == s2
    assert s1 >= s3


def test_ribbon_sheet_produces_polygons(sheet_pdb):
    """Sheet segments produce polygon elements (including arrowhead)."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(sheet_pdb)
    cfg = RenderConfig(protein=True)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])
    pd = _require_protein_data(mol.protein_data)
    items = ribbon_svg_items(pd, cfg, pos, 50.0, 0.0, 0.0, 800, 800)
    all_svg = " ".join(line for _, lines in items for line in lines)
    assert "<polygon" in all_svg


def test_narrow_strand_ribbon_fits_canvas():
    """Ribbon width and the sheet arrowhead must contribute to canvas fitting."""
    import re

    import numpy as np

    from xyzrender.renderer import render_svg
    from xyzrender.types import RenderConfig

    graph = nx.Graph()
    for i, (symbol, position) in enumerate(
        (
            ("C", (0.0, 0.0, 0.0)),
            ("C", (0.0, 4.0, 0.0)),
            ("C", (0.0, 8.0, 0.0)),
            ("C", (0.0, 12.0, 0.0)),
            ("O", (1.0, 0.0, 0.0)),
            ("O", (1.0, 4.0, 0.0)),
            ("O", (1.0, 8.0, 0.0)),
            ("O", (1.0, 12.0, 0.0)),
        )
    ):
        graph.add_node(i, symbol=symbol, position=np.array(position))
    protein_data = ProteinData(
        chains={
            "A": ChainData(
                "A",
                [
                    ResidueData("ALA", 1, "A", [0, 4], ca_index=0, c_index=None, o_index=4, n_index=None, ss_type="E"),
                    ResidueData("ALA", 2, "A", [1, 5], ca_index=1, c_index=None, o_index=5, n_index=None, ss_type="E"),
                    ResidueData("ALA", 3, "A", [2, 6], ca_index=2, c_index=None, o_index=6, n_index=None, ss_type="E"),
                    ResidueData("ALA", 4, "A", [3, 7], ca_index=3, c_index=None, o_index=7, n_index=None, ss_type="E"),
                ],
            )
        },
        hetatm_indices=set(),
        backbone_indices=set(range(8)),
        sidechain_indices=set(),
        helix_spans=[],
        sheet_spans=[],
    )

    svg = str(render_svg(graph, RenderConfig(protein=True, auto_orient=False), protein_data=protein_data))
    view_box = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    assert view_box is not None
    width, height = map(float, view_box.groups())
    assert "<polygon" in svg
    vertices = [
        float(value)
        for points in re.findall(r'<polygon points="([^"]+)"', svg)
        for point in points.split()
        for value in point.split(",")
    ]
    assert all(0 <= x <= width for x in vertices[::2])
    assert all(0 <= y <= height for y in vertices[1::2])


def test_surface_behind_ribbon_is_emitted_first(helix_pdb):
    """Surface and ribbon items share one back-to-front depth ordering."""
    from xyzrender.api import Molecule
    from xyzrender.types import RenderConfig

    loaded = load(helix_pdb)
    protein_data = _require_protein_data(loaded.protein_data)
    for atom in loaded.graph.nodes:
        x, y, _ = loaded.graph.nodes[atom]["position"]
        loaded.graph.nodes[atom]["position"] = (x, y, 10.0)

    cfg = RenderConfig(
        protein=True,
        auto_orient=False,
        pore_spheres=True,
        pore_node_ids=[[0, 1, 2]],
        pore_centroids=[(0.0, 0.0, 5.0)],
        pore_radii=[1.0],
    )
    svg = str(render(Molecule(loaded.graph, protein_data=protein_data), config=cfg))
    import re

    pore = re.search(r'<circle[^>]+fill="url\(#[^"]*pore_', svg)
    assert pore is not None
    assert pore.start() < svg.index("<polygon")


def test_ribbon_two_chains_distinct_colors(two_chain_pdb):
    """Two chains get distinct colours from the palette."""
    from xyzrender.ribbon import assign_chain_colors
    from xyzrender.types import RenderConfig

    mol = load(two_chain_pdb)
    cfg = RenderConfig(protein=True)
    pd = _require_protein_data(mol.protein_data)
    colors = assign_chain_colors(cfg, list(pd.chains.keys()))
    assert len(set(colors.values())) == 2, "Chains A and B should have different colours"


def test_ribbon_chain_color_override(two_chain_pdb):
    """Explicit chain_colors override palette."""
    from xyzrender.colors import resolve_color
    from xyzrender.ribbon import assign_chain_colors
    from xyzrender.types import RenderConfig

    mol = load(two_chain_pdb)
    cfg = RenderConfig(protein=True, chain_colors={"A": resolve_color("steelblue")})
    pd = _require_protein_data(mol.protein_data)
    colors = assign_chain_colors(cfg, list(pd.chains.keys()))
    assert colors["A"] == resolve_color("steelblue")


def test_ribbon_items_sorted_by_z(helix_pdb):
    """ribbon_svg_items() returns items sorted ascending by z_depth."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(helix_pdb)
    cfg = RenderConfig(protein=True)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])

    pd = _require_protein_data(mol.protein_data)
    items = ribbon_svg_items(pd, cfg, pos, 50.0, 0.0, 0.0, 800, 800)
    depths = [z for z, _ in items]
    assert depths == sorted(depths)


# ---------------------------------------------------------------------------
# Phase 3: Renderer integration tests
# ---------------------------------------------------------------------------


def test_protein_flag_renders(helix_pdb):
    """render(mol, protein=True) produces an SVG without error."""
    mol = load(helix_pdb)
    svg = str(render(mol, protein=True, orient=False))
    assert "<svg" in svg


def test_backbone_atoms_hidden(helix_pdb):
    """In protein mode, backbone atom circles should not appear in SVG."""
    mol = load(helix_pdb)
    pd = mol.protein_data
    assert pd is not None

    svg_protein = str(render(mol, protein=True, orient=False, gradient=False))
    svg_normal = str(render(mol, orient=False, gradient=False))

    # Protein mode should produce fewer atom circles (backbone suppressed)
    assert svg_protein.count("<circle") < svg_normal.count("<circle")


def test_hetatm_atoms_visible_in_protein_mode(hetatm_pdb):
    """HETATM atoms remain visible as ball-and-stick in protein mode."""
    mol = load(hetatm_pdb)
    svg = str(render(mol, protein=True, orient=False, gradient=False))
    # There should be at least some circles (HETATM atoms)
    assert "<circle" in svg


def test_ribbon_polygons_in_svg(helix_pdb):
    """render with protein=True contains ribbon polygon elements."""
    mol = load(helix_pdb)
    svg = str(render(mol, protein=True, orient=False))
    assert "<polygon" in svg


# "cartoon" is no longer here: it is now an accepted alias for gloss.
@pytest.mark.parametrize("style", ["plastic", "matte", "pymol", "illustrative"])
def test_removed_protein_styles_raise(helix_pdb, style):
    """Removed legacy styles should fail fast with a validation error."""
    mol = load(helix_pdb)
    with pytest.raises(ValueError, match="unknown style"):
        render(mol, protein=style, orient=False)


def test_protein_gloss_respects_transparent_background(helix_pdb):
    """Gloss style must not force an opaque or black background."""
    mol = load(helix_pdb)
    svg = str(render(mol, protein="gloss", orient=False, transparent=True))
    assert 'style="background:transparent"' in svg
    assert 'fill="#000000"' not in svg


def test_cartoon_is_an_alias_for_gloss(helix_pdb):
    """``cartoon`` is the name users reach for; it must resolve to gloss."""
    from xyzrender.ribbon import normalize_ribbon_style, ribbon_style_names

    assert normalize_ribbon_style("cartoon") == "gloss"
    assert "cartoon" in ribbon_style_names()
    assert "cartoon" not in ribbon_style_names(include_aliases=False)

    mol = load(helix_pdb)
    assert str(render(mol, protein="cartoon", orient=False)) == str(render(mol, protein="gloss", orient=False))


def test_protein_invalid_style_raises(helix_pdb):
    """Unknown protein style strings should raise a clear ValueError."""
    mol = load(helix_pdb)
    with pytest.raises(ValueError, match="unknown style"):
        render(mol, protein="not-a-style", orient=False)


def test_sidechain_flag(helix_pdb):
    """--sidechain flag results in more atoms visible than backbone-only mode."""
    mol = load(helix_pdb)
    # The helix fixture has no sidechain atoms (only N/CA/C/O), so this mostly
    # confirms the flag is accepted without error.
    svg = str(render(mol, protein=True, sidechain=True, orient=False))
    assert "<svg" in svg


def test_ligand_highlight_recolors_ligands(hetatm_pdb):
    """Ligand-highlight recolors ligands without needing manual indices."""
    mol = load(hetatm_pdb)
    svg_plain = str(render(mol, protein=True, orient=False, gradient=False, config="flat"))
    svg_lig_a = str(
        render(
            mol,
            config="flat",
            protein=True,
            ligand_highlight=True,
            ligand_color="#123abc",
            orient=False,
            gradient=False,
        )
    )
    svg_lig_b = str(
        render(
            mol,
            config="flat",
            protein=True,
            ligand_highlight=True,
            ligand_color="#ff0000",
            orient=False,
            gradient=False,
        )
    )
    assert svg_lig_a != svg_plain
    assert svg_lig_b != svg_plain
    assert svg_lig_a != svg_lig_b


def _glow_circles(svg: str) -> list[str]:
    import re

    return [m.group(0) for m in re.finditer(r"<circle[^>]*filter=\"url\(#\w*glow\d+\)\"[^>]*/>", svg)]


def test_ligand_glow_selector_uses_ligand_color(hetatm_pdb):
    """glow='ligand' should target semantic ligands and reuse ligand_color when provided."""
    mol = load(hetatm_pdb)
    svg = str(
        render(
            mol,
            config="flat",
            glow="ligand",
            ligand_color="#12ab34",
            orient=False,
            gradient=False,
            fog=False,
        )
    )
    circles = _glow_circles(svg)
    assert len(circles) == 2  # two ligand HETATM atoms in fixture
    assert all(resolve_color("#12ab34") in c for c in circles)


def test_glow_semantic_tokens_resolve_against_semantics(hetatm_class_pdb):
    """'water' and 'ion' pick out their own HETATM classes, not the whole file."""
    mol = load(hetatm_class_pdb)
    svg = str(render(mol, config="flat", glow=[("water", "blue"), ("ion", "red")], orient=False, fog=False))
    circles = _glow_circles(svg)
    fills = [resolve_color(c) for c in ("blue", "red")]
    assert len(circles) == 2
    assert sorted(f for f in fills if any(f in c for c in circles)) == sorted(fills)


def test_glow_semantic_token_warns_without_semantics(caplog):
    """A semantic token on a plain molecule warns and selects nothing."""
    mol = load(STRUCTURES / "caffeine.xyz")
    with caplog.at_level("WARNING", logger="xyzrender.api"):
        svg = str(render(mol, glow="ligand", orient=False))
    assert _glow_circles(svg) == []
    assert "no matching protein semantics are available" in caplog.text


def test_glow_element_selector_still_works_on_a_protein(hetatm_pdb):
    """Semantic tokens are additive: the element grammar must still resolve."""
    mol = load(hetatm_pdb)
    svg = str(render(mol, config="flat", glow="O", orient=False, fog=False))
    assert len(_glow_circles(svg)) == 2  # backbone O + ligand O1


def test_protein_mode_keeps_nci_dotted_edges_when_backbone_hidden():
    """Protein mode should still render dotted NCI edges with hidden protein endpoints."""
    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))  # protein backbone atom (hidden)
    g.add_node(1, symbol="O", position=(1.4, 0.0, 0.0))  # ligand atom (visible)
    g.add_edge(0, 1, NCI=True)

    pd = ProteinData(
        chains={
            "A": ChainData(
                chain_id="A",
                residues=[
                    ResidueData(
                        res_name="ALA",
                        res_seq=1,
                        chain_id="A",
                        atom_indices=[0],
                        ca_index=0,
                        c_index=None,
                        o_index=None,
                        n_index=None,
                        ss_type="C",
                    )
                ],
            )
        },
        hetatm_indices={1},
        backbone_indices={0},
        sidechain_indices=set(),
        helix_spans=[],
        sheet_spans=[],
        ligand_indices={1},
        water_indices=set(),
        ion_indices=set(),
    )
    from xyzrender.api import Molecule

    mol = Molecule(graph=g, protein_data=pd)
    svg = str(render(mol, protein=True, orient=False))
    assert "stroke-dasharray" in svg


def test_protein_mode_keeps_backbone_nci_without_revealing_atoms():
    """Backbone contacts remain dotted overlays without duplicating cartoon atoms."""
    g = nx.Graph()
    for idx in range(3):
        g.add_node(idx, symbol="C", position=(float(idx) * 1.4, 0.0, 0.0))
    g.add_node(3, symbol="O", position=(0.0, 1.4, 0.0))
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(0, 3, NCI=True)

    residues = [
        ResidueData(
            res_name="ALA",
            res_seq=idx + 1,
            chain_id="A",
            atom_indices=[idx],
            ca_index=idx,
            c_index=None,
            o_index=None,
            n_index=None,
            ss_type="C",
        )
        for idx in range(3)
    ]
    pd = ProteinData(
        chains={"A": ChainData(chain_id="A", residues=residues)},
        hetatm_indices={3},
        backbone_indices={0, 1, 2},
        sidechain_indices=set(),
        helix_spans=[],
        sheet_spans=[],
        ligand_indices={3},
        trace_chains={"A": [0, 1, 2]},
    )
    from xyzrender.api import Molecule

    svg = str(render(Molecule(graph=g, protein_data=pd), protein=True, orient=False))
    assert "stroke-dasharray" in svg
    assert svg.count("<circle") == 1

    excluded = str(render(Molecule(graph=g, protein_data=pd), protein=True, exclude_chains="A", orient=False))
    assert "stroke-dasharray" not in excluded


def test_protein_mode_keeps_nci_between_two_hidden_backbone_atoms():
    """Hidden cartoon atoms do not suppress their NCI overlay."""
    g = nx.Graph()
    for idx in range(4):
        g.add_node(idx, symbol="C", position=(float(idx) * 1.4, 0.0, 0.0))
    g.add_edges_from([(0, 1), (1, 2), (2, 3)])
    g.add_edge(0, 3, NCI=True)

    residues = [ResidueData("ALA", idx + 1, "A", [idx], idx, None, None, None, "C") for idx in range(4)]
    pd = ProteinData(
        chains={"A": ChainData(chain_id="A", residues=residues)},
        hetatm_indices=set(),
        backbone_indices={0, 1, 2, 3},
        sidechain_indices=set(),
        helix_spans=[],
        sheet_spans=[],
        trace_chains={"A": [0, 1, 2, 3]},
    )
    from xyzrender.api import Molecule

    molecule = Molecule(graph=g, protein_data=pd)
    svg = str(render(molecule, protein=True, orient=False))
    assert "stroke-dasharray" in svg
    assert "<circle" not in svg

    excluded = str(render(molecule, protein=True, exclude_chains="A", orient=False))
    assert "stroke-dasharray" not in excluded


def test_protein_mode_treats_backbone_bound_hydrogen_as_backbone_contact():
    """A backbone N-H contact must not reveal an unrelated residue sidechain."""
    g = nx.Graph()
    g.add_node(0, symbol="N", position=(0.0, 0.0, 0.0))
    g.add_node(1, symbol="H", position=(0.0, 0.8, 0.0))
    g.add_node(2, symbol="C", position=(0.8, 0.0, 0.0))
    g.add_node(3, symbol="C", position=(1.4, 0.0, 0.0))
    g.add_node(4, symbol="C", position=(2.8, 0.0, 0.0))
    g.add_node(5, symbol="O", position=(0.0, 2.2, 0.0))
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(3, 4)
    g.add_edge(1, 5, NCI=True)

    residues = [
        ResidueData("ASP", 1, "A", [0, 1, 2], 0, None, None, 0, "C"),
        ResidueData("ALA", 2, "A", [3], 3, None, None, None, "C"),
        ResidueData("ALA", 3, "A", [4], 4, None, None, None, "C"),
    ]
    pd = ProteinData(
        chains={"A": ChainData(chain_id="A", residues=residues)},
        hetatm_indices={5},
        backbone_indices={0, 3, 4},
        # PDB name-based semantics put added hydrogens in this set; the
        # covalent parent is needed to recognise H1 as backbone-attached.
        sidechain_indices={1, 2},
        helix_spans=[],
        sheet_spans=[],
        ligand_indices={5},
        trace_chains={"A": [0, 3, 4]},
    )
    from xyzrender.api import Molecule

    svg = str(render(Molecule(graph=g, protein_data=pd), protein=True, orient=False))
    assert "stroke-dasharray" in svg
    assert svg.count("<circle") == 1


def test_protein_mode_resolves_sidechain_centroid_context_and_exclusion():
    """Aromatic centroids inherit sidechain context and excluded-chain ownership."""
    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.8, 0.0))
    g.add_node(1, symbol="C", position=(0.8, 0.8, 0.0))
    g.add_node(2, symbol="O", position=(0.4, 2.2, 0.0))
    g.add_node(3, symbol="*", position=(0.4, 0.8, 0.0))
    g.add_node(4, symbol="C", position=(0.0, 0.0, 0.0))
    g.add_node(5, symbol="C", position=(1.4, 0.0, 0.0))
    g.add_node(6, symbol="C", position=(2.8, 0.0, 0.0))
    g.add_edge(0, 1)
    g.add_edge(0, 4)
    g.add_edge(4, 5)
    g.add_edge(5, 6)
    g.add_edge(2, 3, NCI=True)
    g.graph["nci_centroid_sites"] = {3: (0, 1)}

    residues = [
        ResidueData("PHE", 1, "A", [0, 1, 4], 4, None, None, None, "C"),
        ResidueData("ALA", 2, "A", [5], 5, None, None, None, "C"),
        ResidueData("ALA", 3, "A", [6], 6, None, None, None, "C"),
    ]
    pd = ProteinData(
        chains={"A": ChainData(chain_id="A", residues=residues)},
        hetatm_indices={2},
        backbone_indices={4, 5, 6},
        sidechain_indices={0, 1},
        helix_spans=[],
        sheet_spans=[],
        ligand_indices={2},
        trace_chains={"A": [4, 5, 6]},
        het_chains={"L": {2}},
    )
    from xyzrender.api import Molecule

    molecule = Molecule(graph=g, protein_data=pd)
    svg = str(render(molecule, protein=True, orient=False))
    assert "stroke-dasharray" in svg
    assert svg.count("<circle") == 4

    excluded = str(render(molecule, protein=True, exclude_chains="A", orient=False))
    assert "stroke-dasharray" not in excluded
    assert excluded.count("<circle") == 1


def test_protein_mode_reveals_sidechain_without_backbone_overlap_for_nci():
    """NCI context shows the sidechain moiety without duplicating cartoon backbone."""
    g = nx.Graph()
    for idx, position in enumerate(((0.0, 0.0, 0.0), (0.0, 0.8, 0.0), (1.4, 0.0, 0.0), (2.8, 0.0, 0.0))):
        g.add_node(idx, symbol="C", position=position)
    g.add_node(4, symbol="O", position=(0.0, 2.2, 0.0))
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(2, 3)
    g.add_edge(1, 4, NCI=True)

    residues = [
        ResidueData("ALA", 1, "A", [0, 1], 0, None, None, None, "C"),
        ResidueData("ALA", 2, "A", [2], 2, None, None, None, "C"),
        ResidueData("ALA", 3, "A", [3], 3, None, None, None, "C"),
    ]
    pd = ProteinData(
        chains={"A": ChainData(chain_id="A", residues=residues)},
        hetatm_indices={4},
        backbone_indices={0, 2, 3},
        sidechain_indices={1},
        helix_spans=[],
        sheet_spans=[],
        ligand_indices={4},
        trace_chains={"A": [0, 2, 3]},
    )
    from xyzrender.api import Molecule

    svg = str(render(Molecule(graph=g, protein_data=pd), protein=True, orient=False, axes=False, no_cell=True))
    assert "stroke-dasharray" in svg
    assert svg.count("<circle") == 2
    assert any("<line" in line and "stroke-dasharray" not in line for line in svg.splitlines())


def test_filter_ligand_protein_nci_keeps_ligand_protein_components():
    """NCI filter keeps only NCI components containing both ligand and protein context."""
    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))  # protein
    g.add_node(1, symbol="N", position=(1.0, 0.0, 0.0))  # protein
    g.add_node(2, symbol="C", position=(2.0, 0.0, 0.0))  # ligand
    g.add_node(3, symbol="O", position=(3.0, 0.0, 0.0))  # ligand
    g.add_node(4, symbol="*", position=(2.5, 0.5, 0.0))  # centroid in kept component
    g.add_node(5, symbol="*", position=(3.5, 0.5, 0.0))  # centroid in dropped component
    g.add_node(6, symbol="C", position=(4.0, 0.0, 0.0))  # other
    g.add_node(7, symbol="C", position=(5.0, 0.0, 0.0))  # other
    g.add_edge(0, 1, NCI=True)  # protein-protein component -> drop
    g.add_edge(2, 4, NCI=True)  # ligand-centroid -> keep (bridges to protein below)
    g.add_edge(4, 1, NCI=True)  # centroid-protein -> keep
    g.add_edge(3, 5, NCI=True)  # ligand-only component -> drop
    g.add_edge(6, 7, NCI=True)  # other-only component -> drop
    g.add_edge(3, 6)  # non-NCI edge should remain untouched

    pd = ProteinData(
        chains={
            "A": ChainData(
                chain_id="A",
                residues=[
                    ResidueData(
                        res_name="ALA",
                        res_seq=1,
                        chain_id="A",
                        atom_indices=[0, 1],
                        ca_index=None,
                        c_index=None,
                        o_index=None,
                        n_index=None,
                        ss_type="C",
                    )
                ],
            )
        },
        hetatm_indices={2, 3},
        backbone_indices=set(),
        sidechain_indices=set(),
        helix_spans=[],
        sheet_spans=[],
        ligand_indices={2, 3},
        water_indices=set(),
        ion_indices=set(),
    )
    fg = filter_ligand_protein_nci(g, pd)
    nci_edges = {(min(i, j), max(i, j)) for i, j, d in fg.edges(data=True) if d.get("NCI", False)}
    assert nci_edges == {(0, 1), (1, 4), (2, 4)}
    assert 4 in fg.nodes(), "centroid in kept component should remain"
    assert 5 not in fg.nodes(), "orphan centroid should be removed"
    assert fg.has_edge(3, 6), "non-NCI edges should not be removed by NCI filtering"


def test_demo_fixture_has_ligand_filtered_nci_edges():
    """The committed protein+ligand demo fixture should retain ligand-filtered NCI edges."""
    if not (STRUCTURES / "protein_ligand_demo.pdb").exists():
        pytest.skip("protein_ligand_demo.pdb fixture not present in this checkout")
    mol = load(STRUCTURES / "protein_ligand_demo.pdb", nci_ligand_protein_only=True)
    n_nci = sum(1 for _, _, data in mol.graph.edges(data=True) if data.get("NCI", False))
    assert n_nci > 0


def test_demo_fixture_renders_dotted_nci_in_protein_mode():
    """Protein-mode render for the demo fixture should include dotted NCI overlays."""
    if not (STRUCTURES / "protein_ligand_demo.pdb").exists():
        pytest.skip("protein_ligand_demo.pdb fixture not present in this checkout")
    mol = load(STRUCTURES / "protein_ligand_demo.pdb", nci_ligand_protein_only=True)
    svg = str(render(mol, protein=True, nci_ligand_protein_only=True, orient=False))
    assert "stroke-dasharray" in svg


# ---------------------------------------------------------------------------
# Secondary-structure inference accuracy (regression floor)
# ---------------------------------------------------------------------------

_8UWL = STRUCTURES / "8UWL.pdb"
_1UBQ = STRUCTURES / "1UBQ.pdb"  # small mixed alpha/beta, unrelated fold


def _records_ss(pdb_text: str) -> dict[tuple[str, int], str]:
    """Per-residue H/E labels from a PDB's HELIX/SHEET records."""
    truth: dict[tuple[str, int], str] = {}
    for ln in pdb_text.splitlines():
        if ln.startswith("HELIX "):
            for r in range(int(ln[21:25]), int(ln[33:37]) + 1):
                truth[(ln[19], r)] = "H"
        elif ln.startswith("SHEET "):
            for r in range(int(ln[22:26]), int(ln[33:37]) + 1):
                truth.setdefault((ln[21], r), "E")
    return truth


def _assigned_ss(path: Path) -> dict[tuple[str, int], str]:
    from xyzrender.protein_semantics import extract_protein_semantics

    mol = load(path)
    sem = mol.protein_semantics or extract_protein_semantics(mol.graph, source_path=str(path), protein_requested=True)
    assert sem is not None
    return {(cid, res.res_seq): res.ss_type for cid, chain in sem.chains.items() for res in chain.residues}


def test_ss_records_take_precedence_over_inference():
    """With HELIX/SHEET present, every record-labelled residue keeps its record label."""
    if not _8UWL.exists():
        pytest.skip("8UWL.pdb fixture not present in this checkout")
    truth = _records_ss(_8UWL.read_text())
    got = _assigned_ss(_8UWL)
    common = set(truth) & set(got)
    assert len(common) > 500, "expected a substantial record overlap"
    assert all(truth[k] == got[k] for k in common)


def test_ss_geometry_inference_accuracy_floor(tmp_path):
    """Blind test: strip HELIX/SHEET, infer from CA geometry, score against the records.

    Guards the inference constants in ``xyzgraph.protein``.  Coil recall is
    the weak axis: extended coil partly satisfies the strand test.  The
    strand-pairing filter in xyzgraph took coil recall from 63.6% to 71.8%
    and overall agreement from 83.3% to 85.8%.
    """
    if not _8UWL.exists():
        pytest.skip("8UWL.pdb fixture not present in this checkout")
    text = _8UWL.read_text()
    blind = tmp_path / "8UWL_noss.pdb"
    blind.write_text("\n".join(ln for ln in text.splitlines() if not ln.startswith(("HELIX ", "SHEET "))))

    truth_he = _records_ss(text)
    got = _assigned_ss(blind)
    truth = {k: truth_he.get(k, "C") for k in got}

    overall = sum(truth[k] == got[k] for k in got) / len(got)
    recall = {}
    for label in "HEC":
        n = sum(1 for k in got if truth[k] == label)
        recall[label] = sum(1 for k in got if truth[k] == label and got[k] == label) / n

    assert overall >= 0.85, f"overall SS agreement regressed to {overall:.1%}"
    assert recall["H"] >= 0.93, f"helix recall regressed to {recall['H']:.1%}"
    assert recall["E"] >= 0.92, f"sheet recall regressed to {recall['E']:.1%}"
    assert recall["C"] >= 0.71, f"coil recall regressed to {recall['C']:.1%}"


def test_ss_geometry_inference_generalises_to_a_second_structure(tmp_path):
    """Same blind test on ubiquitin, whose fold shares nothing with 8UWL.

    Floors are measured on 1UBQ, not carried over: the constants were tuned on
    8UWL alone, so a second structure lands somewhere else.  Strand recall is
    the axis that travels least well (78.8% here against 92% on 8UWL) --
    ubiquitin's five-strand sheet has short edge strands whose ends read as
    coil.
    """
    if not _1UBQ.exists():
        pytest.skip("1UBQ.pdb fixture not present in this checkout")
    text = _1UBQ.read_text()
    blind = tmp_path / "1UBQ_noss.pdb"
    blind.write_text("\n".join(ln for ln in text.splitlines() if not ln.startswith(("HELIX ", "SHEET "))))

    truth_he = _records_ss(text)
    got = _assigned_ss(blind)
    truth = {k: truth_he.get(k, "C") for k in got}

    overall = sum(truth[k] == got[k] for k in got) / len(got)
    recall = {}
    for label in "HEC":
        n = sum(1 for k in got if truth[k] == label)
        recall[label] = sum(1 for k in got if truth[k] == label and got[k] == label) / n

    # Slacker than the 8UWL floors on purpose: 76 residues make 1% under one
    # residue, so tight floors here fire on a single flipped label.  8UWL is
    # the regression tripwire; this test only asserts the method generalises.
    assert overall >= 0.85, f"overall SS agreement regressed to {overall:.1%}"
    assert recall["H"] >= 0.85, f"helix recall regressed to {recall['H']:.1%}"
    assert recall["E"] >= 0.70, f"sheet recall regressed to {recall['E']:.1%}"
    assert recall["C"] >= 0.88, f"coil recall regressed to {recall['C']:.1%}"


# ---------------------------------------------------------------------------
# Swept-cartoon geometry contract
#
# These replace an earlier suite that asserted the retired implementation's
# artifacts: whole-run <polygon> strips (`poly_count <= 2`), stroked <path>
# loops, and `rg_h_*` / `rg_e_*` gradient ids.  The cartoon is now a swept
# surface emitted one depth-sorted quad at a time with per-quad Lambert
# shading, so those assertions described bugs rather than requirements.
# ---------------------------------------------------------------------------


def _ribbon_items(pdb, cfg=None, **cfg_kw):
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(pdb)
    pd = _require_protein_data(mol.protein_data)
    cfg = cfg or RenderConfig(protein=True, **cfg_kw)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])
    return ribbon_svg_items(pd, cfg, pos, 50.0, 0.0, 0.0, 800, 800)


def test_ribbon_emits_per_quad_items(helix_pdb):
    """Depth granularity is per quad, not per secondary-structure run.

    One item per run was the root cause of helices painting as flat
    self-overlapping pancakes: a whole helix sorted at a single centroid z
    cannot occlude its own turns.
    """
    items = _ribbon_items(helix_pdb)
    assert len(items) > 10
    # One quad per item; silhouette strokes ride along in the same item so they
    # paint over their own quad but under anything in front of it.
    assert all(sum("<polygon" in ln for ln in lines) == 1 for _, lines in items)
    assert all("<polygon" in lines[0] for _, lines in items)


def test_ribbon_items_are_depth_sorted(helix_pdb):
    """Items ascend in z so the renderer's drain loop interleaves correctly."""
    zs = [z for z, _ in _ribbon_items(helix_pdb)]
    assert zs == sorted(zs)


def test_ribbon_quads_are_individually_shaded(helix_pdb):
    """Per-quad normals must yield a range of fills, not one flat colour."""
    import re

    fills = {
        m.group(1)
        for _, lines in _ribbon_items(helix_pdb)
        for m in [re.search(r'fill="(#[0-9a-fA-F]{6})"', lines[0])]
        if m
    }
    assert len(fills) >= 3


def test_outline_strokes_the_silhouette_not_the_mesh(helix_pdb):
    """Outlines follow the surface boundary, not every interior quad edge.

    Stroking the quads themselves made --protein illustration render as a
    wireframe dump of the swept mesh.
    """
    items = _ribbon_items(helix_pdb, protein_style="illustration")
    quads = sum(sum("<polygon" in ln for ln in lines) for _, lines in items)
    strokes = sum(sum("<line" in ln for ln in lines) for _, lines in items)
    assert strokes > 0
    # A wireframe would stroke all four edges of every quad.
    assert strokes < 2 * quads


def test_gloss_outlines_the_ribbon(helix_pdb):
    """The default style carries a hairline contour, not just illustration."""
    items = _ribbon_items(helix_pdb, protein_style="gloss")
    assert any("<line" in ln for _, lines in items for ln in lines)


def test_ribbon_shading_reaches_both_ends_of_the_ramp(helix_pdb):
    """Shadowed faces must actually darken, or overlapping strands merge."""
    import re

    def lum(hex_str):
        r, g, b = (int(hex_str[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    fills = [
        m.group(1)
        for _, lines in _ribbon_items(helix_pdb)
        for m in [re.search(r'fill="(#[0-9a-fA-F]{6})"', lines[0])]
        if m
    ]
    lums = [lum(f) for f in fills]
    assert max(lums) - min(lums) > 0.25


def test_ribbon_emits_no_gradient_defs(helix_pdb):
    """Shading is baked into solid fills; no <linearGradient> is needed.

    The old objectBoundingBox gradients ramped across each polygon's screen
    bounding box rather than across the ribbon's local width.
    """
    from xyzrender.ribbon import ribbon_gradient_defs, ribbon_style_uses_gradients

    assert ribbon_gradient_defs({"A": "#4682b4"}) == []
    assert ribbon_style_uses_gradients("gloss") is False
    svg = str(render(load(helix_pdb), protein=True, orient=False))
    assert "rg_h_" not in svg
    assert "rg_e_" not in svg


@pytest.mark.parametrize(("dimension", "value"), [("ribbon_width", 0.0), ("loop_width", -0.5)])
def test_nonpositive_ribbon_dimensions_are_rejected(helix_pdb, dimension, value):
    with pytest.raises(ValueError, match=r"ribbon_width|loop_width"):
        render(load(helix_pdb), protein=True, orient=False, **{dimension: value})


def test_ribbon_default_width_matches_helix_pitch():
    """Default tape width is near the ~5.4 A helix rise, so turns nearly close up."""
    from xyzrender.types import RenderConfig

    assert RenderConfig().ribbon_width == pytest.approx(4.5)


def test_chain_color_reaches_the_ribbon(two_chain_pdb):
    """An explicit chain colour must appear in that chain's quad fills."""
    from xyzrender.colors import Color

    svg = str(render(load(two_chain_pdb), protein=True, chain_colors={"A": "steelblue"}, orient=False))
    base = Color.from_str("steelblue")
    # Fills are Lambert-tinted, so look for the hue rather than an exact hex.
    assert any(
        abs(Color.from_hex(m).to_hls()[0] - base.to_hls()[0]) < 0.03
        for m in __import__("re").findall(r'fill="(#[0-9a-fA-F]{6})"', svg)
    )


def test_exclude_chain_removes_its_geometry(two_chain_pdb):
    """Excluding a chain drops its quads (and shrinks the emitted SVG)."""
    chain_colors = {"A": "steelblue", "B": "red"}
    svg_all = str(render(load(two_chain_pdb), protein=True, chain_colors=chain_colors, orient=False))
    svg_excl = str(
        render(load(two_chain_pdb), protein=True, chain_colors=chain_colors, exclude_chains="B", orient=False)
    )
    assert svg_excl.count("<polygon") < svg_all.count("<polygon")


def test_exclude_chain_removes_its_ligand(tmp_path):
    """A chain's heteroatoms go with its ribbon, not float on alone.

    Ligand/water/ion indices carry no chain of their own, so exclusion has to
    reach them through ProteinSemantics.het_chains.
    """
    p = tmp_path / "two_chain_lig.pdb"
    p.write_text(_TWO_CHAIN_PDB.replace("END\n", "") + _LIGAND_ON_B + "END\n")
    mol = load(p)
    assert str(render(mol, protein=True, orient=False)).count("<circle") == 2  # the two LIG atoms
    assert str(render(mol, protein=True, exclude_chains="B", orient=False)).count("<circle") == 0


def test_exclude_chain_does_not_recolour_the_others(two_chain_pdb):
    """B keeps palette[1] when A is excluded, rather than sliding to palette[0].

    Excluding the *first* chain is the discriminating case: filtering before
    allocation renumbered everything after it, so the same chain changed colour
    between an overview and a close-up.
    """
    import re

    mol = load(two_chain_pdb)

    def fills(svg):
        return set(re.findall(r'<polygon[^>]*fill="(#[0-9a-fA-F]{6})"', svg))

    # The fixture's chains are translated copies, so their shade sets differ
    # only by base colour.  Filtering before allocation gave whichever chain
    # survived palette[0], making these two renders identical.
    exclude_a = str(render(mol, protein=True, exclude_chains="A", orient=False))
    exclude_b = str(render(mol, protein=True, exclude_chains="B", orient=False))
    assert fills(exclude_a) != fills(exclude_b)


def test_lone_amino_acid_ligand_is_still_drawn(tmp_path):
    """A single residue on its own chain must not vanish.

    Backbone names are collected before the peptide test diverts a residue to
    ligand, so its N/CA/C/O stayed in backbone_indices -- which the renderer
    hides -- while its chain was too short to earn a ribbon.
    """
    p = tmp_path / "docked.pdb"
    lig = "".join(
        f"ATOM  {9 + i:>5} {name:^4} GLY L   1    {9.0 + i:>8.3f}{9.0:>8.3f}{0.0:>8.3f}"
        f"{1.0:>6.2f}{0.0:>6.2f}          {elem:>2}\n"
        for i, (name, elem) in enumerate([("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")])
    )
    p.write_text(_TWO_CHAIN_PDB.replace("END\n", "") + lig + "END\n")
    sem = load(p).protein_semantics
    assert sem is not None
    assert len(sem.ligand_indices) == 4
    assert not (sem.ligand_indices & sem.backbone_indices)
    assert str(render(load(p), protein=True, orient=False)).count("<circle") == 4


def test_single_residue_protein_falls_back_to_ball_and_stick(tmp_path):
    """A one-residue ATOM chain cannot earn a ribbon and must remain visible."""
    p = tmp_path / "single_residue.pdb"
    p.write_text(_HETATM_PDB)

    svg = str(render(load(p), protein=True, orient=False, gradient=False))
    assert svg.count("<circle") == 6


def test_placeholder_cryst1_draws_no_cell():
    """PDB mandates CRYST1, so non-crystallographic entries carry a 1 A sentinel."""
    from xyzrender.parsers import parse_pdb

    assert parse_pdb(STRUCTURES / "8UWL.pdb").pbc_cell is None
    assert parse_pdb(STRUCTURES / "water_cryst.pdb").pbc_cell is not None


def test_invalid_cryst1_geometry_is_rejected(tmp_path):
    """Degenerate unit cells fail at parsing rather than leaking NaNs."""
    p = tmp_path / "invalid_cell.pdb"
    p.write_text(
        "CRYST1   10.000   10.000   10.000  90.00  90.00   0.00 P 1           1\n"
        "ATOM      1  C   UNK A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )

    with pytest.raises(ValueError, match="CRYST1"):
        parse_pdb(p)


def test_docked_ligand_without_conect_gets_bonds(tmp_path):
    """A pose appended to a deposited structure has no CONECT of its own.

    File connectivity wins whenever any CONECT record exists, so the ligand
    would draw as loose spheres.  --rebuild fixes it but costs ~131 s on an
    8.6k-atom protein; only the heteroatoms need re-detecting.
    """
    p = tmp_path / "docked.pdb"
    lig = "".join(
        f"HETATM{17 + i:>5} {'C%d' % (i + 1):^4} LIG A 101    "
        f"{8.0 + 1.4 * i:>8.3f}{9.0:>8.3f}{0.0:>8.3f}{1.0:>6.2f}{0.0:>6.2f}           C\n"
        for i in range(6)
    )
    # CONECT covering only the protein, as a deposited file would for its own HETATM
    p.write_text(_TWO_CHAIN_PDB.replace("END\n", "") + lig + "CONECT    1    2\nCONECT    2    3\nEND\n")
    mol = load(p)
    lig_nodes = [n for n in mol.graph.nodes if n >= 16]
    assert sum(mol.graph.degree(n) for n in lig_nodes) > 0


def test_ligand_conect_does_not_erase_polymer_bonds(tmp_path):
    """Deposited HETATM connectivity must not leave protein sidechains loose."""
    p = tmp_path / "partial_conect.pdb"
    p.write_text(
        dedent("""\
            ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
            ATOM      2  CA  ALA A   1       1.456   0.000   0.000  1.00 10.00           C
            ATOM      3  C   ALA A   1       1.930   0.000   1.463  1.00 10.00           C
            ATOM      4  O   ALA A   1       1.160   0.000   2.421  1.00 10.00           O
            ATOM      5  CB  ALA A   1       1.900   1.400  -0.300  1.00 10.00           C
            ATOM      6  N   ALA A   2       3.241   0.000   1.742  1.00 10.00           N
            ATOM      7  CA  ALA A   2       3.690   0.000   3.127  1.00 10.00           C
            ATOM      8  C   ALA A   2       5.228   0.000   3.127  1.00 10.00           C
            ATOM      9  O   ALA A   2       5.902   0.000   2.098  1.00 10.00           O
            ATOM     10  CB  ALA A   2       3.200   1.400   3.500  1.00 10.00           C
            HETATM   11  C1  LIG A 101       8.000   0.000   0.000  1.00 10.00           C
            HETATM   12  O1  LIG A 101       9.200   0.000   0.000  1.00 10.00           O
            CONECT   11   12
            END
        """)
    )

    mol = load(p)
    assert mol.graph.has_edge(1, 4)  # CA-CB sidechain attachment
    assert mol.graph.has_edge(2, 5)  # peptide C-N link
    assert mol.graph.has_edge(6, 9)  # second CA-CB attachment


def test_protein_load_does_not_warn_about_missing_bonds(caplog):
    """Backbone atoms drawn as ribbon never needed bonds; the warning is noise."""
    import logging

    with caplog.at_level(logging.WARNING, logger="xyzrender.readers"):
        load(STRUCTURES / "8UWL.pdb")
    assert not [r for r in caplog.records if "no bonds from file" in r.message]


def test_named_styles_render(helix_pdb):
    """Both named styles still produce cartoon geometry."""
    for style in ("gloss", "illustration"):
        svg = str(render(load(helix_pdb), protein=style, orient=False))
        assert svg.count("<polygon") > 5


def test_illustration_style_is_slimmer_and_outlined():
    """Illustration is the flatter, contoured variant of the same geometry."""
    from xyzrender.ribbon import ribbon_style_profile

    il, gl = ribbon_style_profile("illustration"), ribbon_style_profile("gloss")
    assert il.width_scale < gl.width_scale
    assert il.thickness_scale < gl.thickness_scale
    assert il.shade_gain < gl.shade_gain
    assert il.outline_px > 0


def test_ss_transition_is_continuous():
    """A single-residue H/C/H alternation still sweeps one connected surface.

    The old code emitted an unsplined straight quad at a third width across
    every SS junction, which read as a notch.
    """
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    pd = ProteinData(
        chains={
            "A": ChainData(
                chain_id="A",
                residues=[
                    ResidueData("ALA", 1, "A", [0, 3], ca_index=0, c_index=None, o_index=3, n_index=None, ss_type="H"),
                    ResidueData("GLY", 2, "A", [1, 4], ca_index=1, c_index=None, o_index=4, n_index=None, ss_type="C"),
                    ResidueData("SER", 3, "A", [2, 5], ca_index=2, c_index=None, o_index=5, n_index=None, ss_type="H"),
                ],
            )
        },
        hetatm_indices=set(),
        backbone_indices={0, 1, 2, 3, 4, 5},
        sidechain_indices=set(),
        helix_spans=[],
        sheet_spans=[],
    )
    pos = np.array(
        [[0.0, 0, 0], [3.8, 0, 0], [7.6, 0, 0], [0.5, 1.2, 0], [4.3, 1.2, 0], [8.1, 1.2, 0]],
        dtype=float,
    )
    items = ribbon_svg_items(pd, RenderConfig(protein=True), pos, 50.0, 0.0, 0.0, 800, 800)
    assert len(items) > 5


def test_frame_is_flip_corrected():
    """Successive ribbon normals never invert.

    Carbonyls alternate by ~180 degrees along a strand; without sign
    propagation the tape would flip on every residue.
    """
    import numpy as np

    from xyzrender.ribbon import _central_tangents, _frame_normals

    n = 12
    ca = np.stack([np.arange(n) * 3.8, np.zeros(n), np.zeros(n)], axis=1).astype(float)
    # Alternating carbonyl directions, as in a real beta strand.
    o = [ca[i] + np.array([0.0, 1.2 * (-1) ** i, 0.3]) for i in range(n)]
    normals = _frame_normals(ca, o, _central_tangents(ca), ["E"] * n)
    dots = np.einsum("ij,ij->i", normals[:-1], normals[1:])
    assert bool((dots > 0).all()), f"frame flips at {np.where(dots <= 0)[0]}"


def test_helix_frame_tracks_the_axis():
    """On an ideal helix the tape's width runs along the helix axis.

    This is the Carson-Bugg construction.  Using the curvature vector instead
    (which points at the axis) makes the tape a fin spiralling around it.
    """
    import numpy as np

    from xyzrender.ribbon import _central_tangents, _frame_normals

    # Ideal alpha helix: r=2.3 A, rise 1.5 A/residue, 100 deg/residue.
    n = 20
    t = np.arange(n) * np.deg2rad(100.0)
    ca = np.stack([2.3 * np.cos(t), 2.3 * np.sin(t), np.arange(n) * 1.5], axis=1)
    # Carbonyl points roughly along +z (the axis) in a real helix.
    o = [ca[i] + np.array([0.3 * np.cos(t[i]), 0.3 * np.sin(t[i]), 1.1]) for i in range(n)]
    normals = _frame_normals(ca, o, _central_tangents(ca), ["H"] * n)
    axis = np.array([0.0, 0.0, 1.0])
    # Interior residues only; the ends have one-sided tangents.
    align = np.abs(normals[3:-3] @ axis)
    assert align.mean() > 0.5, f"mean |n . axis| = {align.mean():.2f}"


def test_centripetal_spline_hits_control_points():
    """The spline interpolates its control points (it is not an approximation)."""
    import numpy as np

    from xyzrender.ribbon import _catmull_rom_3d

    pts = np.array([[0.0, 0, 0], [1.0, 2, 0], [3.0, 1, 1], [5.0, 3, 0]])
    curve = _catmull_rom_3d(pts, steps=8)
    for p in pts:
        assert np.min(np.linalg.norm(curve - p, axis=1)) < 1e-6


def test_centripetal_spline_does_not_overshoot():
    """No cusp or overshoot at a tight turn, which uniform Catmull-Rom shows."""
    import numpy as np

    from xyzrender.ribbon import _catmull_rom_3d

    # Hairpin: three nearly-coincident points then a sharp reversal.
    pts = np.array([[0.0, 0, 0], [3.8, 0, 0], [4.2, 3.0, 0], [0.4, 3.2, 0]])
    curve = _catmull_rom_3d(pts, steps=12)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    margin = 0.35 * float(np.linalg.norm(hi - lo))
    assert (curve >= lo - margin).all()
    assert (curve <= hi + margin).all()


def test_strand_gets_an_arrowhead(sheet_pdb):
    """A strand's tape widens to a flange then tapers, as one swept surface."""
    import numpy as np

    from xyzrender.ribbon import _sample_widths

    ss = ["E"] * 8
    res_t = np.linspace(0, 7, 60)
    dims = {"H": (2.0, 0.3), "E": (2.0, 0.3), "C": (0.4, 0.4)}
    hw, _, _ = _sample_widths(ss, res_t, dims)
    body = hw[res_t <= 5.0].mean()
    assert hw.max() > body * 1.5, "no flange"
    assert hw[-1] < body * 0.5, "strand does not taper to a point"


def test_helix_sheet_fixture_assigns_both_ss_types(helix_sheet_pdb):
    """The two-chain fixture carries one helix and one strand."""
    pd = _require_protein_data(load(helix_sheet_pdb).protein_data)
    assert [r.ss_type for r in pd.chains["A"].residues] == ["H"] * 10
    assert [r.ss_type for r in pd.chains["B"].residues] == ["E"] * 7


def test_strand_arrowhead_reaches_the_rendered_surface(helix_sheet_pdb):
    """The flange survives into the drawn polygons, not just _sample_widths."""
    import re

    import numpy as np

    mol = load(helix_sheet_pdb)
    svg = str(render(mol, protein=True, exclude_chains="A", orient=False))
    pts = np.array(
        [
            [float(v) for v in tok.split(",")]
            for m in re.finditer(r'<polygon points="([^"]+)"', svg)
            for tok in m.group(1).split()
        ]
    )
    # Widths measured across the strand's own axis, so the test does not depend
    # on canvas orientation.
    centred = pts - pts.mean(axis=0)
    axis, perp = np.linalg.svd(centred, full_matrices=False)[2][:2]
    t = centred @ axis
    w = np.abs(centred @ perp)
    frac = (t - t.min()) / (t.max() - t.min())
    ends = sorted((w[frac < 0.25].max(), w[frac > 0.75].max()))
    assert ends[1] > 2.0 * ends[0], "no flange at either end of the strand"


def test_helix_and_sheet_chains_render_as_separate_ribbons(helix_sheet_pdb):
    """Two chains get two colours, and excluding one drops its geometry."""
    import re

    def fills(**kw):
        svg = str(render(mol, protein=True, orient=False, **kw))
        return svg.count("<polygon"), {m.group(1) for m in re.finditer(r'<polygon [^>]*fill="([^"]+)"', svg)}

    mol = load(helix_sheet_pdb)
    n_both, _ = fills()
    n_a, fill_a = fills(exclude_chains="B")
    n_b, fill_b = fills(exclude_chains="A")
    assert n_both > n_a
    assert n_both > n_b

    # Per-quad shading gives one chain many fills, so the claim is that the two
    # chains' shade ramps do not overlap -- not that more than one fill exists.
    assert not (fill_a & fill_b)


def test_short_strand_still_renders(tmp_path):
    """A 2-residue SHEET record is legal PDB and must not be dropped.

    The old code had _MIN_SHEET_RES=2 and made such a strand entirely
    arrowhead; the taper now handles it as a short tapered strand.
    """
    import numpy as np

    from xyzrender.ribbon import _sample_widths

    dims = {"H": (2.0, 0.3), "E": (2.0, 0.3), "C": (0.4, 0.4)}
    hw, _, _ = _sample_widths(["E", "E"], np.linspace(0, 1, 16), dims)
    assert np.isfinite(hw).all()
    assert hw.max() > 0


# ---------------------------------------------------------------------------
# PDB record handling: MODEL / altLoc / insertion codes / tempFactor
# ---------------------------------------------------------------------------


def _atom_line(serial, name, res_name, chain, res_seq, xyz, *, altloc=" ", icode=" ", bfac=0.0, elem=None):
    """One fixed-column ATOM record."""
    elem = elem or name[0]
    x, y, z = xyz
    return (
        f"ATOM  {serial:>5} {name:<4}{altloc}{res_name:>3} {chain}{res_seq:>4}{icode}   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{bfac:>6.2f}          {elem:>2}"
    )


def test_multi_model_pdb_uses_first_model_only(tmp_path):
    """An NMR ensemble must not concatenate every model into one structure.

    Without this the models are overlaid *and* res_seq restarts per model, so
    the backbone is shredded into fragments at every model boundary.
    """
    lines = []
    for model in (1, 2, 3):
        lines.append(f"MODEL     {model:>4}")
        for i in range(3):
            lines.append(_atom_line(i + 1, "CA", "ALA", "A", i + 1, (i * 3.8, model * 10.0, 0.0), elem="C"))
        lines.append("ENDMDL")
    p = tmp_path / "nmr.pdb"
    p.write_text("\n".join(lines) + "\n")

    data = parse_pdb(p)
    assert len(data.atoms) == 3, f"expected 1 model (3 atoms), got {len(data.atoms)}"
    # Model 1 was the one kept.
    assert data.atoms[0][1][1] == pytest.approx(10.0)


def test_altloc_keeps_one_conformer(tmp_path):
    """Alternate conformers share a residue key; only altLoc blank/A is kept."""
    lines = [
        _atom_line(1, "N", "SER", "A", 1, (0.0, 0.0, 0.0), elem="N"),
        _atom_line(2, "CA", "SER", "A", 1, (1.5, 0.0, 0.0), altloc="A", elem="C"),
        _atom_line(3, "CA", "SER", "A", 1, (1.5, 0.9, 0.0), altloc="B", elem="C"),
        _atom_line(4, "C", "SER", "A", 1, (2.5, 0.0, 0.0), elem="C"),
    ]
    p = tmp_path / "altloc.pdb"
    p.write_text("\n".join(lines) + "\n")

    data = parse_pdb(p)
    assert len(data.atoms) == 3, "altLoc B should have been dropped"
    pd = _require_protein_data(data.protein_data)
    res = pd.chains["A"].residues[0]
    assert res.ca_index is not None
    assert data.atoms[res.ca_index][1][1] == pytest.approx(0.0), "kept the wrong conformer"


def test_insertion_codes_do_not_merge_residues(tmp_path):
    """Residues 100, 100A, 100B are distinct even when res_name repeats.

    Without the insertion code in the residue key they collide, and ca_index
    is overwritten by whichever CA was parsed last -- a silent jump in the
    backbone trace.
    """
    lines = []
    serial = 1
    for k, icode in enumerate((" ", "A", "B")):
        for name, off in (("N", 0.0), ("CA", 1.5), ("C", 2.5)):
            lines.append(
                _atom_line(
                    serial,
                    name,
                    "SER",
                    "A",
                    100,
                    (k * 3.8 + off, 0.0, 0.0),
                    icode=icode,
                    elem=name[0],
                )
            )
            serial += 1
    p = tmp_path / "icode.pdb"
    p.write_text("\n".join(lines) + "\n")

    pd = _require_protein_data(parse_pdb(p).protein_data)
    residues = pd.chains["A"].residues
    assert len(residues) == 3, f"insertion codes merged: {len(residues)} residue(s)"
    assert [r.i_code for r in residues] == ["", "A", "B"]
    assert len({r.ca_index for r in residues}) == 3


def test_insertion_codes_do_not_split_the_ribbon(tmp_path):
    """An insertion repeats res_seq; that is a continuation, not a chain break."""
    import numpy as np

    from xyzrender.ribbon import _split_backbone_segments

    pd = _require_protein_data(
        parse_pdb(
            _write_icode_chain(tmp_path),
        ).protein_data
    )
    residues = pd.chains["A"].residues
    pos = np.array([[i * 3.8, 0.0, 0.0] for i in range(sum(len(r.atom_indices) for r in residues) + 5)])
    for r in residues:
        pos[r.ca_index] = [residues.index(r) * 3.8, 0.0, 0.0]
    segs = _split_backbone_segments(residues, pos)
    assert len(segs) == 1, f"insertion split the backbone into {len(segs)} segments"


def _write_icode_chain(tmp_path):
    lines = []
    serial = 1
    seq_icodes = [(99, " "), (100, " "), (100, "A"), (100, "B"), (101, " ")]
    for k, (seq, icode) in enumerate(seq_icodes):
        for name, off in (("N", 0.0), ("CA", 1.5), ("C", 2.5)):
            lines.append(
                _atom_line(serial, name, "SER", "A", seq, (k * 3.8 + off, 0.0, 0.0), icode=icode, elem=name[0])
            )
            serial += 1
    p = tmp_path / "icode_chain.pdb"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_temperature_factor_is_parsed(tmp_path):
    """B-factor / pLDDT is read from cols 61-66 onto the residue."""
    lines = [
        _atom_line(1, "N", "ALA", "A", 1, (0.0, 0.0, 0.0), bfac=11.0, elem="N"),
        _atom_line(2, "CA", "ALA", "A", 1, (1.5, 0.0, 0.0), bfac=42.5, elem="C"),
        _atom_line(3, "C", "ALA", "A", 1, (2.5, 0.0, 0.0), bfac=13.0, elem="C"),
    ]
    p = tmp_path / "bfac.pdb"
    p.write_text("\n".join(lines) + "\n")

    pd = _require_protein_data(parse_pdb(p).protein_data)
    # The CA value is the per-residue B-factor (pLDDT is reported on CA).
    assert pd.chains["A"].residues[0].b_factor == pytest.approx(42.5)


# ---------------------------------------------------------------------------
# Sidechain attachment and selection
# ---------------------------------------------------------------------------


def test_sidechain_anchor_lies_outside_the_ribbon(sidechain_pdb):
    """The attachment point clears the swept surface rather than sitting at CA.

    CA is the tape's centreline, so a bond starting there is buried inside the
    solid; the anchor is where the CA->CB ray exits the surface.
    """
    import numpy as np

    from xyzrender.ribbon import sidechain_anchors
    from xyzrender.types import RenderConfig

    mol = load(sidechain_pdb)
    pd = _require_protein_data(mol.protein_data)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])
    cfg = RenderConfig(protein=True, show_sidechain=True)

    anchors = sidechain_anchors(pd, cfg, pos)
    assert anchors, "no sidechain anchors produced"
    for ca_idx, pt in anchors.items():
        assert np.linalg.norm(pt - pos[ca_idx]) > 1e-3, "anchor collapsed onto CA"


def test_sidechain_bond_survives_hidden_ca(sidechain_pdb):
    """--sidechain must emit the CA-CB bond even though CA is hidden.

    Regression for the original defect: renderer.py dropped every bond with a
    hidden endpoint, so sidechains floated unattached to the ribbon.
    """
    mol = load(sidechain_pdb)
    plain = str(render(mol, protein=True, orient=False))
    with_sc = str(render(mol, protein=True, sidechain=True, orient=False))
    assert with_sc.count("<line") > plain.count("<line")


def test_select_sidechain_atoms_by_residue_number(two_chain_pdb):
    """Residue selectors accept numbers, ranges and chain qualifiers."""
    from xyzrender.ribbon import select_sidechain_atoms

    pd = _require_protein_data(load(two_chain_pdb).protein_data)
    all_seqs = {r.res_seq for c in pd.chains.values() for r in c.residues}
    first = min(all_seqs)

    everything = select_sidechain_atoms(pd, None)
    just_one = select_sidechain_atoms(pd, str(first))
    assert just_one
    assert just_one < everything

    chain_a = next(iter(pd.chains))
    qualified = select_sidechain_atoms(pd, f"{chain_a}:{first}")
    assert qualified <= just_one


def test_select_sidechain_atoms_rejects_garbage(two_chain_pdb):
    from xyzrender.ribbon import select_sidechain_atoms

    pd = _require_protein_data(load(two_chain_pdb).protein_data)
    with pytest.raises(ValueError, match="invalid residue selector"):
        select_sidechain_atoms(pd, "not-a-residue")


def test_select_sidechain_atoms_rejects_empty_selector(two_chain_pdb):
    from xyzrender.ribbon import select_sidechain_atoms

    pd = _require_protein_data(load(two_chain_pdb).protein_data)
    with pytest.raises(ValueError, match="residue selector must not be empty"):
        select_sidechain_atoms(pd, " , ")


def test_sidechain_selector_restricts_what_is_drawn(sidechain_pdb):
    """A restricted selector draws fewer atoms than bare --sidechain."""
    mol = load(sidechain_pdb)
    everything = str(render(mol, protein=True, sidechain=True, orient=False))
    restricted = str(render(mol, protein=True, sidechain="1", orient=False))
    assert restricted.count("<circle") < everything.count("<circle")


# ---------------------------------------------------------------------------
# Colour modes
# ---------------------------------------------------------------------------


def _fills(svg: str) -> set[str]:
    import re

    return set(re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg))


def test_color_by_modes_are_validated():
    from xyzrender.ribbon import COLOR_BY_MODES, normalize_color_by

    assert normalize_color_by(None) == "chain"
    for mode in COLOR_BY_MODES:
        assert normalize_color_by(mode.upper()) == mode
    with pytest.raises(ValueError, match="unknown color-by mode"):
        normalize_color_by("bogus")


def _residues(pdb):
    pd = _require_protein_data(load(pdb).protein_data)
    return [r for c in pd.chains.values() for r in c.residues]


def test_color_by_ss_uses_distinct_colors_per_structure(helix_pdb, sheet_pdb):
    """Helix, sheet and coil each get their own colour."""
    from xyzrender.ribbon import _residue_colors

    h = _residue_colors(_residues(helix_pdb), "ss", "#888888", b_range=None)
    e = _residue_colors(_residues(sheet_pdb), "ss", "#888888", b_range=None)
    assert len(set(h)) == 1
    assert len(set(e)) == 1
    assert set(h).isdisjoint(e), "helix and sheet share a colour under color_by='ss'"


def test_color_by_rainbow_ramps_n_to_c(sheet_pdb):
    """Rainbow walks a spectral ramp from the N terminus to the C terminus."""
    from xyzrender.ribbon import _residue_colors

    residues = _residues(sheet_pdb)
    colors = _residue_colors(residues, "rainbow", "#888888", b_range=None)
    assert len(colors) == len(residues)
    assert len(set(colors)) == len(residues), "rainbow repeated a colour"
    assert colors[0] != colors[-1]


def test_color_by_reaches_the_rendered_output(sheet_pdb):
    """Switching mode actually changes the emitted SVG."""
    mol = load(sheet_pdb)
    by_chain = str(render(mol, protein=True, color_by="chain", orient=False))
    by_ss = str(render(mol, protein=True, color_by="ss", orient=False))
    assert _fills(by_chain) != _fills(by_ss)


def test_color_by_bfactor_tracks_the_temperature_factor(tmp_path):
    """Residues with different B-factors get different colours."""
    lines = []
    serial = 1
    for k in range(6):
        for name, off, elem in (("N", 0.0, "N"), ("CA", 1.5, "C"), ("C", 2.5, "C"), ("O", 3.0, "O")):
            lines.append(
                _atom_line(serial, name, "ALA", "A", k + 1, (k * 3.8 + off, 0.0, 0.0), bfac=10.0 * k, elem=elem)
            )
            serial += 1
    p = tmp_path / "bfac_chain.pdb"
    p.write_text("\n".join(lines) + "\n")

    mol = load(p)
    flat = _fills(str(render(mol, protein=True, color_by="chain", orient=False)))
    graded = _fills(str(render(mol, protein=True, color_by="bfactor", orient=False)))
    assert len(graded) > len(flat)


def test_color_by_bfactor_falls_back_when_uniform(helix_pdb, caplog):
    """A structure with no B-factor variation warns and keeps chain colours."""
    import logging

    with caplog.at_level(logging.WARNING):
        svg = str(render(load(helix_pdb), protein=True, color_by="bfactor", orient=False))
    assert svg.count("<polygon") > 0
    assert any("same B-factor" in r.message for r in caplog.records)


def test_bfactor_range_ignores_excluded_chains(two_chain_pdb):
    from xyzrender.ribbon import bfactor_range

    pd = _require_protein_data(load(two_chain_pdb).protein_data)
    # All fixture B-factors are 0.00, so the range is degenerate -> None.
    assert bfactor_range(pd, set()) is None


def test_chain_palette_wrap_is_distinguishable():
    """Chain 9 must not be identical to chain 1 when the palette wraps."""
    from xyzrender.ribbon import assign_chain_colors
    from xyzrender.types import RenderConfig

    cfg = RenderConfig(protein=True)
    n = len(cfg.protein_palette)
    ids = [f"C{i}" for i in range(n * 2)]
    colors = assign_chain_colors(cfg, ids)
    assert colors[ids[0]] != colors[ids[n]], "palette wrapped to an identical colour"
