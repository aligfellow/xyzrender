"""Tests for graph-first protein semantics extraction and fallback gating."""

from __future__ import annotations

from textwrap import dedent

import pytest

from xyzrender import load, render
from xyzrender.protein_semantics import xyzgraph_protein_available
from xyzrender.types import ProteinConfidence

pytestmark = pytest.mark.skipif(
    not xyzgraph_protein_available(),
    reason="installed xyzgraph does not provide xyzgraph.protein",
)


def test_malformed_secondary_structure_span_is_rejected():
    from xyzrender.protein_semantics import _to_xyzrender_semantics

    with pytest.raises(ValueError, match="helix_spans"):
        _to_xyzrender_semantics({"confidence_tier": "insufficient", "helix_spans": [["A", 1]]})


def test_unknown_protein_confidence_is_rejected():
    from xyzrender.protein_semantics import _to_xyzrender_semantics

    with pytest.raises(ValueError, match="unknown protein confidence"):
        _to_xyzrender_semantics({"confidence_tier": "future-tier"})


def test_malformed_extxyz_property_count_is_ignored():
    """Optional protein metadata must not make a coordinate-readable XYZ fail."""
    from xyzrender.protein_semantics import _parse_extxyz_properties

    assert _parse_extxyz_properties("Properties=species:S:1:pos:R:nope") is None


def test_extxyz_canonical_semantics(tmp_path):
    p = tmp_path / "prot.xyz"
    p.write_text(
        dedent(
            """\
            8
            Properties=species:S:1:pos:R:3:atom_name:S:1:res_name:S:1:res_seq:I:1:chain_id:S:1:ss_type:S:1
            N 0.000 0.000 0.000 N ALA 1 A H
            C 1.456 0.000 0.000 CA ALA 1 A H
            C 1.930 0.000 1.463 C ALA 1 A H
            O 1.160 0.000 2.421 O ALA 1 A H
            N 3.241 0.000 1.742 N GLY 2 A H
            C 3.690 0.000 3.127 CA GLY 2 A H
            C 5.228 0.000 3.127 C GLY 2 A H
            O 5.902 0.000 2.098 O GLY 2 A H
            """
        )
    )
    mol = load(p)
    assert mol.protein_semantics is not None
    assert mol.protein_semantics.confidence_tier == ProteinConfidence.FULL_RIBBON
    assert "A" in mol.protein_semantics.chains


def test_mol2_semantics_extraction(tmp_path):
    p = tmp_path / "prot.mol2"
    p.write_text(
        dedent(
            """\
            @<TRIPOS>MOLECULE
            prot
             8 7 0 0 0
            SMALL
            NO_CHARGES
            @<TRIPOS>ATOM
              1 N   0.000 0.000 0.000 N.3  1 ALA1_A  0.0
              2 CA  1.456 0.000 0.000 C.3  1 ALA1_A  0.0
              3 C   1.930 0.000 1.463 C.2  1 ALA1_A  0.0
              4 O   1.160 0.000 2.421 O.2  1 ALA1_A  0.0
              5 N   3.241 0.000 1.742 N.3  2 GLY2_A  0.0
              6 CA  3.690 0.000 3.127 C.3  2 GLY2_A  0.0
              7 C   5.228 0.000 3.127 C.2  2 GLY2_A  0.0
              8 O   5.902 0.000 2.098 O.2  2 GLY2_A  0.0
            @<TRIPOS>BOND
             1 1 2 1
             2 2 3 1
             3 3 4 1
             4 3 5 1
             5 5 6 1
             6 6 7 1
             7 7 8 1
            """
        )
    )
    mol = load(p)
    assert mol.protein_semantics is not None
    assert mol.protein_semantics.confidence_tier == ProteinConfidence.FULL_RIBBON
    assert len(mol.protein_semantics.chains["A"].residues) == 2


def test_mol2_ligand_not_promoted_to_ribbon(tmp_path):
    p = tmp_path / "prot_lig.mol2"
    p.write_text(
        dedent(
            """\
            @<TRIPOS>MOLECULE
            prot_lig
             10 8 0 0 0
            SMALL
            NO_CHARGES
            @<TRIPOS>ATOM
              1 N   0.000 0.000 0.000 N.3  1 ALA1_A  0.0
              2 CA  1.456 0.000 0.000 C.3  1 ALA1_A  0.0
              3 C   1.930 0.000 1.463 C.2  1 ALA1_A  0.0
              4 O   1.160 0.000 2.421 O.2  1 ALA1_A  0.0
              5 N   3.241 0.000 1.742 N.3  2 GLY2_A  0.0
              6 CA  3.690 0.000 3.127 C.3  2 GLY2_A  0.0
              7 C   5.228 0.000 3.127 C.2  2 GLY2_A  0.0
              8 O   5.902 0.000 2.098 O.2  2 GLY2_A  0.0
              9 C1  8.000 0.000 0.000 C.3  3 LIG3_A  0.0
             10 O1  9.200 0.000 0.000 O.2  3 LIG3_A  0.0
            @<TRIPOS>BOND
             1 1 2 1
             2 2 3 1
             3 3 4 1
             4 3 5 1
             5 5 6 1
             6 6 7 1
             7 7 8 1
             8 9 10 1
            """
        )
    )
    mol = load(p)
    sem = mol.protein_semantics
    assert sem is not None
    protein_atoms = {i for ch in sem.chains.values() for r in ch.residues for i in r.atom_indices}
    assert sem.ligand_indices == {8, 9}
    assert 8 not in protein_atoms
    assert 9 not in protein_atoms

    # In protein mode, backbone protein atoms are hidden but ligand atoms should stay visible.
    svg = str(render(mol, protein=True, orient=False, gradient=False, config="flat"))
    assert svg.count("<circle") >= 2


def test_graph_only_trace_fallback(tmp_path):
    p = tmp_path / "trace.xyz"
    p.write_text(
        dedent(
            """\
            16
            trace
            N 0.000 0.000 0.000
            C 1.456 0.000 0.000
            C 1.930 0.000 1.463
            O 1.160 0.000 2.421
            N 3.241 0.000 1.742
            C 3.690 0.000 3.127
            C 5.228 0.000 3.127
            O 5.902 0.000 2.098
            N 5.898 0.000 4.287
            C 7.353 0.000 4.287
            C 7.828 0.000 5.750
            O 7.058 0.001 6.709
            N 9.148 0.000 6.029
            C 9.602 0.000 7.414
            C 11.140 0.000 7.414
            O 11.814 0.000 6.384
            """
        )
    )
    mol = load(p)
    assert mol.protein_semantics is None  # no metadata extracted at load-time
    svg = str(render(mol, protein=True, orient=False, gradient=False, config="flat"))
    # The trace fallback sweeps the same round coil section as a real cartoon
    # coil, so it emits quad polygons rather than the stroked <path> the old
    # implementation used.  No secondary structure is claimed either way.
    assert svg.count("<polygon") > 5


def test_mmcif_is_rejected_with_a_clear_message(tmp_path):
    """ase reads small-molecule CIF only; mmCIF must not surface as StopIteration."""
    from xyzrender.parsers import parse_cif

    p = tmp_path / "prot.cif"
    p.write_text("data_x\nloop_\n_atom_site.group_PDB\n_atom_site.Cartn_x\nATOM 1.0\n")
    with pytest.raises(ValueError, match="mmCIF is not supported"):
        parse_cif(p)
