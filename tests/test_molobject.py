"""Tests for loading RDKit Mol objects (api.load).

These assert the *structure* render() consumes — graph shape, bond orders,
and the ensemble payload — without running a full render.  RDKit is an
optional dependency, so the module is skipped when it is unavailable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit", reason="rdkit required for MolObject tests")

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from xyzrender import load
from xyzrender.api import EnsembleFrames


def _embed(smiles: str, *, n_confs: int = 1, seed: int = 42) -> Chem.Mol:
    """Build a hydrogen-added Mol with *n_confs* embedded 3-D conformers."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=seed)  # ty: ignore[unresolved-attribute]
    return mol


def _water_dimer() -> Chem.Mol:
    """Two waters at a ~1.84 A O-H...O hydrogen bond (deterministic coords)."""
    water = Chem.AddHs(Chem.MolFromSmiles("O"))
    dimer = Chem.RWMol(Chem.CombineMols(water, water))
    conf = Chem.Conformer(dimer.GetNumAtoms())
    coords = [
        (0.0, 0.0, 0.0),  # donor O
        (0.0, -0.96, 0.0),  # donor H (points at acceptor)
        (0.9, 0.3, 0.0),  # donor H
        (0.0, -2.8, 0.0),  # acceptor O
        (0.8, -3.3, 0.0),  # acceptor H
        (-0.8, -3.3, 0.0),  # acceptor H
    ]
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(*xyz))
    dimer.AddConformer(conf)
    return dimer.GetMol()


def test_load_single_conformer_is_render_ready():
    # A multi-conformer Mol renders ONE conformer (not an ensemble) and honours kekule.
    mol = _embed("c1ccccc1", n_confs=5)
    m = load(mol, kekule=True)
    assert m.ensemble is None
    assert m.graph.number_of_nodes() == mol.GetNumAtoms()
    assert {d["bond_order"] for *_, d in m.graph.edges(data=True)} == {1.0, 2.0}


def test_load_ensemble_payload_shape():
    mol = _embed("CCCCCC", n_confs=5)
    m = load(mol, ensemble=True)
    assert isinstance(m.ensemble, EnsembleFrames)
    assert m.ensemble.positions.shape == (5, mol.GetNumAtoms(), 3)


def test_nci_detect_adds_nci_edge():
    m = load(_water_dimer(), nci_detect=True)
    assert any(d.get("NCI") for *_, d in m.graph.edges(data=True))
