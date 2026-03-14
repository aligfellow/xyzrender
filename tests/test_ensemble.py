from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from xyzrender import SVGResult, load, render
from xyzrender.api import EnsembleFrames, _build_ensemble_molecule
from xyzrender.ensemble import _Z_NUDGE, align, merge_graphs

if TYPE_CHECKING:
    from pathlib import Path


def _write_multiframe_xyz(path: Path, frames: list[list[tuple[str, tuple[float, float, float]]]]) -> None:
    lines: list[str] = []
    for frame in frames:
        lines.append(f"{len(frame)}\n")
        lines.append("test frame\n")
        for sym, (x, y, z) in frame:
            lines.append(f"{sym:<3} {x:15.8f} {y:15.8f} {z:15.8f}\n")
    path.write_text("".join(lines))


def _make_traj(tmp_path: Path) -> Path:
    frames = [
        [("H", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.0))],
        [("H", (0.1, 0.0, 0.0)), ("O", (0.0, 0.1, 1.0))],
        [("H", (-0.1, 0.0, 0.0)), ("O", (0.0, -0.1, 1.0))],
    ]
    xyz_path = tmp_path / "traj.xyz"
    _write_multiframe_xyz(xyz_path, frames)
    return xyz_path


# ---------------------------------------------------------------------------
# EnsembleFrames structure
# ---------------------------------------------------------------------------


def test_build_ensemble_molecule(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path)

    assert mol.graph.number_of_nodes() == 2
    assert mol.graph.number_of_edges() == 1
    assert all("overlay" not in data for _, data in mol.graph.nodes(data=True))

    assert mol.ensemble is not None
    ens = mol.ensemble
    assert isinstance(ens, EnsembleFrames)
    assert ens.reference_idx == 0
    assert ens.positions.shape == (3, 2, 3)
    assert len(ens.colors) == 3
    assert len(ens.opacities) == 3

    # Default: no palette → CPK atom colors → all colors None
    assert all(c is None for c in ens.colors)


def test_ensemble_opacity(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, ensemble_opacity=0.4)
    ens = mol.ensemble
    assert ens is not None

    assert ens.opacities[ens.reference_idx] is None
    for i, op in enumerate(ens.opacities):
        if i != ens.reference_idx:
            assert op == 0.4


def test_ensemble_palette_colors(tmp_path: Path) -> None:
    """Explicit palette → non-None hex colors for all conformers."""
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, ensemble_palette="viridis")
    ens = mol.ensemble
    assert ens is not None

    assert all(c is not None and c.startswith("#") for c in ens.colors)


def test_ensemble_single_color_expanded(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, conformer_colors=["#FF0000"])
    ens = mol.ensemble
    assert ens is not None

    assert all(c == "#FF0000" for c in ens.colors)


def test_ensemble_reference_frame_nonzero(tmp_path: Path) -> None:
    """Frame 1 as reference: its positions are unchanged; other frames align onto it."""
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, reference_frame=1)
    ens = mol.ensemble
    assert ens is not None

    assert ens.reference_idx == 1
    # Reference frame positions must be exact (no rotation applied)
    ref_pos_stored = ens.positions[1]
    # Frame 1 in the file: H=(0.1,0,0), O=(0,0.1,1)
    assert ref_pos_stored[0, 0] == pytest.approx(0.1)
    assert ref_pos_stored[1, 1] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# merge_graphs
# ---------------------------------------------------------------------------


def test_merge_graphs_structure(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, ensemble_palette="viridis")
    ens = mol.ensemble
    assert ens is not None

    g = merge_graphs(mol.graph, ens.positions, conformer_colors=ens.colors)

    assert g.number_of_nodes() == 6  # 3 conformers x 2 atoms
    assert g.number_of_edges() == 3  # one bond per conformer

    # Every edge connects two atoms with the same molecule_index
    for i, j, d in g.edges(data=True):
        assert g.nodes[i]["molecule_index"] == g.nodes[j]["molecule_index"] == d["molecule_index"]

    # With palette: reference atoms get no ensemble_color; non-reference atoms do
    ref = [n for n in g.nodes() if g.nodes[n]["molecule_index"] == 0]
    non_ref = [n for n in g.nodes() if g.nodes[n]["molecule_index"] > 0]
    assert all("ensemble_color" not in g.nodes[n] for n in ref)
    assert all(g.nodes[n].get("ensemble_color", "").startswith("#") for n in non_ref)


def test_merge_graphs_no_colors(tmp_path: Path) -> None:
    """Default (no palette): merge_graphs sets no ensemble_color — CPK used by renderer."""
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path)
    ens = mol.ensemble
    assert ens is not None

    g = merge_graphs(mol.graph, ens.positions, conformer_colors=ens.colors)
    assert all("ensemble_color" not in g.nodes[n] for n in g.nodes())


def test_merge_graphs_bond_color_override(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, conformer_colors=["#FF0000"])
    ens = mol.ensemble
    assert ens is not None

    g = merge_graphs(mol.graph, ens.positions, conformer_colors=ens.colors)
    non_ref_edges = [d for _, _, d in g.edges(data=True) if d["molecule_index"] > 0]
    assert all(d.get("bond_color_override", "").startswith("#") for d in non_ref_edges)
    ref_edges = [d for _, _, d in g.edges(data=True) if d["molecule_index"] == 0]
    assert all("bond_color_override" not in d for d in ref_edges)


def test_merge_graphs_z_nudge(tmp_path: Path) -> None:
    """z_nudge=True offsets conformer z-coords; z_nudge=False leaves them exact."""
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path)
    ens = mol.ensemble
    assert ens is not None

    g_nudge = merge_graphs(mol.graph, ens.positions, z_nudge=True)
    g_flat = merge_graphs(mol.graph, ens.positions, z_nudge=False)

    for n in g_nudge.nodes():
        conf_idx = g_nudge.nodes[n]["molecule_index"]
        z_nudge_val = g_nudge.nodes[n]["position"][2]
        z_flat_val = g_flat.nodes[n]["position"][2]
        if conf_idx == 0:
            assert z_nudge_val == pytest.approx(z_flat_val)
        else:
            assert z_nudge_val == pytest.approx(z_flat_val + conf_idx * _Z_NUDGE)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def test_ensemble_render_produces_svg(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    mol = load(xyz_path, ensemble=True, ensemble_palette="spectral", ensemble_opacity=0.5)
    result = render(mol, output=tmp_path / "out.svg")
    assert isinstance(result, SVGResult)
    assert "<svg" in (tmp_path / "out.svg").read_text()


def test_ensemble_render_twice_no_mutation(tmp_path: Path) -> None:
    """render() must not mutate mol — second call must produce an identical graph."""
    xyz_path = _make_traj(tmp_path)
    mol = load(xyz_path, ensemble=True)
    n_nodes = mol.graph.number_of_nodes()
    node_attrs_before = {n: dict(mol.graph.nodes[n]) for n in mol.graph.nodes()}

    render(mol, output=tmp_path / "out1.svg")
    render(mol, output=tmp_path / "out2.svg")

    assert mol.graph.number_of_nodes() == n_nodes
    assert mol.ensemble is not None
    for n in mol.graph.nodes():
        assert mol.graph.nodes[n] == node_attrs_before[n]
    assert "<svg" in (tmp_path / "out1.svg").read_text()
    assert "<svg" in (tmp_path / "out2.svg").read_text()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_ensemble_align_mismatched_atoms() -> None:
    """align() raises when a frame has a different atom count than the reference."""
    frames = [
        {"symbols": ["H", "O"], "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]},
        {"symbols": ["H"], "positions": [[0.1, 0.0, 0.0]]},
    ]
    with pytest.raises(ValueError, match="shape"):
        align(frames)


def test_ensemble_align_out_of_range_reference(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    with pytest.raises(ValueError, match="reference_frame"):
        _build_ensemble_molecule(xyz_path, reference_frame=99)
