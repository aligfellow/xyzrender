from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from xyzrender import SVGResult, load, render
from xyzrender.api import EnsembleFrames, _build_ensemble_molecule, _filter_molecule_atoms
from xyzrender.ensemble import align, merge_graphs
from xyzrender.merge import _Z_NUDGE

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


def _make_triatomic_traj(tmp_path: Path) -> Path:
    frames = [
        [("C", (0.0, 0.0, 0.0)), ("H", (1.0, 0.0, 0.0)), ("H", (0.0, 1.0, 0.0))],
        [("C", (0.1, 0.0, 0.0)), ("H", (1.1, 0.1, 0.0)), ("H", (0.1, 1.0, 0.1))],
        [("C", (-0.1, 0.0, 0.0)), ("H", (0.9, -0.1, 0.0)), ("H", (-0.1, 1.0, -0.1))],
    ]
    xyz_path = tmp_path / "triatomic_traj.xyz"
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

    # Default: spectral palette → non-None hex per conformer
    assert all(c is not None and c.startswith("#") for c in ens.colors)


def test_load_ensemble_detects_ncis_for_each_conformer() -> None:
    mol = load(
        "examples/structures/bimp.v000.xyz",
        ensemble=True,
        max_frames=2,
        nci_detect=True,
    )
    assert mol.ensemble is not None
    assert mol.ensemble.conformer_graphs is not None

    nci_types = [
        {data["nci_type"] for *_edge, data in graph.edges(data=True) if data.get("NCI")}
        for graph in mol.ensemble.conformer_graphs
    ]
    assert "hbond" not in nci_types[0]
    assert "hbond" in nci_types[1]

    merged = merge_graphs(
        mol.graph,
        mol.ensemble.positions,
        conformer_graphs=mol.ensemble.conformer_graphs,
        z_nudge=False,
    )
    assert all(
        any(data.get("NCI") and data.get("molecule_index") == frame_idx for *_edge, data in merged.edges(data=True))
        for frame_idx in range(2)
    )


def test_rebuilt_ensemble_nci_centroids_use_aligned_positions() -> None:
    import numpy as np

    mol = load(
        "examples/structures/bimp.v000.xyz",
        ensemble=True,
        max_frames=2,
        rebuild=True,
        nci_detect=True,
    )
    assert mol.ensemble is not None
    assert mol.ensemble.conformer_graphs is not None

    for frame_idx, graph in enumerate(mol.ensemble.conformer_graphs):
        for centroid, sites in graph.graph.get("nci_centroid_sites", {}).items():
            expected = mol.ensemble.positions[frame_idx][list(sites)].mean(axis=0)
            assert np.allclose(graph.nodes[centroid]["position"], expected)


def test_rebuilt_ensemble_detects_ncis_on_supplied_reference() -> None:
    path = "examples/structures/bimp.v000.xyz"
    mol = load(
        path,
        ensemble=True,
        max_frames=2,
        rebuild=True,
        nci_detect=True,
        reference_mol=load(path),
    )
    assert mol.ensemble is not None
    assert mol.ensemble.conformer_graphs is not None
    reference_graph = mol.ensemble.conformer_graphs[mol.ensemble.reference_idx]
    assert any(data.get("NCI") for *_edge, data in reference_graph.edges(data=True))


def test_ensemble_opacity(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, ensemble_opacity=0.4)
    ens = mol.ensemble
    assert ens is not None

    assert ens.opacities[ens.reference_idx] is None
    for i, op in enumerate(ens.opacities):
        if i != ens.reference_idx:
            assert op == 0.4


def test_ensemble_no_align_preserves_raw_positions(tmp_path: Path) -> None:
    """auto_align=False uses each frame's raw coordinates — no Kabsch."""
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, auto_align=False)
    ens = mol.ensemble
    assert ens is not None

    import numpy as np

    from xyzrender.readers import load_trajectory_frames

    raw = load_trajectory_frames(xyz_path)
    for fi, fr in enumerate(raw):
        expected = np.array(fr["positions"], dtype=float)
        assert np.allclose(ens.positions[fi], expected)


def test_ensemble_palette_colors(tmp_path: Path) -> None:
    """Explicit palette → non-None hex colors for all conformers."""
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, ensemble_color="viridis")
    ens = mol.ensemble
    assert ens is not None

    assert all(c is not None and c.startswith("#") for c in ens.colors)


def test_ensemble_cpk_colors(tmp_path: Path) -> None:
    """'cpk' → no palette override; all conformer colors None (CPK atom colours)."""
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, ensemble_color="cpk")
    ens = mol.ensemble
    assert ens is not None

    assert all(c is None for c in ens.colors)


def test_ensemble_single_color_expanded(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, ensemble_color="#FF0000")
    ens = mol.ensemble
    assert ens is not None

    assert all(c == "#ff0000" for c in ens.colors)


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
    mol = _build_ensemble_molecule(xyz_path, ensemble_color="viridis")
    ens = mol.ensemble
    assert ens is not None

    g = merge_graphs(mol.graph, ens.positions, conformer_colors=ens.colors)

    assert g.number_of_nodes() == 6  # 3 conformers x 2 atoms
    assert g.number_of_edges() == 3  # one bond per conformer

    # Every edge connects two atoms with the same molecule_index
    for i, j, d in g.edges(data=True):
        assert g.nodes[i]["molecule_index"] == g.nodes[j]["molecule_index"] == d["molecule_index"]

    # With palette: reference atoms get no structure_color; non-reference atoms do
    ref = [n for n in g.nodes() if g.nodes[n]["molecule_index"] == 0]
    non_ref = [n for n in g.nodes() if g.nodes[n]["molecule_index"] > 0]
    assert all("structure_color" not in g.nodes[n] for n in ref)
    assert all(g.nodes[n].get("structure_color", "").startswith("#") for n in non_ref)


def test_merge_graphs_preserves_each_conformers_nci_centroids() -> None:
    import networkx as nx
    import numpy as np

    def _frame_graph(nci_type: str, centroid_x: float) -> nx.Graph:
        graph = nx.Graph()
        graph.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))
        graph.add_node(1, symbol="C", position=(2.0, 0.0, 0.0))
        graph.add_node(2, symbol="*", position=(centroid_x, 0.0, 0.0))
        graph.add_edge(0, 2, NCI=True, nci_type=nci_type)
        graph.add_edge(2, 1, NCI=True, nci_type=nci_type)
        graph.graph["nci_centroid"] = [2]
        graph.graph["nci_centroid_sites"] = {2: (0, 1)}
        return graph

    frame_graphs = [_frame_graph("pi_frame_0", 1.0), _frame_graph("pi_frame_1", 2.0)]
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ]
    )

    merged = merge_graphs(
        frame_graphs[1],
        positions,
        conformer_opacities=[None, 0.25],
        conformer_graphs=frame_graphs,
        z_nudge=False,
    )

    for frame_idx, expected_type in enumerate(("pi_frame_0", "pi_frame_1")):
        centroids = [
            node
            for node, data in merged.nodes(data=True)
            if data.get("molecule_index") == frame_idx and data.get("symbol") == "*"
        ]
        assert len(centroids) == 1
        expected_opacity = None if frame_idx == 0 else 0.25
        assert merged.nodes[centroids[0]].get("structure_opacity") == expected_opacity
        nci_types = {
            data["nci_type"]
            for *_edge, data in merged.edges(data=True)
            if data.get("molecule_index") == frame_idx and data.get("NCI")
        }
        assert nci_types == {expected_type}


def test_merge_graphs_cpk_no_structure_color(tmp_path: Path) -> None:
    """'cpk': merge_graphs sets no structure_color override — renderer falls back to CPK."""
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, ensemble_color="cpk")
    ens = mol.ensemble
    assert ens is not None

    g = merge_graphs(mol.graph, ens.positions, conformer_colors=ens.colors)
    assert all("structure_color" not in g.nodes[n] for n in g.nodes())


def test_merge_graphs_bond_color_override(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, ensemble_color="#FF0000")
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
    mol = load(xyz_path, ensemble=True, ensemble_color="spectral", ensemble_opacity=0.5)
    result = render(mol, output=tmp_path / "out.svg")
    assert isinstance(result, SVGResult)
    assert "<svg" in (tmp_path / "out.svg").read_text()


def test_ensemble_rotation_gif_preserves_per_conformer_ncis(tmp_path: Path) -> None:
    from unittest.mock import patch

    from xyzrender import render_gif

    mol = load(
        "examples/structures/bimp.v000.xyz",
        ensemble=True,
        max_frames=2,
        nci_detect=True,
    )

    with patch("xyzrender.gif.render_rotation_gif") as mock_render:
        render_gif(mol, gif_rot="y", output=tmp_path / "ensemble.gif")

    rendered_graph = mock_render.call_args.kwargs["graph"]
    frame_one_types = {
        data["nci_type"]
        for *_edge, data in rendered_graph.edges(data=True)
        if data.get("molecule_index") == 1 and data.get("NCI")
    }
    assert "hbond" in frame_one_types


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


def test_ensemble_exclude_filters_all_frames(tmp_path: Path) -> None:
    xyz_path = _make_triatomic_traj(tmp_path)
    mol = load(xyz_path, ensemble=True)

    filtered = _filter_molecule_atoms(mol, exclude="2")

    assert filtered.graph.number_of_nodes() == 2
    assert filtered.ensemble is not None
    assert filtered.ensemble.positions.shape == (3, 2, 3)
    assert [filtered.graph.nodes[n]["symbol"] for n in filtered.graph.nodes()] == ["C", "H"]
    result = render(filtered, output=tmp_path / "filtered.svg")
    assert isinstance(result, SVGResult)


def test_build_ensemble_with_filtered_reference_filters_trajectory_frames(tmp_path: Path) -> None:
    xyz_path = _make_triatomic_traj(tmp_path)
    ref = _filter_molecule_atoms(load(xyz_path), exclude="2")

    mol = _build_ensemble_molecule(xyz_path, reference_mol=ref)

    assert mol.graph.number_of_nodes() == 2
    assert mol.ensemble is not None
    assert mol.ensemble.positions.shape == (3, 2, 3)


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
