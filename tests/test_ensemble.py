from __future__ import annotations

from typing import TYPE_CHECKING

from xyzrender import SVGResult, load, render
from xyzrender.api import _build_ensemble_molecule

if TYPE_CHECKING:
    from pathlib import Path


def _write_multiframe_xyz(path: Path, frames: list[list[tuple[str, tuple[float, float, float]]]]) -> None:
    """Write a simple multi-frame XYZ file for testing."""
    lines: list[str] = []
    for frame in frames:
        lines.append(f"{len(frame)}\n")
        lines.append("test frame\n")
        for sym, (x, y, z) in frame:
            lines.append(f"{sym:<3} {x:15.8f} {y:15.8f} {z:15.8f}\n")
    path.write_text("".join(lines))


def _make_traj(tmp_path: Path) -> Path:
    base = [
        ("H", (0.0, 0.0, 0.0)),
        ("O", (0.0, 0.0, 1.0)),
    ]
    frames = [
        base,
        [("H", (0.1, 0.0, 0.0)), ("O", (0.0, 0.1, 1.0))],
        [("H", (-0.1, 0.0, 0.0)), ("O", (0.0, -0.1, 1.0))],
    ]
    xyz_path = tmp_path / "traj.xyz"
    _write_multiframe_xyz(xyz_path, frames)
    return xyz_path


def test_build_ensemble_molecule(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path)
    g = mol.graph

    # Three conformers x 2 atoms each
    assert g.number_of_nodes() == 6
    assert g.number_of_edges() == 3  # one bond per conformer

    # All atoms carry molecule_index and no overlay flag
    molecule_indices = {data["molecule_index"] for _, data in g.nodes(data=True)}
    assert molecule_indices == {0, 1, 2}
    assert all("overlay" not in data for _, data in g.nodes(data=True))

    # All bonds share the same molecule_index as their atoms
    for i, j, d in g.edges(data=True):
        mi = g.nodes[i]["molecule_index"]
        mj = g.nodes[j]["molecule_index"]
        assert mi == mj == d["molecule_index"]

    # Default palette (viridis) applies colours to non-reference conformers
    non_ref = [n for n in g.nodes() if g.nodes[n]["molecule_index"] > 0]
    assert all(g.nodes[n].get("ensemble_color") for n in non_ref)
    ref = [n for n in g.nodes() if g.nodes[n]["molecule_index"] == 0]
    assert all("ensemble_color" not in g.nodes[n] for n in ref)


def test_ensemble_via_render_returns_svg(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    mol = load(xyz_path, ensemble=True)
    result = render(mol, output=tmp_path / "out.svg")
    assert isinstance(result, SVGResult)
    out_path = tmp_path / "out.svg"
    assert out_path.exists()
    svg = out_path.read_text()
    assert "<svg" in svg


def test_ensemble_with_palette(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, ensemble_palette="viridis")
    g = mol.graph

    # Non-reference conformer nodes should have ensemble_color set
    non_ref = [n for n in g.nodes() if g.nodes[n]["molecule_index"] > 0]
    assert all(g.nodes[n].get("ensemble_color") for n in non_ref)
    # Reference conformer should NOT have ensemble_color
    ref = [n for n in g.nodes() if g.nodes[n]["molecule_index"] == 0]
    assert all("ensemble_color" not in g.nodes[n] for n in ref)


def test_ensemble_with_single_color(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, conformer_colors=["#FF0000"])
    g = mol.graph

    non_ref = [n for n in g.nodes() if g.nodes[n]["molecule_index"] > 0]
    assert all(g.nodes[n].get("ensemble_color") == "#FF0000" for n in non_ref)
    # Bonds should have bond_color_override
    non_ref_edges = [(i, j, d) for i, j, d in g.edges(data=True) if d["molecule_index"] > 0]
    assert all(d.get("bond_color_override", "").startswith("#") for _, _, d in non_ref_edges)


def test_ensemble_opacity(tmp_path: Path) -> None:
    xyz_path = _make_traj(tmp_path)
    mol = _build_ensemble_molecule(xyz_path, ensemble_opacity=0.4)
    g = mol.graph

    non_ref = [n for n in g.nodes() if g.nodes[n]["molecule_index"] > 0]
    assert all(g.nodes[n].get("ensemble_opacity") == 0.4 for n in non_ref)
    ref = [n for n in g.nodes() if g.nodes[n]["molecule_index"] == 0]
    assert all("ensemble_opacity" not in g.nodes[n] for n in ref)


def test_ensemble_palette_render(tmp_path: Path) -> None:
    """Ensemble with palette colours renders without error."""
    xyz_path = _make_traj(tmp_path)
    mol = load(xyz_path, ensemble=True, ensemble_palette="spectral", ensemble_opacity=0.5)
    result = render(mol, output=tmp_path / "palette.svg")
    assert isinstance(result, SVGResult)
    assert (tmp_path / "palette.svg").exists()
