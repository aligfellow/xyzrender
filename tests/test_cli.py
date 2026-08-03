"""Tests for CLI helpers and dispatch validation."""

import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from xyzrender.cli import _basename, _parse_pairs
from xyzrender.protein_semantics import xyzgraph_protein_available

STRUCTURES = Path(__file__).resolve().parent.parent / "examples" / "structures"
_STRUCTURES = STRUCTURES  # alias: both spellings are used across this module
_CAFFEINE = STRUCTURES / "caffeine.xyz"


def test_basename_from_xyz():
    assert _basename("molecule.xyz", from_stdin=False) == "molecule"


def test_basename_from_path():
    assert _basename("/path/to/caffeine.xyz", from_stdin=False) == "caffeine"


def test_basename_from_out_file():
    assert _basename("calc.out", from_stdin=False) == "calc"


def test_basename_stdin():
    assert _basename(None, from_stdin=True) == "graphic"


def test_basename_stdin_overrides_input():
    assert _basename("molecule.xyz", from_stdin=True) == "graphic"


def test_basename_none_not_stdin():
    assert _basename(None, from_stdin=False) == "graphic"


# ---------------------------------------------------------------------------
# _parse_pairs
# ---------------------------------------------------------------------------


def test_parse_pairs_single():
    assert _parse_pairs("1-6") == [(0, 5)]


def test_parse_pairs_multiple():
    assert _parse_pairs("1-6,3-4") == [(0, 5), (2, 3)]


def test_parse_pairs_empty():
    assert _parse_pairs("") == []
    assert _parse_pairs("   ") == []


def _dummy_molecule():
    from xyzrender.api import Molecule

    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))
    return Molecule(graph=g)


def _dummy_cell_molecule():
    from xyzrender.api import Molecule
    from xyzrender.types import CellData

    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))
    return Molecule(graph=g, cell_data=CellData(lattice=np.eye(3), cell_origin=np.zeros(3)))


def test_cli_protein_without_style_defaults_to_gloss(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--protein"])

    cli.main()
    assert captured["protein"] == "gloss"


def test_cli_protein_without_ghost_flag_defaults_ghosts_off(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_cell_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--protein"])

    cli.main()
    assert captured["ghosts"] is False


def test_cli_protein_with_ghosts_flag_overrides_default(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_cell_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--protein", "--ghosts"])

    cli.main()
    assert captured["ghosts"] is True


def test_cli_non_protein_cell_input_keeps_ghosts_default_on(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_cell_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb")])

    cli.main()
    assert captured["ghosts"] is True


def test_cli_protein_gloss_style(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--protein", "gloss"])

    cli.main()
    assert captured["protein"] == "gloss"


def test_cli_protein_illustration_style(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--protein", "illustration"])

    cli.main()
    assert captured["protein"] == "illustration"


def test_cli_protein_removed_style_fails(monkeypatch):
    import sys

    from xyzrender import cli

    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--protein", "plastic"])
    with pytest.raises(SystemExit):
        cli.main()


@pytest.mark.skipif(
    not xyzgraph_protein_available(),
    reason="installed xyzgraph does not provide xyzgraph.protein",
)
def test_cli_accepts_the_cartoon_alias(monkeypatch, tmp_path):
    import sys

    from xyzrender import cli

    out = tmp_path / "out.svg"
    monkeypatch.setattr(
        sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--protein", "cartoon", "-o", str(out)]
    )
    cli.main()
    assert out.exists()


def test_cli_highlight_ligand_flag_wires_to_render(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--highlight-ligand"])

    cli.main()
    assert captured["ligand_highlight"] is True


def test_cli_ligand_highlight_flag_removed(monkeypatch):
    import sys

    from xyzrender import cli

    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--ligand-highlight"])
    with pytest.raises(SystemExit):
        cli.main()


def test_cli_glow_ligand_wires_to_render(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--glow", "ligand"])

    cli.main()
    assert captured["glow"] == [("ligand",)]


def test_cli_nci_ligand_implies_nci_detect_on_load(monkeypatch):
    import sys

    from xyzrender import cli

    captured_load: dict = {}

    def _fake_load(*args, **kwargs):
        captured_load.update(kwargs)
        return _dummy_molecule()

    monkeypatch.setattr("xyzrender.api.load", _fake_load)
    monkeypatch.setattr("xyzrender.api.render", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--nci-ligand"])

    cli.main()
    assert captured_load["nci_detect"] is True
    assert captured_load["nci_ligand_protein_only"] is True


def test_cli_nci_ligand_implies_nci_detect_for_gif(monkeypatch, tmp_path):
    import sys

    from xyzrender import cli

    captured_gif: dict = {}

    def _fake_render_gif(*args, **kwargs):
        captured_gif.update(kwargs)

    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_molecule())
    monkeypatch.setattr("xyzrender.api.render", lambda *args, **kwargs: None)
    monkeypatch.setattr("xyzrender.api.render_gif", _fake_render_gif)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xyzrender",
            str(STRUCTURES / "water.pdb"),
            "--nci-ligand",
            "--gif-rot",
            "--gif-output",
            str(tmp_path / "out.gif"),
        ],
    )

    cli.main()
    assert captured_gif["detect_nci"] is True
    assert captured_gif["nci_ligand_protein_only"] is True


def test_cli_nci_ligand_protein_only_flag_removed(monkeypatch):
    import sys

    from xyzrender import cli

    monkeypatch.setattr(
        sys,
        "argv",
        ["xyzrender", str(STRUCTURES / "water.pdb"), "--nci-ligand-protein-only"],
    )
    with pytest.raises(SystemExit):
        cli.main()


def test_cli_glow_ligand_wires_to_render_gif(monkeypatch, tmp_path):
    import sys

    from xyzrender import cli

    captured_gif: dict = {}

    def _fake_render_gif(*args, **kwargs):
        captured_gif.update(kwargs)

    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_molecule())
    monkeypatch.setattr("xyzrender.api.render", lambda *args, **kwargs: None)
    monkeypatch.setattr("xyzrender.api.render_gif", _fake_render_gif)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xyzrender",
            str(STRUCTURES / "water.pdb"),
            "--glow",
            "ligand",
            "--gif-rot",
            "--gif-output",
            str(tmp_path / "out.gif"),
        ],
    )

    cli.main()
    assert captured_gif["glow"] == [("ligand",)]


# ---------------------------------------------------------------------------
# CLI dispatch: argparse namespace validation
# ---------------------------------------------------------------------------


def _run_cli(*args: str, expect_error: bool = False) -> subprocess.CompletedProcess:
    """Run xyzrender CLI as a subprocess and return the result."""
    result = subprocess.run(
        [sys.executable, "-c", "from xyzrender.cli import main; main()", *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if expect_error:
        assert result.returncode != 0, f"Expected error but got rc=0: {result.stdout}"
    return result


def test_version_flag():
    result = _run_cli("--version")
    assert result.returncode == 0
    assert "xyzrender" in result.stdout


def test_compact_help():
    result = _run_cli("-h")
    assert result.returncode == 0
    assert "Run 'xyzrender --help' for full details" in result.stdout


def test_full_help():
    result = _run_cli("--help")
    assert result.returncode == 0
    # Full argparse help includes "usage:" header
    assert "usage:" in result.stdout
    assert "--nci-surf" in result.stdout
    assert "--igmh-surf" not in result.stdout


@pytest.mark.skipif(not _CAFFEINE.exists(), reason="fixture not found")
def test_basic_render(tmp_path):
    out = tmp_path / "test.svg"
    result = _run_cli(str(_CAFFEINE), "-o", str(out))
    assert result.returncode == 0
    assert out.exists()
    assert out.read_text().startswith("<?xml") or out.read_text().startswith("<svg")


@pytest.mark.skipif(not (_STRUCTURES / "caffeine_dens.cube").exists(), reason="fixture not found")
def test_cub_density_surface(tmp_path):
    src = _STRUCTURES / "caffeine_dens.cube"
    cub = tmp_path / "caffeine_dens.cub"
    cub.write_bytes(src.read_bytes())
    out = tmp_path / "dens.svg"

    result = _run_cli(str(cub), "--dens", "-o", str(out))

    assert result.returncode == 0
    assert out.exists()


@pytest.mark.skipif(
    not ((_STRUCTURES / "caffeine_dens.cube").exists() and (_STRUCTURES / "caffeine_esp.cube").exists()),
    reason="fixtures not found",
)
def test_cli_esp_uses_shared_cmap_palette(tmp_path):
    dens = _STRUCTURES / "caffeine_dens.cube"
    esp = _STRUCTURES / "caffeine_esp.cube"
    out = tmp_path / "esp.svg"

    result = _run_cli(str(dens), "--esp", str(esp), "--cmap-palette", "coolwarm", "--cbar", "-o", str(out))

    assert result.returncode == 0
    svg = out.read_text()
    assert "#b40426" in svg
    assert "#3b4cc0" in svg


def test_no_input_error():
    result = _run_cli(expect_error=True)
    assert result.returncode != 0


def test_ensemble_overlay_incompatible():
    """--ensemble + --overlay should error."""
    result = _run_cli(str(_CAFFEINE), "--ensemble", "--overlay", str(_CAFFEINE), expect_error=True)
    assert result.returncode != 0


def test_gif_diffuse_ts_incompatible():
    """--gif-diffuse + --gif-ts should error."""
    result = _run_cli(str(_CAFFEINE), "--gif-diffuse", "--gif-ts", expect_error=True)
    assert result.returncode != 0


def test_gif_ts_manual_bonds_skip_auto_ts_load(monkeypatch, tmp_path):
    import sys
    from unittest.mock import patch

    from xyzrender import api
    from xyzrender.cli import main

    source = _STRUCTURES / "sn2.out"
    argv = [
        "xyzrender",
        str(source),
        "--gif-ts",
        "--ts-bond",
        "1-3",
        "-o",
        str(tmp_path / "ts.svg"),
        "-go",
        str(tmp_path / "ts.gif"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with (
        patch.object(api, "load", wraps=api.load) as mock_load,
        patch.object(api, "render"),
        patch.object(api, "render_gif"),
    ):
        main()

    assert mock_load.call_args_list[0].kwargs["ts_detect"] is False


def test_hl_too_many_args():
    """--hl with >2 arguments should error."""
    result = _run_cli(str(_CAFFEINE), "--hl", "1-5", "red", "extra", expect_error=True)
    assert result.returncode != 0
