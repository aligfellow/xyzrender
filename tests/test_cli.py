"""Tests for CLI helpers and dispatch validation."""

import subprocess
import sys
from pathlib import Path

import pytest

from xyzrender.cli import _basename, _parse_pairs

_STRUCTURES = Path(__file__).resolve().parent.parent / "examples" / "structures"
_CAFFEINE = _STRUCTURES / "caffeine.xyz"


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


def test_gif_vib_dispatches_mode_and_scale(monkeypatch, tmp_path):
    import sys
    from unittest.mock import patch

    from xyzrender import api
    from xyzrender.cli import main

    source = _STRUCTURES / "sn2.out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xyzrender",
            str(source),
            "--gif-vib",
            "7",
            "--vib-scale",
            "2.5",
            "--vib-label",
            "-o",
            str(tmp_path / "mode.svg"),
            "-go",
            str(tmp_path / "mode.gif"),
        ],
    )

    with patch.object(api, "render"), patch.object(api, "render_gif") as mock_gif:
        main()

    assert mock_gif.call_args.kwargs["gif_vib"] == 7
    assert mock_gif.call_args.kwargs["vib_scale"] == 2.5
    assert mock_gif.call_args.kwargs["vib_label"] is True
    assert mock_gif.call_args.kwargs["hy"] is None
    assert mock_gif.call_args.kwargs["no_hy"] is False


def test_gif_vib_forwards_no_hy_override(monkeypatch, tmp_path):
    import sys
    from unittest.mock import patch

    from xyzrender import api
    from xyzrender.cli import main

    source = _STRUCTURES / "sn2.out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xyzrender",
            str(source),
            "--gif-vib",
            "0",
            "--no-hy",
            "-o",
            str(tmp_path / "mode.svg"),
            "-go",
            str(tmp_path / "mode.gif"),
        ],
    )

    with patch.object(api, "render"), patch.object(api, "render_gif") as mock_gif:
        main()

    assert mock_gif.call_args.kwargs["no_hy"] is True


@pytest.mark.parametrize(
    "args",
    [
        ("--gif-vib", "-1"),
        ("--gif-vib", "0", "--vib-scale", "0"),
        ("--gif-vib", "0", "--vib-scale", "nan"),
        ("--gif-vib", "0", "--vib-scale", "inf"),
        ("--gif-vib", "0", "--vib-frames", "3"),
    ],
)
def test_gif_vib_rejects_invalid_options(args):
    result = _run_cli(str(_CAFFEINE), *args, expect_error=True)
    assert "error:" in result.stderr


def test_hl_too_many_args():
    """--hl with >2 arguments should error."""
    result = _run_cli(str(_CAFFEINE), "--hl", "1-5", "red", "extra", expect_error=True)
    assert result.returncode != 0
