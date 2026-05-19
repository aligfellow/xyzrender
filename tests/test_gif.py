"""Tests for gif.py — rotation axis parsing and GIF rendering."""

from pathlib import Path

import numpy as np
import pytest

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


# ---------------------------------------------------------------------------
# _rotation_axis — unit tests (no I/O)
# ---------------------------------------------------------------------------


def test_rotation_axis_single():
    from xyzrender.gif import _rotation_axis

    ax, sign = _rotation_axis("x")
    assert np.allclose(ax, [1, 0, 0])
    assert sign == 1.0

    ax, sign = _rotation_axis("y")
    assert np.allclose(ax, [0, 1, 0])

    ax, sign = _rotation_axis("z")
    assert np.allclose(ax, [0, 0, 1])


def test_rotation_axis_negative():
    from xyzrender.gif import _rotation_axis

    ax, sign = _rotation_axis("-y")
    assert np.allclose(ax, [0, 1, 0])
    assert sign == -1.0


def test_rotation_axis_diagonal():
    from xyzrender.gif import _rotation_axis

    ax, _sign = _rotation_axis("xy")
    assert np.allclose(np.linalg.norm(ax), 1.0)
    assert ax[2] == pytest.approx(0.0)

    ax2, _ = _rotation_axis("yx")
    assert not np.allclose(ax, ax2)  # different diagonal


def test_rotation_axis_crystallographic():
    from xyzrender.gif import _rotation_axis

    lat = np.eye(3) * 5.0  # cubic lattice
    ax, _sign = _rotation_axis("111", lattice=lat)
    assert np.allclose(np.linalg.norm(ax), 1.0)
    assert np.allclose(ax, np.array([1, 1, 1]) / np.sqrt(3))


def test_rotation_axis_crystallographic_requires_lattice():
    from xyzrender.gif import _rotation_axis

    with pytest.raises(ValueError, match="lattice"):
        _rotation_axis("110")


# ---------------------------------------------------------------------------
# render_rotation_gif — integration (requires cairosvg)
# ---------------------------------------------------------------------------


def test_render_rotation_gif(tmp_path):
    pytest.importorskip("cairosvg", reason="cairosvg required")
    from xyzrender.gif import render_rotation_gif
    from xyzrender.readers import load_molecule
    from xyzrender.types import RenderConfig

    graph, _ = load_molecule(str(STRUCTURES / "caffeine.xyz"))
    cfg = RenderConfig(auto_orient=False)
    out = str(tmp_path / "rot.gif")
    render_rotation_gif(graph, cfg, out, n_frames=4, fps=5)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


# ---------------------------------------------------------------------------
# render_gif API — rotation GIF via public API
# ---------------------------------------------------------------------------


def test_api_render_gif_rotation(tmp_path):
    pytest.importorskip("cairosvg", reason="cairosvg required")
    from xyzrender import render_gif
    from xyzrender.api import GIFResult

    out = str(tmp_path / "caffeine.gif")
    result = render_gif(
        STRUCTURES / "caffeine.xyz",
        output=out,
        gif_rot="y",
        rot_frames=4,
        gif_fps=5,
        orient=False,
    )
    assert isinstance(result, GIFResult)
    assert result.path.exists()


def test_api_render_gif_bounce(tmp_path):
    pytest.importorskip("cairosvg", reason="cairosvg required")
    from xyzrender import render_gif
    from xyzrender.api import GIFResult

    out = str(tmp_path / "bounce.gif")
    result = render_gif(
        STRUCTURES / "caffeine.xyz",
        output=out,
        gif_bounce=30.0,
        rot_frames=4,
        gif_fps=5,
        orient=False,
    )
    assert isinstance(result, GIFResult)
    assert result.path.exists()
    assert result.path.stat().st_size > 0


def test_api_render_gif_bounce_axis_tuple(tmp_path):
    pytest.importorskip("cairosvg", reason="cairosvg required")
    from xyzrender import render_gif

    out = str(tmp_path / "bounce_x.gif")
    result = render_gif(
        STRUCTURES / "caffeine.xyz",
        output=out,
        gif_bounce=(30.0, "x"),
        rot_frames=4,
        gif_fps=5,
        orient=False,
    )
    assert result.path.exists()


def test_render_gif_bounce_invalid(tmp_path):
    from xyzrender import render_gif

    with pytest.raises(ValueError, match="gif_bounce must be > 0"):
        render_gif(
            STRUCTURES / "caffeine.xyz",
            output=str(tmp_path / "x.gif"),
            gif_bounce=0.0,
        )

    with pytest.raises(ValueError, match="gif_bounce must be > 0"):
        render_gif(
            STRUCTURES / "caffeine.xyz",
            output=str(tmp_path / "x.gif"),
            gif_bounce=-10.0,
        )


def test_render_gif_bounce_gif_rot_conflict(tmp_path):
    from xyzrender import render_gif

    with pytest.raises(ValueError, match="gif_bounce and gif_rot are mutually exclusive"):
        render_gif(
            STRUCTURES / "caffeine.xyz",
            output=str(tmp_path / "x.gif"),
            gif_bounce=(30.0, "x"),
            gif_rot="y",
        )


def test_render_gif_bounce_invalid_axis(tmp_path):
    from xyzrender import render_gif

    with pytest.raises(ValueError, match="invalid gif_bounce axis"):
        render_gif(
            STRUCTURES / "caffeine.xyz",
            output=str(tmp_path / "x.gif"),
            gif_bounce=(30.0, "qq"),
        )


def test_gifresult_save(tmp_path):
    pytest.importorskip("cairosvg", reason="cairosvg required")
    from xyzrender import render_gif

    src = str(tmp_path / "src.gif")
    result = render_gif(
        STRUCTURES / "caffeine.xyz",
        output=src,
        gif_rot="y",
        rot_frames=4,
        gif_fps=5,
        orient=False,
    )
    dest = tmp_path / "copy.gif"
    result.save(dest)
    assert dest.exists()
    assert dest.read_bytes() == Path(src).read_bytes()


# ---------------------------------------------------------------------------
# render_gif — gif_rot branch must forward CLI overrides to the renderer
# ---------------------------------------------------------------------------


def _tiny_molecule():
    import networkx as nx

    from xyzrender.api import Molecule

    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))
    g.add_node(1, symbol="H", position=(1.0, 0.0, 0.0))
    g.add_edge(0, 1, bond_order=1.0)
    return Molecule(graph=g)


def _capture_rotation_cfg():
    """Patch render_rotation_gif and return (context-manager, captured-dict)."""
    from unittest.mock import patch

    captured: dict = {}

    def _spy(graph=None, config=None, output=None, **_):
        captured["cfg"] = config

    return patch("xyzrender.gif.render_rotation_gif", side_effect=_spy), captured


def test_render_gif_rot_applies_vector_color(tmp_path):
    """--vector-color must reach the rotation renderer's config."""
    from xyzrender.api import render_gif
    from xyzrender.colors import resolve_color

    cm, captured = _capture_rotation_cfg()
    with cm:
        render_gif(_tiny_molecule(), gif_rot="y", vector_color="red", output=str(tmp_path / "x.gif"))
    assert captured["cfg"].vector_color == resolve_color("red")


def test_render_gif_rot_applies_surface_overrides(tmp_path):
    """--mo-pos-color / --mo-neg-color / --mo-upsample / --flat-mo / --dens-color
    must reach collect_surf_overrides when gif_rot is the only mode."""
    from unittest.mock import patch

    from xyzrender.api import render_gif

    mol = _tiny_molecule()
    mol.cube_data = object()  # truthy; consumers are patched below

    captured_kwargs: dict = {}

    def _capture(**kw):
        captured_kwargs.update(kw)
        return {}

    cm_rot, _ = _capture_rotation_cfg()
    with (
        patch("xyzrender.config.collect_surf_overrides", side_effect=_capture),
        patch("xyzrender.config.build_surface_params", return_value=(None, None, None, None)),
        cm_rot,
    ):
        render_gif(
            mol,
            gif_rot="y",
            mo=True,
            mo_pos_color="cyan",
            mo_neg_color="magenta",
            mo_upsample=2,
            flat_mo=True,
            dens_color="grey",
            output=str(tmp_path / "x.gif"),
        )
    assert captured_kwargs.get("mo_pos_color") == "cyan"
    assert captured_kwargs.get("mo_neg_color") == "magenta"
    assert captured_kwargs.get("mo_upsample") == 2
    assert captured_kwargs.get("flat_mo") is True
    assert captured_kwargs.get("dens_color") == "grey"


def test_render_gif_rot_respects_auto_align_false(tmp_path):
    """An explicit auto_align=False must override a config-level True
    even on the gif_rot branch (overlay path)."""
    from unittest.mock import patch

    from xyzrender.api import render_gif
    from xyzrender.types import RenderConfig

    mol = _tiny_molecule()
    cfg = RenderConfig()
    cfg.auto_align = True

    cm_rot, captured = _capture_rotation_cfg()
    with patch("xyzrender.api._apply_overlay", return_value=mol), cm_rot:
        render_gif(
            mol,
            config=cfg,
            gif_rot="y",
            overlay=mol,
            auto_align=False,
            output=str(tmp_path / "x.gif"),
        )
    assert captured["cfg"].auto_align is False
