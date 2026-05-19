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


def test_render_gif_rot_applies_overlay_config_without_overlay_kwarg(tmp_path):
    """`overlay_config=` must update cfg.overlay even when `overlay=` is not passed
    (regression for PR #126).

    Mirrors render()'s behaviour at api.py:1118 — `if overlay_config is not None: cfg.overlay = overlay_config`
    runs unconditionally. PR #126 nested the same block under `if overlay is not None:` in render_gif's
    rotation branch, silently dropping the kwarg for callers who override a preset's overlay block
    without re-passing the overlay structure itself.
    """
    from xyzrender.api import render_gif
    from xyzrender.types import OverlayConfig

    cm_rot, captured = _capture_rotation_cfg()
    with cm_rot:
        render_gif(
            _tiny_molecule(),
            gif_rot="y",
            overlay_config=OverlayConfig(color="#abcdef"),
            output=str(tmp_path / "x.gif"),
        )
    assert captured["cfg"].overlay.color == "#abcdef", (
        "overlay_config kwarg silently dropped — render_gif's overlay-block writes "
        "are nested under `if overlay is not None:` instead of running unconditionally "
        "(parity with render())"
    )


def test_render_gif_rot_applies_auto_align_without_overlay_kwarg(tmp_path):
    """`auto_align=False` must update cfg.auto_align even when `overlay=` is not passed
    (regression for PR #126). Same root cause as the overlay_config test above."""
    from xyzrender.api import render_gif
    from xyzrender.types import RenderConfig

    mol = _tiny_molecule()
    cfg = RenderConfig()
    cfg.auto_align = True  # default-true; verify False overrides

    cm_rot, captured = _capture_rotation_cfg()
    with cm_rot:
        render_gif(
            mol,
            config=cfg,
            gif_rot="y",
            auto_align=False,
            output=str(tmp_path / "x.gif"),
        )
    assert captured["cfg"].auto_align is False, (
        "auto_align=False kwarg silently dropped without overlay= — render_gif "
        "nests the write under `if overlay is not None:`"
    )


def test_render_gif_ts_does_not_double_load_molecule(tmp_path):
    """`render_gif(path, gif_ts=True)` must not call `load_molecule` an extra time
    just to populate `ref_graph` (regression for PR #126).

    PR #126 hoisted `ref_graph, _ = load_molecule(str(mol_path))` to run before the
    dispatch switch. The `gif_ts` branch passes `mol_path` straight to
    `render_vibration_gif` and never reads `ref_graph`, so the extra load is wasted
    work — non-trivial for files with cube data.

    Expected call count for `xyzrender.readers.load_molecule`:
      - 1: the initial `load(molecule)` at api.py:1445 (legitimate, needed for cfg setup)
      - 2 (BUG): the extra load at api.py:1593 (the regression — should be lazy)
    """
    from unittest.mock import patch

    from xyzrender.api import render_gif

    src = STRUCTURES / "caffeine.xyz"

    with (
        patch("xyzrender.gif.render_vibration_gif") as mock_vib,
        patch(
            "xyzrender.readers.load_molecule",
            wraps=__import__("xyzrender.readers", fromlist=["load_molecule"]).load_molecule,
        ) as mock_load,
    ):
        render_gif(src, gif_ts=True, output=str(tmp_path / "x.gif"))

    mock_vib.assert_called_once()
    assert mock_load.call_count == 1, (
        f"load_molecule called {mock_load.call_count} times for gif_ts — the second call "
        "(api.py:1593) is wasted because render_vibration_gif uses mol_path directly. "
        "Move the ref_graph load into the rotation/diffuse/trajectory branches that need it."
    )


# ---------------------------------------------------------------------------
# render_gif — additional cfg-kwarg propagation through the gif_rot branch.
# These pin down the parts of cfg most likely to silently break under future
# refactors (Bug-2-shaped regressions).
# ---------------------------------------------------------------------------


def test_render_gif_rot_applies_hull_color(tmp_path):
    """`hull=True, hull_color=X` must propagate through gif_rot — PR #126 routed all
    hull setup through `_apply_hull_pore_workflow`; verify the cfg the rotation
    renderer receives still carries the hull color."""
    from xyzrender.api import render_gif

    cm_rot, captured = _capture_rotation_cfg()
    with cm_rot:
        render_gif(
            _tiny_molecule(),
            gif_rot="y",
            hull=True,
            hull_color="#deadbe",
            output=str(tmp_path / "x.gif"),
        )
    assert captured["cfg"].show_convex_hull is True
    assert "#deadbe" in captured["cfg"].hull_colors


def test_render_gif_rot_applies_radius_scale(tmp_path):
    """`radius_scale=[...]` must reach cfg.radius_scale on the gif_rot branch."""
    from xyzrender.api import render_gif

    cm_rot, captured = _capture_rotation_cfg()
    with cm_rot:
        render_gif(
            _tiny_molecule(),
            gif_rot="y",
            radius_scale=[("H", 0.5)],
            output=str(tmp_path / "x.gif"),
        )
    assert captured["cfg"].radius_scale == [("H", 0.5)]


def test_render_gif_rot_applies_glow(tmp_path):
    """`glow=[1]` (1-indexed) must reach cfg.glow_indices as 0-indexed on the gif_rot branch."""
    from xyzrender.api import render_gif

    cm_rot, captured = _capture_rotation_cfg()
    with cm_rot:
        render_gif(
            _tiny_molecule(),
            gif_rot="y",
            glow=[1, 2],
            output=str(tmp_path / "x.gif"),
        )
    assert captured["cfg"].glow_indices == [0, 1]


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


def test_render_gif_rot_accepts_align_atoms_kwarg(tmp_path):
    """render_gif must accept `align_atoms=[…]` and propagate it to `_apply_overlay`
    (parity with render() — see api.py:1132)."""
    from unittest.mock import patch

    from xyzrender.api import render_gif

    captured_align: dict = {}

    def _spy_overlay(_base, _ov, _cfg, _ov_arg, **kwargs):
        captured_align["align_atoms"] = kwargs.get("align_atoms")
        return _base

    cm_rot, _ = _capture_rotation_cfg()
    mol = _tiny_molecule()
    with patch("xyzrender.api._apply_overlay", side_effect=_spy_overlay), cm_rot:
        render_gif(
            mol,
            gif_rot="y",
            overlay=mol,
            output=str(tmp_path / "x.gif"),
            align_atoms=[1, 2],
        )
    assert captured_align["align_atoms"] == [1, 2], (
        f"render_gif accepted align_atoms but didn't propagate to _apply_overlay: "
        f"got {captured_align.get('align_atoms')!r}"
    )
