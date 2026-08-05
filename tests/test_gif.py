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
# trj_bonds — per-frame bond detection
# ---------------------------------------------------------------------------


def test_orient_frames_preserves_extra_keys():
    """_orient_frames must keep graph / bond_opacities / hull_opacity_factor."""
    from xyzrender.gif import _orient_frames

    frame = {
        "symbols": ["C", "H"],
        "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "graph": object(),
        "bond_opacities": {(0, 1): 0.5},
        "hull_opacity_factor": 0.8,
    }
    out = _orient_frames([frame], np.eye(3))
    assert out[0]["graph"] is frame["graph"]
    assert out[0]["bond_opacities"] == {(0, 1): 0.5}
    assert out[0]["hull_opacity_factor"] == 0.8


def test_rotate_frames_preserves_extra_keys():
    """_rotate_frames must keep graph / bond_opacities / hull_opacity_factor."""
    from xyzrender.gif import _rotate_frames

    frame = {
        "symbols": ["C", "H"],
        "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "graph": object(),
        "bond_opacities": {(0, 1): 0.5},
        "hull_opacity_factor": 0.8,
    }
    out = _rotate_frames([frame], np.eye(3))
    assert out[0]["graph"] is frame["graph"]
    assert out[0]["bond_opacities"] == {(0, 1): 0.5}
    assert out[0]["hull_opacity_factor"] == 0.8


def test_scale_vibration_frames_about_equilibrium():
    from xyzrender.gif import _scale_vibration_frames

    frames = [
        {"symbols": ["H"], "positions": [[1.0, 2.0, 3.0]]},
        {"symbols": ["H"], "positions": [[1.2, 1.5, 4.0]]},
    ]

    scaled = _scale_vibration_frames(frames, 2.0)

    assert scaled[0]["positions"] == [[1.0, 2.0, 3.0]]
    assert np.allclose(scaled[1]["positions"], [[1.4, 1.0, 5.0]])
    assert frames[1]["positions"] == [[1.2, 1.5, 4.0]]


def test_normal_mode_normalization_uses_active_atom_rms():
    from xyzrender.gif import _normal_mode_normalization_scale, _scale_vibration_frames

    frames = [
        {"positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]},
        {"positions": [[0.1, 0.0, 0.0], [1.2, 0.0, 0.0], [2.001, 0.0, 0.0]]},
    ]

    scale = _normal_mode_normalization_scale(frames)
    scaled = _scale_vibration_frames(frames, scale)
    displacement = np.asarray(scaled[1]["positions"]) - np.asarray(scaled[0]["positions"])
    active_rms = np.sqrt(np.mean(np.square(np.linalg.norm(displacement[:2], axis=1))))

    assert active_rms == pytest.approx(0.25)
    assert displacement[0, 0] / displacement[1, 0] == pytest.approx(0.5)


def test_normal_mode_normalization_handles_zero_displacement():
    from xyzrender.gif import _normal_mode_normalization_scale

    frames = [{"positions": [[0.0, 0.0, 0.0]]}, {"positions": [[0.0, 0.0, 0.0]]}]

    assert _normal_mode_normalization_scale(frames) == 1.0


def test_limit_normal_mode_scale_prevents_zero_bond_distance():
    import networkx as nx

    from xyzrender.gif import _limit_normal_mode_scale

    graph = nx.Graph()
    graph.add_edge(0, 1)
    frames = [
        {"positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
        {"positions": [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]},
    ]

    safe_scale = _limit_normal_mode_scale(frames, graph, requested_scale=2.0)

    assert safe_scale == pytest.approx(0.475)
    compressed_bond = 1.0 + safe_scale * (-2.0)
    assert compressed_bond == pytest.approx(0.05)


def test_limit_normal_mode_scale_does_not_restrict_bond_rotation():
    import networkx as nx

    from xyzrender.gif import _limit_normal_mode_scale

    graph = nx.Graph()
    graph.add_edge(0, 1)
    frames = [
        {"positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
        {"positions": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]},
    ]

    # The bond's projection on its original axis becomes negative above scale
    # 1, but its true distance never approaches zero because it rotates away.
    assert _limit_normal_mode_scale(frames, graph, requested_scale=2.0) == 2.0


def test_limit_normal_mode_scale_leaves_safe_amplitude_unchanged():
    import networkx as nx

    from xyzrender.gif import _limit_normal_mode_scale

    graph = nx.Graph()
    graph.add_edge(0, 1)
    frames = [
        {"positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
        {"positions": [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]},
    ]

    assert _limit_normal_mode_scale(frames, graph, requested_scale=2.0) == 2.0


def test_format_vibrational_frequency_preserves_sign():
    from xyzrender.gif import _format_vibrational_frequency

    assert _format_vibrational_frequency(1234.567) == "\u03bd\u0303 = 1234.6 cm⁻¹"
    assert _format_vibrational_frequency(-748.483) == "\u03bd\u0303 = -748.5 cm⁻¹"


def test_add_frequency_label_to_png():
    from io import BytesIO

    from PIL import Image

    from xyzrender.gif import _add_png_label
    from xyzrender.types import RenderConfig

    source = BytesIO()
    Image.new("RGBA", (200, 100), "white").save(source, format="PNG")
    labelled = _add_png_label(source.getvalue(), "\u03bd\u0303 = 1234.6 cm⁻¹", RenderConfig(canvas_size=200))
    image = np.asarray(Image.open(BytesIO(labelled)).convert("RGB"))

    assert np.any(image[:50] < 128)


def test_render_gif_normal_mode_selection_and_scale(tmp_path):
    from unittest.mock import patch

    from xyzrender import render_gif

    frames = [
        {"symbols": ["C", "C"], "positions": [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]},
        {"symbols": ["C", "C"], "positions": [[0.1, 0.0, 0.0], [1.3, 0.0, 0.0]]},
    ]
    with (
        patch("graphrc.load_trajectory", return_value={"frames": frames}) as mock_load,
        patch("graphrc.run_vib_analysis", side_effect=AssertionError("TS analysis must not run")),
        patch("xyzrender.gif._render_frames", return_value=[b"", b""]) as mock_render,
        patch("xyzrender.gif._stitch_gif"),
    ):
        render_gif(
            STRUCTURES / "sn2.out",
            gif_vib=7,
            vib_scale=2.0,
            orient=False,
            output=tmp_path / "mode7.gif",
        )

    assert mock_load.call_args.kwargs["mode"] == 7
    assert mock_load.call_args.kwargs["save_to_disk"] is False
    graph, rendered_frames, config = mock_render.call_args.args
    assert np.allclose(rendered_frames[1]["positions"], [[0.5, 0.0, 0.0], [0.9, 0.0, 0.0]])
    assert not any(data.get("TS") for _i, _j, data in graph.edges(data=True))
    assert config.ts_bonds == []
    assert config.hide_h is False


def test_render_gif_normal_mode_frequency_label(tmp_path):
    from unittest.mock import patch

    from xyzrender import render_gif

    frames = [{"symbols": ["C", "C"], "positions": [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]}]
    with (
        patch("graphrc.load_trajectory", return_value={"frames": frames, "frequencies": [100.0, 1234.567]}),
        patch("xyzrender.gif._render_frames", return_value=[b""]) as mock_render,
        patch("xyzrender.gif._stitch_gif"),
    ):
        render_gif(
            STRUCTURES / "sn2.out",
            gif_vib=1,
            vib_label=True,
            orient=False,
            output=tmp_path / "mode1.gif",
        )

    assert mock_render.call_args.kwargs["frame_label"] == "\u03bd\u0303 = 1234.6 cm⁻¹"


def test_render_gif_normal_mode_rejects_imaginary_frequency(tmp_path):
    from unittest.mock import patch

    from xyzrender import render_gif

    frames = [{"symbols": ["C", "C"], "positions": [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]}]
    with (
        patch("graphrc.load_trajectory", return_value={"frames": frames, "frequencies": [-748.483]}),
        patch("xyzrender.gif._render_frames", side_effect=AssertionError("imaginary mode was rendered")),
        pytest.raises(ValueError, match=r"gif_vib=0.*imaginary frequency.*gif_ts=True explicitly"),
    ):
        render_gif(
            STRUCTURES / "sn2.out",
            gif_vib=0,
            orient=False,
            output=tmp_path / "imaginary.gif",
        )


def test_render_gif_ts_allows_imaginary_frequency(tmp_path):
    from unittest.mock import patch

    import networkx as nx

    from xyzrender import render_gif

    frames = [{"symbols": ["C", "C"], "positions": [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]}]
    graph = nx.Graph()
    graph.add_nodes_from((i, {"symbol": "C", "position": tuple(p)}) for i, p in enumerate(frames[0]["positions"]))
    analysis = {
        "graph": {"ts_graph": graph},
        "trajectory": {"frames": frames, "frequencies": [-748.483]},
    }
    with (
        patch("graphrc.run_vib_analysis", return_value=analysis),
        patch("xyzrender.gif._render_frames", return_value=[b""]) as mock_render,
        patch("xyzrender.gif._stitch_gif"),
    ):
        render_gif(
            STRUCTURES / "sn2.out",
            gif_ts=True,
            orient=False,
            output=tmp_path / "ts.gif",
        )

    mock_render.assert_called_once()


def test_render_gif_normal_mode_respects_no_hy(tmp_path):
    from unittest.mock import patch

    from xyzrender import render_gif

    with patch("xyzrender.gif.render_vibration_gif") as mock_vib:
        render_gif(
            STRUCTURES / "sn2.out",
            gif_vib=0,
            no_hy=True,
            output=tmp_path / "mode0.gif",
        )

    assert mock_vib.call_args.kwargs["config"].hide_h is True


@pytest.mark.parametrize(
    ("ts_bonds", "auto_detect", "expected"),
    [(None, True, []), ([], False, []), ([(1, 3)], False, [(0, 2)])],
)
def test_render_gif_ts_detection_mode(tmp_path, ts_bonds, auto_detect, expected):
    from unittest.mock import patch

    import networkx as nx

    from xyzrender import render_gif

    frames = [{"symbols": ["C", "C", "C"], "positions": [[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0]]}]
    auto_graph = nx.Graph()
    auto_graph.add_nodes_from(
        (i, {"symbol": "C", "position": tuple(position)}) for i, position in enumerate(frames[0]["positions"])
    )
    auto_graph.add_edge(0, 1, TS=True)
    analysis = {"graph": {"ts_graph": auto_graph}, "trajectory": {"frames": frames}}

    with (
        patch("graphrc.load_trajectory", return_value={"frames": frames}) as mock_load,
        patch("graphrc.run_vib_analysis", return_value=analysis) as mock_analysis,
        patch("xyzrender.gif._render_frames", return_value=[b""]) as mock_render,
        patch("xyzrender.gif._stitch_gif"),
    ):
        render_gif(
            STRUCTURES / "sn2.out",
            gif_ts=True,
            ts_bonds=ts_bonds,
            vib_frames=4,
            orient=False,
            output=tmp_path / "ts.gif",
        )

    assert mock_analysis.call_count == int(auto_detect)
    assert mock_load.call_count == int(not auto_detect)
    graph, _frames, config = mock_render.call_args.args
    assert any(data.get("TS") for _i, _j, data in graph.edges(data=True)) is auto_detect
    assert config.ts_bonds == expected


@pytest.mark.parametrize("ts_bonds", [[(1, 1)], [(1, 999)]])
def test_render_gif_ts_rejects_invalid_manual_bonds(tmp_path, ts_bonds):
    from unittest.mock import patch

    from xyzrender import render_gif

    frames = [{"symbols": ["C", "C"], "positions": [[0, 0, 0], [1.4, 0, 0]]}]
    with (
        patch("graphrc.load_trajectory", return_value={"frames": frames}),
        patch("graphrc.run_vib_analysis", side_effect=AssertionError("automatic TS identification ran")),
        pytest.raises(ValueError, match="ts-bond"),
    ):
        render_gif(
            STRUCTURES / "sn2.out",
            gif_ts=True,
            ts_bonds=ts_bonds,
            vib_frames=4,
            orient=False,
            output=tmp_path / "ts.gif",
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"gif_vib": -1}, "zero-based"),
        ({"gif_vib": 0, "vib_scale": 0}, "> 0"),
        ({"gif_vib": 0, "vib_scale": float("nan")}, "finite"),
        ({"gif_vib": 0, "vib_scale": float("inf")}, "finite"),
        ({"gif_vib": 0, "vib_frames": 3}, "multiple of 4"),
        ({"gif_vib": 0, "vib_frames": True}, "multiple of 4"),
    ],
)
def test_render_gif_normal_mode_rejects_invalid_options(tmp_path, kwargs, match):
    from xyzrender import render_gif

    with pytest.raises(ValueError, match=match):
        render_gif(STRUCTURES / "sn2.out", output=tmp_path / "bad.gif", **kwargs)


def test_render_trajectory_gif_trj_bonds(tmp_path):
    """trj_bonds=True must rebuild graphs per frame, and those graphs must
    survive the orient/rotate transforms and reach the worker."""
    from unittest.mock import patch

    from xyzrender.gif import render_trajectory_gif
    from xyzrender.readers import load_trajectory_frames
    from xyzrender.types import RenderConfig

    frames = load_trajectory_frames(STRUCTURES / "sn2.v000.xyz")
    captured: dict = {}

    def _spy(graph, frames, config, **_):
        captured["frames"] = frames
        return [b""] * len(frames)

    with (
        patch("xyzrender.gif._render_frames", side_effect=_spy),
        patch("xyzrender.gif._stitch_gif"),
    ):
        render_trajectory_gif(frames=frames, config=RenderConfig(), output=str(tmp_path / "x.gif"), trj_bonds=True)

    seen = captured["frames"]
    assert all("graph" in f for f in seen), "trj_bonds frames must each carry a 'graph'"
    bond_counts = [f["graph"].number_of_edges() for f in seen]
    # sn2.v000.xyz: bond forms across the SN2 reaction so counts must vary
    assert len(set(bond_counts)) > 1, f"expected varying per-frame bond counts, got {bond_counts}"
    # The graphs must be distinct objects (not all aliased to the last-frame graph)
    assert len({id(f["graph"]) for f in seen}) == len(seen)


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


def test_api_render_gif_rotation_true_uses_y_axis(tmp_path):
    from unittest.mock import patch

    from xyzrender import render_gif

    with patch("xyzrender.gif.render_rotation_gif") as mock_render:
        render_gif(_tiny_molecule(), gif_rot=True, output=tmp_path / "true.gif")

    assert mock_render.call_args.kwargs["axis"] == "y"


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
    must reach build_surface_params when gif_rot is the only mode."""
    from unittest.mock import patch

    from xyzrender.api import render_gif

    mol = _tiny_molecule()
    mol.cube_data = object()  # truthy; consumers are patched below

    captured: dict = {}

    def _capture(cfg, overrides, **_):
        captured["overrides"] = overrides
        return (None, None, None, None)

    cm_rot, _ = _capture_rotation_cfg()
    with (
        patch("xyzrender.config.build_surface_params", side_effect=_capture),
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
    o = captured["overrides"]
    assert o.mo_pos_color == "cyan"
    assert o.mo_neg_color == "magenta"
    assert o.mo_upsample == 2
    assert o.flat_mo is True
    assert o.dens_color == "grey"


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
