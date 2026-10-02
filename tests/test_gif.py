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


def test_render_gif_vibration_filters_detected_nci_types(tmp_path):
    from unittest.mock import patch

    import networkx as nx
    from xyzgraph.nci import NCIData

    from xyzrender import render_gif

    frames = [{"symbols": ["C", "C", "C"], "positions": [[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0]]}]
    ts_graph = nx.Graph()
    ts_graph.add_nodes_from(
        (i, {"symbol": "C", "position": tuple(position)}) for i, position in enumerate(frames[0]["positions"])
    )
    analysis = {"graph": {"ts_graph": ts_graph}, "trajectory": {"frames": frames}}
    detected = [
        NCIData("hbond", (0,), (1,), (), {}, 1.0),
        NCIData("pi_pi_parallel", (1,), (2,), (), {}, 1.0),
    ]

    def _detect(graph):
        graph.graph["ncis"] = detected
        return detected

    with (
        patch("graphrc.run_vib_analysis", return_value=analysis),
        patch("xyzgraph.detect_ncis", side_effect=_detect),
        patch("xyzrender.gif._render_frames", return_value=[b""]) as mock_render,
        patch("xyzrender.gif._stitch_gif"),
    ):
        render_gif(
            STRUCTURES / "sn2.out",
            gif_ts=True,
            detect_nci=["hb"],
            orient=False,
            output=tmp_path / "ts.gif",
        )

    fixed_ncis = mock_render.call_args.kwargs["fixed_ncis"]
    assert [nci.type for nci in fixed_ncis] == ["hbond"]


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


def test_render_trajectory_frame_filters_detected_nci_types():
    from unittest.mock import Mock, patch

    import networkx as nx
    from xyzgraph.nci import NCIData

    from xyzrender.gif import _render_traj_frame
    from xyzrender.types import RenderConfig

    graph = nx.Graph()
    graph.add_nodes_from(
        [
            (0, {"symbol": "C", "position": (0.0, 0.0, 0.0)}),
            (1, {"symbol": "C", "position": (1.4, 0.0, 0.0)}),
        ]
    )
    frame = {"symbols": ["C", "C"], "positions": [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]}
    analyzer = Mock()
    analyzer.detect.return_value = [
        NCIData("hbond", (0,), (1,), (), {}, 1.0),
        NCIData("pi_pi_parallel", (0,), (1,), (), {}, 1.0),
    ]
    captured = {}

    def _build(source, ncis):
        captured["types"] = [nci.type for nci in ncis]
        return source

    with (
        patch("xyzgraph.nci.build_nci_graph", side_effect=_build),
        patch("xyzrender.gif.render_svg", return_value="<svg />"),
        patch("xyzrender.gif.svg_to_png_bytes", return_value=b"png"),
    ):
        _render_traj_frame(
            (0, frame),
            graph,
            RenderConfig(),
            analyzer,
            None,
            False,
            ["hb"],
            None,
            1.0,
            None,
            np.full((0, 3), np.nan),
            np.full((0, 3), np.nan),
        )

    assert captured["types"] == ["hbond"]


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


def test_render_gif_rotation_filters_detected_nci_types(tmp_path):
    from unittest.mock import patch

    from xyzrender import render_gif

    with patch("xyzrender.gif.render_rotation_gif") as mock_render:
        render_gif(
            STRUCTURES / "bimp.v000.xyz",
            gif_rot="y",
            detect_nci=["hb"],
            output=tmp_path / "nci.gif",
        )

    graph = mock_render.call_args.kwargs["graph"]
    nci_types = {data["nci_type"] for *_edge, data in graph.edges(data=True) if data.get("NCI")}
    assert nci_types == {"hbond_bifurcated"}


def test_render_gif_rotation_filters_molecule_without_mutating_it(tmp_path):
    from unittest.mock import patch

    from xyzrender import load, render_gif

    molecule = load(STRUCTURES / "bimp.v000.xyz")

    with patch("xyzrender.gif.render_rotation_gif") as mock_render:
        render_gif(
            molecule,
            gif_rot="y",
            detect_nci=["hb"],
            output=tmp_path / "nci.gif",
        )

    graph = mock_render.call_args.kwargs["graph"]
    nci_types = {data["nci_type"] for *_edge, data in graph.edges(data=True) if data.get("NCI")}
    assert nci_types == {"hbond_bifurcated"}
    assert not any(data.get("NCI") for *_edge, data in molecule.graph.edges(data=True))


def test_render_gif_rotation_detects_ncis_after_atom_filtering(tmp_path):
    from unittest.mock import patch

    from xyzrender import render_gif

    with patch("xyzrender.gif.render_rotation_gif") as mock_render:
        render_gif(
            STRUCTURES / "bimp.v000.xyz",
            gif_rot="y",
            detect_nci="pi",
            only="1-172",
            output=tmp_path / "nci.gif",
        )

    graph = mock_render.call_args.kwargs["graph"]
    assert any(data.get("NCI") for *_edge, data in graph.edges(data=True))


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
