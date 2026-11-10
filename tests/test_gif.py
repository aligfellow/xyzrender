"""Tests for gif.py — rotation axis parsing and GIF rendering."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


def test_trajectory_gif_from_stdin(tmp_path):
    from PIL import Image

    output = tmp_path / "stdin.gif"
    source = (
        "import multiprocessing\n"
        "multiprocessing.set_start_method('spawn')\n"
        "from xyzrender import render_gif\n"
        f"render_gif({str(STRUCTURES / 'sn2.v000.xyz')!r}, gif_trj=True, orient=False, "
        f"canvas_size=100, output={str(output)!r})\n"
    )
    subprocess.run([sys.executable, "-"], input=source, text=True, check=True, capture_output=True, timeout=30)
    with Image.open(output) as image:
        image.seek(1)


def test_gif_worker_exit_propagates():
    source = (
        "import multiprocessing, os\n"
        "multiprocessing.set_start_method('spawn')\n"
        "from xyzrender.gif import _parallel_render\n"
        "_parallel_render(os._exit, [1], 1)\n"
    )
    result = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True, timeout=30, check=False)
    assert result.returncode != 0
    assert "BrokenProcessPool" in result.stderr


def test_trajectory_orientation_uses_matching_frame(tmp_path, monkeypatch):
    from xyzrender import load, render_gif
    from xyzrender.readers import load_trajectory_frames
    from xyzrender.types import RenderConfig

    path = STRUCTURES / "sn2.v000.xyz"
    original = load_trajectory_frames(path)
    captured = {}

    def capture(graph, frames, config, **_):
        captured["frames"] = frames
        captured["graph"] = graph
        return []

    monkeypatch.setattr("xyzrender.gif._render_frames", capture)
    monkeypatch.setattr("xyzrender.gif._stitch_gif", lambda *args: None)
    cfg = RenderConfig(auto_orient=True)
    render_gif(path, gif_trj=True, orient=False, config=cfg, output=tmp_path / "raw.gif")
    for before, after in zip(original, captured["frames"], strict=True):
        np.testing.assert_array_equal(after["positions"], before["positions"])
    assert cfg.auto_orient is True

    reference = load(path).graph
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for node in reference.nodes.values():
        node["position"] = tuple(np.array(node["position"]) @ rotation.T)
    edge = next(iter(reference.edges))
    reference.edges[edge]["TS"] = True
    render_gif(path, gif_trj=True, reference_graph=reference, output=tmp_path / "rotated.gif")
    for before, after in zip(original, captured["frames"], strict=True):
        positions = np.array(before["positions"])
        center = positions.mean(axis=0)
        np.testing.assert_allclose(after["positions"], (positions - center) @ rotation.T + center, atol=1e-10)
    assert captured["graph"].edges[edge]["TS"] is True


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


# ---------------------------------------------------------------------------
# Vibrational modes — amplitude, labels, and fixed camera reference
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_mode_render(monkeypatch):
    """Capture normal-mode frames without rasterising or running TS analysis."""
    from unittest.mock import Mock

    import graphrc

    from xyzrender import gif

    trajectory = {
        "frames": [
            {"symbols": ["C", "C"], "positions": [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]},
            {"symbols": ["C", "C"], "positions": [[0.1, 0.0, 0.0], [1.3, 0.0, 0.0]]},
        ],
        "frequencies": [100.0, 1234.567],
    }
    load_trajectory = Mock(return_value=trajectory)
    render_frames = Mock(return_value=[b"", b""])
    monkeypatch.setattr(graphrc, "load_trajectory", load_trajectory)
    monkeypatch.setattr(graphrc, "run_vib_analysis", Mock(side_effect=AssertionError("TS analysis must not run")))
    monkeypatch.setattr(gif, "_render_frames", render_frames)
    monkeypatch.setattr(gif, "_stitch_gif", Mock())
    return trajectory, load_trajectory, render_frames


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


@pytest.mark.parametrize(
    ("displaced_position", "expected_scale"),
    [
        pytest.param([-1.0, 0.0, 0.0], 0.475, id="bond-compression"),
        pytest.param([0.0, 1.0, 0.0], 2.0, id="bond-rotation"),
        pytest.param([0.9, 0.0, 0.0], 2.0, id="safe-amplitude"),
    ],
)
def test_limit_normal_mode_scale(displaced_position, expected_scale):
    import networkx as nx

    from xyzrender.gif import _limit_normal_mode_scale

    graph = nx.Graph()
    graph.add_edge(0, 1)
    frames = [
        {"positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
        {"positions": [[0.0, 0.0, 0.0], displaced_position]},
    ]

    safe_scale = _limit_normal_mode_scale(frames, graph, requested_scale=2.0)

    assert safe_scale == pytest.approx(expected_scale)


def test_format_vibrational_frequency_preserves_sign():
    from xyzrender.gif import _format_vibrational_frequency

    assert _format_vibrational_frequency(1234.567) == "\u03bd\u0303 = 1234.6 cm⁻¹"
    assert _format_vibrational_frequency(-748.483) == "\u03bd\u0303 = -748.5 cm⁻¹"


def test_render_gif_normal_mode_selection_and_scale(tmp_path, normal_mode_render):
    from xyzrender import render_gif

    _trajectory, mock_load, mock_render = normal_mode_render
    render_gif(
        STRUCTURES / "methanol.out",
        gif_vib=1,
        vib_scale=2.0,
        orient=False,
        output=tmp_path / "mode.gif",
    )

    assert mock_load.call_args.kwargs["mode"] == 1
    assert mock_load.call_args.kwargs["save_to_disk"] is False
    graph, rendered_frames, config = mock_render.call_args.args
    assert np.allclose(rendered_frames[1]["positions"], [[0.5, 0.0, 0.0], [0.9, 0.0, 0.0]])
    assert not any(data.get("TS") for _i, _j, data in graph.edges(data=True))
    assert config.ts_bonds == []


def test_render_gif_normal_mode_frequency_label(tmp_path, normal_mode_render):
    from xyzrender import render_gif
    from xyzrender.types import RenderConfig

    _trajectory, _mock_load, mock_render = normal_mode_render
    config = RenderConfig(padding=10)
    render_gif(
        STRUCTURES / "methanol.out",
        gif_vib=1,
        config=config,
        vib_label=True,
        orient=False,
        output=tmp_path / "mode.gif",
    )

    assert mock_render.call_args.kwargs["frame_label"] == "\u03bd\u0303 = 1234.6 cm⁻¹"

    assert mock_render.call_args.args[2].padding == config.padding == 10


def test_render_gif_normal_mode_rejects_imaginary_frequency(tmp_path, normal_mode_render):
    from xyzrender import render_gif

    trajectory, _mock_load, mock_render = normal_mode_render
    trajectory["frequencies"] = [-748.483]
    with pytest.raises(ValueError, match=r"gif_vib=0.*imaginary frequency.*gif_ts=True explicitly"):
        render_gif(
            STRUCTURES / "methanol.out",
            gif_vib=0,
            orient=False,
            output=tmp_path / "imaginary.gif",
        )
    mock_render.assert_not_called()


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


@pytest.mark.parametrize("no_hy", [False, True])
def test_render_gif_normal_mode_hydrogen_visibility(tmp_path, normal_mode_render, no_hy):
    from xyzrender import render_gif

    _trajectory, _mock_load, mock_render = normal_mode_render
    render_gif(STRUCTURES / "methanol.out", gif_vib=0, no_hy=no_hy, output=tmp_path / "mode.gif")

    assert mock_render.call_args.args[2].hide_h is no_hy


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


@pytest.mark.parametrize("gif_rot", [None, "z"])
def test_render_gif_normal_mode_keeps_shared_camera_center(tmp_path, normal_mode_render, gif_rot):
    """Camera centring must not turn a vibration into motion of stationary atoms."""
    from xyzrender import render_gif

    trajectory, _mock_load, mock_render = normal_mode_render
    trajectory["frames"] = [
        {"symbols": ["C", "C"], "positions": [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]},
        {"symbols": ["C", "C"], "positions": [[0.0, 0.0, 0.0], [1.5, 0.2, 0.1]]},
    ]
    render_gif(
        STRUCTURES / "methanol.out",
        gif_vib=0,
        gif_rot=gif_rot,
        orient=True,
        vib_frames=4,
        output=tmp_path / "vib.gif",
    )

    _graph, rendered_frames, config = mock_render.call_args.args
    initial = np.asarray(rendered_frames[0]["positions"])
    displaced = np.asarray(rendered_frames[1]["positions"])
    np.testing.assert_allclose(initial.mean(axis=0), np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(displaced[0], initial[0], atol=1e-12)
    assert np.linalg.norm(displaced[1] - initial[1]) == pytest.approx(0.25)
    np.testing.assert_allclose(config.fixed_center, [0.0, 0.0], atol=1e-12)
    assert config.auto_orient is False


@pytest.mark.parametrize("axis", [None, np.array([0.0, 1.0, 0.0])])
def test_vibration_viewport_contains_asymmetric_displacements(axis):
    from xyzrender.gif import _fixed_viewport
    from xyzrender.types import RenderConfig

    frames = [
        {"symbols": ["C", "C"], "positions": [[0, 0, 0], [0, 1, 0]]},
        {"symbols": ["C", "C"], "positions": [[0, 0, 0], [0, 3, 0]]},
    ]
    cfg = _fixed_viewport(frames, RenderConfig(), rotation_axis=axis, reference_frame=frames[0])
    assert cfg.fixed_center is not None
    assert cfg.fixed_span is not None
    np.testing.assert_allclose(cfg.fixed_center, [0, 0.5])
    assert cfg.fixed_span > 5.0  # furthest atom is 2.5 Å from the fixed centre


@pytest.mark.parametrize("normal_mode", [False, True])
def test_vibration_dispatch_preserves_loaded_electronic_state(tmp_path, monkeypatch, normal_mode):
    from unittest.mock import Mock

    import networkx as nx

    from xyzrender import render_gif
    from xyzrender.api import Molecule

    graph = nx.Graph(total_charge=1, multiplicity=2)
    graph.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))
    monkeypatch.setattr("xyzrender.api.load", Mock(return_value=Molecule(graph=graph)))
    worker = Mock()
    monkeypatch.setattr("xyzrender.gif.render_vibration_gif", worker)
    render_gif(
        STRUCTURES / "methanol.out",
        gif_vib=0 if normal_mode else None,
        gif_ts=not normal_mode,
        output=tmp_path / "charged.gif",
    )
    assert worker.call_args.kwargs["charge"] == 1
    assert worker.call_args.kwargs["multiplicity"] == 2


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
# Non-uniform atom counts across frames (issue #169)
# ---------------------------------------------------------------------------

_VARIABLE_ATOM_FRAMES = [
    {"symbols": ["H", "H"], "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]},
    {
        "symbols": ["H", "H", "H"],
        "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.80], [0.0, 0.0, 1.60]],
    },
]

_VARIABLE_ATOM_XYZ = """2
frame 0: H2
H  0.000  0.000  0.000
H  0.000  0.000  0.740
3
frame 1: H3
H  0.000  0.000  0.000
H  0.000  0.000  0.800
H  0.000  0.000  1.600
2
frame 2: H2
H  1.000  0.000  0.000
H  1.000  0.000  0.750
2
frame 3: H2
H  0.000  1.000  0.000
H  0.000  1.000  0.730
"""


def test_render_gif_variable_atoms_recommends_opt_in(tmp_path):
    """The public trajectory pipeline explains how to allow intentional count changes."""
    from xyzrender import render_gif

    trajectory = tmp_path / "variable.xyz"
    trajectory.write_text(_VARIABLE_ATOM_XYZ)

    message = (
        "Trajectory has non-uniform atom counts: first frame has 2 atoms, "
        "frame 1 has 3. If this is intentional, use --var-atoms"
    )
    with pytest.raises(ValueError, match=re.escape(message)) as exc_info:
        render_gif(trajectory, gif_trj=True, output=tmp_path / "variable.gif")
    assert str(exc_info.value) == message


def test_render_gif_variable_atoms_opt_in_reaches_public_renderer(tmp_path):
    """var_atoms=True renders every independently sized frame through the public API."""
    from unittest.mock import patch

    from xyzrender import render_gif

    trajectory = tmp_path / "variable.xyz"
    trajectory.write_text(_VARIABLE_ATOM_XYZ)
    captured: dict = {}

    def _spy(graph, frames, config, **_):
        captured["frames"] = frames
        return [b""] * len(frames)

    with (
        patch("xyzrender.gif._render_frames", side_effect=_spy),
        patch("xyzrender.gif._stitch_gif"),
    ):
        render_gif(
            trajectory,
            gif_trj=True,
            var_atoms=True,
            orient=False,
            output=tmp_path / "variable.gif",
        )

    assert [len(frame["symbols"]) for frame in captured["frames"]] == [2, 3, 2, 2]
    assert all("graph" in frame for frame in captured["frames"])


def test_render_trajectory_gif_auto_enables_trj_bonds_for_variable_atoms(tmp_path):
    """Differing atom counts must trigger per-frame graphs even if trj_bonds=False."""
    from unittest.mock import patch

    from xyzrender.gif import render_trajectory_gif
    from xyzrender.types import RenderConfig

    captured: dict = {}

    def _spy(graph, frames, config, **_):
        captured["frames"] = frames
        return [b""] * len(frames)

    with (
        patch("xyzrender.gif._render_frames", side_effect=_spy),
        patch("xyzrender.gif._stitch_gif"),
    ):
        render_trajectory_gif(
            frames=[dict(f) for f in _VARIABLE_ATOM_FRAMES],
            config=RenderConfig(),
            output=str(tmp_path / "x.gif"),
            trj_bonds=False,
        )

    seen = captured["frames"]
    assert all("graph" in f for f in seen), "variable atom counts must trigger per-frame graphs"
    assert [g.number_of_nodes() for g in (f["graph"] for f in seen)] == [2, 3]


def test_render_trajectory_gif_orients_from_matching_reference_frame(tmp_path):
    """Reference orientation must not align a differently sized final frame."""
    from unittest.mock import patch

    import networkx as nx

    from xyzrender.gif import render_trajectory_gif
    from xyzrender.types import RenderConfig

    frames = [
        {
            "symbols": ["C", "N", "O"],
            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        },
        {
            "symbols": ["C", "N", "O", "H"],
            "positions": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
        },
    ]
    reference = nx.Graph()
    for i, (symbol, position) in enumerate(
        zip(frames[0]["symbols"], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], strict=True)
    ):
        reference.add_node(i, symbol=symbol, position=position)

    captured: dict = {}

    def _spy(graph, rendered_frames, config, **_):
        captured["frames"] = rendered_frames
        return [b""] * len(rendered_frames)

    with (
        patch("xyzrender.gif._render_frames", side_effect=_spy),
        patch("xyzrender.gif._stitch_gif"),
    ):
        render_trajectory_gif(
            frames=frames,
            config=RenderConfig(auto_orient=False),
            output=str(tmp_path / "oriented.gif"),
            reference_graph=reference,
        )

    assert [len(frame["positions"]) for frame in captured["frames"]] == [3, 4]
    np.testing.assert_allclose(
        captured["frames"][0]["positions"],
        [[2 / 3, 0.0, 0.0], [2 / 3, 1.0, 0.0], [-1 / 3, 0.0, 0.0]],
        atol=1e-12,
    )

    with pytest.raises(ValueError, match="reference graph and source frame to have the same atom count"):
        render_trajectory_gif(
            frames=frames,
            config=RenderConfig(auto_orient=False),
            output=str(tmp_path / "wrong-reference.gif"),
            reference_graph=reference,
            reference_frame=1,
        )

    mismatched_reference = reference.copy()
    mismatched_reference.nodes[1]["symbol"] = "O"
    mismatched_reference.nodes[2]["symbol"] = "N"
    with pytest.raises(ValueError, match="same atom order"):
        render_trajectory_gif(
            frames=frames,
            config=RenderConfig(auto_orient=False),
            output=str(tmp_path / "wrong-order.gif"),
            reference_graph=mismatched_reference,
        )


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
            frame_label=None,
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


@pytest.mark.parametrize(
    ("mode", "renderer"),
    [
        ({"gif_ts": True}, "render_vibration_gif"),
        ({"gif_trj": True}, "render_trajectory_gif"),
        ({"gif_rot": True}, "render_rotation_gif"),
        ({"gif_diffuse": True}, "render_diffuse_gif"),
    ],
    ids=["ts", "trj", "rot", "diffuse"],
)
def test_render_gif_loads_once(tmp_path, mode, renderer):
    from unittest.mock import patch

    from xyzrender import readers, render_gif

    source = tmp_path / "charged.xyz"
    source.write_text("1\ncharge=-1 multiplicity=1\nCl 0 0 0\n" * 2)
    with (
        patch(f"xyzrender.gif.{renderer}") as mock_render,
        patch("xyzrender.readers.load_molecule", wraps=readers.load_molecule) as mock_load,
    ):
        render_gif(source, **mode, output=tmp_path / "path.gif")

    mock_render.assert_called_once()
    assert mock_load.call_count == 1


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


@pytest.mark.parametrize("var_atoms", [False, True])
def test_trajectory_gif_preserves_electronic_state(tmp_path, monkeypatch, var_atoms):
    from xyzrender import gif, render_gif

    source = tmp_path / "charged.xyz"
    source.write_text("1\ncharge=-1 multiplicity=1\nCl 0 0 0\n" * 2)
    captured = {}
    monkeypatch.setattr(gif, "render_trajectory_gif", lambda **kwargs: captured.update(kwargs))

    render_gif(source, gif_trj=True, trj_bonds=True, var_atoms=var_atoms, output=tmp_path / "path.gif")

    assert captured["charge"] == -1
    assert captured["multiplicity"] == 1
    assert captured["trj_bonds"] is True


@pytest.mark.parametrize("from_path", [False, True], ids=["molecule", "path"])
def test_rotation_preserves_molecule_data(tmp_path, monkeypatch, from_path):
    from xyzrender import api, gif

    source = tmp_path / "cell.xyz"
    source.write_text('2\nLattice="5 0 0 0 5 0 0 0 5" charge=0 multiplicity=1\nH 0 0 0\nH 0.74 0 0\n')
    mol = api.load(source)
    assert mol.cell_data is not None
    captured = {}

    def render(graph, config, **kwargs):
        captured.update(graph=graph, config=config)
        graph.nodes[0]["position"] = (9.0, 9.0, 9.0)
        config.cell_data.lattice[:] = 9.0

    monkeypatch.setattr(gif, "render_rotation_gif", render)
    monkeypatch.setattr(api, "load", lambda _: mol)
    api.render_gif(source if from_path else mol, gif_rot=True, ghosts=False, output=tmp_path / "path.gif")

    assert captured["graph"] is not mol.graph
    assert captured["config"].cell_data is not mol.cell_data
    assert captured["config"].show_cell
    np.testing.assert_array_equal(mol.graph.nodes[0]["position"], (0.0, 0.0, 0.0))
    np.testing.assert_array_equal(mol.cell_data.lattice, np.eye(3) * 5)


@pytest.mark.parametrize("orientation", ["none", "pca", "reference"])
@pytest.mark.parametrize("axis", [None, "z"])
def test_normal_mode_transforms_preserve_stationary_atom(monkeypatch, tmp_path, orientation, axis):
    """Exercise frame workers: camera transforms must preserve a mode's fixed pivot."""
    import copy
    from unittest.mock import patch

    import networkx as nx

    from xyzrender.gif import render_vibration_gif
    from xyzrender.types import RenderConfig

    equilibrium = np.array([[2.0, 3.0, 0.0], [3.4, 3.0, 0.0], [2.0, 4.4, 0.0]])
    frames = []
    for displacement in [0.0, 0.1, 0.0, -0.1]:
        positions = equilibrium.copy()
        positions[1] += [displacement, 2 * displacement, displacement]
        frames.append({"symbols": ["C", "C", "C"], "positions": positions.tolist()})
    reference = None
    if orientation == "reference":
        reference = nx.Graph()
        rotated = equilibrium[:, [1, 0, 2]] * [-1, 1, 1]
        for i, position in enumerate(rotated):
            reference.add_node(i, symbol="C", position=tuple(position))
    rendered = []

    def capture(graph, _config, **_kwargs):
        rendered.append(np.array([graph.nodes[i]["position"] for i in range(3)]))
        return "<svg />"

    def serial_workers(worker, items, _total):
        return [worker(copy.deepcopy(item))[1] for item in items]

    with (
        patch("graphrc.load_trajectory", return_value={"frames": frames, "frequencies": [100.0]}),
        patch("xyzrender.gif._parallel_render", side_effect=serial_workers),
        patch("xyzrender.gif.render_svg", side_effect=capture),
        patch("xyzrender.gif.svg_to_png_bytes", return_value=b"png"),
        patch("xyzrender.gif._stitch_gif"),
    ):
        render_vibration_gif(
            "mode.out",
            RenderConfig(auto_orient=orientation != "none"),
            str(tmp_path / "mode.gif"),
            gif_vib=0,
            auto_detect_ts=False,
            reference_graph=reference,
            axis=axis,
            n_frames=4 if axis else None,
        )
    center = equilibrium.mean(axis=0) if orientation == "none" else np.zeros(3)
    stationary = rendered[0][0]
    for i, positions in enumerate(rendered):
        angle = i * np.pi / 2 if axis else 0.0
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        expected = center + rotation @ (stationary - center)
        np.testing.assert_allclose(positions[0], expected, atol=1e-12)
    if orientation == "reference":
        np.testing.assert_allclose(rendered[0], rotated - rotated.mean(axis=0), atol=1e-12)


@pytest.mark.parametrize("normal_mode", [False, True])
def test_vibration_reference_file_roundtrip(tmp_path, normal_mode):
    """Python ref= saves and reloads orientation before vibration dispatch."""
    from unittest.mock import patch

    from xyzrender import render_gif

    reference_path = tmp_path / "reference.xyz"
    with patch("xyzrender.gif.render_vibration_gif") as worker:
        render_gif(
            STRUCTURES / "methanol.out",
            gif_vib=0 if normal_mode else None,
            gif_ts=not normal_mode,
            ref=reference_path,
            output=tmp_path / "first.gif",
        )
        assert reference_path.is_file()
        saved = worker.call_args.kwargs["reference_graph"]
        assert saved is not None
        saved_positions = np.array([d["position"] for _, d in saved.nodes(data=True)])
        render_gif(
            STRUCTURES / "methanol.out",
            gif_vib=0 if normal_mode else None,
            gif_ts=not normal_mode,
            ref=reference_path,
            orient=False,
            output=tmp_path / "second.gif",
        )
        loaded = worker.call_args.kwargs["reference_graph"]
        assert loaded is not None
        np.testing.assert_allclose([d["position"] for _, d in loaded.nodes(data=True)], saved_positions, atol=1e-6)
        assert worker.call_args.kwargs["config"].auto_orient is False


@pytest.mark.parametrize("axis_name", ["x", "y", "z", "xy", "xz", "yz", "yx", "zx", "zy"])
@pytest.mark.parametrize("fixed_pivot", [False, True])
def test_rotating_camera_contains_every_atom_orbit(axis_name, fixed_pivot):
    """Contain the actual rotation for cardinal/diagonal axes and both pivot policies."""
    from xyzrender.gif import _axis_angle_matrix, _fixed_viewport, _rotation_axis
    from xyzrender.types import RenderConfig

    frames = [
        {"symbols": ["C", "C"], "positions": [[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]},
        {"symbols": ["C", "C"], "positions": [[-9.0, 1.0, 2.0], [12.0, -1.0, -1.0]]},
    ]
    axis, _sign = _rotation_axis(axis_name)
    reference = frames[0] if fixed_pivot else None
    cfg = _fixed_viewport(frames, RenderConfig(), rotation_axis=axis, reference_frame=reference)
    assert cfg.fixed_span is not None
    assert cfg.fixed_center is not None
    for frame in frames:
        positions = np.asarray(frame["positions"])
        pivot = np.asarray(reference["positions"]).mean(axis=0) if reference else positions.mean(axis=0)
        for angle in range(0, 360, 5):
            rotated = (positions - pivot) @ _axis_angle_matrix(axis, angle).T + pivot
            extent = np.abs(rotated[:, :2] - cfg.fixed_center).max()
            assert extent < cfg.fixed_span / 2


def test_nci_centroid_does_not_change_rotation_pivot():
    from unittest.mock import patch

    import networkx as nx

    from xyzrender.gif import _render_traj_frame
    from xyzrender.types import RenderConfig

    positions = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    graph = nx.Graph()
    graph.add_nodes_from((i, {"symbol": "C", "position": tuple(p)}) for i, p in enumerate(positions))

    def decorated(source, _ncis):
        result = source.copy()
        result.add_node(2, symbol="*", position=(1.0, 2.0, 0.0))
        return result

    with (
        patch("xyzgraph.nci.build_nci_graph", side_effect=decorated),
        patch("xyzrender.gif.render_svg", return_value="<svg />") as render_svg,
        patch("xyzrender.gif.svg_to_png_bytes", return_value=b"png"),
    ):
        _render_traj_frame(
            (0, {"symbols": ["C", "C"], "positions": positions}),
            graph,
            RenderConfig(),
            nci_analyzer=None,
            fixed_ncis=[],
            detect_nci_per_frame=False,
            nci_types=None,
            rotation_axis=np.array([0.0, 0.0, 1.0]),
            rotation_sign=1.0,
            angles=np.array([90.0]),
            rf_vec_origins=np.empty((0, 3)),
            rf_vec_dirs=np.empty((0, 3)),
            frame_label=None,
        )
    rendered = render_svg.call_args.args[0]
    np.testing.assert_allclose(
        [rendered.nodes[i]["position"] for i in range(3)], [[0, -1, 0], [0, 1, 0], [-2, 1, 0]], atol=1e-12
    )


@pytest.mark.parametrize(("width", "background"), [(240, None), (800, "#ffffff")])
def test_frequency_label_header_preserves_entire_wide_frame(width, background):
    from io import BytesIO

    from PIL import Image, ImageDraw

    from xyzrender.gif import _add_png_label
    from xyzrender.types import RenderConfig

    height = round(width * 0.6)
    frame = Image.new("RGBA", (width, height), background or (0, 0, 0, 0))
    # A colorbar can occupy the top of a wide frame. No label pixels may
    # overwrite it, and composing a caption must not rescale the scene.
    ImageDraw.Draw(frame).rectangle((width * 3 // 4, 0, width - 1, height - 1), fill="red")
    source = BytesIO()
    frame.save(source, format="PNG")
    labelled = Image.open(
        BytesIO(
            _add_png_label(
                source.getvalue(),
                "\u03bd\u0303 = 320.9 cm⁻¹",
                RenderConfig(canvas_size=width, background=background, label_font_size=40),
            )
        )
    ).convert("RGBA")
    header_height = labelled.height - frame.height
    assert labelled.width == frame.width
    assert header_height > 0
    np.testing.assert_array_equal(labelled.crop((0, header_height, labelled.width, labelled.height)), frame)
    header = np.asarray(labelled)[:header_height]
    ink = np.any(header[:, :, :3] < 128, axis=2) & (header[:, :, 3] > 128)
    y, x = np.where(ink)
    assert len(x) > 0  # visible glyphs, including on a transparent background
    assert x.min() > 0
    assert x.max() < width - 1
    assert y.min() > 0
    assert y.max() < header_height - 1


def test_missing_bond_maps_render_in_rotation_and_rebuilt_trajectory_frames(tmp_path, monkeypatch):
    import copy
    from io import BytesIO
    from xml.etree import ElementTree as ET

    import numpy as np
    from PIL import Image

    from xyzrender import gif, load, render_gif
    from xyzrender.cmap import bond_color_hex
    from xyzrender.types import RenderConfig

    svgs = []
    original_render = gif.render_svg

    def record_svg(*args, **kwargs):
        svg = original_render(*args, **kwargs)
        svgs.append(svg)
        return svg

    def serial(worker, items, total):
        pngs = [b""] * total
        for item in items:
            owned_worker, owned_item = copy.deepcopy((worker, item))
            idx, png = owned_worker(owned_item)
            pngs[idx] = png
        return pngs

    monkeypatch.setattr(gif, "render_svg", record_svg)
    monkeypatch.setattr(gif, "_parallel_render", serial)
    path = tmp_path / "separated.xyz"
    frame = "3\nseparated\nH 0 6 0\nC -2 0 0\nO 2 0 0\n"
    path.write_text(frame * 2)
    cmap_file = tmp_path / "pairs.txt"
    cmap_file.write_text("2 3 1\n")
    config = RenderConfig(
        canvas_size=320, cmap_range=(0, 1), cmap_palette="viridis", fog=False, gradient=False, auto_orient=False
    )
    mol = load(path)
    for name in ("rotation", "trajectory"):
        svgs.clear()
        output = tmp_path / f"{name}.gif"
        if name == "rotation":
            render_gif(mol, config=config, gif_rot="z", rot_frames=2, only="2,3", bond_cmap=cmap_file, output=output)
        else:
            config.bond_cmap = {(1, 2): 1.0}
            render_gif(path, config=config, gif_trj=True, trj_bonds=True, detect_nci=True, output=output)
        assert len(svgs) == 2
        expected = bond_color_hex(1, "viridis", 0, 1)
        for svg in svgs:
            root = ET.fromstring(svg)
            assert any(el.get("stroke") == expected and "stroke-dasharray" in el.attrib for el in root.iter())
        image = Image.open(BytesIO(output.read_bytes())).convert("RGB")
        r, g, b = tuple(int(expected[i : i + 2], 16) for i in (1, 3, 5))
        pixels = np.asarray(image, dtype=int)
        assert np.abs(pixels - (r, g, b)).sum(axis=2).min() < 60
    assert not mol.graph.edges


@pytest.mark.parametrize("bond_map", [{(0, 1): 0.5}, {}])
def test_render_gif_rejects_conflicting_maps_without_writing_output(tmp_path, bond_map):
    from pathlib import Path

    from xyzrender import load, render_gif
    from xyzrender.types import RenderConfig

    mol = load(Path(__file__).resolve().parent.parent / "examples/structures/ethanol.xyz")
    output = tmp_path / "conflicting.gif"
    with pytest.raises(ValueError, match="Atom and bond colormaps are mutually exclusive"):
        render_gif(mol, config=RenderConfig(atom_cmap={0: 0.2}), bond_cmap=bond_map, gif_rot="y", output=output)
    assert not output.exists()
