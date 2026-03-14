"""Tests for the public Python API: load(), render(), build_config(), measure().

Overlays and style params are tested in combinations where possible to
minimise the number of full render calls.
"""

from pathlib import Path

import pytest

from xyzrender import build_config, load, measure, render
from xyzrender.api import Molecule, SVGResult

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


# ---------------------------------------------------------------------------
# Fixtures — loaded once per module to avoid repeated file I/O
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def caffeine():
    return load(STRUCTURES / "caffeine.xyz")


@pytest.fixture(scope="module")
def ethanol():
    return load(STRUCTURES / "ethanol.xyz")


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_returns_molecule(caffeine):
    assert isinstance(caffeine, Molecule)
    assert caffeine.graph.number_of_nodes() > 0
    assert caffeine.cube_data is None
    assert caffeine.cell_data is None
    assert caffeine.oriented is False


def test_load_cube_sets_cube_data():
    mol = load(STRUCTURES / "caffeine_homo.cube")
    assert mol.cube_data is not None
    assert mol.graph.number_of_nodes() > 0


def test_load_cell_sets_cell_data():
    mol = load(STRUCTURES / "caffeine_cell.xyz", cell=True)
    assert mol.cell_data is not None
    assert mol.cell_data.lattice.shape == (3, 3)


def test_load_nci_detect():
    # nci_detect marks NCI edges; molecule must still load correctly
    mol = load(STRUCTURES / "ethanol.xyz", nci_detect=True)
    assert mol.graph.number_of_nodes() > 0


def test_load_smiles():
    pytest.importorskip("rdkit", reason="rdkit required")
    mol = load("CCO", smiles=True)
    assert isinstance(mol, Molecule)
    assert mol.graph.number_of_nodes() > 0


# ---------------------------------------------------------------------------
# SVGResult
# ---------------------------------------------------------------------------


def test_svgresult_str(caffeine):
    result = render(caffeine, orient=False)
    assert isinstance(result, SVGResult)
    assert str(result).startswith("<svg")
    assert "</svg>" in str(result)


def test_svgresult_jupyter_display(caffeine):
    result = render(caffeine, orient=False)
    assert result._repr_svg_().startswith("<svg")


def test_svgresult_save(caffeine, tmp_path):
    result = render(caffeine, orient=False)
    out = tmp_path / "mol.svg"
    result.save(out)
    assert out.exists()
    assert out.read_text().startswith("<svg")


# ---------------------------------------------------------------------------
# render() — basic input types
# ---------------------------------------------------------------------------


def test_render_accepts_path():
    result = render(STRUCTURES / "ethanol.xyz", orient=False)
    assert isinstance(result, SVGResult)


def test_render_accepts_molecule(caffeine):
    result = render(caffeine, orient=False)
    assert isinstance(result, SVGResult)


# ---------------------------------------------------------------------------
# render() — overlays (grouped to share render cost)
# ---------------------------------------------------------------------------


def test_render_ts_and_nci_bonds(caffeine):
    """1-indexed ts_bonds and nci_bonds are accepted and produce valid SVG."""
    result = render(caffeine, ts_bonds=[(1, 5)], nci_bonds=[(2, 7)], orient=False)
    assert str(result).startswith("<svg")


def test_render_vdw_all(ethanol):
    svg = str(render(ethanol, vdw=True, orient=False))
    assert "vg" in svg  # vdw gradient id prefix present


def test_render_vdw_specific(ethanol):
    result = render(ethanol, vdw=[1, 3], orient=False)
    assert str(result).startswith("<svg")


def test_render_idx(caffeine):
    """Index label modes: bool, 's', 'n', 'sn'."""
    for mode in (True, "s", "n", "sn"):
        svg = str(render(caffeine, idx=mode, orient=False))
        assert svg.startswith("<svg")


def test_render_atom_cmap_with_range(caffeine):
    n = caffeine.graph.number_of_nodes()
    cmap = {i + 1: float(i) / n for i in range(n)}
    result = render(caffeine, cmap=cmap, cmap_range=(0.0, 1.0), orient=False)
    assert str(result).startswith("<svg")


# ---------------------------------------------------------------------------
# render() — hydrogen flags
# ---------------------------------------------------------------------------


def test_render_hy_flags(caffeine):
    svg_all = str(render(caffeine, hy=True, orient=False))
    svg_none = str(render(caffeine, no_hy=True, orient=False))
    # Show-all should produce more atoms than hide-all
    assert svg_all.count("<circle") > svg_none.count("<circle")


def test_render_hy_specific(caffeine):
    # Just must not raise
    render(caffeine, hy=[1], orient=False)


# ---------------------------------------------------------------------------
# render() — style params
# ---------------------------------------------------------------------------


def test_render_style_params(ethanol):
    """Multiple style overrides combined in one render."""
    import re

    result = render(
        ethanol,
        canvas_size=300,
        atom_scale=1.2,
        bond_width=5,
        gradient=False,
        fog=False,
        orient=False,
    )
    svg = str(result)
    m = re.search(r'width="(\d+)"', svg)
    assert m is not None
    assert int(m.group(1)) <= 300


def test_render_preset_flat(caffeine):
    result = render(caffeine, config="flat", orient=False)
    assert str(result).startswith("<svg")


def test_render_preset_paton(caffeine):
    result = render(caffeine, config="paton", orient=False)
    assert str(result).startswith("<svg")


# ---------------------------------------------------------------------------
# render() — pre-built RenderConfig
# ---------------------------------------------------------------------------


def test_render_prebuilt_config_reuse(caffeine, ethanol):
    """Same pre-built config applied to two molecules without mutation."""
    cfg = build_config("flat", atom_scale=1.3, gradient=False)
    r1 = render(caffeine, config=cfg)
    r2 = render(ethanol, config=cfg)
    assert str(r1).startswith("<svg")
    assert str(r2).startswith("<svg")
    # Original config is not mutated by render()
    assert cfg.atom_scale == pytest.approx(1.3)


def test_render_prebuilt_config_with_overlay(caffeine):
    """Pre-built config + per-render overlay (ts_bonds + idx)."""
    cfg = build_config("default")
    result = render(caffeine, config=cfg, ts_bonds=[(1, 5)], idx=True, orient=False)
    assert str(result).startswith("<svg")


# ---------------------------------------------------------------------------
# render() — orient flag
# ---------------------------------------------------------------------------


def test_render_orient_false(caffeine):
    result = render(caffeine, orient=False)
    assert str(result).startswith("<svg")


def test_render_mol_oriented_flag_suppresses_pca():
    """mol.oriented=True disables PCA without explicit orient=False."""
    mol = load(STRUCTURES / "caffeine.xyz")
    mol.oriented = True
    result = render(mol)
    assert str(result).startswith("<svg")


# ---------------------------------------------------------------------------
# render() — annotations
# ---------------------------------------------------------------------------


def test_render_inline_labels(caffeine):
    result = render(caffeine, labels=["1 2 d", "1 2 3 a"], orient=False)
    assert str(result).startswith("<svg")


def test_render_label_file():
    result = render(
        STRUCTURES / "sn2.out",
        label_file=str(STRUCTURES / "sn2_label.txt"),
        orient=False,
    )
    assert str(result).startswith("<svg")


# ---------------------------------------------------------------------------
# build_config()
# ---------------------------------------------------------------------------


def test_build_config_orient_param():
    cfg_off = build_config("default", orient=False)
    assert cfg_off.auto_orient is False
    cfg_on = build_config("default", orient=True)
    assert cfg_on.auto_orient is True


def test_build_config_ts_and_nci_bonds():
    """build_config expects 0-indexed pairs (internal convention)."""
    cfg = build_config("default", ts_bonds=[(0, 4)], nci_bonds=[(1, 6)])
    assert cfg.ts_bonds == [(0, 4)]
    assert cfg.nci_bonds == [(1, 6)]


def test_build_config_vdw_indices():
    cfg_all = build_config("default", vdw_indices=[])
    assert cfg_all.vdw_indices == []
    cfg_sel = build_config("default", vdw_indices=[0, 2])
    assert cfg_sel.vdw_indices == [0, 2]
    cfg_off = build_config("default")
    assert cfg_off.vdw_indices is None


def test_build_config_show_indices():
    cfg = build_config("default", show_indices=True, idx_format="s")
    assert cfg.show_indices is True
    assert cfg.idx_format == "s"


def test_build_config_cmap_range():
    cfg = build_config("default", cmap_range=(-1.0, 1.0))
    assert cfg.cmap_range == (-1.0, 1.0)


def test_build_config_returns_render_config():
    from xyzrender.types import RenderConfig

    cfg = build_config("default")
    assert isinstance(cfg, RenderConfig)


# ---------------------------------------------------------------------------
# measure()
# ---------------------------------------------------------------------------


def test_measure_all_keys(caffeine):
    data = measure(caffeine)
    assert set(data.keys()) == {"distances", "angles", "dihedrals"}
    # Distances are 3-tuples: (i, j, Å)
    assert len(data["distances"]) > 0
    _i, _j, d = data["distances"][0]
    assert 0.5 < d < 3.5


def test_measure_modes_subset(caffeine):
    data = measure(caffeine, modes=["d", "a"])
    assert "distances" in data
    assert "angles" in data
    assert "dihedrals" not in data


def test_measure_distances_only(ethanol):
    data = measure(ethanol, modes=["d"])
    assert list(data.keys()) == ["distances"]
    for _, _, d in data["distances"]:
        assert 0.5 < d < 3.5


def test_measure_from_path():
    data = measure(STRUCTURES / "ethanol.xyz")
    assert "distances" in data
    assert len(data["distances"]) > 0


# ---------------------------------------------------------------------------
# render() — SVG structure checks (gradient / fog rendering modes)
# ---------------------------------------------------------------------------


def test_render_gradient_uses_defs_and_circles(caffeine):
    """Gradient mode defines radialGradient in <defs> and renders inline <circle fill=url(#...)>."""
    svg = str(render(caffeine, gradient=True, orient=False))
    assert "<defs>" in svg
    assert "radialGradient" in svg
    assert 'fill="url(#' in svg
    assert "<use" not in svg


def test_render_fog_without_gradient_uses_circles(ethanol):
    """Fog-only mode renders individual circles, not gradient defs."""
    svg = str(render(ethanol, fog=True, gradient=False, orient=False))
    assert "<circle" in svg
    assert "<use" not in svg


def test_render_gradient_and_fog_combined(caffeine):
    svg = str(render(caffeine, gradient=True, fog=True, orient=False))
    assert "<defs>" in svg
    assert "radialGradient" in svg
    assert "<use" not in svg


# ---------------------------------------------------------------------------
# render() — remaining style params
# ---------------------------------------------------------------------------


def test_render_background(ethanol):
    svg = str(render(ethanol, background="#ff0000", orient=False))
    assert "#ff0000" in svg


def test_render_bond_orders_off(caffeine):
    svg = str(render(caffeine, bo=False, orient=False))
    assert "<svg" in svg


def test_render_color_overrides_via_prebuilt_config(ethanol):
    """color_overrides is set on a pre-built config then passed to render()."""
    cfg = build_config("flat")  # no gradient so color appears on atoms directly
    cfg.color_overrides = {"O": "#00ff00"}
    svg = str(render(ethanol, config=cfg, orient=False))
    assert "#00ff00" in svg


# ---------------------------------------------------------------------------
# render() — various molecules
# ---------------------------------------------------------------------------


def test_render_benzene():
    """Aromatic molecule renders without error."""
    svg = str(render(STRUCTURES / "benzene.xyz", bo=True, hy=True, orient=False))
    assert "<line" in svg


def test_render_asparagine():
    svg = str(render(STRUCTURES / "asparagine.xyz", orient=False))
    assert svg.startswith("<svg")


# ---------------------------------------------------------------------------
# render() — format files (end-to-end load + render)
# ---------------------------------------------------------------------------


def test_render_mol2():
    svg = str(render(STRUCTURES / "water_mol2.mol2", orient=False))
    assert svg.startswith("<svg")
    assert "<circle" in svg


def test_render_pdb():
    svg = str(render(STRUCTURES / "ala_phe_ala.pdb", orient=False))
    assert svg.startswith("<svg")
    assert "<circle" in svg


def test_render_sdf():
    pytest.importorskip("rdkit", reason="rdkit required")
    svg = str(render(STRUCTURES / "caffeine_sdf.sdf", orient=False))
    assert svg.startswith("<svg")
    assert "<circle" in svg


@pytest.mark.filterwarnings("ignore::UserWarning:ase")
def test_render_cif_with_cell_data():
    pytest.importorskip("ase", reason="ase required")
    mol = load(STRUCTURES / "caffeine_cif.cif")
    from xyzrender.types import CellData

    assert isinstance(mol.cell_data, CellData)
    svg = str(render(mol, orient=False))
    assert svg.startswith("<svg")
