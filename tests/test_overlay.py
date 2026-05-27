"""Tests for the overlay module and render(overlay=)."""

import copy
from pathlib import Path

import numpy as np
import pytest

from xyzrender import load, render
from xyzrender.config import build_config
from xyzrender.overlay import align, kabsch_align, merge_graphs

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


@pytest.fixture(scope="module")
def caffeine():
    return load(STRUCTURES / "caffeine.xyz")


@pytest.fixture(scope="module")
def ethanol():
    return load(STRUCTURES / "ethanol.xyz")


# ---------------------------------------------------------------------------
# align()
# ---------------------------------------------------------------------------


def test_align_identity_zero_rmsd(caffeine):
    """Aligning a molecule with itself returns near-zero RMSD."""
    g = caffeine.graph
    aligned = align(g, copy.deepcopy(g))
    pos1 = np.array([g.nodes[n]["position"] for n in g.nodes()], dtype=float)
    assert float(np.sqrt(np.mean((aligned - pos1) ** 2))) < 1e-6


def test_align_rotated_molecule(caffeine):
    """Kabsch rotation recovers the original frame after a 90° rotation."""
    g = caffeine.graph
    g2 = copy.deepcopy(g)
    nodes = list(g2.nodes())
    pos = np.array([g2.nodes[n]["position"] for n in nodes], dtype=float)
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    pos_rot = pos @ rot.T
    for k, nid in enumerate(nodes):
        g2.nodes[nid]["position"] = tuple(float(v) for v in pos_rot[k])
    aligned = align(g, g2)
    pos1 = np.array([g.nodes[n]["position"] for n in g.nodes()], dtype=float)
    assert float(np.sqrt(np.mean((aligned - pos1) ** 2))) < 1e-4


def test_align_mismatched_atoms_mcs_fallback():
    """Different atom counts trigger MCS alignment instead of raising."""
    benzene = load(STRUCTURES / "benzene.xyz")
    anthracene = load(STRUCTURES / "anthracene.xyz")
    # Should succeed via MCS fallback, not raise
    aligned = align(benzene.graph, anthracene.graph)
    assert aligned.shape == (anthracene.graph.number_of_nodes(), 3)


# ---------------------------------------------------------------------------
# merge_graphs()
# ---------------------------------------------------------------------------


def test_merge_graphs_structure(caffeine):
    """Merged graph has 2xn nodes; mol2 nodes carry structure_color and bond_color_override."""
    g = caffeine.graph
    aligned = align(g, copy.deepcopy(g))
    merged = merge_graphs(g, copy.deepcopy(g), aligned, build_config("default"))

    n = g.number_of_nodes()
    assert merged.number_of_nodes() == 2 * n

    mol2_nodes = [nid for nid in merged.nodes() if nid >= n]
    assert all(merged.nodes[nid]["molecule_index"] == 1 for nid in mol2_nodes)
    assert all(merged.nodes[nid].get("structure_color", "").startswith("#") for nid in mol2_nodes)

    mol2_edges = [(i, j, d) for i, j, d in merged.edges(data=True) if i >= n or j >= n]
    assert all(d.get("bond_color_override", "").startswith("#") for _, _, d in mol2_edges)


def test_merge_graphs_overlay_atom_scale(caffeine):
    """cfg.overlay.atom_scale stamps structure_atom_scale as an ABSOLUTE value on mol2 nodes."""
    g = caffeine.graph
    aligned = align(g, copy.deepcopy(g))
    cfg = build_config("default")
    cfg.atom_scale = 2.5  # primary
    cfg.overlay.atom_scale = 1.0  # overlay absolute — should NOT multiply with primary
    merged = merge_graphs(g, copy.deepcopy(g), aligned, cfg)
    n = g.number_of_nodes()
    for nid in merged.nodes():
        if nid >= n:
            # Absolute semantics: stamped value is 1.0 regardless of cfg.atom_scale=2.5.
            assert merged.nodes[nid]["structure_atom_scale"] == pytest.approx(1.0)
        else:
            assert "structure_atom_scale" not in merged.nodes[nid]


def test_overlay_bond_color_decouples_from_atom_color(caffeine):
    """`bond_color` wins over the auto-darkened `color` for overlay bonds."""
    from xyzrender import OverlayConfig
    from xyzrender.colors import Color, bond_color_from_atom, resolve_color

    # The auto-darkened hex derived from color="teal" must NOT leak into the SVG
    # when an explicit bond_color is set.
    darkened_teal = bond_color_from_atom(Color.from_str(resolve_color("teal")))
    navy = resolve_color("navy")
    svg = str(
        render(
            caffeine,
            overlay=caffeine,
            overlay_config=OverlayConfig(color="teal", bond_color="navy"),
            orient=False,
        )
    )
    assert navy in svg, "explicit bond_color missing from output"
    assert darkened_teal not in svg, "darkened-teal leaked through despite explicit bond_color"


def test_merge_graphs_overlay_bond_width(caffeine):
    """cfg.overlay.bond_width stamps bond_width_override on every mol2 edge."""
    g = caffeine.graph
    aligned = align(g, copy.deepcopy(g))
    cfg = build_config("default")
    cfg.overlay.bond_width = 2.0
    merged = merge_graphs(g, copy.deepcopy(g), aligned, cfg)
    n = g.number_of_nodes()
    for i, j, d in merged.edges(data=True):
        if i >= n or j >= n:
            assert d["bond_width_override"] == pytest.approx(2.0)
        else:
            assert "bond_width_override" not in d


def test_merge_graphs_overlay_stroke_and_outline(caffeine):
    """Stroke + outline overlay overrides land on the right attrs."""
    g = caffeine.graph
    aligned = align(g, copy.deepcopy(g))
    cfg = build_config("default")
    cfg.overlay.atom_stroke_width = 0.5
    cfg.overlay.atom_stroke_color = "dimgray"
    cfg.overlay.bond_outline_width = 0.0
    cfg.overlay.bond_outline_color = "#123456"
    merged = merge_graphs(g, copy.deepcopy(g), aligned, cfg)
    n = g.number_of_nodes()
    for nid in merged.nodes():
        if nid >= n:
            assert merged.nodes[nid]["structure_atom_stroke_width"] == pytest.approx(0.5)
            assert merged.nodes[nid]["structure_atom_stroke_color"] == "dimgray"
        else:
            assert "structure_atom_stroke_width" not in merged.nodes[nid]
    for i, j, d in merged.edges(data=True):
        if i >= n or j >= n:
            assert d["bond_outline_width_override"] == pytest.approx(0.0)
            assert d["bond_outline_color_override"] == "#123456"
        else:
            assert "bond_outline_width_override" not in d


def test_overlay_atom_scale_absolute_vs_multiplier(caffeine):
    """End-to-end: overlay.atom_scale=1 with primary atom_scale=2.5 must render
    overlay atoms strictly smaller than primary atoms — proves absolute semantics."""
    import re

    cfg = build_config("default", atom_scale=2.5)
    from xyzrender import OverlayConfig

    svg = str(
        render(
            caffeine,
            overlay=caffeine,
            config=cfg,
            overlay_config=OverlayConfig(atom_scale=1.0),
            orient=False,
            gradient=False,
            fog=False,
        )
    )
    # Collect all circle r="..." values; they come in bond-order pairs per atom
    # but the distinct radii bucket should contain two clusters: primary (~2.5x) and overlay (~1.0x).
    radii = sorted({float(m) for m in re.findall(r'<circle[^>]*r="([0-9.]+)"', svg)})
    assert len(radii) >= 2  # at least two size classes
    # Largest primary radius must be > 1.5x the smallest overlay radius.
    assert max(radii) > 1.5 * min(radii), (
        f"expected a clear size gap between primary (atom_scale=2.5) and overlay (atom_scale=1.0); got {radii}"
    )


def test_overlay_no_align_follows_pca_orientation(caffeine):
    """Under --no-align, mol2 inherits the same PCA rotation applied to mol1.

    Without this, mol1 gets PCA-rotated + centred while mol2 stays in the file
    frame, and two already-aligned inputs visibly separate.
    """
    import numpy as np

    from xyzrender.api import Molecule, _apply_overlay

    mol1 = caffeine
    mol2 = load(STRUCTURES / "caffeine.xyz")  # identical file

    cfg = build_config("default")
    cfg.auto_align = False
    # auto_orient defaults to True → PCA runs inside _apply_overlay
    rmol = Molecule(graph=copy.deepcopy(mol1.graph))
    merged_mol = _apply_overlay(
        mol1,
        rmol,
        cfg,
        mol2,
        overlay_color=None,
        overlay_opacity=None,
        align_atoms=None,
        has_surfaces=False,
    )
    n = mol1.graph.number_of_nodes()
    for i in range(n):
        p1 = np.array(merged_mol.graph.nodes[i]["position"])
        p2 = np.array(merged_mol.graph.nodes[n + i]["position"])
        # xy must coincide (z has _Z_NUDGE so allow a small tolerance there).
        assert np.allclose(p1[:2], p2[:2], atol=1e-6)


def test_overlay_no_align_keeps_raw_positions(caffeine, tmp_path):
    """--no-align skips Kabsch; mol2 keeps its original coordinates."""
    # Translate a copy of caffeine so its Kabsch-aligned form would differ from raw.
    mol2 = copy.deepcopy(caffeine)
    for n in mol2.graph.nodes():
        x, y, z = mol2.graph.nodes[n]["position"]
        mol2.graph.nodes[n]["position"] = (x + 5.0, y + 5.0, z + 5.0)

    from xyzrender.api import _apply_overlay

    cfg = build_config("default")
    cfg.auto_align = False
    cfg.auto_orient = False
    from xyzrender import Molecule

    rmol = Molecule(graph=copy.deepcopy(caffeine.graph))
    merged_mol = _apply_overlay(
        caffeine,
        rmol,
        cfg,
        mol2,
        overlay_color=None,
        overlay_opacity=None,
        align_atoms=None,
        has_surfaces=False,
    )
    # First mol2 atom (original id 0 → merged id n) should have the translated position.
    n1 = caffeine.graph.number_of_nodes()
    mol2_first = next(nid for nid in merged_mol.graph.nodes() if nid >= n1)
    raw_x, raw_y, raw_z = mol2.graph.nodes[0]["position"]
    got_x, got_y, got_z = merged_mol.graph.nodes[mol2_first]["position"]
    assert got_x == pytest.approx(raw_x)
    assert got_y == pytest.approx(raw_y)
    # z has a _Z_NUDGE applied; only check it's close to raw
    assert abs(got_z - raw_z) < 1e-2


def test_overlay_unbond_scoped_to_mol2(caffeine):
    """cfg.overlay.unbond removes bonds on the overlay only, not on the base."""
    from xyzrender import OverlayConfig

    base_edges_before = caffeine.graph.number_of_edges()
    svg = str(
        render(
            caffeine,
            overlay=caffeine,
            overlay_config=OverlayConfig(unbond=["all"]),
            orient=False,
        )
    )
    assert svg.startswith("<svg")
    # Base graph must be untouched.
    assert caffeine.graph.number_of_edges() == base_edges_before


def test_merge_graphs_overlay_opacity(caffeine):
    """cfg.overlay.opacity stamps structure_opacity on every mol2 node."""
    g = caffeine.graph
    aligned = align(g, copy.deepcopy(g))
    cfg = build_config("default")
    cfg.overlay.opacity = 0.3
    merged = merge_graphs(g, copy.deepcopy(g), aligned, cfg)

    n = g.number_of_nodes()
    for nid in merged.nodes():
        if nid >= n:
            assert merged.nodes[nid]["structure_opacity"] == pytest.approx(0.3)
        else:
            assert "structure_opacity" not in merged.nodes[nid]


def test_merge_graphs_aggregates_aromatic_rings():
    """Overlay's aromatic rings are translated through id_map and unioned into the merged graph."""
    benzene = load(STRUCTURES / "benzene.xyz")
    g1 = benzene.graph
    g2 = copy.deepcopy(g1)
    aligned = align(g1, g2)
    merged = merge_graphs(g1, g2, aligned, build_config("default"))

    n = g1.number_of_nodes()
    rings = merged.graph.get("aromatic_rings", [])
    # mol1 ring uses IDs 0..n-1; mol2 ring uses IDs n..2n-1 — both should be present.
    assert any(all(a < n for a in ring) for ring in rings), "mol1 ring missing"
    assert any(all(a >= n for a in ring) for ring in rings), (
        "mol2 ring missing (haptic would skip overlay without this)"
    )


# ---------------------------------------------------------------------------
# render(overlay=)
# ---------------------------------------------------------------------------


def test_overlay_does_not_mutate_callers_config(caffeine):
    """Passing a pre-built RenderConfig and then overriding overlay settings via
    flat kwargs must not leak mutations back into the caller's config object.
    """
    from xyzrender import build_config

    my_cfg = build_config("default")
    my_cfg.overlay.color = "teal"
    my_cfg.overlay.opacity = 0.9
    baseline = (my_cfg.overlay.color, my_cfg.overlay.opacity, len(my_cfg.style_regions))

    render(caffeine, overlay=caffeine, config=my_cfg, overlay_color="red", opacity=0.3, orient=False)
    # The user's cfg.overlay should still be teal at 0.9 opacity, style_regions
    # should be untouched even though _apply_overlay ran.
    assert (my_cfg.overlay.color, my_cfg.overlay.opacity, len(my_cfg.style_regions)) == baseline


def test_overlay_full_config_attaches_as_style_region(caffeine):
    """OverlayConfig.config is attached as a StyleRegion over mol2 atoms so
    every RenderConfig field the renderer consults via _acfg takes effect."""
    from xyzrender import OverlayConfig, build_config

    tube = build_config("tube")
    base_default = str(render(caffeine, overlay=caffeine, overlay_config=OverlayConfig(), orient=False))
    with_tube = str(render(caffeine, overlay=caffeine, overlay_config=OverlayConfig(config=tube), orient=False))
    # The inner config must produce a different SVG — otherwise it wasn't applied.
    assert base_default != with_tube


def test_overlay_show_filters_mol2_atoms(caffeine):
    """OverlayConfig.show keeps only matching atoms of the overlay; alignment uses the full scaffold."""
    from xyzrender import OverlayConfig

    # caffeine has 14 heavy atoms (4 of them nitrogens).
    merged = str(render(caffeine, overlay=caffeine, overlay_config=OverlayConfig(show=["N"]), orient=False))
    import re

    assert len(re.findall(r"<circle", merged)) == 14 + 4  # primary + overlay nitrogens


def test_render_overlay_produces_svg(caffeine):
    svg = str(render(caffeine, overlay=STRUCTURES / "caffeine.xyz", orient=False))
    assert svg.startswith("<svg")
    assert "</svg>" in svg


def test_render_overlay_color_appears_in_svg(caffeine):
    from xyzrender.colors import resolve_color

    svg = str(render(caffeine, overlay=caffeine, overlay_color="steelblue", gradient=False, fog=False, orient=False))
    assert resolve_color("steelblue") in svg


def test_mol_color_plus_overlay_color_keeps_both(caffeine):
    """--mol-color on the primary must not over-paint the overlay's structure_color.

    Regression: prior behaviour painted every atom with cfg.mol_color, wiping the
    overlay's colour — rendering both structures in the primary's flat colour.
    """
    from xyzrender.colors import resolve_color

    svg = str(
        render(
            caffeine,
            overlay=caffeine,
            mol_color="steelblue",
            overlay_color="coral",
            orient=False,
            gradient=False,
            fog=False,
        )
    )
    assert resolve_color("steelblue") in svg, "primary mol_color missing"
    assert resolve_color("coral") in svg, "overlay colour over-painted by mol_color"


def test_render_overlay_mutual_exclusion_surface(caffeine):
    with pytest.raises(ValueError, match="mutually exclusive"):
        render(caffeine, overlay=caffeine, dens=True, orient=False)


def test_render_overlay_mutual_exclusion_cell():
    mol = load(STRUCTURES / "caffeine_cell.xyz", cell=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        render(mol, overlay=mol, orient=False)


def test_render_overlay_different_atom_counts():
    """Overlay with different atom counts uses MCS alignment."""
    benzene = load(STRUCTURES / "benzene.xyz")
    anthracene = load(STRUCTURES / "anthracene.xyz")
    svg = str(render(benzene, overlay=anthracene, orient=False))
    assert svg.startswith("<svg")
    assert "</svg>" in svg


def test_render_overlay_very_different_molecules(caffeine, ethanol):
    """Overlay of very different molecules either succeeds (small match) or raises."""
    # Caffeine and ethanol share C/O/H so geometric MCS may find a small match
    try:
        svg = str(render(caffeine, overlay=ethanol, orient=False))
        assert svg.startswith("<svg")
    except ValueError:
        pass  # acceptable if no common substructure found


# ---------------------------------------------------------------------------
# kabsch_align() — shared alignment function
# ---------------------------------------------------------------------------


def test_kabsch_align_identity():
    """Aligning identical positions returns them unchanged."""
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    aligned = kabsch_align(pos, pos.copy())
    assert float(np.sqrt(np.mean((aligned - pos) ** 2))) < 1e-10


def test_kabsch_align_with_align_atoms():
    """Subset alignment: fit on 3 atoms, transform applies to all."""
    ref = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]], dtype=float)
    # Rotate the mobile by 90 deg around z
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    mobile = ref @ rot.T
    aligned = kabsch_align(ref, mobile, align_atoms=[0, 1, 2])
    # The first 3 atoms should be well-aligned
    assert float(np.sqrt(np.mean((aligned[:3] - ref[:3]) ** 2))) < 1e-4
    # All atoms should be aligned since it's a rigid rotation
    assert float(np.sqrt(np.mean((aligned - ref) ** 2))) < 1e-4


def test_kabsch_align_too_few_atoms_raises():
    """align_atoms with fewer than 3 indices should raise."""
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    with pytest.raises(ValueError, match="at least 3"):
        kabsch_align(pos, pos.copy(), align_atoms=[0, 1])


def test_render_overlay_with_align_atoms(caffeine):
    """Overlay with align_atoms renders without error."""
    # Use 1-indexed atoms for the render API
    svg = str(render(caffeine, overlay=caffeine, align_atoms=[1, 2, 3], orient=False))
    assert svg.startswith("<svg")


def test_render_overlay_opacity_via_api(caffeine):
    """render(overlay=..., opacity=0.3) produces SVG with transparent overlay atoms."""
    svg = str(render(caffeine, overlay=caffeine, opacity=0.3, orient=False))
    assert svg.startswith("<svg")
    # fill-opacity or opacity="0.30" appears for the overlay atoms/bonds.
    assert 'opacity="0.30"' in svg


def _write_metallo_benzene(path: Path) -> None:
    """Benzene ring (D6h) with an Fe atom centred above it — triggers haptic."""
    # Planar benzene C6 (r=1.4 Å) + 6 Hs (r=2.5 Å) + Fe at z=1.65 Å.
    coords = []
    import math

    for k in range(6):
        ang = k * math.pi / 3
        coords.append(("C", 1.4 * math.cos(ang), 1.4 * math.sin(ang), 0.0))
    for k in range(6):
        ang = k * math.pi / 3
        coords.append(("H", 2.5 * math.cos(ang), 2.5 * math.sin(ang), 0.0))
    coords.append(("Fe", 0.0, 0.0, 1.65))

    lines = [str(len(coords)), "metallo-benzene"]
    lines.extend(f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, x, y, z in coords)
    path.write_text("\n".join(lines) + "\n")


def test_render_overlay_haptic_applies_to_mol2(tmp_path, caffeine):
    """--haptic replaces eta-coordination bonds on the OVERLAY molecule, not just the base.

    Without aromatic-ring aggregation during merge, _iter_pi_groups can't see mol2's ring
    and silently skips the overlay — so this test catches regressions in that pathway.
    """
    mb_path = tmp_path / "metallo_benzene.xyz"
    _write_metallo_benzene(mb_path)

    # Overlay the metal-benzene onto caffeine; haptic should introduce a centroid dummy atom
    # whose bonds attach mol2's Fe to a "*" node (rather than 6 Fe-C bonds).
    from xyzrender import load as _load
    from xyzrender.bond_rules import apply_bond_rules
    from xyzrender.overlay import align as _align
    from xyzrender.overlay import merge_graphs as _merge_graphs

    mb = _load(mb_path)
    aligned = _align(caffeine.graph, mb.graph)
    merged = _merge_graphs(caffeine.graph, copy.deepcopy(mb.graph), aligned, build_config("default"))

    cfg = build_config("default")
    cfg.haptic = True
    apply_bond_rules(merged, cfg)

    centroids = [nid for nid in merged.nodes() if merged.nodes[nid].get("symbol") == "*"]
    assert centroids, "haptic did not create a centroid node on the overlay"

    # The centroid node + its bond must inherit the overlay's per-structure
    # style so the haptic bond renders in the overlay colour, not the default.
    for c in centroids:
        assert merged.nodes[c].get("molecule_index") == 1, "centroid missing overlay molecule_index"
        assert merged.nodes[c].get("structure_color", "").startswith("#")
        ((_, bond_data),) = [(nb, merged.edges[c, nb]) for nb in merged.neighbors(c)]
        assert bond_data.get("molecule_index") == 1
        assert bond_data.get("bond_color_override", "").startswith("#")


def test_render_overlay_with_haptic_public_api(tmp_path):
    """Public-API equivalent of the test above — `render(mol, overlay=other, haptic=True)`
    must produce haptic centroids on BOTH the base and the overlay, not just the base.

    `cfg.haptic` is treated as a global setting that runs post-merge on the full graph
    (see api.py:2445 `_ov_cfg.haptic = False  # haptic is global; runs post-merge…`).
    This is the seed test for the "haptic on align/overlay" class of bugs the user
    flagged — extend it as new failure modes are found."""
    from unittest.mock import patch

    mb_path = tmp_path / "metallo_benzene.xyz"
    _write_metallo_benzene(mb_path)
    base = load(mb_path)  # base = metallo-benzene → triggers haptic on base
    overlay_mol = load(mb_path)  # overlay = also metallo-benzene → triggers haptic on overlay

    captured: dict = {}

    def _spy(graph, cfg, **_kwargs):
        captured["graph"] = graph
        captured["cfg"] = cfg
        return "<svg/>"

    with patch("xyzrender.renderer.render_svg", side_effect=_spy):
        render(base, overlay=overlay_mol, haptic=True, orient=False)

    g = captured["graph"]
    centroids = [n for n in g.nodes() if g.nodes[n].get("symbol") == "*"]
    assert len(centroids) == 2, (
        f"Expected one haptic centroid each on base + overlay (=2), got {len(centroids)}. "
        "render(haptic=True, overlay=…) must apply haptic to the merged graph so the "
        "overlay's eta-coordination is collapsed too — not just the base."
    )

    # Exactly one of those centroids must carry molecule_index=1 (the overlay's),
    # and one must carry molecule_index=0 (the base's). If both are 0, the
    # overlay was haptic-processed but not stamped with its structure index.
    indices = sorted(g.nodes[c].get("molecule_index", 0) for c in centroids)
    assert indices == [0, 1], (
        f"Haptic centroids should carry distinct molecule_index values [0, 1] for base + overlay; got {indices}"
    )


# ---------------------------------------------------------------------------
# --overlay-ts / --overlay-ts-bond
# ---------------------------------------------------------------------------


def test_overlay_ts_bond_marks_edge_in_merged_graph(caffeine):
    """Manual overlay-ts-bond translates overlay-local indices through the id_map."""
    from xyzrender import OverlayConfig

    g = caffeine.graph
    n = g.number_of_nodes()
    cfg = build_config("default")
    cfg.overlay = OverlayConfig(ts_bonds=[(0, 1)])  # overlay atoms 1-2 (0-indexed)
    aligned = align(g, copy.deepcopy(g))
    merged = merge_graphs(g, copy.deepcopy(g), aligned, cfg)

    # Overlay's atom-0/atom-1 → merged nodes n+0, n+1.  Edge should carry TS=True.
    assert merged.has_edge(n, n + 1)
    assert merged[n][n + 1].get("TS") is True

    # Primary atoms 0-1 stay un-TS-stamped — the overlay flag is scoped to the overlay.
    if g.has_edge(0, 1):
        assert merged[0][1].get("TS", False) is False


def test_overlay_ts_bond_out_of_range_raises(caffeine):
    """Out-of-range overlay-ts-bond pair surfaces a clear ValueError."""
    from xyzrender import OverlayConfig

    g = caffeine.graph
    cfg = build_config("default")
    cfg.overlay = OverlayConfig(ts_bonds=[(0, 999)])  # 999 is way past caffeine's atom count
    aligned = align(g, copy.deepcopy(g))

    with pytest.raises(ValueError, match="overlay-ts-bond"):
        merge_graphs(g, copy.deepcopy(g), aligned, cfg)


def test_overlay_ts_bond_new_edge_inherits_overlay_colour(caffeine):
    """A TS bond between non-bonded overlay atoms (forming bond) inherits overlay colour."""
    from xyzrender import OverlayConfig

    g = caffeine.graph
    n = g.number_of_nodes()
    # Pick two overlay atoms that are NOT covalently bonded in caffeine.
    non_bonded = next((i, j) for i in range(min(n, 10)) for j in range(i + 1, min(n, 10)) if not g.has_edge(i, j))
    cfg = build_config("default")
    cfg.overlay = OverlayConfig(color="hotpink", ts_bonds=[non_bonded])
    aligned = align(g, copy.deepcopy(g))
    merged = merge_graphs(g, copy.deepcopy(g), aligned, cfg)

    u, v = n + non_bonded[0], n + non_bonded[1]
    assert merged.has_edge(u, v)
    assert merged[u][v]["TS"] is True
    # New TS edge must carry bond_color_override; otherwise it falls back to default.
    assert "bond_color_override" in merged[u][v]
    assert merged[u][v]["bond_color_override"].startswith("#")


def test_overlay_no_ts_bond_leaves_overlay_edges_unmarked(caffeine):
    """Without overlay-ts-bond / --overlay-ts, no overlay edge gets TS=True."""
    g = caffeine.graph
    n = g.number_of_nodes()
    aligned = align(g, copy.deepcopy(g))
    merged = merge_graphs(g, copy.deepcopy(g), aligned, build_config("default"))
    overlay_edges = [(i, j, d) for i, j, d in merged.edges(data=True) if i >= n or j >= n]
    assert not any(d.get("TS") for _, _, d in overlay_edges)
