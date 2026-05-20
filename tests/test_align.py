"""Alignment strategies.

Covers:

* ``align_with_selection`` — user-driven: takes selector-resolved atom sets,
  runs MCS on the induced subgraph with K-subset Kabsch fallback.
* ``best_fit_align`` — spec-free PCA-seeded ICP fallback.
* ``kabsch_with_pivot`` — Kabsch with exact pivot coincidence.
* ``_align_metal_fragments`` — auto-path organometallic overlay (per-M-pairing).
"""

from __future__ import annotations

import copy

import networkx as nx
import numpy as np
import pytest

from xyzrender.align import (
    _align_metal_fragments,
    align_with_selection,
    best_fit_align,
    kabsch_with_pivot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotation_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rigid_transform_graph(g: nx.Graph, rot: np.ndarray, translate: np.ndarray) -> nx.Graph:
    h = copy.deepcopy(g)
    for n in h.nodes():
        p = np.array(h.nodes[n]["position"], dtype=float) @ rot.T + translate
        h.nodes[n]["position"] = tuple(p)
    return h


def _build_mxy_complex(metal: str, ligands: list[str]) -> nx.Graph:
    """Octahedral-ish metal + ligands; metal bonded to every ligand."""
    g = nx.Graph()
    g.add_node(0, symbol=metal, position=(0.0, 0.0, 0.0))
    axes = [(2.0, 0, 0), (-2.0, 0, 0), (0, 2.0, 0), (0, -2.0, 0), (0, 0, 2.0), (0, 0, -2.0)]
    for i, lig in enumerate(ligands):
        g.add_node(i + 1, symbol=lig, position=axes[i % len(axes)])
        g.add_edge(0, i + 1)
    return g


def _build_alkoxide() -> nx.Graph:
    """Li-O-C-C-H chain + an isolated Li + isolated O (no Li-O bond)."""
    g = nx.Graph()
    atoms = [
        ("Li", (0.0, 0.0, 0.0)),
        ("O", (1.8, 0.0, 0.0)),
        ("C", (3.0, 0.0, 0.0)),
        ("C", (3.0, 1.5, 0.0)),
        ("H", (4.0, 1.5, 0.0)),
        ("Li", (-1.5, 1.0, 0.0)),
        ("O", (0.0, 5.0, 0.0)),
    ]
    for i, (s, p) in enumerate(atoms):
        g.add_node(i, symbol=s, position=p)
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    return g


# ---------------------------------------------------------------------------
# kabsch_with_pivot
# ---------------------------------------------------------------------------


def test_kabsch_with_pivot_locks_pivot_centroid_exactly() -> None:
    """Pivot centroid coincides post-transform; anchors drive rotation."""
    ref = np.array(
        [
            [0.0, 0.0, 0.0],  # pivot
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot = _rotation_z(60.0)
    mob = ref @ rot.T + np.array([5.0, -2.0, 3.0])
    aligned = kabsch_with_pivot(ref, mob, [0, 1, 2, 3], [0, 1, 2, 3], [0], [0])
    assert np.allclose(aligned[0], ref[0], atol=1e-9)
    assert np.allclose(aligned, ref, atol=1e-9)


def test_kabsch_with_pivot_uses_centroid_of_multiple_pivots() -> None:
    """When the pivot set has >1 atom, the centroid coincides exactly."""
    ref = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    mob = ref + np.array([3.0, 5.0, -1.0])
    aligned = kabsch_with_pivot(ref, mob, [0, 1, 2], [0, 1, 2], [0, 1], [0, 1])
    assert np.allclose(aligned[:2].mean(axis=0), ref[:2].mean(axis=0), atol=1e-9)


# ---------------------------------------------------------------------------
# best_fit_align
# ---------------------------------------------------------------------------


def test_best_fit_align_recovers_rigid_transform_on_same_molecule() -> None:
    """Self-rotation of an asymmetric graph → ICP converges to RMSD≈0."""
    g_ref = nx.Graph()
    for i, (s, p) in enumerate(
        [
            ("Pt", (0.0, 0.0, 0.0)),
            ("Cl", (2.0, 0.0, 0.0)),
            ("Br", (-2.5, 0.3, 0.2)),
            ("P", (0.1, 2.1, 0.5)),
            ("N", (-0.4, -1.8, -0.7)),
            ("C", (0.0, 0.0, 2.2)),
        ]
    ):
        g_ref.add_node(i, symbol=s, position=p)

    rot = _rotation_z(50.0)
    g_mob = _rigid_transform_graph(g_ref, rot, np.array([2.0, -1.0, 3.0]))
    _, rmsd = best_fit_align(g_ref, g_mob)
    assert rmsd < 1e-6


def test_best_fit_align_returns_finite_for_different_molecules() -> None:
    """Different graphs → best-effort, no crash."""
    g1 = nx.Graph()
    for i, (s, p) in enumerate([("C", (0, 0, 0)), ("C", (1.5, 0, 0)), ("O", (3, 0, 0))]):
        g1.add_node(i, symbol=s, position=p)
    g1.add_edges_from([(0, 1), (1, 2)])

    g2 = nx.Graph()
    for i, (s, p) in enumerate(
        [
            ("N", (0, 0, 0)),
            ("C", (1, 1, 0)),
            ("C", (2, 0, 0)),
            ("F", (3, -1, 0)),
        ]
    ):
        g2.add_node(i, symbol=s, position=p)
    g2.add_edges_from([(0, 1), (1, 2), (2, 3)])

    aligned, rmsd = best_fit_align(g1, g2)
    assert aligned.shape == (4, 3)
    assert np.isfinite(rmsd)
    assert np.all(np.isfinite(aligned))


# ---------------------------------------------------------------------------
# align_with_selection — MCS-on-subgraph
# ---------------------------------------------------------------------------


def test_align_with_selection_uses_mcs_when_subgraph_has_edges() -> None:
    """Selected atoms form a connected subgraph → MCS-driven anchor pairing.

    Rigid self-rotation of an asymmetric Pt(Cl,Br,P,N) — selector picks all
    5 atoms; the induced subgraph has 4 metal-ligand edges; MCS pairs them.
    """
    g_ref = _build_mxy_complex("Pt", ["Cl", "Br", "P", "N"])
    rot = _rotation_z(37.0)
    g_mob = _rigid_transform_graph(g_ref, rot, np.array([5.0, -1.5, 2.0]))

    atoms = list(g_ref.nodes())
    _, paired_ref, _, _, rmsd = align_with_selection(
        g_ref,
        g_mob,
        atoms,
        atoms,
    )
    assert rmsd < 1e-6
    assert len(paired_ref) >= 3


def test_align_with_selection_handles_size_mismatch_via_subset() -> None:
    """When the selected sets differ in size, K-subset Kabsch finds best match."""
    g_ref = _build_mxy_complex("Pt", ["Cl", "Cl", "Cl", "P", "N"])
    g_mob = _build_mxy_complex("Pt", ["Cl", "P"])
    rot = _rotation_z(20.0)
    g_mob = _rigid_transform_graph(g_mob, rot, np.zeros(3))

    _, paired_ref, paired_mob, _, rmsd = align_with_selection(
        g_ref,
        g_mob,
        list(g_ref.nodes()),
        list(g_mob.nodes()),
    )
    assert len(paired_ref) == len(paired_mob)
    assert len(paired_ref) == 3  # min(6, 3) anchors
    assert rmsd < 1e-6


def test_align_with_selection_errors_below_min_atoms() -> None:
    """Fewer than 3 atoms on either side → ValueError (Kabsch needs ≥3)."""
    g_ref = _build_mxy_complex("Pt", ["Cl", "P"])
    g_mob = _build_mxy_complex("Pt", ["Cl", "P"])
    with pytest.raises(ValueError, match="need ≥ 3"):
        align_with_selection(g_ref, g_mob, [0, 1], [0, 1])


# ---------------------------------------------------------------------------
# _align_metal_fragments — auto-path organometallic
# ---------------------------------------------------------------------------


def test_align_metal_fragments_single_metal_each() -> None:
    """One metal per side → trivial pairing, metal-pivot Kabsch on coord shell."""
    g_ref = _build_mxy_complex("Pt", ["Cl", "Br", "P", "N"])
    rot = _rotation_z(45.0)
    g_mob = _rigid_transform_graph(g_ref, rot, np.array([2.0, 1.0, -3.0]))

    aligned, _, _, _, rmsd = _align_metal_fragments(g_ref, g_mob)
    assert rmsd < 1e-9
    # Pivot on Pt → Pt coincides exactly post-transform.
    pt_ref_pos = np.asarray(g_ref.nodes[0]["position"])
    assert np.allclose(aligned[0], pt_ref_pos, atol=1e-9)


def test_align_metal_fragments_multiple_metals_picks_best_pairing() -> None:
    """When ref has multiple metals, per-pairing search finds the best.

    Cross-element ligand sets force the algorithm to score each metal pairing
    by full-mol RMSD and pick the lowest.
    """
    # ref has Fe + Co, both with 3-Cl shell; mob has Fe with Cl,Br,I shell.
    g_ref = nx.Graph()
    for i, (s, p) in enumerate(
        [
            ("Fe", (0, 0, 0)),
            ("Cl", (2, 0, 0)),
            ("Cl", (-2, 0, 0)),
            ("Cl", (0, 2, 0)),
            ("Co", (10, 0, 0)),
            ("Cl", (12, 0, 0)),
            ("Cl", (8, 0, 0)),
            ("Cl", (10, 2, 0)),
        ]
    ):
        g_ref.add_node(i, symbol=s, position=p)
    g_ref.add_edges_from([(0, 1), (0, 2), (0, 3), (4, 5), (4, 6), (4, 7)])

    g_mob = nx.Graph()
    for i, (s, p) in enumerate(
        [
            ("Fe", (0, 0, 0)),
            ("Cl", (2, 0, 0)),
            ("Cl", (-2, 0, 0)),
            ("Cl", (0, 2, 0)),
        ]
    ):
        g_mob.add_node(i, symbol=s, position=p)
    g_mob.add_edges_from([(0, 1), (0, 2), (0, 3)])

    _, _, _, mismatches, rmsd = _align_metal_fragments(g_ref, g_mob)
    # Both Fe-Fe (mismatch=0) and Fe-Co (cross-pair) candidates tried; same-element wins.
    assert mismatches == 0
    assert rmsd < 1e-6


def test_align_metal_fragments_errors_without_metals() -> None:
    """_align_metal_fragments precondition: both graphs must have metals."""
    g = nx.Graph()
    for i, s in enumerate(["C", "C", "O"]):
        g.add_node(i, symbol=s, position=(float(i), 0.0, 0.0))
    with pytest.raises(ValueError, match="must contain metal"):
        _align_metal_fragments(g, g)


# ---------------------------------------------------------------------------
# Integration with overlay.align() — selector-driven path
# ---------------------------------------------------------------------------


def test_overlay_align_dispatches_string_to_selector_path() -> None:
    """overlay.align() with a string spec routes through resolve_atom_indices + align_with_selection."""
    from xyzrender.overlay import align as overlay_align

    g_ref = _build_mxy_complex("Pt", ["Cl", "Br", "P", "N"])
    rot = _rotation_z(30.0)
    g_mob = _rigid_transform_graph(g_ref, rot, np.array([2.0, 0.0, 0.0]))

    aligned = overlay_align(g_ref, g_mob, align_atoms="M,L")
    ref_pos = np.array([g_ref.nodes[n]["position"] for n in g_ref.nodes()])
    assert np.allclose(aligned, ref_pos, atol=1e-6)


def test_selector_specific_element_tokens_stay_literal() -> None:
    """Element-symbol tokens (Li, O, …) are literal: ALL atoms of that element.

    Only category tokens (L, het) are graph-context-aware.  Specific
    elements never get narrowed by graph connectivity.
    """
    from xyzrender.selectors import resolve_atom_indices

    g = _build_alkoxide()
    atoms = resolve_atom_indices("Li,O", g)
    syms = sorted(g.nodes[n]["symbol"] for n in atoms)
    assert syms == ["Li", "Li", "O", "O"]  # both Lis + both Os, no narrowing


def test_selector_l_narrows_to_coord_shell_when_metals_present() -> None:
    """L = "ligand" = atoms bonded to a metal when graph has metals."""
    from xyzrender.selectors import resolve_atom_indices

    g = _build_alkoxide()  # has Li atoms (metals)
    atoms = resolve_atom_indices("L", g)
    # Only the O bonded to Li (atom 1) qualifies as a ligand.  The isolated O
    # (atom 6) and C/H atoms not bonded to Li are excluded.
    assert 1 in atoms  # O bonded to Li → ligand
    assert 6 not in atoms  # isolated O → not a ligand
    # Only atoms bonded to Li[0] are in the coord shell
    syms_bonded_to_li = {g.nodes[n]["symbol"] for n in atoms}
    assert "Li" not in syms_bonded_to_li  # Li is metal, not ligand


def test_selector_l_falls_back_to_all_nonmetals_when_no_metals() -> None:
    """L on a pure-organic graph = all non-metals (no narrowing possible)."""
    from xyzrender.selectors import resolve_atom_indices

    g = nx.Graph()
    for i, (s, p) in enumerate(
        [
            ("C", (0, 0, 0)),
            ("C", (1.5, 0, 0)),
            ("O", (3, 0, 0)),
            ("H", (0, 1, 0)),
        ]
    ):
        g.add_node(i, symbol=s, position=p)
    g.add_edges_from([(0, 1), (1, 2), (0, 3)])
    atoms = resolve_atom_indices("L", g)
    syms = sorted(g.nodes[n]["symbol"] for n in atoms)
    assert syms == ["C", "C", "H", "O"]
