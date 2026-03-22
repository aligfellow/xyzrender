"""Diffuse / assembly frame generation for GIF animations.

Atoms scatter outward with a blend of radial and isotropic noise, then
(when reversed) reassemble into the molecule — a denoising aesthetic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx


def parse_anchor(anchor: str | list[int] | None) -> set[int] | None:
    """Parse anchor specification to a set of 0-indexed atom indices.

    Accepts a 1-indexed string (``"1-5,8"``) or 1-indexed ``list[int]``.
    Returns ``None`` if *anchor* is ``None``.
    """
    if anchor is None:
        return None
    from xyzrender.utils import parse_atom_indices

    return set(parse_atom_indices(anchor))


def diffuse_frames(
    graph: nx.Graph,
    n_frames: int = 60,
    noise: float = 0.3,
    bonds: str = "fade",
    reverse: bool = True,
    seed: int = 42,
    anchor: set[int] | None = None,
) -> list[dict]:
    """Generate frames with progressive noise on atom positions.

    Each atom gets a drift direction (blend of radial outward + random) and
    per-frame isotropic noise that scales with displacement, giving a noisy
    random-walk feel rather than a smooth glide.

    *noise* controls the per-frame random walk strength (default 0.3).

    *anchor* atoms (0-indexed) stay at their original positions throughout.

    When *reverse* is True (default), the frame list is reversed so playback
    shows atoms assembling from noise into the molecule (denoising aesthetic).

    Each frame dict includes a ``"bond_opacities"`` mapping
    ``{(i,j): float}`` for bonds whose current length exceeds the
    equilibrium length — the renderer uses this to fade stretched bonds.
    """
    _sigma = 4.0
    _radial_bias = 0.5

    nodes = list(graph.nodes())
    positions = np.array([graph.nodes[n]["position"] for n in nodes])
    symbols = [graph.nodes[n]["symbol"] for n in nodes]
    n_atoms = len(nodes)
    anchor = anchor or set()

    centroid = positions.mean(axis=0)
    rng = np.random.default_rng(seed)

    # Radial unit vectors (away from centroid)
    radial = positions - centroid
    norms = np.linalg.norm(radial, axis=1, keepdims=True)
    at_center = (norms < 1e-8).flatten()
    random_dirs = rng.standard_normal(radial.shape)
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
    radial[at_center] = random_dirs[at_center]
    norms[at_center.reshape(-1, 1)] = 1.0
    r_hat = radial / norms

    # Per-atom drift direction = blend of radial + isotropic
    iso_noise = rng.standard_normal(positions.shape)
    iso_noise /= np.linalg.norm(iso_noise, axis=1, keepdims=True)
    drift = _sigma * (_radial_bias * r_hat + (1 - _radial_bias) * iso_noise)

    # Anchor mask: 0 for anchored atoms, 1 for mobile
    mobile = np.array([0.0 if i in anchor else 1.0 for i in range(n_atoms)]).reshape(-1, 1)

    # Build cumulative random walk
    walk = np.zeros_like(positions)
    walk_history = [walk.copy()]
    for _fi in range(1, n_frames):
        step = rng.standard_normal(positions.shape) * noise * _sigma / max(n_frames, 1) ** 0.5
        walk = walk + step
        walk_history.append(walk.copy())

    # Equilibrium bond lengths for opacity fading
    eq_lengths: dict[tuple[int, int], float] = {}
    for i, j in graph.edges():
        d = float(np.linalg.norm(positions[i] - positions[j]))
        eq_lengths[(i, j)] = eq_lengths[(j, i)] = d

    frames = []
    for fi in range(n_frames):
        t = fi / max(n_frames - 1, 1)
        d = t * t * (3 - 2 * t)  # smoothstep: slow at both ends
        displacement = mobile * (d * drift + d * d * walk_history[fi])
        perturbed = positions + displacement

        # Bond opacity per mode
        bond_opacities: dict[tuple[int, int], float] = {}
        if bonds == "hide":
            for i, j in eq_lengths:
                if i < j:
                    bond_opacities[(i, j)] = bond_opacities[(j, i)] = 0.0
        elif bonds == "fade":
            for (i, j), eq_len in eq_lengths.items():
                if i >= j:
                    continue
                cur_len = float(np.linalg.norm(perturbed[i] - perturbed[j]))
                ratio = cur_len / max(eq_len, 0.1)
                if ratio > 1.0:
                    op = max(0.0, np.exp(-6.0 * (ratio - 1.0)))
                    bond_opacities[(i, j)] = bond_opacities[(j, i)] = op

        frames.append(
            {
                "symbols": symbols,
                "positions": perturbed.tolist(),
                "bond_opacities": bond_opacities,
            }
        )

    if reverse:
        frames.reverse()
    return frames
