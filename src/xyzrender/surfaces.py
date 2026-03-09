"""Library-callable surface builders.

Each ``compute_*_surface()`` function handles the full surface pipeline:

1. Apply PCA auto-orientation (if ``cfg.auto_orient`` is set) and compute
   the Kabsch rotation to align the volumetric grid with the atom positions.
2. Build the 2-D surface contours / raster.
3. Store the result on *cfg* (mutates in-place).

These are the entry points for programmatic use (notebooks, scripts).
The CLI uses them via :mod:`xyzrender.cli`; the high-level :func:`~xyzrender.api.render`
function wraps them further.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

    from xyzrender.cube import CubeData
    from xyzrender.types import DensParams, ESPParams, MOParams, NCIParams, RenderConfig


def compute_mo_surface(
    graph: nx.Graph,
    cube: CubeData,
    cfg: RenderConfig,
    params: MOParams,
) -> None:
    """Build MO contours and store on ``cfg.mo_contours``.

    Applies PCA auto-orientation (if ``cfg.auto_orient``) with a ``-30°``
    x-axis tilt to separate above/below-plane orbital lobes.

    Parameters
    ----------
    graph:
        Molecular graph.  Node positions may be updated in-place by PCA.
    cube:
        Gaussian cube file containing the molecular orbital data.
    cfg:
        Render configuration.  ``mo_contours``, ``flat_mo``, and
        ``auto_orient`` are updated in-place.
    params:
        MO surface parameters (isovalue, colors, blur, upsampling, flat).
    """
    from xyzrender.mo import build_mo_contours
    from xyzrender.utils import resolve_orientation

    rot, atom_centroid, curr_centroid = resolve_orientation(graph, cube, cfg, tilt_degrees=-30.0)
    cfg.mo_contours = build_mo_contours(
        cube,
        params,
        rot=rot,
        atom_centroid=atom_centroid,
        target_centroid=curr_centroid,
    )
    cfg.flat_mo = params.flat


def compute_dens_surface(
    graph: nx.Graph,
    cube: CubeData,
    cfg: RenderConfig,
    params: DensParams,
) -> None:
    """Build density contours and store on ``cfg.dens_contours``.

    Parameters
    ----------
    graph:
        Molecular graph.  Node positions may be updated in-place by PCA.
    cube:
        Gaussian cube file containing the electron density data.
    cfg:
        Render configuration.  ``dens_contours`` and ``auto_orient`` are
        updated in-place.
    params:
        Density surface parameters (isovalue, color).
    """
    from xyzrender.dens import build_density_contours
    from xyzrender.types import resolve_color
    from xyzrender.utils import resolve_orientation

    rot, atom_centroid, curr_centroid = resolve_orientation(graph, cube, cfg)
    cfg.dens_contours = build_density_contours(
        cube,
        isovalue=params.isovalue,
        color=resolve_color(params.color),
        rot=rot,
        atom_centroid=atom_centroid,
        target_centroid=curr_centroid,
    )


def compute_esp_surface(
    graph: nx.Graph,
    dens_cube: CubeData,
    esp_cube: CubeData,
    cfg: RenderConfig,
    params: ESPParams,
) -> None:
    """Build an ESP surface and store on ``cfg.esp_surface``.

    Parameters
    ----------
    graph:
        Molecular graph.  Node positions may be updated in-place by PCA.
    dens_cube:
        Gaussian cube file containing the electron density (used for the
        isosurface and atom geometry).
    esp_cube:
        Gaussian cube file containing the electrostatic potential values
        mapped onto the density isosurface.
    cfg:
        Render configuration.  ``esp_surface`` and ``auto_orient`` are
        updated in-place.
    params:
        ESP surface parameters (isovalue of the density isosurface).
    """
    from xyzrender.esp import build_esp_surface
    from xyzrender.utils import resolve_orientation

    rot, atom_centroid, curr_centroid = resolve_orientation(graph, dens_cube, cfg)
    cfg.esp_surface = build_esp_surface(
        dens_cube,
        esp_cube,
        params,
        rot=rot,
        atom_centroid=atom_centroid,
        target_centroid=curr_centroid,
    )


def compute_nci_surface(
    graph: nx.Graph,
    dens_cube: CubeData,
    grad_cube: CubeData,
    cfg: RenderConfig,
    params: NCIParams,
) -> None:
    """Build NCI contours and store on ``cfg.nci_contours``.

    Parameters
    ----------
    graph:
        Molecular graph.  Node positions may be updated in-place by PCA.
    dens_cube:
        Gaussian cube file containing the electron density (sign(lambda2)*rho
        values for NCI coloring, and atom geometry).
    grad_cube:
        Gaussian cube file containing the reduced density gradient (RDG)
        values used to locate NCI interaction regions.
    cfg:
        Render configuration.  ``nci_contours`` and ``auto_orient`` are
        updated in-place.
    params:
        NCI surface parameters (isovalue, color, color_mode, dens_cutoff).
    """
    from xyzrender.nci import build_nci_contours
    from xyzrender.utils import resolve_orientation

    rot, atom_centroid, curr_centroid = resolve_orientation(graph, dens_cube, cfg)
    cfg.nci_contours = build_nci_contours(
        grad_cube,
        dens_cube,
        params,
        rot=rot,
        atom_centroid=atom_centroid,
        target_centroid=curr_centroid,
    )
