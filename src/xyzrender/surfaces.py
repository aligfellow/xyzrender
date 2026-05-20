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
    from xyzrender.api import Molecule
    from xyzrender.cube import CubeData
    from xyzrender.types import DensParams, ESPParams, MOParams, NCIParams, RenderConfig


def compute_mo_surface(
    mol: Molecule,
    cube: CubeData,
    cfg: RenderConfig,
    params: MOParams,
) -> None:
    """Build MO contours and store on ``cfg.mo_contours``.

    PCA-orients *mol* (if not already) with a ``-30`` degree x-axis tilt to
    separate above/below-plane orbital lobes, then Kabsch-aligns the cube grid
    to the (possibly rotated) atom positions.

    Parameters
    ----------
    mol:
        Molecule.  Orientation is applied via :meth:`Molecule.orient` if
        ``cfg.auto_orient`` is set and the molecule is not already oriented.
    cube:
        Gaussian cube file containing the molecular orbital data.
    cfg:
        Render configuration.  ``mo_contours`` and ``flat_mo`` are updated
        in-place.
    params:
        MO surface parameters (isovalue, colors, blur, upsampling, flat).
    """
    from xyzrender.mo import build_mo_contours
    from xyzrender.utils import align_cube_to_atoms

    if cfg.auto_orient:
        mol.orient(tilt_degrees=-30.0)
        cfg.auto_orient = False
    rot, atom_centroid, target_centroid = align_cube_to_atoms(cube, mol.graph)
    cfg.mo_contours = build_mo_contours(
        cube,
        params,
        rot=rot,
        atom_centroid=atom_centroid,
        target_centroid=target_centroid,
        surface_style=cfg.surface_style,
    )
    cfg.flat_mo = params.flat


def compute_dens_surface(
    mol: Molecule,
    cube: CubeData,
    cfg: RenderConfig,
    params: DensParams,
) -> None:
    """Build density contours and store on ``cfg.dens_contours``.

    Parameters
    ----------
    mol:
        Molecule.  Orientation is applied via :meth:`Molecule.orient` if
        ``cfg.auto_orient`` is set.
    cube:
        Gaussian cube file containing the electron density data.
    cfg:
        Render configuration.  ``dens_contours`` is updated in-place.
    params:
        Density surface parameters (isovalue, color).
    """
    from xyzrender.colors import resolve_color
    from xyzrender.dens import build_density_contours
    from xyzrender.utils import align_cube_to_atoms

    if cfg.auto_orient:
        mol.orient()
        cfg.auto_orient = False
    rot, atom_centroid, target_centroid = align_cube_to_atoms(cube, mol.graph)
    cfg.dens_contours = build_density_contours(
        cube,
        isovalue=params.isovalue,
        color=resolve_color(params.color),
        rot=rot,
        atom_centroid=atom_centroid,
        target_centroid=target_centroid,
        surface_style=cfg.surface_style,
    )


def compute_esp_surface(
    mol: Molecule,
    dens_cube: CubeData,
    esp_cube: CubeData,
    cfg: RenderConfig,
    params: ESPParams,
) -> None:
    """Build an ESP surface and store on ``cfg.esp_surface``.

    Parameters
    ----------
    mol:
        Molecule.  Orientation is applied via :meth:`Molecule.orient` if
        ``cfg.auto_orient`` is set.
    dens_cube:
        Gaussian cube file containing the electron density (used for the
        isosurface and atom geometry).
    esp_cube:
        Gaussian cube file containing the electrostatic potential values
        mapped onto the density isosurface.
    cfg:
        Render configuration.  ``esp_surface`` is updated in-place.
    params:
        ESP surface parameters (isovalue of the density isosurface).
    """
    from xyzrender.colors import DEFAULT_ESP_PALETTE
    from xyzrender.esp import build_esp_surface
    from xyzrender.utils import align_cube_to_atoms

    if cfg.auto_orient:
        mol.orient()
        cfg.auto_orient = False
    rot, atom_centroid, target_centroid = align_cube_to_atoms(dens_cube, mol.graph)
    cfg.esp_surface = build_esp_surface(
        dens_cube,
        esp_cube,
        params,
        palette=cfg.cmap_palette or DEFAULT_ESP_PALETTE,
        rot=rot,
        atom_centroid=atom_centroid,
        target_centroid=target_centroid,
        esp_range=cfg.cmap_range,
        esp_symm=cfg.cmap_symm,
    )


def compute_nci_surface(
    mol: Molecule,
    dens_cube: CubeData,
    grad_cube: CubeData,
    cfg: RenderConfig,
    params: NCIParams,
    *,
    surface_mode: str = "auto",
    iso_was_explicit: bool = False,
) -> None:
    """Build NCI contours and store on ``cfg.nci_contours``.

    Parameters
    ----------
    mol:
        Molecule.  Orientation is applied via :meth:`Molecule.orient` if
        ``cfg.auto_orient`` is set.
    dens_cube:
        Gaussian cube file containing the electron density (sign(lambda2)*rho
        values for NCI coloring, and atom geometry).
    grad_cube:
        Gaussian cube file containing the reduced density gradient (RDG)
        values used to locate NCI interaction regions.
    cfg:
        Render configuration.  ``nci_contours`` is updated in-place.
    params:
        NCI surface parameters (isovalue, color, color_mode, dens_cutoff).
    surface_mode:
        Surface interpretation mode: ``auto``, ``low_field``, or ``high_field``.
    iso_was_explicit:
        Whether the isovalue came from an explicit user override.
    """
    from xyzrender.nci import build_nci_contours
    from xyzrender.utils import align_cube_to_atoms

    if cfg.auto_orient:
        mol.orient()
        cfg.auto_orient = False
    rot, atom_centroid, target_centroid = align_cube_to_atoms(dens_cube, mol.graph)
    cfg.nci_contours = build_nci_contours(
        grad_cube,
        dens_cube,
        params,
        rot=rot,
        atom_centroid=atom_centroid,
        target_centroid=target_centroid,
        surface_style=cfg.surface_style,
        surface_mode=surface_mode,
        iso_was_explicit=iso_was_explicit,
    )
