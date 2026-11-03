"""MCP server that lets agents render molecules with xyzrender.

``mcp`` is an optional extra (Python 3.10+). The import stays inside
:func:`build_server` so the core package still imports on a minimal install.
"""

from __future__ import annotations

from typing import Any, Literal

from .agent import (
    AgentResult,
)
from .agent import (
    inspect_structure as _inspect_structure,
)
from .agent import (
    measure_geometry as _measure_geometry,
)
from .agent import (
    render_animation as _render_animation,
)
from .agent import (
    render_molecule as _render_molecule,
)

_INSTRUCTIONS = (
    "Use render_molecule for static SVG/PNG/PDF figures (prefer this over GIFs). "
    "Use render_gif only when the user asks for an animation. "
    "Use measure_geometry for distances/angles/dihedrals as JSON. "
    "Use inspect_structure to learn atom counts, formula, cube/cell/TS/NCI hints before rendering. "
    "Paths should be absolute when possible. "
    "Atom indices and bond pairs are 1-indexed (e.g. ts_bonds=['1-2']). "
    "ts / nci enable automatic graph overlays; nci_surf is a volumetric RDG cube path (different). "
    "Never open an interactive viewer."
)


def build_server() -> Any:
    """Return a configured MCP server. Requires ``pip install 'xyzrender[mcp]'``."""
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise ImportError("The xyzrender MCP server needs the mcp extra: pip install 'xyzrender[mcp]'") from exc

    server = MCPServer("xyzrender", instructions=_INSTRUCTIONS)

    @server.tool(
        annotations=ToolAnnotations(
            title="Inspect structure",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def inspect_structure(
        input_path: str,
        smiles: bool = False,
        charge: int = 0,
        multiplicity: int | None = None,
        ts_detect: bool = False,
        nci_detect: bool = False,
        cell: bool = False,
        ensemble: bool = False,
    ) -> AgentResult:
        """Summarise a molecular input (atom count, formula, cube/cell/TS/NCI flags)."""
        return _inspect_structure(
            input_path,
            smiles=smiles,
            charge=charge,
            multiplicity=multiplicity,
            ts_detect=ts_detect,
            nci_detect=nci_detect,
            cell=cell,
            ensemble=ensemble,
        )

    @server.tool(
        annotations=ToolAnnotations(
            title="Measure geometry",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def measure_geometry(
        input_path: str,
        modes: list[str] | None = None,
        smiles: bool = False,
        charge: int = 0,
        multiplicity: int | None = None,
    ) -> AgentResult:
        """Return bond distances, angles, and/or dihedrals as JSON.

        ``modes`` is a subset of ``["d", "a", "t"]`` (distances, angles, dihedrals).
        """
        return _measure_geometry(
            input_path,
            modes=modes,
            smiles=smiles,
            charge=charge,
            multiplicity=multiplicity,
        )

    @server.tool(
        annotations=ToolAnnotations(
            title="Render molecule",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def render_molecule(
        input_path: str,
        output: str,
        smiles: bool = False,
        config: str = "default",
        charge: int = 0,
        multiplicity: int | None = None,
        hy: bool = False,
        hy_atoms: str | None = None,
        ts: bool = False,
        ts_bonds: list[str] | None = None,
        nci: bool = False,
        nci_bonds: list[str] | None = None,
        overlay: str | None = None,
        auto_align: bool | None = None,
        align_atoms: str | None = None,
        vdw: bool = False,
        vdw_atoms: str | None = None,
        mo: bool = False,
        dens: bool = False,
        esp: str | None = None,
        nci_surf: str | None = None,
        iso: float | None = None,
        transparent: bool = False,
        background: str | None = None,
        ref: str | None = None,
        idx: bool = False,
        ensemble: bool = False,
    ) -> AgentResult:
        """Render a static SVG, PNG, or PDF figure to ``output``.

        Prefer PNG/SVG for Cursor preview. ``ts``/``nci`` are automatic graph
        overlays; ``nci_surf`` is a volumetric NCI cube path.
        """
        return _render_molecule(
            input_path,
            output,
            smiles=smiles,
            config=config,
            charge=charge,
            multiplicity=multiplicity,
            hy=hy_atoms if hy_atoms is not None else hy,
            ts=ts,
            ts_bonds=ts_bonds,
            nci=nci,
            nci_bonds=nci_bonds,
            overlay=overlay,
            auto_align=auto_align,
            align_atoms=align_atoms,
            vdw=vdw_atoms if vdw_atoms is not None else (True if vdw else None),
            mo=mo,
            dens=dens,
            esp=esp,
            nci_surf=nci_surf,
            iso=iso,
            transparent=transparent,
            background=background,
            ref=ref,
            idx=idx,
            ensemble=ensemble,
        )

    @server.tool(
        annotations=ToolAnnotations(
            title="Render GIF",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def render_gif(
        input_path: str,
        output: str,
        mode: Literal["rot", "bounce", "ts", "trj", "diffuse"] = "rot",
        smiles: bool = False,
        config: str = "default",
        charge: int = 0,
        multiplicity: int | None = None,
        hy: bool = False,
        hy_atoms: str | None = None,
        ts: bool = False,
        ts_bonds: list[str] | None = None,
        nci: bool = False,
        overlay: str | None = None,
        vdw: bool = False,
        vdw_atoms: str | None = None,
        gif_fps: int = 10,
        rot_frames: int = 60,
        rot_axis: str = "y",
        bounce_degrees: float = 30.0,
        transparent: bool = False,
        background: str | None = None,
        ref: str | None = None,
    ) -> AgentResult:
        """Render a GIF animation. Prefer ``render_molecule`` unless animation is requested."""
        return _render_animation(
            input_path,
            output,
            mode=mode,
            smiles=smiles,
            config=config,
            charge=charge,
            multiplicity=multiplicity,
            hy=hy_atoms if hy_atoms is not None else hy,
            ts=ts,
            ts_bonds=ts_bonds,
            nci=nci,
            overlay=overlay,
            vdw=vdw_atoms if vdw_atoms is not None else (True if vdw else None),
            gif_fps=gif_fps,
            rot_frames=rot_frames,
            rot_axis=rot_axis,
            bounce_degrees=bounce_degrees,
            transparent=transparent,
            background=background,
            ref=ref,
        )

    return server


def main() -> None:
    """Serve xyzrender tools over stdio."""
    build_server().run()


if __name__ == "__main__":
    main()
