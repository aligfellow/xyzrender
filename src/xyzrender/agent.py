"""Non-interactive helpers for agents that render molecules with xyzrender."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal, TypedDict

from xyzrender.api import load, measure, render, render_gif
from xyzrender.utils import parse_atom_indices

_GifMode = Literal["rot", "bounce", "ts", "trj", "diffuse"]
_STATIC_SUFFIXES = {".svg", ".png", ".pdf", ".tif", ".tiff"}


class AgentResult(TypedDict, total=False):
    """JSON-ready outcome of one agent helper call."""

    success: bool
    action: str
    output: str | None
    format: str | None
    error: str | None
    n_atoms: int
    n_bonds: int
    formula: str
    symbols: list[str]
    has_cube: bool
    has_cell: bool
    has_ensemble: bool
    ts_edges: int
    nci_edges: int
    distances: list[list[float | int]]
    angles: list[list[float | int]]
    dihedrals: list[list[float | int]]


def inspect_structure(
    input_path: str,
    *,
    smiles: bool = False,
    charge: int = 0,
    multiplicity: int | None = None,
    ts_detect: bool = False,
    nci_detect: bool = False,
    cell: bool = False,
    ensemble: bool = False,
) -> AgentResult:
    """Load a structure and return a compact summary for agent planning."""
    try:
        path = _resolve_input(input_path, smiles=smiles)
        mol = load(
            path if not smiles else input_path,
            smiles=smiles,
            charge=charge,
            multiplicity=multiplicity,
            ts_detect=ts_detect,
            nci_detect=nci_detect,
            cell=cell,
            ensemble=ensemble,
        )
    except Exception as exc:
        return _failure("inspect", str(exc), output=None)

    symbols = [str(mol.graph.nodes[i].get("symbol", "?")) for i in mol.graph.nodes()]
    real = [s for s in symbols if s != "*"]
    counts = Counter(real)
    formula = "".join(f"{el}{counts[el] if counts[el] > 1 else ''}" for el in sorted(counts))
    ts_edges = sum(1 for _, _, data in mol.graph.edges(data=True) if data.get("TS"))
    nci_edges = sum(1 for _, _, data in mol.graph.edges(data=True) if data.get("NCI"))
    return {
        "success": True,
        "action": "inspect",
        "output": None,
        "format": None,
        "n_atoms": len(real),
        "n_bonds": mol.graph.number_of_edges(),
        "formula": formula,
        "symbols": real,
        "has_cube": mol.cube_data is not None,
        "has_cell": mol.cell_data is not None,
        "has_ensemble": mol.ensemble is not None,
        "ts_edges": ts_edges,
        "nci_edges": nci_edges,
    }


def measure_geometry(
    input_path: str,
    *,
    modes: list[str] | None = None,
    smiles: bool = False,
    charge: int = 0,
    multiplicity: int | None = None,
) -> AgentResult:
    """Return bond distances / angles / dihedrals as JSON-friendly lists."""
    try:
        path = _resolve_input(input_path, smiles=smiles)
        mol = load(
            path if not smiles else input_path,
            smiles=smiles,
            charge=charge,
            multiplicity=multiplicity,
        )
        raw = measure(mol, modes=modes)
    except Exception as exc:
        return _failure("measure", str(exc), output=None)

    result: AgentResult = {
        "success": True,
        "action": "measure",
        "output": None,
        "format": None,
        "n_atoms": mol.graph.number_of_nodes(),
    }
    for key in ("distances", "angles", "dihedrals"):
        if key in raw:
            result[key] = [_jsonable_row(row) for row in raw[key]]  # type: ignore[literal-required]
    return result


def render_molecule(
    input_path: str,
    output: str,
    *,
    smiles: bool = False,
    config: str = "default",
    charge: int = 0,
    multiplicity: int | None = None,
    hy: bool | str = False,
    ts: bool = False,
    ts_bonds: list[list[int]] | list[str] | None = None,
    nci: bool = False,
    nci_bonds: list[list[int]] | list[str] | None = None,
    overlay: str | None = None,
    auto_align: bool | None = None,
    align_atoms: str | list[int] | None = None,
    vdw: bool | str | list[int] | None = None,
    mo: bool = False,
    dens: bool = False,
    esp: str | None = None,
    nci_surf: str | None = None,
    iso: float | None = None,
    transparent: bool = False,
    background: str | None = None,
    ref: str | None = None,
    idx: bool | str = False,
    ensemble: bool = False,
) -> AgentResult:
    """Render a static SVG/PNG/PDF figure and return the absolute output path."""
    out = Path(output).expanduser().resolve()
    if out.suffix.lower() not in _STATIC_SUFFIXES:
        return _failure(
            "render",
            f"Unsupported output suffix {out.suffix!r}; use .svg, .png, .pdf, or .tiff",
            output=str(out),
        )
    try:
        path = _resolve_input(input_path, smiles=smiles)
        mol = load(
            path if not smiles else input_path,
            smiles=smiles,
            charge=charge,
            multiplicity=multiplicity,
            ts_detect=ts and not ts_bonds,
            nci_detect=nci and not nci_bonds,
            ensemble=ensemble,
        )
        vdw_arg = _resolve_vdw(vdw, mol.graph)
        hy_arg = _resolve_hy(hy)
        out.parent.mkdir(parents=True, exist_ok=True)
        render(
            mol,
            config=config,
            output=out,
            hy=hy_arg,
            ts_bonds=_parse_pairs(ts_bonds),
            nci_bonds=_parse_pairs(nci_bonds),
            overlay=overlay,
            auto_align=auto_align,
            align_atoms=align_atoms,
            vdw=vdw_arg,
            mo=mo,
            dens=dens,
            esp=esp,
            nci=nci_surf,
            iso=iso,
            transparent=transparent,
            background=background,
            ref=ref,
            idx=idx,
        )
    except Exception as exc:
        return _failure("render", str(exc), output=str(out))

    if not out.is_file():
        return _failure("render", f"Render finished but output missing: {out}", output=str(out))
    return {
        "success": True,
        "action": "render",
        "output": str(out),
        "format": out.suffix.lower().lstrip("."),
        "n_atoms": mol.graph.number_of_nodes(),
    }


def render_animation(
    input_path: str,
    output: str,
    *,
    mode: _GifMode = "rot",
    smiles: bool = False,
    config: str = "default",
    charge: int = 0,
    multiplicity: int | None = None,
    hy: bool | str = False,
    ts: bool = False,
    ts_bonds: list[list[int]] | list[str] | None = None,
    nci: bool = False,
    overlay: str | None = None,
    vdw: bool | str | list[int] | None = None,
    gif_fps: int = 10,
    rot_frames: int = 60,
    rot_axis: str = "y",
    bounce_degrees: float = 30.0,
    transparent: bool = False,
    background: str | None = None,
    ref: str | None = None,
) -> AgentResult:
    """Render a GIF animation and return the absolute output path."""
    out = Path(output).expanduser().resolve()
    if out.suffix.lower() != ".gif":
        return _failure("gif", f"GIF output must end with .gif, got {out.suffix!r}", output=str(out))
    try:
        path = _resolve_input(input_path, smiles=smiles)
        mol = load(
            path if not smiles else input_path,
            smiles=smiles,
            charge=charge,
            multiplicity=multiplicity,
            ts_detect=(mode == "ts" or ts) and not ts_bonds,
            nci_detect=nci,
            ensemble=mode == "trj",
        )
        vdw_arg = _resolve_vdw(vdw, mol.graph)
        hy_arg = _resolve_hy(hy)
        out.parent.mkdir(parents=True, exist_ok=True)
        # render_gif has no vdw= kwarg; set indices on a RenderConfig instead.
        gif_config: str | Any = config
        if vdw_arg is not None:
            from xyzrender.config import build_config

            cfg = build_config(config) if isinstance(config, str) else config
            cfg.vdw_indices = [] if vdw_arg is True else [i - 1 for i in vdw_arg]
            gif_config = cfg
        kwargs: dict[str, Any] = {
            "config": gif_config,
            "output": out,
            "hy": hy_arg,
            "ts_bonds": _parse_pairs(ts_bonds),
            "gif_fps": gif_fps,
            "rot_frames": rot_frames,
            "transparent": transparent,
            "background": background,
            "ref": ref,
            "overlay": overlay,
            "detect_nci": nci,
        }
        if mode == "rot":
            kwargs["gif_rot"] = rot_axis
        elif mode == "bounce":
            kwargs["gif_bounce"] = bounce_degrees
        elif mode == "ts":
            kwargs["gif_ts"] = True
        elif mode == "trj":
            kwargs["gif_trj"] = True
        elif mode == "diffuse":
            kwargs["gif_diffuse"] = True
        else:
            return _failure(
                "gif",
                f"Unknown GIF mode {mode!r}. Use rot, bounce, ts, trj, or diffuse.",
                output=str(out),
            )
        render_gif(mol, **kwargs)
    except Exception as exc:
        return _failure("gif", str(exc), output=str(out))

    if not out.is_file():
        return _failure("gif", f"GIF finished but output missing: {out}", output=str(out))
    return {
        "success": True,
        "action": "gif",
        "output": str(out),
        "format": "gif",
        "n_atoms": mol.graph.number_of_nodes(),
    }


def _resolve_input(input_path: str, *, smiles: bool) -> Path | str:
    if smiles:
        return input_path
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    return path


def _resolve_hy(hy: bool | str) -> bool | list[int]:
    if isinstance(hy, bool):
        return hy
    return parse_atom_indices(hy, one_indexed=True)


def _resolve_vdw(vdw: bool | str | list[int] | None, graph: Any) -> bool | list[int] | None:
    if vdw is None or isinstance(vdw, bool):
        return vdw
    if isinstance(vdw, list):
        return vdw
    from xyzrender.selectors import resolve_atom_indices

    # render() expects 1-indexed indices when given a list
    return sorted(i + 1 for i in resolve_atom_indices(vdw, graph))


def _parse_pairs(pairs: list[list[int]] | list[str] | None) -> list[tuple[int, int]] | None:
    if pairs is None:
        return None
    parsed: list[tuple[int, int]] = []
    for item in pairs:
        if isinstance(item, str):
            left, right = item.replace(",", "-").split("-", maxsplit=1)
            parsed.append((int(left), int(right)))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            parsed.append((int(item[0]), int(item[1])))
        else:
            raise ValueError(f"Bond pair must be [i, j] or 'i-j', got {item!r}")
    return parsed


def _jsonable_row(row: tuple[Any, ...] | list[Any]) -> list[float | int]:
    out: list[float | int] = []
    for value in row:
        if isinstance(value, bool):
            out.append(int(value))
        elif isinstance(value, int):
            out.append(int(value))
        else:
            out.append(float(value))
    return out


def _failure(action: str, error: str, *, output: str | None) -> AgentResult:
    return {
        "success": False,
        "action": action,
        "output": output,
        "format": None,
        "error": error,
    }
