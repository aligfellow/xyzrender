from __future__ import annotations

from typing import TYPE_CHECKING

from xyzrender import SVGResult, load, render

if TYPE_CHECKING:
    from pathlib import Path


def _has_svg_tag(svg: str, tag: str) -> bool:
    return f"<{tag}" in svg


def test_skeletal_style_no_atom_circles(tmp_path: Path) -> None:
    mol = load("examples/structures/ethanol.xyz")
    out = tmp_path / "ethanol_skeletal.svg"
    result = render(mol, config="skeletal", output=out)
    assert isinstance(result, SVGResult)
    svg = out.read_text()
    # No atom circles should be drawn for skeletal style
    assert not _has_svg_tag(svg, "circle")


def test_skeletal_style_non_carbon_labels(tmp_path: Path) -> None:
    mol = load("examples/structures/ethanol.xyz")
    out = tmp_path / "ethanol_skeletal_labels.svg"
    render(mol, config="skeletal", output=out)
    svg = out.read_text()
    # Expect at least one text label for O and H atoms
    assert "<text" in svg
    assert "O" in svg
