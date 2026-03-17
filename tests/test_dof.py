"""Tests for --dof (depth-of-field blur)."""

from pathlib import Path

import pytest

from xyzrender import load, render

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


@pytest.fixture(scope="module")
def caffeine():
    return load(STRUCTURES / "caffeine.xyz")


def test_dof_filter_defs_in_svg(caffeine):
    """DoF should inject feGaussianBlur filter definitions."""
    svg = str(render(caffeine, dof=True, orient=False))
    assert "feGaussianBlur" in svg
    assert 'filter id="' in svg


def test_dof_applied_to_atoms(caffeine):
    """Atom circles should reference dof filters."""
    svg = str(render(caffeine, dof=True, orient=False))
    assert 'filter="url(#' in svg


def test_dof_not_applied_to_bonds(caffeine):
    """Bonds should NOT have dof filters (blur makes thin lines look wrong)."""
    svg = str(render(caffeine, dof=True, orient=False))
    # Bond lines don't have filter attributes — only circles do
    for line in svg.split("\n"):
        if "<line " in line:
            assert "filter=" not in line


def test_dof_custom_strength(caffeine):
    """Custom dof_strength should scale the max blur."""
    svg = str(render(caffeine, dof=True, dof_strength=6.0, orient=False))
    assert 'stdDeviation="6.00"' in svg


def test_dof_off_by_default(caffeine):
    """No dof filters when dof=False."""
    svg = str(render(caffeine, orient=False))
    assert "feGaussianBlur" not in svg


def test_dof_with_fog(caffeine):
    """DoF + fog should work together."""
    svg = str(render(caffeine, dof=True, fog=True, orient=False))
    assert "feGaussianBlur" in svg
