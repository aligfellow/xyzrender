"""Tests for color blending logic."""

import numpy as np

from xyzrender.colors import blend_fog


def test_fog_preserves_at_zero():
    assert blend_fog("#abcdef", np.array([255, 255, 255]), 0.0) == "#abcdef"


def test_fog_full_strength():
    # Fog is capped at 70% to prevent atoms disappearing into the background
    assert blend_fog("#000000", np.array([255, 255, 255]), -1.0) == "#b2b2b2"


def test_fog_sign_irrelevant():
    assert blend_fog("#000000", np.array([255, 255, 255]), 0.5) == blend_fog("#000000", np.array([255, 255, 255]), -0.5)
