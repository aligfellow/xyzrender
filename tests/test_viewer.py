"""Tests for viewer integration."""

import pytest


def test_rotate_with_viewer_missing_vmol(monkeypatch):
    """ImportError with helpful message when vmol is not installed."""
    import builtins

    import networkx as nx

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "vmol":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from xyzrender.viewer import rotate_with_viewer

    g = nx.Graph()
    g.add_node(0, symbol="H", position=(0.0, 0.0, 0.0))

    with pytest.raises(ImportError, match="Interactive viewer requires vmol"):
        rotate_with_viewer(g)
