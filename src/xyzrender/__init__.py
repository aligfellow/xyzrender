"""Publication-quality molecular graphics from the command line."""

import logging

from xyzrender.io import load_molecule
from xyzrender.renderer import render_svg
from xyzrender.types import RenderConfig

__all__ = ["RenderConfig", "configure_logging", "load_molecule", "render_svg"]


def configure_logging(*, verbose: bool = False, debug: bool = False) -> None:
    """Enable console logging for the xyzrender package."""
    pkg_logger = logging.getLogger("xyzrender")
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        pkg_logger.addHandler(handler)
    if debug:
        pkg_logger.setLevel(logging.DEBUG)
    elif verbose:
        pkg_logger.setLevel(logging.INFO)
    else:
        pkg_logger.setLevel(logging.WARNING)
