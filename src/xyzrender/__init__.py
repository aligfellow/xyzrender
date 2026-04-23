"""Publication-quality molecular graphics."""

import logging
from importlib.metadata import version

from xyzrender.annotations import load_cmap
from xyzrender.api import Molecule, load, measure, orient, render, render_gif
from xyzrender.config import build_config
from xyzrender.types import GIFResult, OverlayConfig, RenderConfig, StyleRegion, SVGResult

__version__ = version("xyzrender")

__all__ = [
    "GIFResult",
    "Molecule",
    "OverlayConfig",
    "RenderConfig",
    "SVGResult",
    "StyleRegion",
    "__version__",
    "build_config",
    "configure_logging",
    "load",
    "load_cmap",
    "measure",
    "orient",
    "render",
    "render_gif",
]


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
