# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "xyzrender"
copyright = "2026, Dr Alister S. Goodfellow"
author = "Dr Alister S. Goodfellow"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # pulls docstrings into docs automatically
    "sphinx.ext.napoleon",  # understands numpy/Google style docstrings
    "sphinx.ext.viewcode",  # adds [source] links to API pages
    "sphinx_autodoc_typehints",  # renders type hints nicely
    "myst_parser",  # lets you write pages in Markdown as well as RST
]

templates_path = ["_templates"]
exclude_patterns = []

suppress_warnings = ["image.not_readable"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# ---------------------------------------------------

import sys
from pathlib import Path

# Path to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
