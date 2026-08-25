"""Sphinx configuration for the Cognitive Calibration documentation."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "Cognitive Calibration"
author = "MultipleSclerosisMonitoring"
release = "0.1.0"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",
]

autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

html_theme = "alabaster"
html_title = "Cognitive Calibration Documentation"
html_static_path = []
html_show_sourcelink = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}
mermaid_output_format = "raw"
mermaid_init_js = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"
