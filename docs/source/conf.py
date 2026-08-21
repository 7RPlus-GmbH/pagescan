"""Sphinx configuration for pagescan documentation."""
from __future__ import annotations

import sys
from pathlib import Path

# -- Project root on sys.path so autodoc can import pagescan -----------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pagescan  # noqa: E402

# -- Project information -----------------------------------------------------
project = "pagescan"
author = "7R+ GmbH"
copyright = "2026, 7R+ GmbH"
release = pagescan.__version__
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
    "attrs_inline",
    "linkify",
]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_typehints = "description"  # render type hints in the description, not the signature
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# The docs carry the same mark as the site. The currentcolor cut inherits the
# theme's text colour, so it works in both light and dark without a second
# file; favicon.svg is the 16px cut with its own stroke weight, not a scaled
# master.
html_logo = "_static/pagescan-mark.svg"
html_favicon = "_static/favicon.svg"
html_css_files = ["custom.css"]
html_title = f"pagescan {release}"

html_theme_options = {
    "github_url": "https://github.com/7RPlus-GmbH/pagescan",
    "show_prev_next": True,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pagescan/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Hugging Face",
            "url": "https://huggingface.co/7rplus/pagescan-weights",
            "icon": "fa-solid fa-cube",
        },
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
}

html_context = {
    "github_user": "7RPlus-GmbH",
    "github_repo": "pagescan",
    "github_version": "main",
    "doc_path": "docs/source",
}

# -- Copy button -------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
