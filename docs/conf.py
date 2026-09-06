# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from importlib.metadata import version as package_version

project = "opytimark"
copyright = "2020, Gustavo de Rosa"
author = "Gustavo de Rosa"
release = package_version("opytimark")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
autosummary_generate = True
exclude_patterns = ["_build"]
html_theme = "alabaster"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "special-members": "__call__",
}
autodoc_member_order = "bysource"
autoclass_content = "both"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
