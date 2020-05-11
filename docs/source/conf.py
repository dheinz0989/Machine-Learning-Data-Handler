# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[2].joinpath("src")

sys.path.insert(0,str(src_path))

# -- Project information -----------------------------------------------------

project = 'Machine Learning Data Handler'
copyright = '2020, Dominik Heinz'
author = 'Dominik Heinz'

# The full version, including alpha/beta/rc tags
release = '1.2'

autodoc_default_options = {
    'members': True,
    'private-members' : True,
    'show-inheritance': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

master_doc ='ML_Data_Handler'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
today_fmt = '%d of %B %Y at %H:%M'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinxdoc'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ['_static']
#html_static_path = []

