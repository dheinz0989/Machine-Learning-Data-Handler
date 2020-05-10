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
#sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, r'C:\Daten\Trainings\ML_Data_Prep_backup\src')
src_path = Path(__file__).resolve().parents[2].joinpath("src")
#subfolder = [x for x in src_path.iterdir() if x.is_dir()]
#for folder in subfolder:
#    sys.path.insert(0, str(folder))
sys.path.insert(0,str(src_path))

# -- Project information -----------------------------------------------------

project = 'Machine Learning Data Handler'
copyright = '2020, Dominik Heinz'
author = 'Dominik Heinz'

# The full version, including alpha/beta/rc tags
release = '1.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.graphviz',
    'sphinx_autodoc_typehints'
]

master_doc ='ML_Data_Handler'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
today_fmt = '%d of %B %Y at %H:%M'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autodoc_default_flags = ['members', 'undoc-members', 'private-members', 'inherited-members', 'show-inheritance', 'autofunction']

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'nature'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ['_static']
#html_static_path = []

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)