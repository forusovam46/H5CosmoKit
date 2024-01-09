# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'H5CosmoKit'
copyright = '2024, Magdalena Forusova'
author = 'Magdalena Forusova'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', # for reading your python files and automatically extract docstrings
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon', # allows sphinx to understand google-style docstrings
    "myst_nb"
]

templates_path = ['_templates']
exclude_patterns = []

language = 'y'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# -- Add package to path -----------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../../H5CosmoKit/'))
