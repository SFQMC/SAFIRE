# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

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
#import os
#import sys
#sys.path.insert(0, os.path.abspath('extensions'))


# -- Project information -----------------------------------------------------

project = 'SAFIRE'

# The full version, including alpha/beta/rc tags
version = '0.1.0'
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
import sys, os
sys.path.append(os.path.abspath('extensions'))

import sphinx_rtd_theme

extensions = [
    'sphinxcontrib.bibtex', 
    'sphinx_rtd_theme', 
    'myst_nb',
    'sphinx.ext.mathjax',
    'sphinx.ext.apidoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'numpydoc',
    'notebook_header',
    ]
bibtex_bibfiles = ['bibs/afqmc.bib']

def setup(app):
    # Add CSS for bold references
    app.add_css_file('custom.css')

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb'
}


suppress_warnings = [
]

numpydoc_show_class_members = False

numfig = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'afqmctools/api/modules.rst' # we only have one module, so we do not need this
]

apidoc_modules = [
    {
        'path': '../utils/afqmctools',
        'destination': 'afqmctools/api/',
        'exclude_patterns': ['**/qe_driver.py', '**/hamiltonian/hubbard.py'],
    },
]

autodoc_default_options = {
    'exclude-members': 'warn,jit',
}

mathjax3_config = {
      "chtml": {
          "mtextInheritFont": True,
      },
  }

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'sphinx_rtd_theme'
html_theme = 'pydata_sphinx_theme' # possible theme - images need be compatible with dark and light modes

# Base URL of the published docs. Used by Sphinx for canonical links and by the
# notebook_header extension to rewrite {doc} cross-references into absolute links
# in the downloadable .ipynb notebooks.
html_baseurl = 'https://safire.flatironinstitute.org/docs/dev/'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

html_theme_options = {
    "logo": {
        "image_light": "images/logo.svg",
        "link": "https://safire.flatironinstitute.org/",
    },
    "globaltoc_maxdepth": 3,        # -1 for unlimited
    "globaltoc_collapse": False,    # expand whole tree or not
    "globaltoc_includehidden": False, # show :hidden: entries
    "navigation_depth": 3,
    "header_links_before_dropdown": 6,
    # Show this many levels expanded on page load (default: 1)
    "show_nav_level": 2,
    # Maximum depth rendered in the left nav (default: 4)
    # Remove the light/dark theme switcher (default: ["theme-switcher", "navbar-icon-links"])
    "navbar_end": ["navbar-icon-links"],
    "navbar_align": "left",
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/SFQMC/SAFIRE",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
   ]
}

# Force light mode (disables dark mode / the auto switcher default).
html_context = {
    "default_mode": "light",
}

## -- options for MyST-NB (rendering jupyter notebooks) ----------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath"
]
# required to have equations inside of text cells.
myst_dmath_double_inline = True

nb_execution_mode = "off"

myst_heading_anchors = 3
