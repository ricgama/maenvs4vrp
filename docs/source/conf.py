import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.abspath(__file__)))))

sys.path.append(BASE_DIR)  # append the path to system

#sys.path.insert(0,os.path.abspath('........'))
#sys.path.append(os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'maenvs4vrp'
copyright = '2025, maenvs4vrp'
author = 'marlvrp'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_math_dollar',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'sphinx_copybutton']


mathjax_config = {
    'tex2jax': {
        'inlineMath': [ ["\\(","\\)"] ],
        'displayMath': [["\\[","\\]"] ],
    },
}

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}


napoleon_google_docstring = True #allows the use of Google's docstring style
napoleon_use_param = False #disables the use of :param: tags in docstrings, used to document the parameters of a function or method
napoleon_use_ivar = True #allows the use of :ivar: tags in docstrings,used to document the instance variables of a class

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {
    '.md': 'markdown',
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
}

# The encoding of source files.
#
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

exclude_patterns = []

pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

html_static_path = ['_static']

html_css_files = [
    'custom.css',
]
