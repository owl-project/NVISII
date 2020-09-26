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
import subprocess
sys.path.insert(0, os.path.abspath('../../install/'))
subprocess.run('wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin; sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600; sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub; sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"; sudo apt-get update; sudo apt-get --no-install-recommends -y install cuda-toolkit-10-2', shell=True)
sys.path.insert(0, '/usr/local/cuda/lib64/stubs/')

#print(os.listdir("/home/docs/checkouts/readthedocs.org/user_builds/visii/conda/latest/"))
#print(os.listdir("/home/docs/checkouts/readthedocs.org/user_builds/visii/conda/latest/pkgs/"))
#print(os.listdir("/home/docs/checkouts/readthedocs.org/user_builds/visii/conda/latest/pkgs/cuda-toolkit/"))
#print(os.listdir("/home/docs/checkouts/readthedocs.org/user_builds/visii/conda/latest/pkgs/cuda-toolkit/lib/"))
#print(os.listdir("/home/docs/checkouts/readthedocs.org/user_builds/visii/conda/latest/pkgs/cuda-toolkit/lib/stubs/"))

# -- Project information -----------------------------------------------------

project = 'ViSII'
copyright = '2020, Nate Morrical'
author = 'Nate Morrical'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

def setup(app):
    app.add_css_file('custom.css')
