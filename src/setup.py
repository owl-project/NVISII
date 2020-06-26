# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md
from setuptools import setup, dist
import wheel

import visii

# force setuptools to recognize that this is
# actually a binary distribution
class BinaryDistribution(dist.Distribution):
    def has_ext_modules(foo):
        return True

setup(
    # This package is called visii
    name='visii',

    packages = ['visii'], # include the package "visii" 

    # make sure the shared library is included
    package_data = {'': ("*.dll", "*.pyd", ".so")},
    include_package_data=True,

    description='',

    # See class BinaryDistribution that was defined earlier
    distclass=BinaryDistribution,

    # This gets the version from the most recent git tag, potentially concatinating a commit hash at the end.
    use_scm_version={
        'fallback_version': '0.0.0-dev0',
        "root" : "..",
        "relative_to" : __file__ 
    },

    # Note, below might give "WARNING: The wheel package is not available." if wheel isnt installed
    setup_requires=['setuptools_scm'], # discouraged

    author='Nate Morrical',
    author_email='',
    maintainer='',
    maintainer_email='',
    

    zip_safe=False,
    python_requires = "~=" + str(visii._built_major_version) + "." + str(visii._built_minor_version),
)
