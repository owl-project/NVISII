# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md
from setuptools import setup, dist
import wheel 
import os
# required to geneerate a platlib folder required by audittools
from setuptools.command.install import install
# for generating a wheel version from git tag
from setuptools_scm import get_version

class InstallPlatlib(install):
    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib

# force setuptools to recognize that this is
# actually a binary distribution
class BinaryDistribution(dist.Distribution):
    def is_pure(self):
        return False
    def has_ext_modules(foo):
        return True

# This gets the version from the most recent git tag, potentially concatinating 
# a commit hash at the end.
current_version = get_version(
    root = "..", 
    relative_to = __file__,
    fallback_version='0.0.0-dev0'
)

optix_version = os.environ.get("OPTIX_VERSION", None)
if optix_version:
    current_version = current_version + "." + optix_version

print(current_version)

setup(
    # This package is called nvisii
    name='nvisii',

    install_requires = ['numpy=1.20.0'],

    packages = ['nvisii', "nvisii.importers"], # include the package "nvisii" 

    # make sure the shared library is included
    package_data = {'': ("*.dll", "*.pyd", "*.so")},
    include_package_data=True,

    description='',

    # See class BinaryDistribution that was defined earlier
    distclass=BinaryDistribution,

    version = current_version,

    author='Nate Morrical',
    author_email='',
    maintainer='',
    maintainer_email='',
    
    python_requires = ">=2.7",
    cmdclass={'install': InstallPlatlib},
)
