import os

# Override sdist to always produce .zip archive
from distutils.command.sdist import sdist as _sdist

from setuptools import setup, find_packages

class sdistzip(_sdist):
    def initialize_options(self):
        _sdist.initialize_options(self)
        self.formats = ['zip', 'gztar']

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VERSION')) as version_file:
    version = version_file.read().strip()

setup(name='cyanureGPU',
      version=version,
      author="Julien Mairal",
      author_email="julien.mairal@inria.fr",
      license='bsd-3-clause',
      url="https://inria-thoth.github.io/cyanure_gpu/welcome.html",
      description='optimization toolbox for machine learning',
      install_requires=['scikit-learn', 'torch<2.3.0', 'numpy==1.26.4'],
      packages=find_packages(),
      cmdclass={'sdist': sdistzip},
      long_description="Cyanure is an open-source Python software package. This is a partial re implementation of Cyanure package. It provides a simple Python API, which should be fully compatible with scikit-learn.")

