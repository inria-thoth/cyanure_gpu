import platform
import struct

import contextlib
import os
import sys

# Override sdist to always produce .zip archive
from distutils.command.sdist import sdist as _sdist

from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import numpy

class sdistzip(_sdist):
    def initialize_options(self):
        _sdist.initialize_options(self)
        self.formats = ['zip', 'gztar']

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VERSION')) as version_file:
    version = version_file.read().strip()

setup(name='cyanure',
      version=version,
      author="Julien Mairal",
      author_email="julien.mairal@inria.fr",
      license='bsd-3-clause',
      url="https://inria-thoth.github.io/cyanure/welcome.html",
      description='optimization toolbox for machine learning',
      install_requires=["scipy<=1.8.1;python_version<'3.11'", "scipy>=1.8.1;python_version>='3.11'", "numpy>=1.23.5;python_version>='3.11'", "numpy<=1.23.5;python_version<'3.11'",'scikit-learn'],
      packages=find_packages(),
      cmdclass={'sdist': sdistzip},
      py_modules=['cyanure'],
      long_description="Cyanure is an open-source C++ software package with a Python 3 interface. The goal of Cyanure is to provide state-of-the-art solvers for learning linear models, based on stochastic variance-reduced stochastic optimization with acceleration mechanisms and Quasi-Newton principles. Cyanure can handle a large variety of loss functions (logistic, square, squared hinge, multinomial logistic) and regularization functions (l2, l1, elastic-net, fused Lasso, multi-task group Lasso). It provides a simple Python API, which should be fully compatible with scikit-learn.")

