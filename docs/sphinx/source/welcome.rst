Welcome to Cyanure GPU's documentation!
=======================================

Cyanure is an open-source python software package with a Python 3 interface.
It is a GPU reimplementation of certain solvers from the original `Cyanure <https://github.com/inria-thoth/cyanure>`_ library, It is 
The goal of Cyanure is to provide state-of-the-art solvers for learning linear models,
based on stochastic variance-reduced stochastic optimization with
acceleration mechanisms and Quasi-Newton principles.
Cyanure can handle different loss functions (logistic, square,
multinomial logistic) and regularization functions (:math:`\ell_2`,
:math:`\ell_1`).
It provides a simple Python API, which should be fully compatible with scikit-learn.

The code is written by Thomas Ryckeboer (Inria, Univ. Grenoble-Alpes), and 
a documentation is provided in pdf in the following arXiv document

* Julien Mairal. `Cyanure: An Open-Source Toolbox for Empirical Risk Minimization for Python, C++, and soon more <https://arxiv.org/abs/1912.08165>`_ arXiv:1912.08165.  2019 

Main features
-------------
Cyanure is build upon several goals and principles:
   * **Cyanure is memory efficient**. Compared to the C++ version, the library only works with dense matrix. The library works by default with float32 to be able 
   to be able to feat as much data as possible on a GPU. (There should not have a significant drop in performances using float32 instead of float64). There can be intermediate matrix during calculation 
   but it is for the sake of computation speed. When fitting an
   intercept, there is no need to add a column of 1's and there is no matrix copy as well. 
   * **Cyanure implements fast algorithms.** Cyanure builds upon two algorithmic principles: (i) variance-reduced stochastic optimization; (ii) Nesterov of Quasi-Newton acceleration. Variance-reduced stochastic optimization algorithms are now popular, but tend to perform poorly when the objective function is badly conditioned. We observe large gains when combining these approaches with Quasi-Newton. 
   * **Cyanure only depends on Pytorch.** Cyanure depends on usual machine learning libraries numpy, scipy, Pytorch and scikit-learn for Python.
   * **Cyanure can handle many combinations of loss and regularization functions.** Cyanure can handle a vast combination of loss functions (logistic, square, multiclass logistic) with regularization functions (:math:`\ell_2`, :math:`\ell_1`).
   * **Cyanure provides optimization guarantees.** We believe that reproducibility is important in research. For this reason, knowing if you have solved your problem when the algorithm stops is important. Cyanure provides such a guarantee with a mechanism called duality gap.
   * **Cyanure is easy to use.** All the classes of the library are compatible with the [SKLEARN]_'s API, in order to use Cyanure with minimum effort.

Besides all these nice features, Cyanure has also probably some drawbacks, which we will let you discover by yourself.  


License
=======
Cyanure GPU is distributed under BSD-3-Clause license. Even though this is non-legally binding, the author kindly ask users to cite the previous arXiv document in their publications, as well as the publication related to the algorithm they have chosen (see References section). 
Note that if you have chosen the solver 'auto', you are likely to use [QNING]_ or [CATALYST]_ combined with [ISTA]_.


Installation
============

The recommanded installation procedure is to use either conda or pip.

For conda the package is available on conda forge.

You can either install with:

 `conda install -c conda-forge cyanureGPU`

 or 

 `pip install cyanureGPU`

You can also install cyanureGPU from the sources. 

.. image:: logo-inria-scientifique-couleur.jpg 
   :width: 35%
.. image:: erc-logo.gif
   :width: 15%
.. image:: logo_miai.jpg
   :width: 20%
