References
==========

We provide here various references regarding the solvers implemented in Cyanure. 

Accelerators
------------
Cyanure uses two types of accelerators. The QNing approach builds upon Quasi-Newton principles and was introduced in

.. [QNING] H. Lin, J. Mairal and Z. Harchaoui. `An Inexact Variable Metric Proximal Point Algorithm for Generic Quasi-Newton Acceleration <https://arxiv.org/pdf/1610.00960v4>`_. SIAM Journal on Optimization. 29(2), pages 1408–1443, 2019.

Catalyst uses Nesterov's acceleration, and was introduced in

.. [CATALYST] H. Lin, J. Mairal and Z. Harchaoui. `Catalyst Acceleration for First-order Convex Optimization: from Theory to Practice <https://arxiv.org/abs/1712.05654>`_. Journal of Machine Learning Research (JMLR). 18(212), pages 1–54, 2018.

Batch algorithms
----------------
Cyanure also implements ISTA and FISTA with line-search, as described in 

.. [FISTA] A. Beck and M. Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM journal on imaging sciences, 2(1), 183-202. 2009.

It is perhaps worth noting that qing-ista seems to perform always better than fista in all our experiments (see benchmark section).

Other frameworks
----------------
Even though Cyanure does not depend on it, our goal is to make it easy to use within Scikit-learn

.. [SKLEARN] F. Pedregosa, G. Varoquaux, A. Gramfort and others. `Scikit-learn: Machine learning in Python <https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf>`_. Journal of machine learning research, 12(Oct), 2825-2830. 2011.

Other solvers in our comparisons include also 

.. [LIBLINEAR]  Fan, R. E., Chang, K. W., Hsieh, C. J., Wang, X. R., & Lin, C. J. (2008). LIBLINEAR: A library for large linear classification. Journal of machine learning research, 9(Aug), 1871-1874.
.. [LBFGS] Nocedal, J. (1980). "Updating Quasi-Newton Matrices with Limited Storage". Mathematics of Computation. 35 (151): 773–782
