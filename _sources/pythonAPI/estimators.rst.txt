.. automodule:: cyanure_pytorch.estimators

Estimators
----------

The link between the regularization parameter C of scikit-learn and :math:`\lambda`  is :math:`C=\frac{1}{n \lambda}`
The :math:`\frac{1}{n}` factor is handled inside the library. There is no need to scale the C factor.

The Regression Class
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Regression
   :members:
   :inherited-members:
   :show-inheritance:
   :autosummary-inherited-members:
   :member-order: bysource
   :exclude-members: get_metadata_routing, set_fit_request, set_score_request

The Classifier Class
^^^^^^^^^^^^^^^^^^^^


.. autoclass:: Classifier
   :members:
   :inherited-members:
   :show-inheritance:
   :autosummary-inherited-members:
   :member-order: bysource
   :exclude-members: get_metadata_routing, set_fit_request, set_score_request


Pre-configured classes
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LinearSVC
   :members:
   :show-inheritance:
   :autosummary-inherited-members:
   :member-order: bysource
   :exclude-members: get_metadata_routing, set_fit_request, set_score_request


.. autoclass:: LogisticRegression
   :members:
   :show-inheritance:
   :autosummary-inherited-members:
   :member-order: bysource
   :exclude-members: get_metadata_routing, set_fit_request, set_score_request


.. autoclass:: Lasso
   :members:
   :show-inheritance:
   :autosummary-inherited-members:
   :member-order: bysource
   :exclude-members: get_metadata_routing, set_fit_request, set_score_request


.. autoclass:: L1Logistic
    :members:
    :show-inheritance:
    :autosummary-inherited-members:
    :member-order: bysource
    :exclude-members: get_metadata_routing, set_fit_request, set_score_request


