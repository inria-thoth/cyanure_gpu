import pytest


from sklearn.exceptions import ConvergenceWarning
from libsvmdata import fetch_libsvm
import scipy.sparse
import numpy as np
import warnings

from cyanure_pytorch.estimators import L1Logistic, Lasso, Regression, fit_large_feature_number


@pytest.fixture
def dataset_finance():
    X, y = fetch_libsvm('finance')
    X = scipy.sparse.csr_matrix(X)
    return X, y

@pytest.fixture
def dataset_rcv1():
    X, y = fetch_libsvm('rcv1.binary')
    if (scipy.sparse.issparse(X) and
                scipy.sparse.isspmatrix_csc(X)):
        X = scipy.sparse.csr_matrix(X)
    return X, y

@pytest.mark.parametrize(
    "estimator",
    [ L1Logistic(verbose=False)]
)
def test_active_set_rcv1(estimator, dataset_rcv1):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    X, y = dataset_rcv1[0], dataset_rcv1[1]

    n_samples = X.shape[0]
    estimator.lambda_1= 2697 / n_samples
    estimator.fit_intercept = False

    estimator.max_iter = 500

    estimator.fit(X, y)