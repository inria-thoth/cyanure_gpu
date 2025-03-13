import numpy as np

from cyanureGPU.estimators import Regression, fit_large_feature_number

def test_fit_large_feature_number():
    # Test case with large number of features and dense matrix
    X = np.random.rand(100, 1001)
    labels = np.random.randint(2, size=100)
    estimator = Regression(lambda_1=0.1, fit_intercept=True, tol=1e-4, verbose=True)
    aux = Regression(lambda_1=0.1, fit_intercept=True, tol=1e-4, verbose=True)
    type(estimator).__name__ = "Lasso"
    type(aux).__name__ = "Lasso"
    result = fit_large_feature_number(estimator, aux, X, labels)
    assert isinstance(result, Regression)
    assert result.coef_.shape == (1001,)
    assert result.intercept_ is not None
    assert result.intercept_ != 0
