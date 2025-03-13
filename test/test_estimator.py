import pytest

from sklearn.utils.estimator_checks import check_estimator

from cyanureGPU.estimators import LogisticRegression, Regression, Classifier, L1Logistic, Lasso


@pytest.mark.parametrize(
    "estimator",
    [LogisticRegression(verbose=False), Regression(verbose=False), Classifier(verbose=False), Lasso(verbose=False), L1Logistic(verbose=False)]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
