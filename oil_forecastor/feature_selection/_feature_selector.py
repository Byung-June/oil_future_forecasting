# https://scikit-learn.org/stable/modules/feature_selection.html#
from sklearn.feature_selection import SeleckKBest
from sklearn.feature_selection import chi2


def select():
    pass


def _chi_square_selector(X_train, y_train, k=20):
    return SeleckKBest(chi2, k).fit_transform(X_train, y_train)
