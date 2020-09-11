# https://scikit-learn.org/stable/modules/feature_selection.html#
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
import warnings

warnings.filterwarnings('ignore')


def selector(
    X_train, X_test, y_train, y_test, n_features,
    method='f-classif'
):
    """[summary]

    Args:
        X_train (array): n_X_train_samples x None
        X_test (array): n_test_samples x None
        y_train (array): n_y_train_samples x None
        y_test (array): n_y_test_samples x None
        n_features (int): number of features
        method (str, optional): [description]. Defaults to 'f-classif'.

    Raises:
        Exception: [description]

    Returns:
        tuple: X_train_reduce, X_test_reduced, y_train_reduced, y_test_reduced
    """
    if method == 'f-classif':
        raise Exception("f-classif cannot be used for regression purpose")
    elif method == 'f-regression':
        return _f_regression_selector(X_train, X_test, y_train, y_test,
                                      n_features)
    elif method == 'mutual-info-regression':
        return _mutual_info_regression_selector(X_train, X_test,
                                                y_train, y_test,
                                                n_features)
    else:
        raise Exception("Not supported feature selectio method"
                        "Do not use underscore _")


def _f_classif_square_selector(X_train, X_test, y_train, y_test, n_features):
    if n_features > X_train.shape[0]:
        n_features = X_train.shape[0]
    kbest = SelectKBest(f_classif, n_features)
    kbest = kbest.fit(X_train, y_train)
    X_train = kbest.transform(X_train)
    X_test = kbest.transform(X_test)
    return X_train, X_test, y_train, y_test


def _f_regression_selector(X_train, X_test, y_train, y_test, n_features):
    if n_features > X_train.shape[0]:
        n_features = X_train.shape[0]
    kbest = SelectKBest(f_regression, n_features)
    kbest.fit(X_train, y_train)
    X_train = kbest.transform(X_train)
    X_test = kbest.transform(X_test)
    return X_train, X_test, y_train, y_test


def _mutual_info_regression_selector(X_train, X_test, y_train, y_test,
                                     n_features):
    if n_features > X_train.shape[0]:
        n_features = X_train.shape[0]
    kbest = SelectKBest(mutual_info_regression, n_features)
    kbest.fit(X_train, y_train)
    X_train = kbest.transform(X_train)
    X_test = kbest.transform(X_test)
    return X_train, X_test, y_train, y_test
