# https://scikit-learn.org/stable/modules/feature_selection.html#
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


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
        return _f_classif_square_selector(
            X_train, X_test, y_train, y_test, n_features
        )
    else:
        raise Exception("Not supported feature selectio method"
                        "Do not use underscore _")


def _f_classif_square_selector(X_train, X_test, y_train, y_test, n_features):
    kbest = SelectKBest(f_classif, n_features)
    kbest.fit(X_train, y_train)
    X_train = X_train[:, kbest.get_support()]
    X_test = X_test[:, kbest.get_support()]
    return X_train, X_test, y_train, y_test
