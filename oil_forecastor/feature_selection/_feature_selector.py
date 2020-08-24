# https://scikit-learn.org/stable/modules/feature_selection.html#
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def selector(
    X_train, X_test, y_train, y_test, n_features,
    method='f-classif'
):
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
