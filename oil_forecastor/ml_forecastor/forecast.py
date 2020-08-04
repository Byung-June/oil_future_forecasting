import numpy as np
import pandas as pd
from ..model_selection import rolling_sample
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from functools import wraps  # for debugging purpose


def rolling(func):
    @wraps(func)
    def train_model_wrapper(self, *args, **kwargs):
        y_list = []
        for i, time_idx in enumerate(pd.to_datetime(self.data['date'])):
            if time_idx < self.start_time:
                continue
            train_test = self._data_helper(i)
            y_pred_test = func(self, train_test)
            y_list.append(y_pred_test)
        y_list = np.array(y_list)
        return y_list
    return train_model_wrapper


class MLForecast():
    def __init__(self, data, n_windows, n_samples):
        self.data = data
        self.n_windows, self.n_samples = n_windows, n_samples
        start_time = n_windows + n_samples
        self.start_time = pd.to_datetime(data['date'].iloc[start_time])

    def _data_helper(self, time_idx):

        data_tuple = rolling_sample(
                self.data, self.n_windows, self.n_samples, time_idx
            )
        X_train = np.stack(
            [elt.values for elt in data_tuple[0]], axis=-1
        )
        X_test = np.expand_dims(data_tuple[1].values, axis=-1)
        X_train = X_train.reshape(X_train.shape[-1], -1)
        X_test = X_test.reshape(X_test.shape[-1], -1)
        y_train = np.array(data_tuple[2]).reshape(-1, 1)
        y_test = np.array(data_tuple[3]).reshape(-1, 1)
        return X_train, X_test, y_train, y_test

    @rolling
    def linear_reg(self, train_test=None):
        X_train, X_test, y_train, y_test = train_test
        linear_regressor = LinearRegression()
        linear_regressor.fit(X_train, y_train)
        y_pred = linear_regressor.predict(X_test)
        return y_pred, y_test

    @rolling
    def lasso(self, train_test):
        X_train, X_test, y_train, y_test = train_test
        lasso_gridsearch = GridSearchCV(
            Lasso(),
            verbose=0, param_grid={"alpha": np.logspace(-2, 1, 10)},
            scoring='r2'
        )
        lasso_gridsearch.fit(X_train, y_train)
        y_pred = lasso_gridsearch.predict(X_test)
        return y_pred, y_test

    @rolling
    def decision_tree_reg(self, train_test):
        X_train, X_test, y_train, y_test = train_test
        dtr_gridsearch = GridSearchCV(
            DecisionTreeRegressor(),
            verbose=2, param_grid={"max_depth": [i + 1 for i in range(20)]},
            scoring='r2'
        )
        dtr_gridsearch.fit(X_train, y_train)
        y_pred = dtr_gridsearch.predict(X_test)
        return y_pred, y_test

    @rolling
    def grad_boost_reg(self, train_test):
        X_train, X_test, y_train, y_test = train_test
        dtr_gridsearch = GridSearchCV(
            GradientBoostingRegressor(),
            verbose=2,
            param_grid={
                'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                'max_depth': [3, 4, 5]
            },
            scoring='r2', n_jobs=4
        )
        dtr_gridsearch.fit(X_train, y_train)
        y_pred = dtr_gridsearch.predict(X_test)
        return y_pred, y_test

    @rolling
    def hist_grad_boost_reg(self, train_test):
        X_train, X_test, y_train, y_test = train_test
        hgbr_gridsearch = GridSearchCV(
            HistGradientBoostingRegressor(),
            verbose=2,
            param_grid={
                'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                'max_depth': [3, 4, 5]
            },
            scoring='r2', n_jobs=4
        )
        hgbr_gridsearch.fit(X_train, y_train)
        y_pred = hgbr_gridsearch.predict(X_test)
        return y_pred, y_test

    @rolling
    def pcr(self, train_test):
        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-fa-model-selection-py
        n_components = np.arange(1, 21)
        X_train, X_test, y_train, y_test = train_test
        linear_regressor = LinearRegression()
        pcr_scores = []
        for i in n_components:
            print(i)
            pca = PCA(n_components=i)
            X_train_reduced = pca.fit_transform(X_train)
            pcr_score = cross_val_score(
                linear_regressor, X_train_reduced, y_train,
                scoring='r2'
            ).mean()
            pcr_scores.append(pcr_score)

        n_components_pcr = n_components[np.argmax(pcr_scores)]
        linear_regressor_pcr = LinearRegression()

        pca_pcr = PCA(n_components_pcr)
        X_train_reduced = pca_pcr.fit_transform(X_train)
        linear_regressor_pcr.fit(X_train_reduced, y_train)
        X_test_reduced = pca_pcr.fit_transform(X_test)
        y_pred = linear_regressor_pcr.predict(X_test_reduced)
        return y_pred, y_test

    @rolling
    def rand_forest_reg(self, train_test):
        X_train, X_test, y_train, y_test = train_test
        rfr_gridsearch = GridSearchCV(
            RandomForestRegressor(),
            verbose=2, param_grid={
                "max_depth": [4, 5, 6],
                "min_samples_split": [2, 3, 4],
                "min_samples_leaf": [1, 2, 3]
            },
            scoring='r2', n_jobs=4
        )
        rfr_gridsearch.fit(X_train, y_train)
        y_pred = rfr_gridsearch.predict(X_test)
        return y_pred, y_test

    def svr(self):
        raise Exception("Too slow")
