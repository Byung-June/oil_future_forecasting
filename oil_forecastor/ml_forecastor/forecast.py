import numpy as np
import pandas as pd
from functools import wraps  # for debugging purpose
from tqdm import tqdm
import multiprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from ..model_selection import rolling_train_test_split
from ..feature_selection import selector

n_cpus = max(multiprocessing.cpu_count() - 2, 4)


def rolling(func):
    @wraps(func)
    def train_model_wrapper(
        self,
        n_features=np.inf, method=None, *args, **kwargs
    ):
        y_pred_list = []
        sample_date = pd.to_datetime(
            self.data.index[:self.end_time_idx]
        )
        for i, time_idx in enumerate(tqdm(sample_date)):
            if time_idx < self.start_time:
                continue

            train_test = self._data_helper(i, n_features, method)
            y_pred = func(self, train_test, n_features, method)
            y_pred_list.append(y_pred)
        y_pred_list = np.array(y_pred_list).reshape(-1, 1)
        return y_pred_list
    return train_model_wrapper


class MLForecast():
    def __init__(self, data, n_windows, n_samples, start_time, end_time):
        self.data = data
        self.n_windows, self.n_samples = n_windows, n_samples
        self.start_time = pd.to_datetime(data.index[start_time])
        self.end_time_idx = end_time
        self.end_time = pd.to_datetime(data.index[end_time])
        self.verbose = 0

    def _data_helper(self, time_idx, n_features, method):

        data_tuple = rolling_train_test_split(
                self.data, self.n_windows, self.n_samples, time_idx
            )
        X_train = np.stack(
            [elt.values for elt in data_tuple[0]], axis=-1
        )
        X_test = np.expand_dims(data_tuple[1].values, axis=-1)
        X_train = X_train.reshape(X_train.shape[-1], -1)
        X_test = X_test.reshape(X_test.shape[-1], -1)
        y_train = np.array(data_tuple[2])
        y_test = np.array(data_tuple[3])

        if n_features < np.inf:
            X_train, X_test, y_train, y_test = selector(
                X_train, X_test, y_train, y_test, n_features, method
            )
        return X_train, X_test, y_train, y_test

    @rolling
    def linear_reg(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        linear_regressor = LinearRegression()
        linear_regressor.fit(X_train, y_train)
        y_pred = linear_regressor.predict(X_test)
        return y_pred

    @rolling
    def lasso(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        lasso_gridsearch = GridSearchCV(
            Lasso(max_iter=3000, tol=5e-2, selection='random'),
            verbose=self.verbose, param_grid={"alpha": np.logspace(-3, 2, 60)},
            scoring='r2', n_jobs=n_cpus
        )
        lasso_gridsearch.fit(X_train, y_train)
        y_pred = lasso_gridsearch.predict(X_test)
        return y_pred

    @rolling
    def decision_tree_reg(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        dtr_gridsearch = GridSearchCV(
            DecisionTreeRegressor(),
            verbose=self.verbose,
            param_grid={"max_depth": [i + 1 for i in range(20)]},
            scoring='r2', n_jobs=n_cpus
        )
        dtr_gridsearch.fit(X_train, y_train)
        y_pred = dtr_gridsearch.predict(X_test)
        return y_pred

    @rolling
    def grad_boost_reg(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        dtr_gridsearch = GridSearchCV(
            GradientBoostingRegressor(),
            verbose=self.verbose,
            param_grid={
                'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                'max_depth': [2, 3, 4, 5, 6]
            },
            scoring='r2', n_jobs=n_cpus
        )
        dtr_gridsearch.fit(X_train, y_train)
        y_pred = dtr_gridsearch.predict(X_test)
        return y_pred

    @rolling
    def hist_grad_boost_reg(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        hgbr_gridsearch = GridSearchCV(
            HistGradientBoostingRegressor(),
            verbose=self.verbose,
            param_grid={
                'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                'max_depth': [2, 3, 4, 5, 6]
            },
            scoring='r2', n_jobs=n_cpus
        )
        hgbr_gridsearch.fit(X_train, y_train)
        y_pred = hgbr_gridsearch.predict(X_test)
        return y_pred

    @rolling
    def pcr(self, train_test, n_features=np.inf, method=None):
        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-fa-model-selection-py
        n_components = np.arange(1, 21)
        X_train, X_test, y_train, y_test = train_test
        linear_regressor = LinearRegression()
        pcr_scores = []

        pca = PCA()
        X_train_reduced = pca.fit_transform(X_train)
        for i in n_components:
            pcr_score = cross_val_score(
                linear_regressor, X_train_reduced[:, :i], y_train,
                scoring='r2'
            ).mean()
            pcr_scores.append(pcr_score)

        n_components_pcr = n_components[np.argmax(pcr_scores)]
        linear_regressor_pcr = LinearRegression()

        linear_regressor_pcr.fit(
            X_train_reduced[:, :n_components_pcr], y_train
        )
        X_test_reduced = pca.transform(X_test)[:, :n_components_pcr]
        y_pred = linear_regressor_pcr.predict(X_test_reduced)
        return y_pred

    @rolling
    def rand_forest_reg(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        rfr_gridsearch = GridSearchCV(
            RandomForestRegressor(),
            verbose=self.verbose, param_grid={
                "max_depth": [2, 3, 4, 5, 6],
                "min_samples_split": [2, 3, 4],
                "min_samples_leaf": [2, 3]
            },
            scoring='r2', n_jobs=n_cpus
        )
        rfr_gridsearch.fit(X_train, y_train)
        y_pred = rfr_gridsearch.predict(X_test)
        return y_pred

    @rolling
    def svr(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        svr_gridsearch = GridSearchCV(
            SVR(kernel='rbf', gamma=0.1, cache_size=10000),
            verbose=self.verbose,
            param_grid={
                'C': [1, 2, 3, 4, 5],
                'gamma': np.logspace(-2, 1, 5)
            },
            scoring='r2', n_jobs=n_cpus
        )
        svr_gridsearch.fit(X_train, y_train)
        y_pred = svr_gridsearch.predict(X_test)
        return y_pred

    @rolling
    def kernel_ridge(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        kr_gridsearch = GridSearchCV(
            KernelRidge(kernel='rbf', gamma=0.1),
            verbose=self.verbose,
            param_grid={"alpha": [1, 2, 5, 10, 20, 50, 100],
                        "gamma": np.logspace(-2, 3, 6)},
            scoring='r2', n_jobs=n_cpus
        )
        kr_gridsearch.fit(X_train, y_train)
        y_pred = kr_gridsearch.predict(X_test)
        return y_pred
