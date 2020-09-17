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
# from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from ..model_selection import rolling_train_test_split
from ..feature_selection import selector
# from ..model_selection._utility import adf_test
import copy
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')

n_cpus = max(multiprocessing.cpu_count() - 2, 4)


def recover(i, diff_order, y_pred, y_test):
    y_t = y_test[-1]
    y_t_before = y_test[-2]
    y_t_2 = y_test[-3]
    if diff_order == 0:
        y_pred_recovered = y_pred
    elif diff_order == 1:
        y_pred_recovered = y_pred + y_t
    elif diff_order == 2:
        y_pred_recovered = y_pred + 2 * y_t - y_t_before
    elif diff_order == 3:
        y_pred_recovered = y_pred - 3 * y_t + 3 * y_t_before - y_t_2
    else:
        raise ValueError("Too high diff order")
    return y_pred_recovered


def auto_diff(train_test):
    train_test = list(train_test)
    # diff_order = max(adf_test(train_test[2]))
    diff_order = 0
    train_test[0] = train_test[0][diff_order:, ...]
    train_test[-2] = np.diff(train_test[-2], diff_order)
    return train_test, diff_order


def rolling(func):
    @wraps(func)
    def train_model_wrapper(self,
                            n_features=np.inf, method=None, *args, **kwargs):
        sample_date = pd.to_datetime(self.data.index[:self.end_time_idx])
        df = pd.DataFrame(np.nan,
                          index=self.data.index,
                          columns=['y_pred', 'y_test'])
        # add one day because the test date is one day
        # after the last date of thetraining dataset
        for i, time_idx in enumerate(tqdm(sample_date)):
            if time_idx < self.start_time:
                continue
            train_test = self._data_helper(i, n_features, method)
            train_test, diff_order = auto_diff(train_test)
            y_train = train_test[-2]
            y_test = train_test[-1].reshape(-1, 1)

            d_y_pred = func(self, train_test, n_features, method)
            d_y_pred = d_y_pred.reshape(1, 1)
            y_pred = recover(i, diff_order, d_y_pred, y_train)

            df['y_test'].iloc[i+1] = y_test.flatten()
            df['y_pred'].iloc[i+1] = y_pred.flatten()
        return df
    return train_model_wrapper


class MLForecast():
    def __init__(self, data, n_windows, n_samples,
                 start_time, end_time, scaler):
        self.data = data
        self.y_test = copy.deepcopy(self.data['y_test'])
        self.n_windows, self.n_samples = n_windows, n_samples
        self.start_time = pd.to_datetime(data.index[start_time])
        self.end_time_idx = end_time
        self.end_time = pd.to_datetime(data.index[end_time])
        self.verbose = 0
        self.scaler = scaler

    def _scaler(self, X_train, y_train, index,
                method='robust'):
        if method == 'robust':
            X_transformer = RobustScaler().fit(X_train[:, index])
        elif method == 'none':
            X_transformer = None
        return X_transformer

    def _data_helper(self, time_idx, n_features, method):
        data_tuple = rolling_train_test_split(
                self.data, self.n_windows, self.n_samples, time_idx
            )
        X_train = np.stack(
            [elt.values for elt in data_tuple[0]], axis=-1
        )
        X_test = np.expand_dims(data_tuple[1].values, axis=-1)
        X_train = X_train.reshape(-1, X_train.shape[-1])
        X_train = np.transpose(X_train)
        X_test = X_test.reshape(X_test.shape[-1], -1)
        y_train = np.array(data_tuple[2]).reshape(-1, 1)
        y_test = np.array(data_tuple[3]).reshape(1, 1)

        std = X_train.std(axis=0) > 1000
        X_transformer = self._scaler(X_train, y_train, std,
                                     method=self.scaler)

        X_train_scaled = X_train
        X_test_scaled = X_test
        if X_transformer is not None:
            X_train_scaled[:, std]\
                = X_transformer.transform(X_train[:, std])
            X_test_scaled[:, std] = X_transformer.transform(X_test[:, std])
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # if True:
        #     kpca = KernelPCA(100)
        #     kpca = kpca.fit(X_train_scaled)
        #     X_train_scaled = kpca.transform(X_train_scaled)
        #     X_test_scaled = kpca.transform(X_test_scaled)

        if n_features < np.inf:
            X_train_scaled, X_test_scaled, y_train, y_test\
                = selector(X_train_scaled, X_test_scaled,
                           y_train, y_test,
                           n_features, method)
        y_train, y_test = y_train.flatten(), y_test.flatten()
        return X_train_scaled, X_test_scaled, y_train, y_test

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
            verbose=self.verbose, param_grid={"alpha": np.logspace(-3, 2, 15)},
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
            param_grid={
                "max_depth": [2, 3, 5, 10, 15, 20],
                "min_samples_split": [0.1, 0.3],
                "min_samples_leaf": [0.1, 0.2, 0.3, 0.5]
            },
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
                "learning_rate": [0.05, 0.1, 0.2, 0.3],
                "min_samples_split": [0.1, 0.3],
                'n_estimators': [20, 50, 100],
                'max_depth': [2, 3, 5, 10]
            },
            scoring='r2', n_jobs=n_cpus
        )
        dtr_gridsearch.fit(X_train, y_train)
        y_pred = dtr_gridsearch.predict(X_test)
        return y_pred

    @rolling
    def pcr(self, train_test, n_features=np.inf, method=None):
        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-fa-model-selection-py
        max_components = min(n_features, 30)
        n_components = np.arange(1, max_components)
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
                'n_estimators': [20, 50, 100],
                "max_depth": [2, 5, 10],
                "min_samples_split": [0.1, 0.3],
                "min_samples_leaf": [0.1, 0.2, 0.3, 0.5]
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
                'C': [0.1, 1, 3, 5],
                'gamma': np.logspace(-3, 1, 8)
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
                        "gamma": np.logspace(-3, 1, 8)},
            scoring='r2', n_jobs=n_cpus
        )
        kr_gridsearch.fit(X_train, y_train)
        y_pred = kr_gridsearch.predict(X_test)
        return y_pred
