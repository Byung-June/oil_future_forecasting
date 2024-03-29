import numpy as np
import pandas as pd
from functools import wraps  # for debugging purpose
from tqdm import tqdm
import multiprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression, \
    BayesianRidge, HuberRegressor, ARDRegression, ElasticNet
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
import pmdarima as pm
import copy
from sklearn.preprocessing import RobustScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
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
            train_test, y_transformer = self._data_helper(i, n_features,
                                                          method)
            train_test, diff_order = auto_diff(train_test)
            y_train = train_test[-2]
            y_test = train_test[-1].reshape(-1, 1)

            d_y_pred = func(self, train_test, n_features, method)
            d_y_pred = d_y_pred.reshape(1, 1)
            y_pred = recover(i, diff_order, d_y_pred, y_train)

            if self.scaler != 'none':
                y_test = y_transformer.inverse_transform(y_test)
                y_pred = y_transformer.inverse_transform(y_pred)

            df['y_test'].iloc[i+1] = y_test.flatten()
            df['y_pred'].iloc[i+1] = y_pred
        return df
    return train_model_wrapper


def lagged_features_extractor(X_train, X_test, num_lagged=1):
    X_train_lagged = X_train[:, :num_lagged, :]
    X_train_exo = X_train[:, num_lagged:, :]
    X_test_lagged = X_test[:, :num_lagged, :]
    X_test_exo = X_test[:, num_lagged:, :]
    return X_train_lagged, X_train_exo, X_test_lagged, X_test_exo


def _shape_manager(data):
    data = data.reshape(-1, data.shape[-1])
    data = np.transpose(data)
    return data


class MLForecast():
    def __init__(self, data, n_windows, n_samples,
                 start_time, end_time, scaler, n_columns,
                 random_state=0):
        self.data = data
        self.y_test = copy.deepcopy(self.data['y_test'])
        self.n_windows, self.n_samples = n_windows, n_samples
        self.start_time = pd.to_datetime(data.index[start_time])
        self.end_time_idx = end_time
        self.end_time = pd.to_datetime(data.index[end_time])
        self.verbose = 0
        self.scaler = scaler
        self.n_columns = n_columns
        self.random_state = random_state

    def _scaler(self, X_train, y_train, index,
                method='robust'):
        if method == 'robust':
            X_transformer = RobustScaler().fit(X_train[:, index])
            y_transformer = RobustScaler().fit(y_train)
        elif method == 'none':
            X_transformer = None
            y_transformer = None
        return X_transformer, y_transformer

    def _data_helper(self, time_idx, n_features, method):
        data_tuple = rolling_train_test_split(self.data,
                                              self.n_windows,
                                              self.n_samples,
                                              time_idx)
        X_train_ = np.stack([elt.values for elt in data_tuple[0]], axis=-1)
        X_test_ = np.expand_dims(data_tuple[1].values, axis=-1)
        X_train_lagged, X_train_exo, X_test_lagged, X_test_exo\
            = lagged_features_extractor(X_train_, X_test_,
                                        num_lagged=self.n_columns)

        X_train_lagged = _shape_manager(X_train_lagged)
        X_train_exo = _shape_manager(X_train_exo)
        X_test_lagged = _shape_manager(X_test_lagged)
        X_test_exo = _shape_manager(X_test_exo)
        y_train = np.array(data_tuple[2]).reshape(-1, 1)
        y_test = np.array(data_tuple[3]).reshape(1, 1)

        std = X_train_exo.std(axis=0) > 0.0
        X_transformer, y_transformer = self._scaler(X_train_exo, y_train, std,
                                                    method=self.scaler)

        X_train_exo_scaled = X_train_exo
        X_test_exo_scaled = X_test_exo
        if X_transformer is not None:
            X_train_exo_scaled[:, std]\
                = X_transformer.transform(X_train_exo[:, std])
            X_test_exo_scaled[:, std]\
                = X_transformer.transform(X_test_exo[:, std])
            y_train_scaled = y_transformer.transform(y_train)
            y_test_scaled = y_transformer.transform(y_test)
        else:
            X_train_exo_scaled = X_train_exo
            X_test_exo_scaled = X_test_exo
            y_train_scaled = y_train
            y_test_scaled = y_test

        if n_features < np.inf:
            (X_train_exo_scaled, X_test_exo_scaled,
             y_train_scaled, y_test_scaled)\
                = selector(X_train_exo_scaled, X_test_exo_scaled,
                           y_train_scaled, y_test_scaled,
                           n_features, method)
        y_train_scaled, y_test_scaled\
            = y_train_scaled.flatten(), y_test_scaled.flatten()
        X_train_scaled = np.concatenate([X_train_lagged, X_train_exo_scaled],
                                        axis=-1)
        X_test_scaled = np.concatenate([X_test_lagged, X_test_exo_scaled],
                                       axis=-1)
        return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled),\
            y_transformer

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
            Lasso(max_iter=100000, tol=1e-2, random_state=self.random_state),
            verbose=self.verbose,
            param_grid={"alpha": np.logspace(-3, 2, 15)},
            scoring='neg_mean_squared_error', n_jobs=n_cpus
        )
        lasso_gridsearch.fit(X_train, y_train)
        y_pred = lasso_gridsearch.predict(X_test)
        # lasso = Lasso()
        # lasso.fit(X_train, y_train)
        # y_pred = lasso.predict(X_test)
        return y_pred

    @rolling
    def ard(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        ard_gridsearch = GridSearchCV(
            ARDRegression(),
            verbose=self.verbose,
            param_grid={"alpha_1": np.logspace(-7, -5, 3),
                        "alpha_2": np.logspace(-7, -5, 3),
                        "lambda_1": np.logspace(-7, -5, 3),
                        "lambda_2": np.logspace(-7, -5, 3)}
        )
        ard_gridsearch.fit(X_train, y_train)
        y_pred = ard_gridsearch.predict(X_test)
        return y_pred

    @rolling
    def bayes(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        bayesian_gridsearch = GridSearchCV(
            BayesianRidge(),
            verbose=self.verbose,
            param_grid={"alpha_1": np.logspace(-7, -5, 3),
                        "alpha_2": np.logspace(-7, -5, 3),
                        "lambda_1": np.logspace(-7, -5, 3),
                        "lambda_2": np.logspace(-7, -5, 3)}
        )
        bayesian_gridsearch.fit(X_train, y_train)
        y_pred = bayesian_gridsearch.predict(X_test)
        return y_pred

    @rolling
    def huber(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        els = HuberRegressor()
        els.fit(X_train, y_train)
        y_pred = els.predict(X_test)
        return y_pred

    @rolling
    def pcr(self, train_test, n_features=np.inf, method=None):
        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-fa-model-selection-py
        max_components = min(n_features, 30)
        n_components = np.arange(1, max_components)
        X_train, X_test, y_train, y_test = train_test
        linear_regressor = LinearRegression()
        pcr_scores = []

        if n_features == 0:
            X_train, X_test, y_train, y_test = train_test
            linear_regressor = LinearRegression()
            linear_regressor.fit(X_train, y_train)
            y_pred = linear_regressor.predict(X_test)
            return y_pred

        pca = PCA(random_state=self.random_state)
        X_train_reduced = pca.fit_transform(X_train[:, self.n_columns:])
        for i in n_components:
            pcr_score = cross_val_score(
                linear_regressor,
                np.concatenate([X_train[:, :self.n_columns],
                                X_train_reduced[:, :i]],
                               axis=-1),
                y_train,
                scoring='neg_mean_squared_error'
            ).mean()
            pcr_scores.append(pcr_score)

        n_components_pcr = n_components[np.argmax(pcr_scores)]
        linear_regressor_pcr = LinearRegression()

        linear_regressor_pcr.fit(
            np.concatenate([X_train[:, :self.n_columns],
                            X_train_reduced[:, :n_components_pcr]],
                           axis=-1),
            y_train)
        X_test_reduced\
            = pca.transform(X_test[:, self.n_columns:])[:, :n_components_pcr]
        X_test_reduced = np.concatenate([X_test[:, :self.n_columns],
                                         X_test_reduced],
                                        axis=-1)
        y_pred = linear_regressor_pcr.predict(X_test_reduced)
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
            scoring='neg_mean_squared_error', n_jobs=n_cpus
        )
        svr_gridsearch.fit(X_train, y_train)
        y_pred = svr_gridsearch.predict(X_test)
        # svr = SVR(kernel='rbf')
        # svr.fit(X_train, y_train)
        # y_pred = svr.predict(X_test)
        return y_pred

    @rolling
    def kernel_ridge(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        kr_gridsearch = GridSearchCV(
            KernelRidge(kernel='rbf', gamma=0.1),
            verbose=self.verbose,
            param_grid={"alpha": [1, 2, 5, 10, 20, 50, 100],
                        "gamma": np.logspace(-3, 1, 8)},
            scoring='neg_mean_squared_error', n_jobs=n_cpus
        )
        kr_gridsearch.fit(X_train, y_train)
        y_pred = kr_gridsearch.predict(X_test)
        # kr = KernelRidge(kernel='rbf')
        # kr.fit(X_train, y_train)
        # y_pred = kr.predict(X_test)
        return y_pred

    @rolling
    def decision_tree_reg(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        dtr_gridsearch = GridSearchCV(
            DecisionTreeRegressor(random_state=self.random_state),
            verbose=self.verbose,
            param_grid={
                "max_depth": [2, 3, 5, 10, 15, 20],
                "min_samples_split": [2, 4, 8],
                "min_samples_leaf": [1, 2, 4]
            },
            scoring='neg_mean_squared_error', n_jobs=n_cpus
        )
        dtr_gridsearch.fit(X_train, y_train)
        y_pred = dtr_gridsearch.predict(X_test)
        # dtr = DecisionTreeRegressor()
        # dtr.fit(X_train, y_train)
        # y_pred = dtr.predict(X_test)
        return y_pred

    @rolling
    def rand_forest_reg(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        rfr_gridsearch = GridSearchCV(
            RandomForestRegressor(random_state=self.random_state),
            verbose=self.verbose, param_grid={
                'n_estimators': [200],
                "max_depth": [2, 5, 10],
                "min_samples_split": [2, 4, 8],
                "min_samples_leaf": [1, 2, 4]
            },
            scoring='neg_mean_squared_error', n_jobs=n_cpus
        )
        rfr_gridsearch.fit(X_train, y_train)
        y_pred = rfr_gridsearch.predict(X_test)
        # rfr = RandomForestRegressor()
        # rfr.fit(X_train, y_train)
        # y_pred = rfr.predict(X_test)
        return y_pred

    @rolling
    def grad_boost_reg(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        dtr_gridsearch = GridSearchCV(
            GradientBoostingRegressor(random_state=self.random_state),
            verbose=self.verbose,
            param_grid={
                "learning_rate": [0.1, 0.2, 0.3],
                "min_samples_split": [2, 4, 8],
                'n_estimators': [200],
                'max_depth': [2, 3, 5, 10]
            },
            scoring='neg_mean_squared_error', n_jobs=n_cpus
        )
        dtr_gridsearch.fit(X_train, y_train)
        y_pred = dtr_gridsearch.predict(X_test)
        # gbr = GradientBoostingRegressor(n_estimators=200,
        #                                 min_samples_split=4)
        # gbr.fit(X_train, y_train)
        # y_pred = gbr.predict(X_test)
        return y_pred

    @rolling
    def elastic_net(self, train_test, n_features=np.inf, method=None):
        X_train, X_test, y_train, y_test = train_test
        linear_regressor = ElasticNet()
        linear_regressor.fit(X_train, y_train)
        y_pred = linear_regressor.predict(X_test)
        return y_pred

    def arima_pipe(self, train_test, n_features=0):
        if n_features > 50:
            n_features = 0
        X_train, X_test, y_train, y_test = train_test
        d_ = 0
        if n_features > 0:
            arima_train = pm.auto_arima(y_train, exogenous=X_train, d=d_,
                                        seasonal=False, with_intercept=True,
                                        information_criterion='bic',
                                        trace=False, suppress_warnings=True,
                                        stepwise=False, error_action='ignore')
            y_pred = arima_train.predict(n_periods=1, exogenous=X_test)
        else:
            arima_train = pm.auto_arima(y_train, d=d_,
                                        seasonal=False, with_intercept=True,
                                        information_criterion='bic',
                                        trace=False, suppress_warnings=True,
                                        stepwise=False, error_action='ignore')
            y_pred = arima_train.predict(n_periods=1)
        residual = arima_train.resid()
        return y_pred, residual

    @rolling
    def pipeline(self, train_test, n_features=0, method=None):
        X_train, X_test, y_train, y_test = train_test

        # lin_reg = LinearRegression()
        # lin_reg.fit(X_train, y_train)
        # y_pred_train = lin_reg.predict(X_train)

        y_pred, residual = self.arima_pipe(train_test, n_features=n_features)

        # residual = y_train - y_pred_train
        residual = np.expand_dims(residual, axis=-1)

        rfr = RandomForestRegressor()
        rfr.fit(X_train, residual.flatten())

        # y_pred = lin_reg.predict(X_test)
        residual_pred = rfr.predict(X_test)
        return y_pred + residual_pred

    @rolling
    def pipeline_grid(self, train_test, n_features=0, method=None):
        X_train, X_test, y_train, y_test = train_test
        y_pred, residual = self.arima_pipe(train_test, n_features=n_features)

        residual = np.expand_dims(residual, axis=-1)

        rfr_gridsearch = GridSearchCV(
            RandomForestRegressor(random_state=self.random_state),
            verbose=self.verbose, param_grid={
                'n_estimators': [200],
                "max_depth": [2, 5, 10],
                "min_samples_split": [2, 4, 8],
                "min_samples_leaf": [1, 2, 4]
            },
            scoring='neg_mean_squared_error', n_jobs=n_cpus
        )
        rfr_gridsearch.fit(X_train, residual.flatten())

        # y_pred = lin_reg.predict(X_test)
        residual_pred = rfr_gridsearch.predict(X_test)
        return y_pred + residual_pred

    @rolling
    def arima(self, train_test, n_features=0, method=None):
        if n_features > 50:
            n_features = 0
        X_train, X_test, y_train, y_test = train_test
        d_ = 0
        if n_features > 0:
            arima_train = pm.auto_arima(y_train, exogenous=X_train, d=d_,
                                        seasonal=False, with_intercept=True,
                                        information_criterion='bic',
                                        trace=False, suppress_warnings=True,
                                        stepwise=False, error_action='ignore')
            y_pred = arima_train.predict(n_periods=1, exogenous=X_test)
        else:
            arima_train = pm.auto_arima(y_train, d=d_,
                                        seasonal=False, with_intercept=True,
                                        information_criterion='bic',
                                        trace=False, suppress_warnings=True,
                                        stepwise=False, error_action='ignore')
            y_pred = arima_train.predict(n_periods=1)
        return y_pred
