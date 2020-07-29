import numpy as np
from ..model_selection import rolling_sample
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


def rolling(func):
    def train_model_wrapper(self):
        y_list = []
        for time_idx in self.data['date'].iloc[self.start_time]:
            train_test = self._data_helper(time_idx)
            y_pred_test = func(self, train_test)
            y_list.append(y_pred_test)
        y_list = np.array(y_list)
        return y_list
    return rolling


class MLForecast():
    def __init__(self, data, n_windows, n_samples):
        self.data = data
        self.start_time = n_windows + n_samples

    def _data_helper(self, time_idx):
        data_tuple = rolling_sample(
                self.data, self.n_windows, self.n_samples, time_idx
            )
        train_test_split = [
            np.concatenate([elt.values for elt in data_list], axis=-1)
            for data_list in data_tuple
        ]
        return train_test_split

    @rolling
    def linear_reg(self, train_test):
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

    def hist_grad_boost_reg(self, train_test):
        X_train, X_test, y_train, y_test = train_test
        hdtr_gridsearch = GridSearchCV(
            HistGradientBoostingRegressor(),
            verbose=2,
            param_grid={
                'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                'max_depth': [3, 4, 5]
            },
            scoring='r2', n_jobs=4
        )
        hdtr_gridsearch.fit(X_train, y_train)
        y_pred = hdtr_gridsearch.predict(X_test)
        return y_pred, y_test

    def pcr(self):
        raise NotImplementedError("PCR is not implemented.")

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
        raise NotImplementedError("SVR is not implemented.")
