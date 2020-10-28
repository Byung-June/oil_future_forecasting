import numpy as np
import pandas as pd
import pmdarima as pm
from ..model_selection._utility import adf_test, get_features, flatten_x_train, flatten_x_test
from ..feature_selection._feature_selector import selector
from arch.univariate import arch_model
import datetime as dt


def arima(x_train, x_test, y_train, y_test, t_, forecast_period, feature_num, fix_feature_num=0):
    # d_ = max(adf_test(y_train))
    d_ = 0
    x_train_pos = []
    if feature_num > 0:
        if fix_feature_num == 0:
            x_train = flatten_x_train(x_train)
            x_test = flatten_x_test(x_test)
            x_train, x_test, y_train, y_test = selector(
                x_train,
                x_test,
                y_train,
                y_test,
                feature_num
            )
        else:
            fix_x_train = flatten_x_train([tab.iloc[:, :fix_feature_num] for tab in x_train])
            fix_x_test = flatten_x_test(x_test.iloc[:, :fix_feature_num])
            x_train = flatten_x_train([tab.iloc[:, fix_feature_num:] for tab in x_train])
            x_test = flatten_x_test(x_test.iloc[:, fix_feature_num:])
            x_train, x_test, y_train, y_test = selector(
                x_train,
                x_test,
                y_train,
                y_test,
                feature_num - fix_feature_num
            )
            x_train = [np.append(x, y) for x, y in zip(fix_x_train.to_numpy(), x_train)]
            x_test = [np.append(fix_x_test.to_numpy(), x_test)]

        arima_train = pm.auto_arima(y_train, exogenous=x_train, d=d_,
                                    seasonal=False, with_intercept=True, information_criterion='bic', trace=False,
                                    suppress_warnings=True, stepwise=False, error_action='ignore')
    else:
        arima_train = pm.auto_arima(y_train, d=d_,
                                    seasonal=False, with_intercept=True, information_criterion='bic', trace=False,
                                    suppress_warnings=True, stepwise=False, error_action='ignore')
    params = arima_train.params()
    orders = arima_train.get_params()['order']
    pred, conf_int = arima_train.predict(n_periods=forecast_period, exogenous=x_test, return_conf_int=True)
    return [t_, pred[0], y_test, x_train_pos, params, orders, conf_int[0]]