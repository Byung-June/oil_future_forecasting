import numpy as np
import pmdarima as pm
from ..model_selection._utility import adf_test


def arima(x_train, x_test, y_train, y_test, t_, forecast_period, feature_num):
    d_ = max(adf_test(y_train))
    x_train = np.array([x.to_numpy().flatten() for x in x_train])
    x_train = x_train[:, 0:feature_num]

    x_test = np.array([x_test.to_numpy().flatten()])
    x_test = x_test[:, 0:feature_num]
    # print('xtest', x_test, x_test.shape)

    arima_train = pm.auto_arima(y_train, exogenous=x_train, d=d_,
                                seasonal=False, with_intercept=True, information_criterion='aicc', trace=False,
                                suppress_warnings=True, stepwise=False, error_action='ignore')
    params = arima_train.params()
    orders = arima_train.get_params()['order']
    # print('params: ', params, len(params), orders)

    # print('arima result', arima_train.summary())
    pred, conf_int = arima_train.predict(n_periods=forecast_period, exogenous=x_test, return_conf_int=True)
    # print('result:', pred, conf_int, y_test, params)
    return [t_, pred[0], y_test, params, orders, conf_int[0]]


def garch(x_train, x_test, y_train, y_test, t_, forecast_period, feature_num):
