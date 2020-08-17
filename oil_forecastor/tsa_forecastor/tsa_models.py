import numpy as np
import pandas as pd
import pmdarima as pm
from ..model_selection._utility import adf_test, get_features, flatten_x_train, flatten_x_test
from arch.univariate import arch_model
import datetime as dt


def arima(x_train, x_test, y_train, y_test, t_, forecast_period, feature_num):
    d_ = max(adf_test(y_train))
    x_train_pos = []
    if feature_num > 0:
        _, x_train_pos = get_features(flatten_x_train(x_train), y_train, n_features=feature_num)
        x_train = flatten_x_train(x_train)[:, x_train_pos]
        x_test = flatten_x_test(x_test)[:, x_train_pos]

        arima_train = pm.auto_arima(y_train, exogenous=x_train, d=d_,
                                    seasonal=False, with_intercept=True, information_criterion='bic', trace=False,
                                    suppress_warnings=True, stepwise=False, error_action='ignore')
    else:
        arima_train = pm.auto_arima(y_train, d=d_,
                                    seasonal=False, with_intercept=True, information_criterion='bic', trace=False,
                                    suppress_warnings=True, stepwise=False, error_action='ignore')
    params = arima_train.params()
    orders = arima_train.get_params()['order']
    # print('params: ', params, len(params), orders)

    pred, conf_int = arima_train.predict(n_periods=forecast_period, exogenous=x_test, return_conf_int=True)
    return [t_, pred[0], y_test, x_train_pos, params, orders, conf_int[0]]


def garch(x_train, x_test, y_train, y_test, t_, forecast_period, feature_num, model='arx-garch'):
    d_ = max(adf_test(y_train))
    x_train_pos = []
    if feature_num > 0:
        x_train = x_train.append(x_test)
        y_train = y_train.append(y_test)
        _, x_train_pos = get_features(flatten_x_train(x_train), y_train, n_features=feature_num)
        x_train = flatten_x_train(x_train).iloc[:, x_train_pos]
        x_test = flatten_x_test(x_test).iloc[:, x_train_pos]
        y_train = pd.DataFrame(y_train, index=x_train.index)

        if model == 'arx-gjr-garch':
            am = arch_model(y_train, x=x_train, mean='ARX', p=1, o=1, q=1)

        elif model == 'arx-tgarch':
            am = arch_model(y_train, x=x_train, mean='ARX', p=1, o=1, q=1, power=1)

        else:
            # 'arx-garch'
            am = arch_model(y_train, x=x_train, mean='ARX')

    else:
        if model == 'arx-gjr-garch':
            am = arch_model(y_train, mean='ARX', p=1, o=1, q=1)

        elif model == 'arx-tgarch':
            am = arch_model(y_train, mean='ARX', p=1, o=1, q=1, power=1)

        else:
            # 'arx-garch'
            am = arch_model(y_train, mean='ARX')

    res = am.fit(disp='off')
    # print(res.summary())
    forecast = res.forecast(horizon=forecast_period)
    # print('mean result', type(forecast.mean), forecast.mean)
    # print('variance', type(forecast.variance), forecast.variance)
    return [t_, forecast.mean, y_test, x_train_pos, forecast.variance]

