import numpy as np
import datetime
from sklearn.tree import DecisionTreeRegressor
from pmdarima.arima.utils import ndiffs

__all__ = ['rolling_sample', 'get_features', 'adf_test']


def rolling_sample(data_, window_num, sample_num, time):
    """
    :param data_: Input data
    :param window_num: number of window in one sample
    :param sample_num: number of samples to get parameters for forecasting
    :param time: current time
        if type(time) == int:
             int in range(window_num + sample_num - 2, len(data_) - 1)
        if type(time) == str: %Y-%m-%d format -> converted to int in algorithm
        if type(time) == Datetime -> converted to int in algorithm
    :return:
        data_x_train_time: time X sample X (window X variable Dataframe)
        data_x_test_time: time X 1 X (window X variable Dataframe)
        data_y_train_time: time X sample X 1
        data_y_test_time: time X 1 X 1
    """
    data_y = data_.iloc[:, 0]
    data_x = data_.iloc[:, 1:]
    if type(time) == str:
        time = datetime.datetime.strptime(time, '%Y-%m-%d')
    if type(time) == datetime.datetime:
        time = data_y.index.get_loc(time)

    data_y_train = []
    data_x_train = []
    for sample in np.arange(sample_num):
        start = time - window_num - sample_num + sample + 2
        end = time - sample_num + sample + 2

        data_y_roll = data_y.iloc[end-1]
        data_x_roll = data_x.iloc[start:end, :]

        data_y_train.append(data_y_roll)
        data_x_train.append(data_x_roll)

    data_y_test = data_y.iloc[time + 1]
    data_x_test = data_x.iloc[time - window_num + 2: time + 2, :]

    return (
        data_x_train, data_x_test,
        data_y_train, data_y_test
    )


def get_features(X_train, y_train, n_features=20):
    ranking = np.zeros([X_train.shape[1], 2])
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    for i, value in enumerate(model.feature_importances_):
        ranking[i, 0] = i
        ranking[i, 1] = -value
    ranking = ranking[ranking[:, 1].argsort()]
    ranking[:, 1] = - ranking[:, 1]
    return ranking[:n_features, :], np.array(ranking[:n_features, 0], dtype=np.int)


def adf_test(data_):
    '''
    :param data_: data to test differencing order
    :return:
        n_adf: ADF test
        n_kpss: KPSS test
        n_pp: PP test
    '''
    n_adf = ndiffs(data_, test='adf')
    n_kpss = ndiffs(data_, test='kpss')
    n_pp = ndiffs(data_, test='pp')
    return n_adf, n_kpss, n_pp