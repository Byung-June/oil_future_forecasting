import numpy as np
import pandas as pd
import datetime


__all__ = ['rolling_sample']


def rolling_sample(data_, window_num, sample_num, time):
    """
    :param data_: Input data
    :param window_num: number of window in one sample
    :param sample_num: number of samples to get parameters for forecasting
    :param time: current time
        if type(time) == int: int in range(window_num + sample_num - 2, len(data_) - 1)
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


if __name__=='__main__':

    idx = pd.date_range('2018-01-01', periods=7, freq='D')
    ts = pd.DataFrame(range(len(idx)), index=idx, columns=['value1'])
    ts2 = pd.DataFrame(range(1, len(idx) + 1), index=idx, columns=['value2'])
    ts3 = pd.DataFrame(range(2, len(idx) + 2), index=idx, columns=['value3'])
    ts = pd.merge(ts, ts2, left_index=True, right_index=True)
    ts = pd.merge(ts, ts3, left_index=True, right_index=True)
    print(ts)
    data_x_train, data_x_test, data_y_train, data_y_test = rolling_sample(ts, 2, 4, '2018-01-05')
    print('x_train', data_x_train)
    print('x_test', data_x_test)
    print('y_train', data_y_train)
    print('y_test', data_y_test)
    print('done!')