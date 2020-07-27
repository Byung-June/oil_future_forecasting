import numpy as np

__all__ = ['rolling_sample']


def rolling_sample(data_, window_num, sample_num):
    """
    :param data_: Input data
    :param window_num: number of window in one sample
    :param sample_num: number of samples to get parameters for forecasting
    :return:
        data_x_train_time: time X sample X (window X variable Dataframe)
        data_x_test_time: time X 1 X (window X variable Dataframe)
        data_y_train_time: time X sample X 1
        data_y_test_time: time X 1 X 1
    """
    data_y = data_.iloc[:, 0]
    data_x = data_.iloc[:, 1:]

    total_sample_num = len(data_y) - 1
    data_x_train_time = []
    data_x_test_time = []
    data_y_train_time = []
    data_y_test_time = []
    for time in np.arange(window_num + sample_num - 2, total_sample_num):
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

        data_x_train_time.append(data_x_train)
        data_x_test_time.append(data_x_test)
        data_y_train_time.append(data_y_train)
        data_y_test_time.append(data_y_test)

    return (
        data_x_train_time, data_x_test_time,
        data_y_train_time, data_y_test_time
    )
