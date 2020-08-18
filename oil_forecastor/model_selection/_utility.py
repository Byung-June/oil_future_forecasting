import numpy as np
import pandas as pd
import datetime
from sklearn.tree import DecisionTreeRegressor
from pmdarima.arima.utils import ndiffs
from skimage.restoration import denoise_bilateral, denoise_wavelet
from matplotlib import pyplot as plt


__all__ = ['rolling_train_test_split',
           'get_features', 'adf_test',
           'flatten_x_train',
           'flatten_x_test',
           'denoising_func']


def rolling_train_test_split(data_, window_num, sample_num, time):
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
    if data_.columns[0] == 'date':
        data_ = data_.set_index('date')
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


def flatten_x_train(x_train_):
    return pd.DataFrame([x.to_numpy().flatten() for x in x_train_], index=[x.index[-1] for x in x_train_])


def flatten_x_test(x_test_):
    return pd.DataFrame([x_test_.to_numpy().flatten()], index=[x_test_.index[-1]])


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


def denoising_func(data_, filter):
    if filter == 'wavelet_db1':
        data_['y_pred'] = denoise_wavelet(np.array(data_['y_pred']), wavelet='db1', mode='soft', wavelet_levels=2,
                                          multichannel=True, rescale_sigma=True)
    elif filter == 'wavelet_db2':
        data_['y_pred'] = denoise_wavelet(np.array(data_['y_pred']), wavelet='db2', mode='soft', wavelet_levels=2,
                                          multichannel=True, rescale_sigma=True)
    elif filter == 'bilateral':
        data_['y_pred'] = np.exp(
            denoise_bilateral(
                np.array(
                    [x * np.log(1 + abs(x)) if x > 0 else - x * np.log(1 + abs(x)) for x in np.array(data_['y_pred'])]
                ),
                sigma_spatial=1000, multichannel=False)
        ) - 1

    elif filter == 'moving_average':
        data_['y_pred'] = data_.y_pred.rolling(window=5).mean()

    else:
        pass

    return data_


if __name__ == '__main__':

    idx = pd.date_range('2018-01-01', periods=7, freq='D')
    ts = pd.DataFrame(range(len(idx)), index=idx, columns=['value1'])
    ts2 = pd.DataFrame(range(1, len(idx) + 1), index=idx, columns=['value2'])
    ts3 = pd.DataFrame(range(2, len(idx) + 2), index=idx, columns=['value3'])
    ts = pd.merge(ts, ts2, left_index=True, right_index=True)
    ts = pd.merge(ts, ts3, left_index=True, right_index=True)
    print(ts)
    data_x_train, data_x_test, data_y_train, data_y_test = rolling_train_test_split(ts, 2, 4, '2018-01-05')
    print('x_train', type(data_x_train), data_x_train, data_x_train.append(data_x_test))
    print('x_train2', type(data_x_train[-1]), data_x_train[-1], data_x_train[-1].index[-1])
    print('x_test', type(data_x_test), data_x_test)
    print('y_train', type(data_y_train), data_y_train, data_y_train.append(data_y_test))
    print('y_test', type(data_y_test), data_y_test)
    print('done!')

    print('xtrain flatten', flatten_x_train(data_x_train))
    a, b = get_features(flatten_x_train(data_x_train), data_y_train, n_features=2)
    print('features', a, b, flatten_x_train(data_x_train).iloc[:, b])
    a, b = get_features(flatten_x_train(data_x_train), data_y_train, n_features=3)
    print('features', a, b, flatten_x_train(data_x_train).iloc[:, b])

    # data = pd.read_csv('data_input_normalized_stationary.csv', index_col=0)
    # data = data.iloc[:, [0]]
    # print(data)
    #
    # denoised_wavelet = denoise_wavelet(np.array(data), wavelet='db2', mode='soft', wavelet_levels=2, multichannel=True,
    #                                    rescale_sigma=True)
    # print(denoised_wavelet)
    #
    # denoised_wavelet2 = denoise_wavelet(np.array(data), wavelet='db1', mode='soft', wavelet_levels=2, multichannel=True,
    #                                     rescale_sigma=True)
    # print(denoised_wavelet2)
    #
    # log_data = np.array([x * np.log(1 + abs(x)) if x > 0 else - x * np.log(1 + abs(x)) for x in np.array(data)])
    # print(log_data)
    # denoised_bilateral = np.exp(denoise_bilateral(log_data, sigma_spatial=1000, multichannel=False)) - 1
    # print(denoised_bilateral)
    #
    # denoised_ma = data.y_pred.rolling(window=5).mean()
    #
    # # plt.plot(data[:1000])
    # # plt.plot(denoised_wavelet[:1000])
    # # plt.plot(denoised_wavelet2[:1000])
    # # plt.plot(denoised_bilateral[:1000])
    # # plt.plot(denoised_ma[:1000])
    # # plt.show()
    #
    # data['wavelet'] = denoised_wavelet
    # data['wavelet2'] = denoised_wavelet2
    # data['bilateral'] = denoised_bilateral
    # data['moving_average'] = denoised_ma
    #
    # data['price'] = data['y_pred'] + 1
    # data['price'] = data['price'].cumprod()
    # data['price_wavelet'] = data['wavelet'] + 1
    # data['price_wavelet'] = data['price_wavelet'].cumprod()
    # data['price_wavelet2'] = data['wavelet2'] + 1
    # data['price_wavelet2'] = data['price_wavelet2'].cumprod()
    # data['price_bilateral'] = data['bilateral'] + 1
    # data['price_bilateral'] = data['price_bilateral'].cumprod()
    # data['price_moving_average'] = data['moving_average'] + 1
    # data['price_moving_average'] = data['price_moving_average'].cumprod()
    #
    # plt.plot(data['price'][:1000])
    # plt.plot(data['price_wavelet'][:1000])
    # plt.plot(data['price_wavelet2'][:1000])
    # plt.plot(data['price_bilateral'][:1000])
    # plt.plot(data['price_moving_average'][:1000])
    # plt.show()
    #
    #
    # print(data)
    # data.to_csv('denoise.csv')