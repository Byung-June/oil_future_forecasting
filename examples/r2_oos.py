import pandas as pd
import numpy as np
import glob
from sklearn.metrics import r2_score
from scipy.stats import t
from statsmodels.tsa.stattools import acovf
import os


def r2_oos_func(data_, type='tsa'):
    if type=='tsa':
        data_ = pd.read_csv(data_, index_col=1)
        data_ = data_.drop(columns=['Unnamed: 0'])
        data_ = data_.dropna(axis=0)
    else:
        # type=='ml'
        data_ = pd.read_csv(data_, index_col=0)
        data_ = data_.dropna(axis=0)

    y_pred = data_.iloc[:, 0]
    y_test = data_.iloc[:, 1]
    r2_oos = [r2_score(y_test, y_pred)]

    for n_dates in [120, 60, 36, 24, 12]:
        y_pred = data_.iloc[-n_dates:, 0]
        y_test = data_.iloc[-n_dates:, 1]
        r2_oos.append(r2_score(y_test, y_pred))

    # data_true = pd.read_csv('../data_preprocessing/vol_ml_data_M.csv', index_col=0)
    # y_true = data_true[['y_test']].loc[y_pred.index]
    # r2_oos_true = r2_score(y_true, y_pred)
    return r2_oos


def r2_oos_ml(path='../results'):
    names = glob.glob(path + '/*.csv')
    dict_ = dict()
    for name in names:
        df = pd.read_csv(name)
        df = df.set_index('date')
        df = df.dropna()
        name_wo_csv = name.replace('.csv', '')
        for n_dates in [255, 255 * 3, 255 * 5, 255 * 10, 10 ** 32]:
            df_ladder = df.iloc[-n_dates:, :]
            r_square, r_square_filtered\
                = evaluation(df_ladder, delete_outlier=True)
            print(r_square)
            if n_dates > 100000:
                n_dates = 100000
            dict_[name_wo_csv + '_' + str(n_dates)] = r_square
            dict_[name_wo_csv + '_zero_' + str(n_dates)]\
                = r_square_filtered
    print(dict_)
    df_r2 = pd.DataFrame(dict_, index=[0])
    print(df_r2)
    df_r2.to_csv('r_2.csv')


def evaluation(df, y_true, delete_outlier=False):
    if y_true is not None:
        df = pd.concat([df, y_true], axis=1)
    df = df.dropna()

    y_pred = df['y_pred'].values.flatten()
    y_test = df['y_test'].values.flatten()

    if delete_outlier:
        y_err = np.abs(y_pred - y_test)
        y_max_err = np.percentile(y_err, 99.9)

        y_pred = y_pred[y_err < y_max_err]
        y_test = y_test[y_err < y_max_err]

    r2_true = 0.0
    if y_true is not None:
        y_true = df['y_true'].values.flatten()
        r2_true = r2_score(y_true, y_pred)

    return r2_score(y_test, y_pred), r2_true


def db_test(true, pred1, pred2, h, err_type='MSE'):
    assert (len(true) == len(pred1)) and (len(true) == len(pred2)), print('check length')
    true = np.array(true)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)

    err1 = true - pred1
    err2 = true - pred2

    if err_type == 'MSE':
        d = np.power(err1, 2) - np.power(err2, 2)
    elif err_type == 'MAD':
        d = np.abs(err1) - np.abs(err2)
    elif err_type == 'MAPE':
        d = np.abs(np.divide(err1, true)) - np.abs(np.divide(err2, true))
    else:
        d = []
        raise TypeError

    def autocovariance(dd, length, lag):
        return np.sum([((dd[i + lag]) - np.mean(dd)) * (dd[i] - np.mean(dd))
                       for i in np.arange(0, length-lag)]) \
               / float(length)

    T = float(len(d))
    gamma = [autocovariance(d, len(d), lag) for lag in range(0, h)]
    V_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    harvey_adj = ((T + 1 - 2 * h + h * (h - 1) / T) / T) ** (0.5)
    DM_stat = harvey_adj * (V_d ** (-0.5) * np.mean(d))
    p_value = 2 * t.cdf(-abs(DM_stat), df=T - 1)
    return DM_stat, p_value


if __name__ == "__main__":
    path_tsa = os.getcwd()
    path_ml = 'D:\Dropbox/6_git_repository\oil_future_forecasting/results/vol_ml_data_M'
    file = '\*.csv'

    data_list_tsa = glob.glob(path_tsa + file)
    for data in data_list_tsa:
        print(data)
        print(r2_oos_func(data))

    data_list_ml = glob.glob(path_ml + file)
    for data in data_list_ml:
        print(data)
        print(r2_oos_func(data, type='ml'))

