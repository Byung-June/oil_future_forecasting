import pandas as pd
import numpy as np
import glob
from sklearn.metrics import r2_score
from scipy.stats import t, norm
from statsmodels.tsa.stattools import adfuller
import os
import re


def dm_test(true, pred1, pred2, h, err_type='MSE'):
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
        raise TypeError

    def auto_covariance(dd, length, lag):
        return np.sum([((dd[t]) - np.mean(dd)) * (dd[t - lag] - np.mean(dd))
                       for t in np.arange(lag, length)]) \
               / float(length)

    T = float(len(d))
    gamma = [auto_covariance(d, len(d), lag) for lag in range(0, h)]
    V_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    harvey_adj = ((T + 1 - 2 * h + h * (h - 1) / T) / T) ** 0.5
    DM_stat = harvey_adj * (V_d ** (-0.5) * np.mean(d))
    p_value = 2 * t.cdf(-abs(DM_stat), df=T - 1)
    return DM_stat, p_value


def pt_test(true, pred):
    assert (len(true) == len(pred)), print('check length')
    true = np.where(true > 0, 1, -1)
    pred = np.where(pred > 0, 1, -1)
    n = len(true)

    def p_hat(y, x):
        return np.sum(np.where(np.multiply(y, x) > 0, 1, 0)) / n

    P_hat = p_hat(true, pred)
    P_hat_y = p_hat(true, 1)
    P_hat_x = p_hat(pred, 1)
    P_hat_star = P_hat_y * P_hat_x + (1 - P_hat_y) * (1 - P_hat_x)
    V_hat = P_hat_star * (1 - P_hat_star) / n
    V_hat_star = (np.power(2 * P_hat_y - 1, 2) * P_hat_x * (1 - P_hat_x) / n) \
                 + (np.power(2 * P_hat_x - 1, 2) * P_hat_y * (1 - P_hat_y) / n) \
                 + 4 / np.power(n, 2) * P_hat_y * P_hat_x * (1 - P_hat_y) * (1 - P_hat_x)

    pt_stat = (P_hat - P_hat_star) / np.sqrt(V_hat - V_hat_star)
    p_value = 1 - norm.cdf(pt_stat)
    return P_hat, p_value


def r2_oos_func(data_, type='ARIMA', test_type='ladder'):
    if type=='ARIMA':
        data_ = pd.read_csv(data_, index_col=1)
        data_ = data_.drop(columns=['Unnamed: 0'])
        data_ = data_.dropna(axis=0)
    else:
        # type=='ML'
        data_ = pd.read_csv(data_, index_col=0)
        data_ = data_.dropna(axis=0)

    y_pred = data_.iloc[:, 0]
    y_test = data_.iloc[:, 1]
    r2_oos = [r2_score(y_test, y_pred)]
    unit_root_test = [adfuller(y_pred - y_test)[1]]

    # only for w=5, s=5

    bench = 'D:\Dropbox/6_git_repository\oil_future_forecasting/results\logvol_results\logvol_ml_data_W' \
            + '/res_linear_reg_windows_1_samples_52_scaler_none_features_0_none_without_epu_None.csv'
    bench = pd.read_csv(bench, index_col=0).dropna(axis=0)
    bench = bench.iloc[:, 0]
    # assert (len(y_test) == len(bench)), print('check benchmark file time')
    d_result = dm_test(y_test[-len(bench):], bench, y_pred[-len(bench):], h=1, err_type='MSE')[1]

    y_t = y_test.shift(1)
    pt_y = y_test - y_t
    pt_x = y_pred - y_t
    pt_result = pt_test(pt_y, pt_x)
    p_hat = [pt_result[0]]
    pt_p_value = [pt_result[1]]
    time = []
    step = 5
    T = len(y_test) // step


    # Time Consistency Test
    if test_type=='step':

        for n_dates in np.arange(1, step + 1)[::-1]:
            start = - T * n_dates
            end = - T * (n_dates - 1)
            if end != 0:
                y_pred = data_.iloc[start:end, 0]
                y_test = data_.iloc[start:end, 1]
                r2_oos.append(r2_score(y_test, y_pred))
                unit_root_test.append(adfuller(y_pred - y_test)[1])

                y_test = pt_y.iloc[start:end]
                y_pred = pt_x.iloc[start:end]
                pt_result = pt_test(y_test, y_pred)
                p_hat.append(pt_result[0])
                pt_p_value.append(pt_result[1])
                time.append(y_test.index[0])
            else:
                y_pred = data_.iloc[start:, 0]
                y_test = data_.iloc[start:, 1]
                r2_oos.append(r2_score(y_test, y_pred))
                unit_root_test.append(adfuller(y_pred - y_test)[1])

                y_test = pt_y.iloc[start:]
                y_pred = pt_x.iloc[start:]
                pt_result = pt_test(y_test, y_pred)
                p_hat.append(pt_result[0])
                pt_p_value.append(pt_result[1])
                time.append(y_test.index[0])

    if test_type == 'ladder':
        for n_dates in (np.array([10, 5, 3, 2, 1]) * 52):
            y_pred = data_.iloc[-n_dates:, 0]
            y_test = data_.iloc[-n_dates:, 1]
            r2_oos.append(r2_score(y_test, y_pred))
            unit_root_test.append(adfuller(y_pred - y_test)[1])

            y_test = pt_y.iloc[-n_dates:]
            y_pred = pt_x.iloc[-n_dates:]
            pt_result = pt_test(y_test, y_pred)
            p_hat.append(pt_result[0])
            pt_p_value.append(pt_result[1])
            time.append(y_test.index[0])
    unit_root_test = [(np.round(x * 10**-np.floor(np.log10(x)), 3), np.floor(np.log10(x))) for x in unit_root_test]
    r2_result = [str(x) + ' (' + str(y[0]) + 'e' + str(y[1]) + ')' for x, y in zip(np.round(r2_oos, 3), unit_root_test)]
    pt_result = [str(x) + ' (' + str(y) + ')' for x, y in zip(np.round(p_hat, 3), np.round(pt_p_value, 3))]

    return (
        time,
        r2_result,
        pt_result,
        np.round(d_result, 3)
    )


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

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return r2_score(y_test, y_pred), r2_true, mape


def result_table(data_list, test_type, data_type='ARIMA'):
    table = []
    for data in data_list:
        text = re.split(r'\\', data)[-1]
        text = re.split('.cs', text)[0]
        text = re.split('_', text)
        if data_type == 'ARIMA':
            model = text[1]
            if 'epu' in text: model = model + '_without_epu'
            w, s, f = text[-3:]
            if int(f) > 0: f = int(f)-2
            else: f = f + '^*'
        elif data_type == 'ML':
            model = text[1]
            b = []
            for x in text:
                try:
                    b.append(int(x))
                except ValueError:
                    pass
            if len(b) == 2:
                f = 'all'
            else:
                f = b[2]
        else:
            raise ValueError

        _, r2_result, pt_result, dm_result = r2_oos_func(data, type=data_type, test_type=test_type)
        row = [model, f]
        row.extend(r2_result)
        row.extend([pt_result[0], dm_result])
        table.append(row)
    table = pd.DataFrame(table, columns=['Model', 'Number of Feature Selection',
                                         '$R^2_{oos}$', '$R^2_{oos, 10}$', '$R^2_{oos, 5}$', '$R^2_{oos, 3}$',
                                         '$R^2_{oos, 2}$', '$R^2_{oos, 1}$', 'PT test', 'DM test'])
    print(table)
    return table

if __name__ == "__main__":
    path_tsa = os.getcwd()
    file = '\*.csv'
    test_type = 'ladder'
    # test_type = 'step'

    # print('--------------------------------------')
    # print('ARIMAX')
    # print('--------------------------------------')
    # data_list_tsa = glob.glob(path_tsa + file)
    # arima_result = result_table(data_list_tsa, test_type=test_type, data_type='ARIMA')
    # arima_result.to_csv('result_table//arima_result.csv')

    print('--------------------------------------')
    print('Machine Learning')
    print('--------------------------------------')

    # path_ml = 'C:/Users/junelap/Dropbox/6_git_repository/oil_future_forecasting/results/vol_ml_data_M'
    path_ml = 'D:\Dropbox/6_git_repository\oil_future_forecasting/results\logvol_results\logvol_ml_data_no_Q_W'
    data_list_ml = glob.glob(path_ml + file)
    ml_result = result_table(data_list_ml, test_type=test_type, data_type='ML')
    ml_result.to_csv('result_table//ml_result.csv')

    print('--------------------------------------')
    print('Machine Learning without EPU')
    print('--------------------------------------')

    # path_ml = 'C:/Users/junelap/Dropbox/6_git_repository/oil_future_forecasting/results/vol_ml_data_M_no_epu'
    path_ml = 'D:\Dropbox/6_git_repository\oil_future_forecasting/results/results\logvol_ml_data_W_no_epu_no_Q'
    data_list_ml = glob.glob(path_ml + file)
    ml_result_without_epu = result_table(data_list_ml, test_type=test_type, data_type='ML')
    ml_result_without_epu.to_csv('result_table//ml_without_epu_result.csv')
