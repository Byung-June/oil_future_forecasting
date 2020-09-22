import pandas as pd
import numpy as np
import glob
from sklearn.metrics import r2_score
import os


def r2_oos_func(data_):
    data_ = pd.read_csv(data_, index_col=1)
    data_ = data_.drop(columns=['Unnamed: 0'])
    data_['3'] = np.abs(data_['1'] - data_['2'])
    # print('percentile', data, np.percentile(data_['3'], 99.9), np.percentile(data_['3'], 0.1))
    # data_ = data_[data_['3'] < np.percentile(data_['3'], 99.9)]
    y_pred = data_.iloc[:, 0]
    y_test = data_.iloc[:, 1]
    r2_oos_test = r2_score(y_test, y_pred)

    data_true = pd.read_csv('../data_preprocessing/vol_ml_data_M.csv', index_col=0)
    # print(data_true)
    y_true = data_true[['y_test']].loc[y_pred.index]
    # print('y_true', y_true)
    r2_oos_true = r2_score(y_true, y_pred)
    return r2_oos_test, r2_oos_true


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


if __name__ == "__main__":
    path = os.getcwd()
    file = '\*.csv'
    data_list = glob.glob(path + file)
    for data in data_list:
        print(data, r2_oos_func(data))
    # r2_oos_ml(path=path, file=file)
