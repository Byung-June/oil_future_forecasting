import pandas as pd
import numpy as np
import glob
from sklearn.metrics import r2_score


def r2_oos_func(data_):
    data_ = pd.read_csv(data_, index_col=1)
    data_ = data_.drop(columns=['Unnamed: 0'])
    pred = data_.iloc[:, 0]
    y = data_.iloc[:, 1]
    r2_oos = 1 - sum(pow(pred-y, 2)) / sum(pow(y, 2))
    # print(sum(pow(pred-y, 2)))
    # print(sum(pow(y-data_.iloc[:, 1].mean(), 2)))
    return r2_oos, data_


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


def evaluation(df, delete_outlier=True):
    df = df.dropna()
    # y_test_no_prefilter = df['y_test'].values.flatten()
    y_pred_zero = df['y_pred_zero'].values.flatten()

    y_pred = df['y_pred'].values.flatten()
    if delete_outlier:
        mean = y_pred.mean()
        std = y_pred.std()
        y_pred = np.where(y_pred >= mean + 3 * std,
                          mean + 3 * std, y_pred)
        y_pred = np.where(y_pred <= mean - 3 * std,
                          mean - 3 * std, y_pred)
    y_true = df['y_true'].values.flatten()
    return r2_score(y_true, y_pred), r2_score(y_true, y_pred_zero)


if __name__ == '__main__':
    r2_oos_ml()
