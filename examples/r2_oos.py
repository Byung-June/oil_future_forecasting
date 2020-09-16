import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import r2_score


def r2_oos_func(data_):
    data_ = pd.read_csv(data_, index_col=1)
    data_ = data_.drop(columns=['Unnamed: 0'])
    y_pred = data_.iloc[:, 0]
    y_test = data_.iloc[:, 1]
    r2_oos = r2_score(y_test, y_pred)
    return r2_oos


def ladder_test(y_test, y_pred, n_dates):
    assert n_dates > 0
    n_dates = - n_dates
    y_test = y_test[n_dates:].flatten()
    y_pred = y_pred[n_dates:].flatten()
    assert y_pred.shape == y_test.shape
    r_square = r2_score(y_test, y_pred)
    return r_square


def r2_oos_ml(path='../results', file='/*.npz'):
    names = glob.glob(path + file)
    y_test_names = [name for name in names if 'y_test' in name]
    y_pred_names = [name for name in names if 'y_test' not in name]

    dict_ = dict()
    for y_pred_name in y_pred_names:
        substring_pred_idx = y_pred_name.find('windows_')
        substring_pred = y_pred_name[substring_pred_idx:]
        y_test_name = None
        for elt in y_test_names:
            if substring_pred in elt:
                y_test_name = elt
        assert y_test_name is not None
        y_pred = np.load(y_pred_name)['arr_0']
        y_test = np.load(y_test_name)['arr_0']

        for n_dates in [255, 255 * 3, 255 * 5, 255 * 10, 10 ** 32]:
            r_square = ladder_test(y_test, y_pred, n_dates)
            name_ = (y_pred_name + '_date_' + str(n_dates) + '_r2').replace(
                        '.npz', '')
            dict_[name_] = r_square
    df = pd.DataFrame(dict_, index=[0])
    df.to_csv('r_2.csv')


if __name__ == "__main__":
    path = os.getcwd()
    file = '\*.csv'
    data_list = glob.glob(path + file)
    for data in data_list:
        df = pd.read_csv(data)
        print(data, r2_oos_func(data))
    # r2_oos_ml(path=path, file=file)

