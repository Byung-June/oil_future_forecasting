import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def add_name(list_, name):
    list_ = list(list_)
    for i, elt in enumerate(list_):
        list_[i] = elt + name
    return list_


def make_lag(df, num_lags=10, fillna=True):
    concat = []
    if fillna:
        df = df.fillna(method='ffill')
    for lag in range(num_lags):
        lagged_df = df.shift(lag)
        name = "_lag" + str(lag)
        columns = add_name(lagged_df.columns, name)
        lagged_df.columns = columns
        concat.append(lagged_df)
    concat = pd.concat(concat, axis=1)
    return concat


def make_diff_lag(df, num_diffs=1, num_lags=10, fillna=True):
    concat = []
    if fillna:
        df = df.fillna(method='ffill')
    for lag in range(num_lags):
        lagged_df = df.diff(num_diffs).shift(lag)
        name = '_lag' + str(lag) + '_diff' + str(num_diffs)
        columns = add_name(lagged_df.columns, name)
        lagged_df.columns = columns
        concat.append(lagged_df)
    concat = pd.concat(concat, axis=1)
    return concat


def make_pct_change_lag(df, num_diffs=1, num_lags=10, fillna=True):
    concat = []
    if fillna:
        df = df.fillna(method='ffill')
    for lag in range(num_lags):
        lagged_df = df.pct_change(num_diffs).shift(lag)
        name = '_lag' + str(lag) + '_pct_change' + str(num_diffs)
        columns = add_name(lagged_df.columns, name)
        lagged_df.columns = columns
        concat.append(lagged_df)
    concat = pd.concat(concat, axis=1)
    return concat


def make_moving_average(df, num_rolls=10, fillna=True):
    if fillna:
        df = df.fillna(method='ffill')
    name = '_ma' + str(num_rolls)
    columns = add_name(df.columns, name)
    df = df.rolling(window=num_rolls).mean()
    df.columns = columns
    return df


def make_momentum(df, num_rolls=10, fillna=True):
    if fillna:
        df = df.fillna(method='ffill')
    name = '_momentum' + str(num_rolls)
    columns = add_name(df.columns, name)
    df = df.pct_change() + 1.0
    df = df.rolling(window=num_rolls).apply(np.prod, raw=True) - 1.0
    df.columns = columns
    return df


def make_std(df, num_rolls=10, fillna=True):
    if fillna:
        df = df.fillna(method='ffill')
    name = '_std' + str(num_rolls)
    columns = add_name(df.columns, name)
    df = df.rolling(window=num_rolls).std()
    df.columns = columns
    return df


def make_skew(df, num_rolls=10, fillna=True):
    if fillna:
        df = df.fillna(method='ffill')
    name = '_skew' + str(num_rolls)
    columns = add_name(df.columns, name)
    df = df.rolling(window=num_rolls).skew()
    df.columns = columns
    return df


def make_kurt(df, num_rolls=10, fillna=True):
    if fillna:
        df = df.fillna(method='ffill')
    name = '_kurtosis' + str(num_rolls)
    columns = add_name(df.columns, name)
    df = df.rolling(window=num_rolls).kurt()
    df.columns = columns
    return df


def lagging(
    df,
    num_diff_list=[1, 2, 3], num_lags=10,
    num_rolls_list=[10, 20, 30], scrambled=False
):
    concat = []
    concat.append(make_lag(df, num_lags))

    for num_diff in num_diff_list:
        concat.append(
            make_diff_lag(
                df, num_diffs=num_diff, num_lags=num_lags
            )
        )
        concat.append(
            make_pct_change_lag(
                df, num_diffs=num_diff, num_lags=num_lags
            )
        )
    for num_rolls in num_rolls_list:
        concat.append(make_moving_average(df, num_rolls=num_rolls))
        concat.append(make_momentum(df, num_rolls=num_rolls))
        concat.append(make_std(df, num_rolls=num_rolls))
        concat.append(make_skew(df, num_rolls=num_rolls))
        concat.append(make_kurt(df, num_rolls=num_rolls))
    concat = pd.concat(concat, axis=1)
    if scrambled:
        concat = shuffle(concat, random_state=1)
    return concat


def get_r2_score(y_true, y_pred):
    assert y_pred.shape == y_true.shape
    numerator = (y_pred - y_true) ** 2
    denominator = y_true ** 2
    return 1 - (numerator.sum() / denominator.sum())


if __name__ == "__main__":
    xls = pd.ExcelFile('petro_spot_price.xls')
    df1 = pd.read_excel(xls, 'Data 1')
    df1 = df1[2:]
    df1.columns = ["date", "wti", "brent"]
    df1["date"] = pd.to_datetime(df1["date"])
    df1.set_index(df1.columns[0], inplace=True)
    df = lagging(df1)
    print(df)
