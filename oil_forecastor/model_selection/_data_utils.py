from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from ..model_selection._utility import adf_test


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
    concat = [make_lag(df, num_lags)]

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


def scaler_with_nan(dataframe):
    for name in dataframe.columns:
        data = dataframe[name].values
        not_inf = ~np.isinf(data)
        scaled = np.full_like(data, np.nan)
        data[not_inf] = data[not_inf] - data[not_inf].mean()

        assert np.isnan(data[not_inf].mean()) == False
        assert np.isinf(data[not_inf].mean()) == False

        scaled_data = scale(data[not_inf])
        scaled[not_inf] = scaled_data
        scaled[data == np.inf] = np.absolute(scaled_data).max() * 1.5
        scaled[data == -np.inf] = -np.absolute(scaled_data).max() * 1.5
        dataframe[name] = scaled
    return dataframe


def make_train_test(path, scale=False):
    df = pd.read_csv(path)
    df.columns = [name.lower() for name in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.crude_future = df.crude_future
    df = df.rename(columns={'crude_future': 'pred'})
    df = df.dropna()

    if scale:
        scaled_df = scaler_with_nan(df)
    else:
        scaled_df = df
    y_ = scaled_df['pred'].values
    X_ = scaled_df.drop('pred', axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(
        X_, y_,
        test_size=0.33, random_state=1
    )

    return X_train, X_test, y_train, y_test


def stationary_df(df_):
    col_stationary_index = np.zeros(len(df_.columns))
    col_stationary_diff = np.zeros(len(df_.columns))

    for col_num in np.arange(len(df_.columns)):
        col = df_.iloc[:, col_num]
        # print(col)
        try:
            col_diff = adf_test(col)
            col_diff = max(col_diff)
            if col_diff > 0:
                col_stationary_index[col_num] = 1
                col_stationary_diff[col_num] = col_diff

        except ValueError:
            col_stationary_index[col_num] = 2

    return col_stationary_index, col_stationary_diff


def make_stationary(df_, col_stationary_index_, col_stationary_diff_):
    for col_num in np.arange(1, len(df_.columns)):
        col_index = col_stationary_index_[col_num]
        col_diff = col_stationary_diff_[col_num]

        if col_index == 0:
            if col_diff > 0:
                print('%s th column is wrong' % str(col_num))
        elif col_index == 1:
            df_.iloc[:, col_num] = df_.iloc[:, col_num].diff(col_diff)
        else:
            print('%s th column is wrong with string type input' % str(col_num))
    return df_


if __name__ == "__main__":
    xls = pd.ExcelFile('petro_spot_price.xls')
    df1 = pd.read_excel(xls, 'Data 1')
    df1 = df1[2:]
    df1.columns = ["date", "wti", "brent"]
    df1["date"] = pd.to_datetime(df1["date"])
    df1.set_index(df1.columns[0], inplace=True)
    df = lagging(df1)
    print(df)
