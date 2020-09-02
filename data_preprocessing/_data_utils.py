from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from oil_forecastor.model_selection._utility import adf_test


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


def df_slicing(df_, start, end):
    df_ = df_[start:end]
    df_ = df_.replace(['NA', 'NaN'], np.nan)
    df_ = df_.dropna(axis=1)
    return df_


def fill_bs_date(index, df_, method='ffill'):
    dummy = pd.DataFrame(data=np.zeros(len(index)), index=index, columns=['dummy'])
    dummy.index.name = 'date'
    df_ = pd.merge(df_, dummy, left_index=True, right_index=True, how='outer')
    df_ = df_.replace(['NA', 'NaN'], np.nan)
    if method == 'ffill':
        df_ = df_.fillna(method='ffill')
    elif method == 'zero':
        df_ = df_.fillna(0)
    else:
        pass
    df_ = df_.loc[index]
    df_ = df_.drop(columns=['dummy'])
    return df_


def future_rolling(df_, method='additive'):
    index_list = [1 if date.day > 25 else 0 for date in df_.index]
    index_list2 = np.zeros(len(index_list))
    for num in np.arange(1, len(index_list)):
        if index_list[num] == 0:
            try:
                if index_list[num+1] == 1:
                    index_list2[num-3] = 1
            except IndexError:
                pass

    rolling_date = [date_ for date_, index in zip(df_.index, index_list2) if index ==1]
    rolling_date = pd.to_datetime(rolling_date)
    df_index = pd.DataFrame(np.ones(len(rolling_date)), index=rolling_date, columns=['index'])
    df_ = pd.merge(df_, df_index, left_index=True, right_index=True, how='outer')
    print('rolling', df_index)

    if method == 'additive':
        df_['spread'] = 0
        for date_, i in zip(df_.index, np.arange(len(df_.index))):
            if df_.loc[date_, 'index'] == 1:
                df_.loc[date_, 'spread'] = df_.loc[date_, 'y_test'] - df_.loc[df_.index[i-1], 'y_test']
        df_['cum_spread'] = df_['spread'].cumsum()
        df_['y_test'] = df_['y_test'] - df_['cum_spread'] + df_.loc[df_.index[-1], 'cum_spread']

    elif method == 'productive':
        df_['spread'] = 1
        for date_, i in zip(df_.index, np.arange(len(df_.index))):
            if df_.loc[date_, 'index'] == 1:
                val = df_.loc[df_.index[i-1], 'y_test'] / df_.loc[df_.index[i-1], 'y_test_dummy']
                if val > 0:
                    df_.loc[date_, 'spread'] = val
                else:
                    df_.loc[date_, 'spread'] = 1

        df_['cum_spread'] = df_['spread'].cumprod()
        # print(df_[['index', 'spread', 'cum_spread']][df_['index'] == 1])
        df_['y_test'] = df_['y_test'] / df_['cum_spread'] * df_.loc[df_.index[-1], 'cum_spread']

    else:
        print('correct method needed')
    df_ = df_.drop(['y_test_dummy', 'index', 'spread', 'cum_spread'], axis=1)
    print('rol', df_)
    return df_


def make_float(df):
    df = df.replace(".", np.nan)
    df = df.astype(float)
    return df