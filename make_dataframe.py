import glob
from pandas.tseries.offsets import BDay, MonthEnd, QuarterEnd
import matplotlib.pylab as plt
from oil_forecastor.model_selection._data_utils import *


def set_date_as_index(df):
    df.columns = [name.lower() for name in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def make_float(df):
    df = df.replace(".", np.nan)
    df = df.astype(float)
    return df


def read_files(paths, fillna=True):
    csv_list_d_ = []
    csv_list_w_ = []
    csv_list_m_ = []
    csv_list_q_ = []
    xls_list_d_ = []
    xls_list_m_ = []
    xls_list_q_ = []

    for path in paths:
        csv_files = glob.glob(path + "/*.csv")
        xls_files = glob.glob(path + "/*.xls")

        for elt in csv_files:
            print(elt, type(elt))
            df = pd.read_csv(elt, index_col=0)
            df.index = pd.to_datetime(df.index)

            df = make_float(df)
            df = df.replace([0, 'NA', 'NaN'], np.nan)

            if elt[-5] == 'D':
                if fillna:
                    df = df.fillna(method='ffill')
                    df = df.loc[~df.index.duplicated(keep='first')]
                csv_list_d_.append(df)
            elif elt[-5] == 'W':
                df.index = df.index
                df = df.loc[~df.index.duplicated(keep='first')]
                csv_list_w_.append(df)
            elif elt[-5] == 'M':
                df.index = df.index + MonthEnd(0)
                df = df.loc[~df.index.duplicated(keep='first')]
                csv_list_m_.append(df)
            elif elt[-5] == "Q":
                df.index = df.index + QuarterEnd(0)
                df = df.loc[~df.index.duplicated(keep='first')]
                csv_list_q_.append(df)
            else:
                pass

        for elt in xls_files:
            try:
                df = pd.read_excel(elt, index_col=0)
                df.index = pd.to_datetime(df.index)
                df = make_float(df)
                df = df.replace([0, 'NA', 'NaN'], np.nan)

                if elt[-5] == 'D':
                    if fillna:
                        df = df.fillna(method='ffill')
                    df = df.drop_duplicates(inplace=True)
                    xls_list_d_.append(df)
                elif elt[-5] == 'M':
                    df.index = df.index + MonthEnd(0)
                    df = df.drop_duplicates(inplace=True)
                    xls_list_m_.append(df)
                elif elt[-5] == "Q":
                    df.index = df.index + QuarterEnd(0)
                    df = df.drop_duplicates(inplace=True)
                    xls_list_q_.append(df)
                else:
                    pass
            except Exception:
                pass

    return csv_list_d_, csv_list_w_, csv_list_m_, csv_list_q_, xls_list_d_, xls_list_m_, xls_list_q_


def fill_bs_date(index, df_, method='ffill'):
    # print('before', len(df_.index), df_.index)
    # bd_list = pd.date_range(df_.index[1], end=df_.index[-1], freq=BDay())
    # print('after, len', len(bd_list), bd_list)

    dummy = pd.DataFrame(data=np.zeros(len(index)), index=index, columns=['dummy'])
    dummy.index.name = 'date'
    df_ = pd.merge(df_, dummy, left_index=True, right_index=True, how='right')

    df_ = df_.replace([0, 'NA', 'NaN'], np.nan)
    if method == 'ffill':
        df_ = df_.fillna(method='ffill')
    elif method == 'zero':
        df_ = df_.fillna(0)
    else:
        pass

    # match_series = np.is_busday(df_.index[1:].values.astype('datetime64[D]'))
    # df_ = df_[np.insert(match_series, 0, 'False')]

    return df_


def df_slicing(df_, start, end):
    df_ = df_[start:end]
    # df_ = df_.drop(df_.loc[:, list((100 * (df_.isnull().sum() / len(df_.index)) > 70))].columns, 1)
    df_ = df_.replace([0, 'NA', 'NaN'], np.nan)
    df_ = df_.dropna(axis=1)
    return df_


def merge_to_df(df_, df_list_):
    try:
        df_ = df_[~df_.index.isin([0, 'NA', 'NaN', 'NaT'])]
    except KeyError:
        pass

    for df_merge in df_list_:
        try:
            df_merge = df_merge[~df_merge.index.isin([0, 'NA', 'NaN', 'NaT'])]
        except KeyError:
            pass
        df_ = pd.merge(df_, df_merge, left_index=True, right_index=True, how='outer')
    return df_


def future_rolling(df_, method='additive'):
    index_list = [1 if date.day > 25 else 0 for date in df_.index]
    index_list2 = np.zeros(len(index_list))
    # print(len(index_list))
    for num in np.arange(1, len(index_list)):
        if index_list[num] == 0:
            try:
                if index_list[num+1] == 1:
                    index_list2[num-3] = 1
            except IndexError:
                pass


    rolling_date = [date_ for date_, index in zip(df_.index, index_list2) if index ==1]
    rolling_date = pd.to_datetime(rolling_date)
    # print(rolling_date)
    df_index = pd.DataFrame(np.ones(len(rolling_date)), index=rolling_date, columns=['index'])
    # df_index.to_csv('test.csv')
    # print(df_index)
    df_ = pd.merge(df_, df_index, left_index=True, right_index=True, how='outer')

    if method == 'additive':
        df_['spread'] = 0
        for date_, i in zip(df_.index, np.arange(len(df_.index))):
            if df_.loc[date_, 'index'] == 1:
                df_.loc[date_, 'spread'] = df_.iloc[i, 0] - df_.iloc[i, 1]
        df_['cum_spread'] = df_['spread'].cumsum()
        df_['crude_future'] = df_['crude_future'] + df_['cum_spread'] - df_.iloc[-1, -1]

    elif method == 'productive':
        df_['spread'] = 1
        for date_, i in zip(df_.index, np.arange(len(df_.index))):
            if df_.loc[date_, 'index'] == 1:
                df_.loc[date_, 'spread'] = df_.iloc[i, 0] / df_.iloc[i, 1]
        df_['cum_spread'] = df_['spread'].cumprod()
        df_['crude_future'] = df_['crude_future'] * df_['cum_spread'] / df_.iloc[-1, -1]

    else:
        print('correct method needed')

    df_ = df_.drop(['crude_future2', 'index', 'spread', 'cum_spread'], axis=1)
    # plt.plot(df_['crude_future'])
    # plt.show()

    return df_


if __name__ == "__main__":
    start_date = '2002-01-31'
    end_date = '2020-06-01'
    path_price = ["data/1.Price", "data/2.Supply_Demand"] ## , "data/3.future_market"
    csv_list_d, csv_list_w, csv_list_m, csv_list_q, xls_list_d, xls_list_m, xls_list_q = read_files(path_price, fillna=True)

    ### Stationary -> Freq -> Normalize
    # Day freq (rolling -> tech indicator gen -> stationary gen (excluding tech indicator) -> merge)
    df_d = pd.concat(csv_list_d, axis=1, join='outer').loc[csv_list_d[0].index]
    df_d = df_slicing(fill_bs_date(df_d.index, df_d), start=start_date, end=end_date)
    df_d = future_rolling(df_d, method='productive').dropna()
    b_index = df_d.index

    # df_d_diff_lag = make_diff_lag(df_d, num_diffs=1, num_lags=10)
    # df_moving_avg_5 = make_moving_average(df_d, num_rolls=5)
    # df_moving_avg_20 = make_moving_average(df_d, num_rolls=20)
    # df_mom_5 = make_momentum(df_d, num_rolls=5)
    # df_mom_20 = make_momentum(df_d, num_rolls=20)
    # df_std_5 = make_std(df_d, num_rolls=5)
    # df_std_20 = make_std(df_d, num_rolls=20)
    # df_skew_5 = make_skew(df_d, num_rolls=5)
    # df_skew_20 = make_skew(df_d, num_rolls=20)
    # df_kurt_5 = make_kurt(df_d, num_rolls=5)
    # df_kurt_20 = make_kurt(df_d, num_rolls=20)
    # df_tech_merge_list = [df_d_diff_lag, df_moving_avg_5, df_moving_avg_20, df_std_5, df_std_20, df_skew_5, df_skew_20,
    #                     df_kurt_5, df_kurt_20]

    df_y = df_d.iloc[:, [0]].pct_change().shift(-1)
    df_y['crude_future_lag0'] = df_y['crude_future'].shift(1)
    df_y['crude_future_lag1'] = df_y['crude_future'].shift(2)
    df_y['crude_future_lag2'] = df_y['crude_future'].shift(3)
    df_y['crude_future_lag3'] = df_y['crude_future'].shift(4)
    df_y['crude_future_lag4'] = df_y['crude_future'].shift(5)

    col_stationary_index, col_stationary_diff = stationary_df(df_d.iloc[:, 1:])
    df_d = pd.merge(df_y, make_stationary(df_d.iloc[:, 1:], col_stationary_index, col_stationary_diff),
                    left_index=True, right_index=True, how='outer')
    # df_d = fill_bs_date(b_index, merge_to_df(df_d, df_tech_merge_list))
    df_d = fill_bs_date(b_index, df_d)
    # df_d.to_csv('test3.csv')

    # Week
    df_w = pd.concat(csv_list_w, axis=1, join='outer', sort=True)
    print('week', df_w)

    # Month
    print(csv_list_m)
    df_m = pd.concat(csv_list_m, axis=1, join='outer', sort=True)
    df_m['CRUDEOIL_closingstock_total'] = df_m['CRUDEOIL_closingstock_AS'] + df_m['CRUDEOIL_closingstock_EU'] \
                                          + df_m['CRUDEOIL_closingstock_OC'] + df_m['CRUDEOIL_closingstock_SA'] \
                                          + df_m['CRUDEOIL_closingstock_US']

    df_m['CRUDEOIL_export_total'] = df_m['CRUDEOIL_export_AS'] + df_m['CRUDEOIL_export_EU'] \
                                          + df_m['CRUDEOIL_export_OC'] + df_m['CRUDEOIL_export_SA'] \
                                          + df_m['CRUDEOIL_export_US'] + df_m['CRUDEOIL_export_CN']

    df_m = df_slicing(df_m.fillna(method='ffill'), start=start_date, end=end_date)
    col_stationary_index, col_stationary_diff = stationary_df(df_m)
    # print('month', col_stationary_index, col_stationary_diff, len(col_stationary_index), len(col_stationary_diff))
    df_m = make_stationary(df_m, col_stationary_index, col_stationary_diff)
    # print('df_m', df_m)

    # Quarter
    df_q = pd.concat(csv_list_q, axis=1, join='outer', sort=True)
    df_q = df_slicing(df_q.fillna(method='ffill'), start=start_date, end=end_date)
    col_stationary_index, col_stationary_diff = stationary_df(df_q)
    # print('quarter', col_stationary_index, col_stationary_diff, len(col_stationary_index), len(col_stationary_diff))
    df_q = make_stationary(df_q, col_stationary_index, col_stationary_diff)
    # print('df_q', df_q)

    # Merge (Freq)
    df = pd.merge(df_d, df_w, left_index=True, right_index=True, how='left')
    df = pd.merge(df, df_m, left_index=True, right_index=True, how='left')
    df = pd.merge(df, df_q, left_index=True, right_index=True, how='left')
    df = df.fillna(method='ffill')
    df = df.dropna(how='all', axis=1)

    df = df[['crude_future', 'crude_future_lag0', 'crude_future_lag1', 'crude_future_lag2', 'crude_future_lag3',
             'crude_future_lag4', 'ngl_furture1', 'ngl_spot', 'wti_spot', 'brent_spot',
             'CRUDEOIL_closingstock_total', 'CRUDEOIL_export_total', 'CRUDEOIL_prod_total', 'production_cap_total',
             'cur', 'consump_world_M', 'CRUDEOIL_import_total', 'PPI_CHN', 'PPI_US', 'PPI_EU']]

    df.to_csv('df_whole0.csv')

    # Merge EPU data
    df2 = pd.read_csv('df_whole_new.csv', index_col='date')
    df2 = df_slicing(fill_bs_date(b_index, df2), start='2002-01-31', end='2020-06-01').dropna(how='all', axis=1)
    # print(df2.index, df2)

    df_input = pd.merge(df, df2, left_index=True, right_index=True, how='outer').dropna(how='all', axis=1)
    df_input.index.name = 'date'
    # df_input.to_csv('data_input.csv')
    df_input = df_input.dropna()

    df_x = scaler_with_nan(df_input.iloc[:, 6:])
    df_input = pd.merge(df_input.iloc[:, :6], df_x, left_index=True, right_index=True, how='outer')
    print(df_input, len(df_input.columns))
    df_input.to_csv('ml_data_input_normalized_stationary.csv')


