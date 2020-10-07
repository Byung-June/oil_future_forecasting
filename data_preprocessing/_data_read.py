import pandas as pd
import numpy as np
from _data_utils import stationary_df, make_stationary
from _data_utils import df_slicing, fill_bs_date, future_rolling, make_float
from _data_utils import scaler_with_nan
from _data_jodi import jodi_read
from ml_data_preprocessing.make_data import make_data
from oil_forecastor.model_selection._utility import rolling_train_test_split, denoising_func
import copy
from sklearn.metrics import r2_score
import math
from scipy.special import gamma


def read_data_url(url, sheet_name_list, col_name_list, freq='D'):
    data = pd.read_excel(url, sheet_name=sheet_name_list)
    df_list_ = []

    for sheet_name, col_name in zip(sheet_name_list, col_name_list):
        df_ = data[sheet_name]
        df_.columns = column_fill_name(df_.columns, col_name)
        df_ = df_[pd.to_numeric(df_[col_name[1]], errors='coerce').notnull()]
        df_ = df_.set_index('date')
        if freq == 'M':
            # df_.index = df_.index.to_period('M').to_timestamp('M').shift(1, freq='D')
            pass
        try:
            df_ = df_.drop(['_'], axis=1)
        except KeyError:
            pass
        df_list_.append(df_)
    return df_list_


def read_data_csv(filename, freq='M'):
    df_ = pd.read_csv(filename, index_col=0)
    df_.index = pd.to_datetime(df_.index)

    df_ = make_float(df_)
    df_ = df_.replace([0, 'NA', 'NaN'], np.nan)

    if freq == 'M':
        df_.index = df_.index.to_period('M').to_timestamp('M').shift(1, freq='D')
    df_ = df_.loc[~df_.index.duplicated(keep='first')]
    return df_


def column_fill_name(columns, new_columns):
    if len(columns) > len(new_columns):
        new_columns.extend(['_'] * (len(columns) - len(new_columns)))
    return new_columns


def resample_df(df_, freq='W', method='last'):
    df_2 = copy.deepcopy(df_)
    df_3 = copy.deepcopy(df_2.iloc[:, 0])
    if method == 'sum':
        df_2 = df_2.assign(date=df_.index).resample(freq).sum()
    else:
        df_2 = df_2.assign(date=df_.index).resample(freq).last()
    m = df_3.index.to_period(freq)
    df_3 = df_3.reset_index().groupby(m).last().set_index('date')
    df_2.index = df_3.index
    return df_2


def gen_data(filename, freq='D', start_date='2002-03-30', end_date='2020-06-01',
             scaler=False, filter=None, y_type='return',
             bi_w=5, bi_sig_d=10, bi_sig_i=1):
    url_list_daily = [
        [
            'https://www.eia.gov/dnav/pet/xls/PET_PRI_FUT_S1_D.xls',
            ['Data 1'],
            [['date', 'y_test', 'y_test_dummy']]
        ],
        [
            'https://www.eia.gov/dnav/pet/xls/PET_PRI_SPT_S1_D.xls',
            ['Data 1'],
            [['date', 'wti_spot_daily', 'brent_spot_daily']]
        ],
        [
            'https://www.eia.gov/dnav/ng/xls/NG_PRI_FUT_S1_D.xls',
            ['Data 1', 'Data 2'],
            [['date', 'ngl_spot_daily'], ['date', 'ngl_furture_daily']]
        ]
    ]

    url_list_weekly = [
        [
            'https://www.eia.gov/dnav/pet/hist_xls/WPULEUS3w.xls',
            ['Data 1'],
            [['date', 'cur_weekly']]]
    ]

    url_list_monthly = [
        [
            'https://fred.stlouisfed.org/graph/fredgraph.xls?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=off&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=CHNPIEATI01GYM&scale=left&cosd=1999-01-01&coed=2020-06-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2020-08-28&revision_date=2020-08-28&nd=1999-01-01',
            [0],
            [['date', 'PPI_china_monthly']]
        ],
        [
            'https://fred.stlouisfed.org/graph/fredgraph.xls?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=PCUOMFGOMFG&scale=left&cosd=1984-12-01&coed=2020-07-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2020-08-28&revision_date=2020-08-28&nd=1984-12-01',
            [0],
            [['date', 'PPI_usa_monthly']]
        ],
        [
            'https://fred.stlouisfed.org/graph/fredgraph.xls?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=off&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=EU28PIEATI01GPM&scale=left&cosd=2000-02-01&coed=2020-06-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2020-08-28&revision_date=2020-08-28&nd=2000-02-01',
            [0],
            [['date', 'PPI_eu_monthly']]
        ]
    ]

    df_d = []
    df_w = []
    df_m = [jodi_read(),
            read_data_csv('Liquid_Fuels_Consumption_Total_World_M.csv'),
            read_data_csv('OPEC_production_capacity_total_M.csv')]

    for url, sheets, columns in url_list_daily:
        df_d.extend(read_data_url(url, sheets, columns))

    for url, sheets, columns in url_list_weekly:
        df_w.extend(read_data_url(url, sheets, columns))

    for url, sheets, columns in url_list_monthly:
        df_m.extend(read_data_url(url, sheets, columns, freq='M'))

    df_d = pd.concat(df_d, axis=1, join='outer').loc[df_d[0].index]
    df_d = fill_bs_date(df_d.index, df_d)
    df_d = future_rolling(df_d, method='productive').dropna()
    df_d = df_d[df_d['y_test'] > 0]
    df_d.index.name = 'date'

    df_y = df_d.iloc[:, [0]]
    df_y = denoising_func(df_y, filter=filter)
    if y_type == 'logvol':
        df_y['y_test'] = 100 * np.log(df_y['y_test']).diff()
        df_y['cumulative_return'] = df_y['y_test']
        df_y['y_test'] = np.square(df_y['y_test'])

    if freq == 'W':
        df_y = resample_df(df_y, freq='W', method='sum')
    if freq == 'M':
        df_y = resample_df(df_y, freq='M', method='sum')
    b_index = df_y.index
    df_y['y_test'] = np.sqrt(df_y['y_test'])
    df_y['cumulative_return'] = np.abs(df_y['cumulative_return'])

    if freq == 'D':
        df_y['crude_oil_realized_W'] = df_y['y_test'].rolling(5).mean()
        df_y['crude_oil_realized_M'] = df_y['y_test'].rolling(22).mean()
    if freq == 'W':
        df_y['crude_oil_realized_M'] = df_y['y_test'].rolling(4).mean()
        df_y['crude_oil_realized_Q'] = df_y['y_test'].rolling(13).mean()

    df_y['crude_future_daily_lag0'] = df_y['y_test'].values
    df_y['y_test'] = df_y['y_test'].shift(-1)
    print('weekly realized vol', df_y)

    col_stationary_index, col_stationary_diff = stationary_df(df_d.iloc[:, 1:])
    df_d = pd.merge(df_y, make_stationary(df_d.iloc[:, 1:], col_stationary_index, col_stationary_diff),
                    left_index=True, right_index=True, how='outer')
    df_d = fill_bs_date(b_index, df_d)

    # Week
    df_w = df_w[0]
    df_w = df_slicing(fill_bs_date(b_index, df_w).dropna(), start=start_date, end=end_date)
    if freq == 'M':
        df_w = resample_df(df_w, freq='M')
    col_stationary_index, col_stationary_diff = stationary_df(df_w)
    df_w = make_stationary(df_w, col_stationary_index, col_stationary_diff)

    # Month
    df_m = pd.concat(df_m, axis=1, join='outer', sort=True)
    df_m['CRUDEOIL_closingstock_total'] = df_m['CRUDEOIL_closingstock_AS'] + df_m['CRUDEOIL_closingstock_EU'] \
                                          + df_m['CRUDEOIL_closingstock_OC'] + df_m['CRUDEOIL_closingstock_SA'] \
                                          + df_m['CRUDEOIL_closingstock_US']

    df_m['CRUDEOIL_export_total'] = df_m['CRUDEOIL_export_AS'] + df_m['CRUDEOIL_export_EU'] \
                                    + df_m['CRUDEOIL_export_OC'] + df_m['CRUDEOIL_export_SA'] \
                                    + df_m['CRUDEOIL_export_US'] + df_m['CRUDEOIL_export_CN']

    df_m['CRUDEOIL_import_total'] = df_m['CRUDEOIL_import_AS'] + df_m['CRUDEOIL_import_EU'] \
                                    + df_m['CRUDEOIL_import_OC'] + df_m['CRUDEOIL_import_SA'] \
                                    + df_m['CRUDEOIL_import_US'] + df_m['CRUDEOIL_import_CN']

    df_m['CRUDEOIL_prod_total'] = df_m['CRUDEOIL_prod_AS'] + df_m['CRUDEOIL_prod_EU'] \
                                  + df_m['CRUDEOIL_prod_OC'] + df_m['CRUDEOIL_prod_SA'] \
                                  + df_m['CRUDEOIL_prod_US'] + df_m['CRUDEOIL_prod_CN']

    col_stationary_index, col_stationary_diff = stationary_df(df_m)
    df_m = make_stationary(df_m, col_stationary_index, col_stationary_diff)
    df_m = df_slicing(fill_bs_date(b_index, df_m), start=start_date, end=end_date)

    # Merge (Freq)
    df = pd.merge(df_d, df_w, left_index=True, right_index=True, how='left')
    df = pd.merge(df, df_m, left_index=True, right_index=True, how='left')
    df = df[['y_test', 'crude_future_daily_lag0', 'crude_oil_realized_M', 'crude_oil_realized_Q',
             'cumulative_return',
             'wti_spot_daily', 'ngl_spot_daily',
             'ngl_furture_daily', 'brent_spot_daily',
             'cur_weekly',
             'CRUDEOIL_closingstock_total', 'CRUDEOIL_export_total', 'CRUDEOIL_import_total',
             'CRUDEOIL_prod_total', 'production_cap_total', 'consump_world_M',
             'PPI_china_monthly', 'PPI_usa_monthly', 'PPI_eu_monthly']]
    df = df.fillna(method='ffill')

    # Merge EPU data
    df2 = make_data()
    df2 = df_slicing(fill_bs_date(b_index, df2), start=start_date, end=end_date).dropna(how='all', axis=1)

    df_input = pd.merge(df, df2, left_index=True, right_index=True, how='outer').dropna(how='all', axis=1)
    df_input.index.name = 'date'
    df_input = df_input.dropna()

    if scaler:
        df_x = scaler_with_nan(df_input.iloc[:, 6:])
        df_input = pd.merge(df_input.iloc[:, :6], df_x, left_index=True, right_index=True, how='outer')

    df_input.to_csv(filename)
    return df_input


if __name__ == '__main__':
    # gen_data(filename='return_ml_data_W.csv', freq='W', filter=None, y_type='return')
    # gen_data(filename='return_ma_ml_data_W.csv', freq='W', filter='moving_average', y_type='return')

    gen_data(filename='logvol_ml_data_W.csv', freq='W', filter=None, y_type='logvol')
    # gen_data(filename='logvol_ml_data_W.csv', freq='W', filter=None, y_type='logvol')

    # gen_data(filename='vol_ml_data_W.csv', freq='W', filter=None, y_type='vol')
    # gen_data(filename='vol_ml_data_M.csv', freq='M', filter=None, y_type='vol')
