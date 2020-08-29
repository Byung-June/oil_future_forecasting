import pandas as pd
from data_preprocessing._data_utils import *
from data_preprocessing.data_jodi import jodi_read
from pandas.tseries.offsets import MonthEnd


def read_data_url(url, sheet_name_list, col_name_list, freq='D'):
    data = pd.read_excel(url, sheet_name=sheet_name_list)
    df_list_ = []

    for sheet_name, col_name in zip(sheet_name_list, col_name_list):
        df_ = data[sheet_name]
        df_.columns = column_fill_name(df_.columns, col_name)
        df_ = df_[pd.to_numeric(df_[col_name[1]], errors='coerce').notnull()]
        df_ = df_.set_index('date')
        if freq == 'M':
            df_.index = df_.index.to_period('M').to_timestamp('M')
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

    if freq=='M':
        df_.index = df_.index.to_period('M').to_timestamp('M')
    df_ = df_.loc[~df_.index.duplicated(keep='first')]
    return df_


def column_fill_name(columns, new_columns):
    if len(columns) > len(new_columns):
        new_columns.extend(['_'] * (len(columns) - len(new_columns)))
    return new_columns


if __name__ == "__main__":
    # %%
    url_list_daily = [
        [
            'https://www.eia.gov/dnav/pet/xls/PET_PRI_FUT_S1_D.xls',
            ['Data 1'],
            [['date', 'y_test']]
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

    start_date = '2002-03-30'
    end_date = '2020-06-01'
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
    b_index = df_d.index

    # %%
    df_y = df_d.iloc[:, [0]].pct_change().shift(-1)
    df_y['crude_future_daily_lag0'] = df_y['y_test'].shift(1)
    df_y['crude_future_daily_lag1'] = df_y['y_test'].shift(2)
    df_y['crude_future_daily_lag2'] = df_y['y_test'].shift(3)
    df_y['crude_future_daily_lag3'] = df_y['y_test'].shift(4)
    df_y['crude_future_daily_lag4'] = df_y['y_test'].shift(5)
    col_stationary_index, col_stationary_diff = stationary_df(df_d.iloc[:, 1:])
    df_d = pd.merge(df_y, make_stationary(df_d.iloc[:, 1:], col_stationary_index, col_stationary_diff),
                    left_index=True, right_index=True, how='outer')
    df_d = fill_bs_date(b_index, df_d)
    # print('df_d', df_d.columns)

    # Week
    df_w = df_w[0]
    df_w = df_slicing(fill_bs_date(b_index, df_w).dropna(), start=start_date, end=end_date)
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

    df_m = df_slicing(fill_bs_date(b_index, df_m), start=start_date, end=end_date)
    col_stationary_index, col_stationary_diff = stationary_df(df_m)
    df_m = make_stationary(df_m, col_stationary_index, col_stationary_diff)

    # Merge (Freq)
    df = pd.merge(df_d, df_w, left_index=True, right_index=True, how='left')
    df = pd.merge(df, df_m, left_index=True, right_index=True, how='left')
    df = df[['y_test', 'crude_future_daily_lag0', 'crude_future_daily_lag1',
             'crude_future_daily_lag2', 'crude_future_daily_lag3',
             'crude_future_daily_lag4', 'wti_spot_daily', 'ngl_spot_daily',
             'ngl_furture_daily', 'brent_spot_daily',
             'cur_weekly',
             'CRUDEOIL_closingstock_total', 'CRUDEOIL_export_total', 'CRUDEOIL_import_total',
             'CRUDEOIL_prod_total', 'production_cap_total', 'consump_world_M',
             'PPI_china_monthly', 'PPI_usa_monthly', 'PPI_eu_monthly']]
    df = df.fillna(method='ffill')

    # Merge EPU data
    df2 = pd.read_csv('df_whole_new.csv', index_col='date')
    df2 = df_slicing(fill_bs_date(b_index, df2), start=start_date, end=end_date).dropna(how='all', axis=1)

    df_input = pd.merge(df, df2, left_index=True, right_index=True, how='outer').dropna(how='all', axis=1)
    df_input.index.name = 'date'
    df_input = df_input.dropna()

    df_x = scaler_with_nan(df_input.iloc[:, 6:])
    df_input = pd.merge(df_input.iloc[:, :6], df_x, left_index=True, right_index=True, how='outer')
    # print('final', df_input, len(df_input.columns))
    df_input.to_csv('ml_data_input_normalized_stationary.csv')