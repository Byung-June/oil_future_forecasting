import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np


def jodi_read():
    # JODI database
    df_jodi = pd.read_csv('jodi_raw_M.csv')
    df_jodi.columns = ['area', 'Date', 'product', 'flow', 'unit', 'value', 'code']
    dict_flow = {'INDPROD': 'prod', 'DIRECUSE': 'direcuse', 'TOTEXPSB': 'export', 'TOTIMPSB': 'import',
                 'REFINOBS': 'refinein', 'STOCKCH': 'stockch', 'STATDIFF': 'statdiff', 'TRANSBAK': 'transfer',
                 'OSOURCES': 'osource','CLOSTLV': 'closingstock'}
    df_jodi.flow = df_jodi.flow.replace(dict_flow)

    country_continent = pd.read_csv('country_continent.csv', index_col=0)
    country_continent = country_continent.dropna(axis=0)
    country_continent = country_continent.to_dict()
    country_continent = country_continent['continent_code']
    # print('country_contin', country_continent)

    df_jodi['area'] = df_jodi.area.replace(country_continent)
    df_jodi = df_jodi.dropna()

    area_list = ['AS', 'AS', 'EU', 'NA', 'OC', 'SA', 'CN', 'US']
    df_jodi = df_jodi[df_jodi['area'].isin(area_list)]
    df_jodi = df_jodi[df_jodi['unit'] == 'KBBL']
    df_jodi = df_jodi.sort_values(by=['product', 'flow'])
    df_jodi = df_jodi.groupby(['Date', 'area', 'product', 'flow']).sum()
    df_jodi = df_jodi.reset_index()
    df_jodi['area'] = df_jodi['product'] + '_' + df_jodi['flow'] + '_' + df_jodi['area']
    df_jodi = pd.pivot_table(data=df_jodi, index='Date', columns='area', values='value')
    df_jodi.index = pd.to_datetime(df_jodi.index, format="%Y-%m") + MonthEnd(1)
    zero_rule = df_jodi.replace([0, 'NA'], np.nan).apply(lambda x: any(~x.isnull()))
    df_jodi = df_jodi.loc[:, zero_rule]
    # df_jodi.to_csv('2.jodi_M.test.csv')

    return df_jodi

