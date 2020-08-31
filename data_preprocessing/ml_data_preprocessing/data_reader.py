import numpy as np
import pandas as pd
import glob
from pmdarima.arima import ndiffs
from pandas.tseries.offsets import QuarterBegin, QuarterEnd


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
    csv_list = []
    xls_list = []

    for path in paths:
        csv_files = glob.glob(path + "/*.csv")
        xls_files = glob.glob(path + "/*.xls")

        for elt in csv_files:
            df = pd.read_csv(elt)
            df = set_date_as_index(df)
            df = make_float(df)
            if fillna:
                df = df.fillna(method='ffill')
            csv_list.append(df)

        for elt in xls_files:
            try:
                df = pd.read_excel(elt)
                df = set_date_as_index(df)
                df = make_float(df)
                if fillna:
                    df = df.fillna(method='ffill')
                xls_files.append(df)
            except Exception:
                pass

    return csv_list, xls_list


def make_stationary(df):
    columns = df.columns
    for name in columns:
        x = df[name].values
        # print(name, x)
        d_kpss = ndiffs(x, test='kpss')
        d_adf = ndiffs(x, test='adf')
        d_pp = ndiffs(x, test='pp')
        d_ = max(d_kpss, d_adf, d_pp)
        if d_ > 0:
            new_name = name + '_diff' + str(d_)
            df[new_name] = df[name].diff(d_)
            df = df.drop(columns=[name])
    df = df.dropna()
    return df


def read_data(path, sheet=False, header='infer'):
    if sheet is False:
        try:
            df = pd.read_excel(path, header=header)
        except Exception:
            try:
                df = pd.read_csv(path, header=header)
            except Exception as e:
                raise Exception(e)
    else:
        try:
            excel_file = pd.ExcelFile(path)
            assert sheet in excel_file.sheet_names
            df = excel_file.parse(sheet, header=header)
        except Exception:
            raise Exception("Can not read sheet")

    df.columns = [name.lower() for name in df.columns]

    if 'year2' in df.columns:
        drop_columns = ['year2']
    else:
        drop_columns = []
    for elt in df.columns:
        if 'unnamed' in elt:
            drop_columns.append(elt)
    df.drop(columns=drop_columns, inplace=True)

    first_valid = df.iloc[:, 1].first_valid_index()
    last_valid = df.iloc[:, 1].last_valid_index() + 1
    df = df.iloc[first_valid:last_valid]
    df.columns = df.columns.str.replace('.', '_')
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('__', '_')
    df.dropna(inplace=True)
    return df


def make_monthly_date(df, offset=True):
    datetime = pd.to_datetime(
        (
            df['year'].astype(int) * 100
            + df['month'].astype(int)
        ).astype(str),
        format='%Y%m'
    )
    if offset:
        datetime += pd.tseries.offsets.MonthBegin(1)
    else:
        datetime = datetime
    df['date'] = datetime
    df.drop(columns=['year', 'month'], inplace=True)
    df.set_index('date', inplace=True)
    df.columns = [elt + '_monthly' for elt in df.columns]
    df = df.dropna()
    return df


def make_quarterly_date(df, offset=True):
    df['year'] = df['year'].str.lower()
    df['year'] = df['year'].str.replace(r'(q\d)-(\d+)', r'\2-\1')
    if offset:
        # Bug that quarterbegin is March 01
        df['date'] = pd.to_datetime(df['year'])\
            + pd.tseries.offsets.DateOffset(days=1)\
            + pd.tseries.offsets.QuarterBegin(1, startingMonth=1)
    else:
        df['date'] = pd.to_datetime(df['year'])
    df.drop(columns=['year'], inplace=True)
    df.set_index('date', inplace=True)
    # Manually shift because of QuarterBegin bug
    df.columns = [elt + '_quarterly' for elt in df.columns]
    df = df.dropna()
    return df


def make_daily_date(df):
    datetime = pd.to_datetime(
        (
            df['year'].astype(int) * 10000
            + df['month'].astype(int) * 100
            + df['day'].astype(int)
        ).astype(str),
        format='%Y%m%d'
    )
    df['date'] = datetime
    df.drop(columns=['year', 'month', 'day'], inplace=True)
    df.set_index('date', inplace=True)
    df.columns = [elt + '_daily' for elt in df.columns]
    df = df.dropna()
    return df


# If date of low frequency data is specified, assume It is announced
# before the start of the market
# If not specified, assume it is announced after the market is closed
def daily_data(df, freq, offset=True, fill_method='ffill'):
    drop_columns = []
    for elt in df.columns:
        if 'unnamed' in elt:
            drop_columns.append(elt)
    df.drop(columns=drop_columns, inplace=True)

    if freq.lower() == 'monthly':
        try:
            df = make_monthly_date(df, offset=offset)
        except Exception:
            print("set monthly date as index")
            datetime = pd.to_datetime(df['date'])
            df['date'] = datetime
            df.set_index('date', inplace=True)
            df.columns = [elt + '_monthly' for elt in df.columns]
        df = make_stationary(df)
        if offset:
            daily_datetime = pd.date_range(
                df.index[0] + pd.tseries.offsets.MonthBegin(1),
                df.index[-1] + pd.tseries.offsets.MonthEnd(1),
                freq='D'
            )
        else:
            daily_datetime = pd.date_range(
                df.index[0] + pd.tseries.offsets.MonthBegin(1),
                df.index[-1] + pd.tseries.offsets.MonthEnd(1),
                freq='D'
            )
        df = df.reindex(daily_datetime, method=fill_method)
    elif freq.lower() == 'daily':
        try:
            df = make_daily_date(df)
        except Exception:
            print("set daily date as index")
            datetime = pd.to_datetime(df['date'])
            df['date'] = datetime
            df.set_index('date', inplace=True)
            df.columns = [elt + '_daily' for elt in df.columns]
        df = make_stationary(df)
        daily_datetime = pd.date_range(
            df.index[0],
            df.index[-1],
            freq='D'
        )
        df = df.reindex(daily_datetime, method=fill_method)
    elif freq.lower() == 'quarterly':
        try:
            df = make_quarterly_date(df)
        except Exception:
            print("set quarterly date as index")
            datetime = pd.to_datetime(df['date'])
            df['date'] = datetime
            df.set_index('date', inplace=True)
            df.columns = [elt + '_quarterly' for elt in df.columns]
        df = make_stationary(df)
        if offset:
            daily_datetime = pd.date_range(
                df.index[0] + QuarterBegin(1, startingMonth=1),
                df.index[-1] + QuarterEnd(1, startingMonth=1),
                freq='D'
            )
        else:
            daily_datetime = pd.date_range(
                df.index[0],
                df.index[-1] + pd.tseries.offsets.QuarterEnd(1),
                freq='D'
            )
        df = df.reindex(daily_datetime, method=fill_method)
        df = df.dropna()
    else:
        print("Type frequency")
    daily_datetime = pd.date_range(
        df.index[0], df.index[-1],
        freq='D'
    )
    df = df.reindex(daily_datetime, method=fill_method)

    drop_columns = []
    for elt in df.columns:
        if 'unnamed' in elt:
            drop_columns.append(elt)
    df.drop(columns=drop_columns, inplace=True)
    return df


def get_nonfinancial():
    print('monthly epu')
    monthly_epu = read_data('../../data/epu/All_Country_data.xlsx')
    daily_epu = daily_data(monthly_epu, 'monthly')
    daily_epu.columns = ['epu_' + elt for elt in daily_epu.columns]

    print('daily_infectious')
    daily_infectious = read_data('../../data/epu/All_Infectious_EMV_Data.csv')
    daily_infectious = daily_data(daily_infectious, 'daily')
    daily_infectious.columns = [
        'daily_infectious_' + elt for elt in daily_infectious.columns]

    print('categorical_epu')
    categorical_epu = read_data('../../data/epu/Categorical_EPU_Data.xlsx')
    categorical_epu = daily_data(categorical_epu, 'monthly')
    categorical_epu.columns = [
        'categorical_epu_' + elt for elt in categorical_epu.columns]

    print('eurq_data')
    eurq_data = read_data('../../data/epu/EURQ_data.xlsx', sheet='EURQ')
    eurq_data = daily_data(eurq_data, 'monthly')
    eurq_data.columns = ['eurq_data_' + elt for elt in eurq_data.columns]

    print('trade_unc')
    trade_uncertainty_data = read_data(
        '../../data/epu/Trade_Uncertainty_Data.xlsx')
    trade_uncertainty_data = daily_data(trade_uncertainty_data, 'monthly')
    trade_uncertainty_data.columns = [
        'trade_uncertainty_' + elt for elt in trade_uncertainty_data.columns
    ]

    print('wpui')
    wpui_data = read_data(
        '../../data/epu/WPUI_Data.xlsx', sheet='F1', header=1)
    wpui_data = daily_data(wpui_data, 'quarterly')
    wpui_data.columns = [
        'wpui_' + elt for elt in wpui_data.columns
    ]

    print('wui')
    wui_data = read_data(
        '../../data/epu/WUI_Data-1.xlsx', sheet='F1', header=2)
    wui_data = daily_data(wui_data, 'quarterly')
    wui_data.columns = [
        'wui_' + elt for elt in wui_data.columns
    ]

    print('fed_funds')
    fed_funds = read_data('../../data/financial_market_monthly/FEDFUNDS.csv')
    fed_funds = daily_data(fed_funds, 'monthly', offset=False)

    print('msci')
    msci_data = read_data('../../data/financial_market_monthly/msciworld.xls')
    msci_data = daily_data(msci_data, 'monthly', offset=False)

    df_non_financial = pd.concat(
        [
                daily_epu, daily_infectious, categorical_epu, eurq_data,
                trade_uncertainty_data, wpui_data, wui_data,
                fed_funds, msci_data
        ], axis=1
    )

    print('non-financial data')
    df_non_financial.index.name = 'date'
    return df_non_financial


def get_financial():
    print('finance data')
    path = ["../../data/financial_market"]
    csv_list, xls_list = read_files(path)
    df = pd.concat([*csv_list], axis=1, sort=False)
    df = df.drop(df.tail(3).index)
    df = df.fillna(method='ffill')
    df.columns = [elt + '_daily' for elt in df.columns]

    df.index.name = 'date'
    df.dropna(inplace=True)
    df = make_stationary(df)
    df.dropna(inplace=True)
    df.index.name = 'date'
    return df
