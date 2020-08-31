import pandas as pd
from utils import make_diff_lag, make_pct_change_lag
from utils import make_momentum, make_moving_average
from utils import make_std, make_skew, make_kurt
from data_reader import get_financial
from data_reader import get_nonfinancial
from hand_select import hand_select


def drop_non_daily(df):
    non_daily_columns = []
    for elt in df.columns:
        if '_daily' not in elt:
            non_daily_columns.append(elt)
    df = df.drop(columns=non_daily_columns)
    return df


def naive_datetime(df):
    datetime = pd.to_datetime(df['date'])
    df['date'] = datetime
    df.set_index('date', inplace=True)
    df.index.name = 'date'
    return df


def make_technical(df):
    df_diff = make_diff_lag(df, num_diffs=1, num_lags=1)
    df_pct_change = make_pct_change_lag(df, num_diffs=1, num_lags=1)
    df_momentum = make_momentum(df, num_rolls=30)
    df_ma = make_moving_average(df, num_rolls=30)
    df_std = make_std(df, num_rolls=30)
    df_skew = make_skew(df, num_rolls=30)
    df_kurt = make_kurt(df, num_rolls=30)
    df_technical = pd.concat(
        [
            df_diff, df_pct_change,
            df_momentum, df_ma,
            df_std, df_skew, df_kurt
        ],
        axis=1
    )
    df_technical = drop_non_daily(df_technical)
    return df_technical


def make_data():
    nonfinancial_data = get_nonfinancial()
    financial_data = get_financial()
    nonfinancial_data.index.name = 'date'
    financial_data.index.name = 'date'

    df = pd.concat([nonfinancial_data, financial_data], axis=1)
    df.index.name = 'date'

    df_selected = hand_select(df)
    return df_selected


if __name__ == '__main__':
    df_selected = make_data()
    df_selected.to_csv('df_selected.csv')
