name_list = [
    'GEPU_current', 'GEPU_ppp',

    'daily_infect_emv_index',

    'economic_policy_uncertainty', 'monetary_policy', 'fiscal_policy',
    'taxes', 'government_spending', 'health_care', 'national_security',
    'entitlement_programs', 'regulation', 'financial_regulation',
    'trade_policy', 'sovereign_debt',

    'EURQ_US',

    # 'US_Trade_policy_Uncertainty', 'Japanese_trade_policy_uncertainty',
    # 'trade_policy_EMV_Fraction',

    'total_count_of_uncertainty_surrounding_pandemics',

    'WUI',

    'adj_close',

    'dexjpus', 'dexuseu', 'dexusuk', 'dexchus',
    'fedfunds', 'world_standard',

    'date'
]
name_list = [elt.lower() for elt in name_list]


def hand_select(df):
    columns = df.columns

    selected = []
    for col_name in columns:
        for elt in name_list:
            if col_name in elt or elt in col_name:
                if 'eurq_us_old' not in col_name:
                    selected.append(col_name)
    df = df[selected]
    df = df.loc[:, ~df.columns.duplicated()]
    return df
