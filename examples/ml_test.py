import numpy as np
import os
import copy
from oil_forecastor.ml_forecastor.forecast import MLForecast
import pandas as pd
import argparse
import warnings
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-path',
    default='../data/ml_data.csv', type=str,
    help="path to data"
)
parser.add_argument(
    '--without-epu', default=False, action='store_true'
)
parser.add_argument('--filter-method',
                    default='moving_average', type=str)
parser.add_argument('--ignore-warnings', default=True, action='store_false')
parser.add_argument('--use-unfiltered', default=False, action='store_true')
parser.add_argument('--plot-test-data', default=False, action='store_true')
parser.add_argument('--selected-inputs', default=True, action='store_false')
arguments = parser.parse_args()

if arguments.ignore_warnings:
    warnings.filterwarnings('ignore')


def evaluation(df, delete_outlier=True):
    df = df.dropna()
    # y_test_before_filtered = df['y_test'].values.flatten()
    y_pred_before_recovered = df['y_pred_before_recovered'].values.flatten()
    y_filtered_test = df['y_test_filtered'].values.flatten()

    y_pred = df['y_pred'].values.flatten()
    if delete_outlier:
        mean = y_pred.mean()
        std = y_pred.std()
        y_pred = np.where(y_pred >= mean + 3 * std,
                          mean + 3 * std, y_pred)
        y_pred = np.where(y_pred <= mean - 3 * std,
                          mean - 3 * std, y_pred)
    y_true = df['y_true'].values.flatten()
    return r2_score(y_true, y_pred),\
        r2_score(y_filtered_test, y_pred_before_recovered)


def make_name(name, sw_tuple, n_features, args):
    n_samples, n_windows = sw_tuple
    without_epu = str(args.without_epu)
    filter_method = args.filter_method
    path = "../results/" + name + "_windows_" + str(n_windows)
    path += "_samples_" + str(n_samples)
    if n_features > 100:
        path += "_whole"
    else:
        path += "_" + str(n_features)
    path += "_" + filter_method + "_" + "without_epu_" + without_epu
    return path


def main(exogenous, filter_method, n_features, sw_tuple):
    # exogenous = exogenous.drop('y_true', axis=1)
    y_test_before_filtered = copy.deepcopy(exogenous['y_test']).to_frame()
    y_test_filtered = copy.deepcopy(exogenous['y_test_filtered']).to_frame()
    exogenous = exogenous.drop('y_test', axis=1)
    exogenous = exogenous.drop('y_test_filtered', axis=1)
    if filter_method != 'none':
        filtered = pd.concat([exogenous, y_test_filtered], axis=1)
    else:
        filtered = pd.concat([exogenous, y_test_before_filtered], axis=1)
        filtered = filtered.rename(columns={'y_test': 'y_test_filtered'})

    n_samples, n_windows = sw_tuple
    start_time = n_windows + n_samples - 2
    end_time = len(filtered) - 1

    ml_forecast = MLForecast(
        filtered, n_windows, n_samples, start_time, end_time)

    print("linear_reg")
    res_linear_reg = ml_forecast.linear_reg(
        n_features=n_features, method='f-classif'
    )
    res_linear_reg = pd.concat([res_linear_reg, y_test_before_filtered],
                               axis=1)
    r2_test, r2_filtered_test = evaluation(res_linear_reg)
    print('r2 test {}, r2 filtered test {}'.format(r2_test, r2_filtered_test))
    name_lin_reg = make_name("res_linear_reg", sw_tuple, n_features, arguments)
    res_linear_reg.to_csv(name_lin_reg + ".csv")

    print("lasso")
    res_lasso = ml_forecast.lasso(
        n_features=n_features, method='f-classif'
    )
    res_lasso = pd.concat([res_lasso, y_test_before_filtered],
                          axis=1)
    r2_test, r2_filtered_test = evaluation(res_lasso)
    print('r2 test {}, r2 filtered test {}'.format(r2_test, r2_filtered_test))
    name_lasso_reg\
        = make_name("res_lasso_reg", sw_tuple, n_features, arguments)
    res_lasso.to_csv(name_lasso_reg + ".csv")

    print("pcr")
    res_pcr = ml_forecast.pcr()
    res_pcr = pd.concat([res_pcr, y_test_before_filtered],
                        axis=1)
    r2_test, r2_filtered_test = evaluation(res_pcr)
    print('r2 test {}, r2 filtered test {}'.format(r2_test, r2_filtered_test))
    name_lin_reg = make_name("res_pcr", sw_tuple, n_features, arguments)
    res_pcr.to_csv(name_lin_reg + ".csv")

    print("svr")
    if n_features > 100:
        res_svr = ml_forecast.svr(n_features=50, method='f-classif')
    else:
        res_svr = ml_forecast.svr(n_features=n_features, method='f-classif')
    res_svr = pd.concat([res_svr, y_test_before_filtered],
                        axis=1)
    r2_test, r2_filtered_test = evaluation(res_svr)
    print('r2 test {}, r2 filtered test {}'.format(r2_test, r2_filtered_test))
    name_lin_reg = make_name("res_svr", sw_tuple, n_features, arguments)
    res_svr.to_csv(name_lin_reg + ".csv")

    print("kr")
    if n_features > 100:
        res_kr = ml_forecast.kernel_ridge(n_features=50, method='f-classif')
    else:
        res_kr = ml_forecast.kernel_ridge(
            n_features=n_features, method='f-classif'
        )
    res_kr = pd.concat([res_kr, y_test_before_filtered],
                       axis=1)
    r2_test, r2_filtered_test = evaluation(res_kr)
    print('r2 test {}, r2 filtered test {}'.format(r2_test, r2_filtered_test))
    name_lin_reg = make_name("res_kr", sw_tuple, n_features, arguments)
    res_kr.to_csv(name_lin_reg + ".csv")

    print("dtr")
    res_dtr = ml_forecast.decision_tree_reg(
        n_features=n_features, method='f-classif'
    )
    res_dtr = pd.concat([res_dtr, y_test_before_filtered],
                        axis=1)
    r2_test, r2_filtered_test = evaluation(res_dtr)
    print('r2 test {}, r2 filtered test {}'.format(r2_test, r2_filtered_test))
    name_lin_reg = make_name("res_dtr", sw_tuple, n_features, arguments)
    res_dtr.to_csv(name_lin_reg + ".csv")

    print("rfr")
    res_rfr = ml_forecast.rand_forest_reg()
    res_rfr = pd.concat([res_rfr, y_test_before_filtered],
                        axis=1)
    r2_test, r2_filtered_test = evaluation(res_rfr)
    print('r2 test {}, r2 filtered test {}'.format(r2_test, r2_filtered_test))
    name_lin_reg = make_name("res_rfr", sw_tuple, n_features, arguments)
    res_rfr.to_csv(name_lin_reg + ".csv")

    print("gbr")
    res_gbr = ml_forecast.grad_boost_reg()
    res_gbr = pd.concat([res_gbr, y_test_before_filtered],
                        axis=1)
    r2_test, r2_filtered_test = evaluation(res_gbr)
    print('r2 test {}, r2 filtered test {}'.format(r2_test, r2_filtered_test))
    name_lin_reg = make_name("res_gbr", sw_tuple, n_features, arguments)
    res_gbr.to_csv(name_lin_reg + ".csv")

    # print("hgbr")
    # res_hgbr = ml_forecast.hist_grad_boost_reg()
    # res_hgbr = pd.concat([res_hgbr, y_test_before_filtered],
    #                      axis=1)
    # r2_test, r2_filtered_test = evaluation(res_hgbr)
    # print('r2 test {}, r2 filtered test {}'.format(r2_test,
    # r2_filtered_test))
    # name_lin_reg = make_name("res_hgbr", sw_tuple, n_features, arguments)
    # res_hgbr.to_csv(name_lin_reg + ".csv")


if __name__ == '__main__':
    path = arguments.data_path
    if arguments.without_epu:
        path.replace('_with_epu', '-without_epu')
    exogenous = pd.read_csv(path)
    exogenous = exogenous.set_index('date')
    exogenous.index = pd.DatetimeIndex(exogenous.index)

    try:
        os.mkdir('../results/')
    except Exception as e:
        print(e)

    if 'y_test' not in exogenous.columns:
        if 'crude_future' in exogenous.columns:
            exogenous = exogenous.rename(
                columns={'crude_future': 'y_test_filtered'})
        else:
            raise Exception("There must be a true data")
    else:
        if 'curde_future' in exogenous.columns:
            exogenous = exogenous.drop('crude_future', axis=1)

    for filter_method in ['not_none', 'none']:
        for n_features in [np.inf, 10]:
            for sw_tuple in [(45, 22), (15, 5)]:
                copied = copy.deepcopy(exogenous)
                main(copied, filter_method, n_features, sw_tuple)
