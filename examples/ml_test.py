import numpy as np
import os
import copy
from oil_forecastor.ml_forecastor.forecast import MLForecast
from oil_forecastor.model_selection import denoising_func
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


def evaluation(test, pred, true_variation=True):
    if true_variation:
        return r2_score(test, pred)
    else:
        pred = pred.flatten()
        test = test.flatten()
        diff = (pred - test) ** 2
        numerator = diff.sum()
        denominator = (test - test.mean() ** 2).sum()
        r2_measure = 1 - numerator / denominator
        assert ~np.isnan(r2_measure.mean())
        return r2_measure


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
    path += "_" + filter_method + "_" + "without_epu_" + without_epu + ".npz"
    return path


def main(exogenous, filter_method, n_features, sw_tuple):
    # you are reusing exogenous....
    y_test_before_filtered = copy.deepcopy(exogenous['y_test'])
    if filter_method != 'none':
        filtered = denoising_func(exogenous, filter_method)
    else:
        filtered = exogenous

    n_samples, n_windows = sw_tuple
    start_time = n_windows + n_samples - 2
    end_time = len(filtered) - 1

    y_test_before_filtered = y_test_before_filtered[start_time:end_time].values
    if filter_method != 'none':
        y_test_filtered = filtered['y_test'][start_time:end_time].values
        print(
            evaluation(y_test_before_filtered, y_test_filtered)
        )

    ml_forecast = MLForecast(
        filtered, n_windows, n_samples, start_time, end_time)

    y_test_before_filtered = np.expand_dims(y_test_before_filtered, axis=-1)

    print("linear_reg")
    res_linear_reg = ml_forecast.linear_reg(
        n_features=n_features, method='f-classif'
    )
    np.savez(
        make_name("res_linear_reg", sw_tuple, n_features, arguments),
        np.concatenate([y_test_before_filtered, res_linear_reg], axis=-1)
    )
    print(evaluation(y_test_before_filtered, res_linear_reg))

    print("rfr")
    res_rfr = ml_forecast.rand_forest_reg()
    np.savez(
        make_name("res_rfr", sw_tuple, n_features, arguments),
        np.concatenate([y_test_before_filtered, res_rfr])
    )
    print(evaluation(y_test_before_filtered, res_rfr))

    print("lasso")
    res_lasso = ml_forecast.lasso(
        n_features=n_features, method='f-classif'
    )
    np.savez(
        make_name("res_lasso", sw_tuple, n_features, arguments),
        np.concatenate([y_test_before_filtered, res_lasso])
    )
    print(evaluation(y_test_before_filtered, res_lasso))

    print("svr")
    if n_features > 100:
        res_svr = ml_forecast.svr(n_features=50, method='f-classif')
    else:
        res_svr = ml_forecast.svr(n_features=n_features, method='f-classif')
    np.savez(
        make_name("res_svr", sw_tuple, n_features, arguments),
        np.concatenate([y_test_before_filtered, res_svr])
    )
    print(evaluation(y_test_before_filtered, res_svr))

    print("kr")
    if n_features > 100:
        res_kr = ml_forecast.kernel_ridge(n_features=50, method='f-classif')
    else:
        res_kr = ml_forecast.kernel_ridge(
            n_features=n_features, method='f-classif'
        )
    np.savez(
        make_name("res_kr", sw_tuple, n_features, arguments),
        np.concatenate([y_test_before_filtered, res_kr])
    )
    print(evaluation(y_test_before_filtered, res_kr))

    print("dtr")
    res_dtr = ml_forecast.decision_tree_reg(
        n_features=n_features, method='f-classif'
    )
    np.savez(
        make_name("res_dtr", sw_tuple, n_features, arguments),
        np.concatenate([y_test_before_filtered, res_dtr])
    )
    print(evaluation(y_test_before_filtered, res_dtr))

    print("gbr")
    res_gbr = ml_forecast.grad_boost_reg()
    np.savez(
        make_name("res_gbr", sw_tuple, n_features, arguments),
        np.concatenate([y_test_before_filtered, res_gbr])
    )
    print(evaluation(y_test_before_filtered, res_gbr))

    print("hgbr")
    res_hgbr = ml_forecast.hist_grad_boost_reg()
    np.savez(
        make_name("res_hgbr", sw_tuple, n_features, arguments),
        np.concatenate([y_test_before_filtered, res_hgbr])
    )
    print(evaluation(y_test_before_filtered, res_hgbr))

    print("pcr")
    res_pcr = ml_forecast.pcr()
    np.savez(
        make_name("res_pcr", sw_tuple, n_features, arguments),
        np.concatenate([y_test_before_filtered, res_pcr])
    )
    print(evaluation(y_test_before_filtered, res_pcr))


if __name__ == '__main__':
    path = arguments.data_path
    if arguments.without_epu:
        path.replace('_with_epu', '-without_epu')
    exogenous = pd.read_csv(path)
    exogenous = exogenous.set_index('date')

    try:
        os.mkdir('../results/')
    except Exception as e:
        print(e)

    if 'y_test' not in exogenous.columns:
        if 'crude_future' in exogenous.columns:
            exogenous = exogenous.rename(columns={'crude_future': 'y_test'})
        else:
            raise Exception("There must be a true data")
    else:
        if 'curde_future' in exogenous.columns:
            exogenous = exogenous.drop('crude_future', axis=1)

    for filter_method in ['moving_average', 'none', 'wavelet_db1']:
        for n_features in [np.inf, 10]:
            for sw_tuple in [(45, 22), (15, 5)]:
                copied = copy.deepcopy(exogenous)
                main(copied, filter_method, n_features, sw_tuple)
