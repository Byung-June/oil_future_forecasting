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
    default='../data/ml_data_input_normalized_stationary.csv', type=str,
    help="path to data"
)
parser.add_argument('--filter-method', default='moving_average', type=str)
parser.add_argument('--ignore-warnings', default=True, action='store_false')
parser.add_argument('--n-windows', default=5, type=int)
parser.add_argument('--n-samples', default=52, type=int)
parser.add_argument('--use-unfiltered', default=False, action='store_true')
parser.add_argument('--plot-test-data', default=False, action='store_true')
args = parser.parse_args()

if args.ignore_warnings:
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


def make_name(name):
    path = "../results/" + name + "_" + str(args.n_windows)
    path += "_" + str(args.n_samples) + ".npz"
    return path


if __name__ == '__main__':
    exogenous = pd.read_csv(args.data_path)
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

    y_test_before_filtered = copy.deepcopy(exogenous['y_test'])
    if args.filter_method != 'none':
        filtered = denoising_func(exogenous, args.filter_method)
        exogenous = filtered

    start_time = args.n_windows + args.n_samples - 2
    end_time = len(exogenous) - 1

    y_test_before_filtered = y_test_before_filtered[start_time:end_time].values
    if args.filter_method != 'none':
        y_test_filtered = exogenous['y_test'][start_time:end_time].values
        if args.plot_test_data:
            from matplotlib import pyplot as plt
            plt.plot(y_test_filtered)
            plt.show()
            plt.plot(y_test_before_filtered)
            plt.show()
        print(
            evaluation(y_test_before_filtered, y_test_filtered)
        )

    ml_forecast = MLForecast(
        exogenous, args.n_windows, args.n_samples, start_time, end_time)
    res_linear_reg = ml_forecast.linear_reg()
    np.savez(make_name("res_linear_reg"), res_linear_reg)
    print(evaluation(y_test_before_filtered, res_linear_reg))
    res_lasso = ml_forecast.lasso()
    np.savez(make_name("res_lasso"), res_lasso)
    print(evaluation(y_test_before_filtered, res_lasso))
    res_svr = ml_forecast.svr(n_features=20, method='f-classif')
    np.savez(make_name("res_svr"), res_svr)
    print(evaluation(y_test_before_filtered, res_svr))
    res_kr = ml_forecast.kernel_ridge(n_features=20, method='f-classif')
    np.savez(make_name("res_kr"), res_kr)
    print(evaluation(y_test_before_filtered, res_kr))
    res_dtr = ml_forecast.decision_tree_reg()
    np.savez(make_name("res_dtr"), res_dtr)
    print(evaluation(y_test_before_filtered, res_dtr))
    res_gbr = ml_forecast.grad_boost_reg()
    np.savez(make_name("res_gbr"), res_gbr)
    print(evaluation(y_test_before_filtered, res_gbr))
    res_hgbr = ml_forecast.hist_grad_boost_reg()
    np.savez(make_name("res_hgbr"), res_hgbr)
    print(evaluation(y_test_before_filtered, res_hgbr))
    res_pcr = ml_forecast.pcr()
    np.savez(make_name("res_pcr"), res_pcr)
    print(evaluation(y_test_before_filtered, res_pcr))
    res_rfr = ml_forecast.rand_forest_reg()
    np.savez(make_name("res_rfr"), res_rfr)
    print(evaluation(y_test_before_filtered, res_rfr))
