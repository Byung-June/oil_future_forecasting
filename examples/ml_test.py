import numpy as np
from oil_forecastor.ml_forecastor.forecast import MLForecast
import pandas as pd
import argparse
import warnings

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-path',
    default='../data/ml_data_input_normalized_stationary.csv', type=str,
    help="path to data"
)
parser.add_argument('--ignore-warnings', default=True, action='store_false')
parser.add_argument('--n-windows', default=5, type=int)
parser.add_argument('--n-samples', default=52, type=int)
args = parser.parse_args()

if args.ignore_warnings:
    warnings.filterwarnings('ignore')


def evaluation(res):
    pred = res[:, 0]
    test = res[:, 1]
    diff = (pred - test) ** 2
    numerator = diff.sum()
    denominator = (test ** 2).sum()
    return 1 - numerator / denominator


def make_name(name):
    path = "../results/" + name + "_" + str(args.n_windows)
    path += "_" + str(args.n_samples) + ".npz"
    return path


if __name__ == '__main__':
    exogenous = pd.read_csv(args.data_path)
    exogenous = exogenous.set_index('date')

    ml_forecast = MLForecast(exogenous, args.n_windows, args.n_samples)
    res_linear_reg = ml_forecast.linear_reg()
    np.savez(make_name("res_linear_reg"), res_linear_reg)
    print(evaluation(res_linear_reg))
    res_lasso = ml_forecast.lasso()
    np.savez(make_name("res_lasso"), res_lasso)
    print(evaluation(res_lasso))
    res_svr = ml_forecast.svr(n_features=20, method='f-classif')
    np.savez(make_name("res_svr"), res_svr)
    print(evaluation(res_svr))
    res_kr = ml_forecast.kernel_ridge(n_features=20, method='f-classif')
    np.savez(make_name("res_kr"), res_kr)
    print(evaluation(res_kr))
    res_dtr = ml_forecast.decision_tree_reg()
    np.savez(make_name("res_dtr"), res_dtr)
    print(evaluation(res_dtr))
    res_gbr = ml_forecast.grad_boost_reg()
    np.savez(make_name("res_gbr"), res_gbr)
    print(evaluation(res_gbr))
    res_hgbr = ml_forecast.hist_grad_boost_reg()
    np.savez(make_name("res_hgbr"), res_hgbr)
    print(evaluation(res_hgbr))
    res_pcr = ml_forecast.pcr()
    np.savez(make_name("res_pcr"), res_pcr)
    print(evaluation(res_pcr))
    res_rfr = ml_forecast.rand_forest_reg()
    np.savez(make_name("res_rfr"), res_rfr)
    print(evaluation(res_rfr))
