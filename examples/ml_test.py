import numpy as np
from oil_forecastor.ml_forecastor.forecast import MLForecast
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-path',
    default='../data/data_input_normalized_stationary.csv', type=str,
    help="path to data"
)
args = parser.parse_args()


def evaluation(res):
    pred = res[:, 0]
    test = res[:, 1]
    diff = (pred - test) ** 2
    numerator = diff.sum()
    denominator = (test ** 2).sum()
    return 1 - numerator / denominator


if __name__ == '__main__':
    exogenous = pd.read_csv(args.data_path)

    ml_forecast = MLForecast(exogenous, 5, 52)
    res_linear_reg = ml_forecast.linear_reg()
    np.savez("res_linear_reg.npz", res_linear_reg)
    print(evaluation(res_linear_reg))
    res_lasso = ml_forecast.lasso()
    np.savez("res_lasso.npz", res_lasso)
    print(evaluation(res_lasso))
    res_dtr = ml_forecast.decision_tree_reg()
    np.savez("res_dtr.npz", res_dtr)
    print(evaluation(res_dtr))
    res_gbr = ml_forecast.grad_boost_reg()
    np.savez("res_gbr.npz", res_gbr)
    print(evaluation(res_gbr))
    res_hgbr = ml_forecast.hist_grad_boost_reg()
    np.savez("res_hgbr.npz", res_hgbr)
    print(evaluation(res_hgbr))
    res_pcr = ml_forecast.pcr()
    np.savez("res_pcr.npz", res_pcr)
    print(evaluation(res_pcr))
    res_rfr = ml_forecast.rand_forest_reg()
    np.savez("res_rfr.npz", res_rfr)
    print(evaluation(res_rfr))
    res_svr = ml_forecast.svr(n_features=20, method='f-classif')
    np.savez("res_svr.npz", res_svr)
    print(evaluation(res_svr))
    res_kr = ml_forecast.kr(n_features=20, method='f-classif')
    np.savez("res_kr.npz", res_kr)
    print(evaluation(res_kr))
