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
    return numerator / denominator


if __name__ == '__main__':
    exogenous = pd.read_csv(args.data_path)

    ml_forecast = MLForecast(exogenous, 5, 52)
    res_linear_reg = ml_forecast.linear_reg()
    res_lasso = ml_forecast.lasso()
    res_dtr = ml_forecast.decision_tree_reg()
    res_gbr = ml_forecast.grad_boost_reg()
    res_hgbr = ml_forecast.hist_grad_boost_reg()
    res_pcr = ml_forecast.pcr()
    res_rfr = ml_forecast.rand_forest_reg()
