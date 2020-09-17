import numpy as np
import os
import copy
from oil_forecastor.ml_forecastor.forecast import MLForecast
import pandas as pd
import argparse
import warnings
from oil_forecastor.model_selection import denoising_func
from r2_oos import evaluation
import glob


parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-path',
    default='../data', type=str,
    help="path to data"
)
parser.add_argument(
    '--without-epu', default=False, action='store_true'
)
parser.add_argument('--filter-method',
                    default='none', type=str)
parser.add_argument('--ignore-warnings', default=True, action='store_false')
parser.add_argument('--use-unfiltered', default=False, action='store_true')
parser.add_argument('--plot-test-data', default=False, action='store_true')
parser.add_argument('--selected-inputs', default=True, action='store_false')
parser.add_argument('--prefilter', default=False, type=bool)
parser.add_argument('--n-windows', default=22, type=int)
parser.add_argument('--n-samples', default=45, type=int)
parser.add_argument('--selector', default='f-regression', type=str)
parser.add_argument('--scaler', default='none', type=str)
arguments = parser.parse_args()

if arguments.ignore_warnings:
    warnings.filterwarnings('ignore')


def make_name(name, csv_name, sw_tuple, n_features, args):
    n_samples, n_windows = sw_tuple
    without_epu = str(args.without_epu)
    filter_method = args.filter_method
    path = "../results/" + csv_name + '/'\
        + name + "_windows_" + str(n_windows)
    path += "_samples_" + str(n_samples)
    if n_features > 100:
        path += "_whole"
    else:
        path += "_" + str(n_features)
    path += "_" + filter_method + "_" + "without_epu_" + without_epu
    return path


def main(exogenous, filter_method, n_features, sw_tuple, csv_name,
         prefiltered=False):
    filtered = denoising_func(exogenous, filter_method)

    n_samples, n_windows = sw_tuple
    start_time = n_windows + n_samples - 2
    end_time = len(filtered) - 1

    ml_forecast = MLForecast(
        filtered, n_windows, n_samples, start_time, end_time, arguments.scaler)

    print("linear_reg")
    res_linear_reg = ml_forecast.linear_reg(
        n_features=n_features, method=arguments.selector
    )
    r2_test, r2_filtered_test = evaluation(res_linear_reg)
    print('r2 test {}, r2 zero return {}'.format(r2_test, r2_filtered_test))
    name_lin_reg = make_name("res_linear_reg", csv_name,
                             sw_tuple, n_features, arguments)
    res_linear_reg.to_csv(name_lin_reg + ".csv")

    print("lasso")
    res_lasso = ml_forecast.lasso(
        n_features=n_features, method=arguments.selector
    )
    r2_test, r2_filtered_test = evaluation(res_lasso)
    print('r2 test {}, r2 zero return {}'.format(r2_test, r2_filtered_test))
    name_lasso_reg\
        = make_name("res_lasso_reg", csv_name, sw_tuple, n_features, arguments)
    res_lasso.to_csv(name_lasso_reg + ".csv")

    print("pcr")
    res_pcr = ml_forecast.pcr()
    r2_test, r2_filtered_test = evaluation(res_pcr)
    print('r2 test {}, r2 zero return {}'.format(r2_test, r2_filtered_test))
    name_lin_reg = make_name("res_pcr", csv_name,
                             sw_tuple, n_features, arguments)
    res_pcr.to_csv(name_lin_reg + ".csv")

    print("svr")
    if n_features > 100:
        res_svr = ml_forecast.svr(n_features=50, method=arguments.selector)
    else:
        res_svr = ml_forecast.svr(n_features=n_features,
                                  method=arguments.selector)
    r2_test, r2_filtered_test = evaluation(res_svr)
    print('r2 test {}, r2 zero return {}'.format(r2_test, r2_filtered_test))
    name_lin_reg = make_name("res_svr", csv_name,
                             sw_tuple, n_features, arguments)
    res_svr.to_csv(name_lin_reg + ".csv")

    print("kr")
    if n_features > 100:
        res_kr = ml_forecast.kernel_ridge(n_features=50,
                                          method=arguments.selector)
    else:
        res_kr = ml_forecast.kernel_ridge(
            n_features=n_features, method=arguments.selector
        )
    r2_test, r2_filtered_test = evaluation(res_kr)
    print('r2 test {}, r2 zero return {}'.format(r2_test, r2_filtered_test))
    name_lin_reg = make_name("res_kr", csv_name,
                             sw_tuple, n_features, arguments)
    res_kr.to_csv(name_lin_reg + ".csv")

    print("dtr")
    res_dtr = ml_forecast.decision_tree_reg(
        n_features=n_features, method=arguments.selector
    )
    r2_test, r2_filtered_test = evaluation(res_dtr)
    print('r2 test {}, r2 zero return {}'.format(r2_test, r2_filtered_test))
    name_lin_reg = make_name("res_dtr", csv_name,
                             sw_tuple, n_features, arguments)
    res_dtr.to_csv(name_lin_reg + ".csv")

    print("rfr")
    res_rfr = ml_forecast.rand_forest_reg()
    r2_test, r2_filtered_test = evaluation(res_rfr)
    print('r2 test {}, r2 zero return {}'.format(r2_test, r2_filtered_test))
    name_lin_reg = make_name("res_rfr", csv_name,
                             sw_tuple, n_features, arguments)
    res_rfr.to_csv(name_lin_reg + ".csv")

    print("gbr")
    res_gbr = ml_forecast.grad_boost_reg()
    r2_test, r2_filtered_test = evaluation(res_gbr)
    print('r2 test {}, r2 zero return {}'.format(r2_test, r2_filtered_test))
    name_lin_reg = make_name("res_gbr", csv_name,
                             sw_tuple, n_features, arguments)
    res_gbr.to_csv(name_lin_reg + ".csv")

    # print("hgbr")
    # res_hgbr = ml_forecast.hist_grad_boost_reg()
    # res_hgbr = pd.concat([res_hgbr, y_test_no_prefilter],
    #                      axis=1)
    # r2_test, r2_filtered_test = evaluation(res_hgbr)
    # print('r2 test {}, r2 zero return {}'.format(r2_test,
    # r2_filtered_test))
    # name_lin_reg = make_name("res_hgbr", sw_tuple, n_features, arguments)
    # res_hgbr.to_csv(name_lin_reg + ".csv")


if __name__ == '__main__':
    path = arguments.data_path
    if arguments.without_epu:
        path.replace('_with_epu', '-without_epu')

    paths = glob.glob(arguments.data_path + '/*.csv')
    for path in paths:
        exogenous = pd.read_csv(path)
        exogenous = exogenous.set_index('date')
        exogenous.index = pd.DatetimeIndex(exogenous.index)
        exo_dropcolumn = [elt for elt in exogenous.columns if 'Unnamed' in elt]
        exogenous = exogenous.drop(columns=exo_dropcolumn)

        try:
            os.makedirs('../results/'
                        + os.path.basename(path).replace('.csv', ''))
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

        for filter_method in [arguments.filter_method]:
            for n_features in [np.inf, 50, 10]:
                for sw_tuple in [(arguments.n_samples, arguments.n_windows),
                                 (15, 5)]:
                    copied = copy.deepcopy(exogenous)
                    main(copied, filter_method, n_features, sw_tuple,
                         os.path.basename(path).replace('.csv', ''))
