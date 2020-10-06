import numpy as np
import os
import copy
from oil_forecastor.ml_forecastor.forecast import MLForecast
import pandas as pd
import argparse
import warnings
from oil_forecastor.model_selection import denoising_func
from r2_oos import evaluation
import sys
import glob
from sklearn.exceptions import ConvergenceWarning

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-path',
    default='../data', type=str,
    help="path to data"
)
parser.add_argument('--without-epu', type=bool)
parser.add_argument('--filter-method',
                    default='none', type=str)
parser.add_argument('--ignore-warnings', default=True, type=bool)
parser.add_argument('--use-unfiltered', default=False, action='store_true')
parser.add_argument('--plot-test-data', default=False, action='store_true')
parser.add_argument('--selected-inputs', default=True, action='store_false')
parser.add_argument('--prefilter', default=False, type=bool)
parser.add_argument('--n-windows', default=1, type=int)
parser.add_argument('--n-samples', default=52, type=int)
parser.add_argument('--selector', default='f-regression', type=str)
parser.add_argument('--scaler', default='none', type=str)
parser.add_argument('--true-path',
                    default='../data/your_path.csv', type=str,
                    help='path to the data which is unfiltered')
parser.add_argument('--n-columns',
                    default=3, type=int,
                    help='# of first n-columns that must be included')
parser.add_argument('--run-arima', default=False, type=bool)
arguments = parser.parse_args()

if arguments.scaler != 'none':
    print("The model uses scaler. The performance could drop. Are you sure?")
    get_str = input().lower()
    if get_str in ['yes', 'y']:
        print("Using {} scaler".format(arguments.scaler))
    else:
        print("Terminate")
        sys.exit()
else:
    print("Proceed without scaler")


if arguments.ignore_warnings:
    print("Ignore sklearn warnings")
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
else:
    print("Do not ignore sklearn warnings")


def make_name(name, csv_name, sw_tuple, n_features, args):
    n_samples, n_windows = sw_tuple
    without_epu = str(args.without_epu)
    filter_method = args.filter_method
    path = "../results/" + csv_name + '/'\
        + name + "_windows_" + str(n_windows)
    path += "_samples_" + str(n_samples)
    path += "_scaler_" + args.scaler
    if n_features > 100:
        path += "_whole"
    else:
        path += "_features_" + str(n_features)
    path += "_" + filter_method + "_" + "without_epu_" + without_epu
    return path


def main(exogenous, filter_method, n_features, sw_tuple, csv_name,
         y_true=None):
    filtered = denoising_func(exogenous, filter_method)

    n_samples, n_windows = sw_tuple
    start_time = n_windows + n_samples - 2
    end_time = len(filtered) - 1

    ml_forecast = MLForecast(
        filtered, n_windows, n_samples, start_time, end_time,
        arguments.scaler, arguments.n_columns)

    print("linear_reg")
    res_linear_reg = ml_forecast.linear_reg(n_features=n_features,
                                            method=arguments.selector)
    r2_test, r2_true, mape = evaluation(res_linear_reg, y_true)
    print('r2 test {}, r2 true {}, mape {}'.format(r2_test, r2_true, mape))
    name_lin_reg = make_name("res_linear_reg", csv_name,
                             sw_tuple, n_features, arguments)
    res_linear_reg.to_csv(name_lin_reg + ".csv")

    if arguments.run_arima:
        print("arima")
        res_pipe = ml_forecast.arima(n_features=n_features,
                                     method=arguments.selector)
        r2_test, r2_true, mape = evaluation(res_pipe, y_true)
        print('r2 test {}, r2 true {}, mape {}'.format(r2_test, r2_true, mape))
        name_lin_reg = make_name("arima", csv_name,
                                 sw_tuple, n_features, arguments)
        res_pipe.to_csv(name_lin_reg + ".csv")

    print("lasso")
    res_lasso = ml_forecast.lasso(n_features=n_features,
                                  method=arguments.selector)
    r2_test, r2_true, mape = evaluation(res_lasso, y_true)
    print('r2 test {}, r2 true {}, mape {}'.format(r2_test, r2_true, mape))
    name_lasso_reg\
        = make_name("res_lasso_reg", csv_name, sw_tuple, n_features, arguments)
    res_lasso.to_csv(name_lasso_reg + ".csv")

    print("pcr")
    res_pcr = ml_forecast.pcr(n_features=n_features,
                              method=arguments.selector)
    r2_test, r2_true, mape = evaluation(res_pcr, y_true)
    print('r2 test {}, r2 true {}, mape {}'.format(r2_test, r2_true, mape))
    name_lin_reg = make_name("res_pcr", csv_name,
                             sw_tuple, n_features, arguments)
    res_pcr.to_csv(name_lin_reg + ".csv")

    print("dtr")
    res_dtr = ml_forecast.decision_tree_reg(n_features=n_features,
                                            method=arguments.selector)
    r2_test, r2_true, mape = evaluation(res_dtr, y_true)
    print('r2 test {}, r2 true {}, mape {}'.format(r2_test, r2_true, mape))
    name_lin_reg = make_name("res_dtr", csv_name,
                             sw_tuple, n_features, arguments)
    res_dtr.to_csv(name_lin_reg + ".csv")

    print("rfr")
    res_rfr = ml_forecast.rand_forest_reg(n_features=n_features,
                                          method=arguments.selector)
    r2_test, r2_true, mape = evaluation(res_rfr, y_true)
    print('r2 test {}, r2 true {}, mape {}'.format(r2_test, r2_true, mape))
    name_lin_reg = make_name("res_rfr", csv_name,
                             sw_tuple, n_features, arguments)
    res_rfr.to_csv(name_lin_reg + ".csv")

    print("gbr")
    res_gbr = ml_forecast.grad_boost_reg(n_features=n_features,
                                         method=arguments.selector)
    r2_test, r2_true, mape = evaluation(res_gbr, y_true)
    print('r2 test {}, r2 true {}, mape {}'.format(r2_test, r2_true, mape))
    name_lin_reg = make_name("res_gbr", csv_name,
                             sw_tuple, n_features, arguments)
    res_gbr.to_csv(name_lin_reg + ".csv")


if __name__ == '__main__':
    paths = glob.glob(arguments.data_path + '/*.csv')
    for path in paths:
        arguments.true_path = path
        y_true = pd.read_csv(arguments.true_path)
        y_true = y_true.set_index('date')
        y_true.index = pd.DatetimeIndex(y_true.index)
        y_true = y_true['y_test']
        y_true = y_true.rename('y_true')

        exogenous = pd.read_csv(path)
        exogenous = exogenous.set_index('date')
        exogenous.index = pd.DatetimeIndex(exogenous.index)
        exo_dropcolumn = [elt for elt in exogenous.columns if 'Unnamed' in elt]
        exogenous = exogenous.drop(columns=exo_dropcolumn)
        try:
            dropcolumns = ['crude_future_daily_lag1',
                           'crude_future_daily_lag2',
                           'crude_future_daily_lag3',
                           'crude_future_daily_lag4']
            exogenous = exogenous.drop(columns=dropcolumns)
        except Exception as e:
            print(e)
        assert exogenous.columns[0] == 'y_test'
        assert exogenous.columns[1] == 'crude_future_daily_lag0'

        if '_no_Q_W' in path:
            print("Without quarterly")
            arguments.n_columns = 2
        else:
            print("With quaterly")
            arguments.n_columns = 3

        if arguments.n_columns == 2:
            exogenous.columns[2] == "crude_oil_realized_M"
            exogenous.columns[3] != "crude_oil_realized_Q"
        elif arguments.n_columns == 3:
            exogenous.columns[2] == "crude_oil_realized_M"
            exogenous.columns[3] == "crude_oil_realized_Q"
        elif arguments.n_columns > 3:
            raise ValueError("Too many fixed columns ", arguments.n_columns)
        else:
            print("Using only one fixed ", arguments.n_columns)

        crude_future_columns = [elt for elt in exogenous.columns
                                if 'crude_future' in elt]
        if len(crude_future_columns) > 1:
            raise Exception("data has same values"
                            "delete crude_future_lag_n, n > 0")

        if arguments.true_path != 'none':
            assert len(exogenous) == len(y_true)
            for i in range(len(exogenous)):
                assert exogenous.index[i] == y_true.index[i]

        try:
            os.makedirs('../results/'
                        + os.path.basename(path).replace('.csv', ''))
        except Exception as e:
            print(e)

        for filter_method in [arguments.filter_method]:
            for sw_tuple in [(arguments.n_samples, arguments.n_windows)]:
                for n_features in [np.inf, 0, 10, 20]:
                    copied = copy.deepcopy(exogenous)
                    main(copied, filter_method, n_features, sw_tuple,
                         os.path.basename(path).replace('.csv', ''), y_true)
