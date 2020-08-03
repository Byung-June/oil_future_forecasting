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


if __name__ == '__main__':
    exogenous = pd.read_csv(args.data_path)

    ml_forecast = MLForecast(exogenous, 5, 52)
    ml_forecast.linear_reg()
