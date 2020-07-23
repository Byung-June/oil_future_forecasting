from oil_forecastor.model_selection import rolling_sample
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-path',
    default='../data/data_input_normalized_stationary.csv', type=str,
    help="path to data"
)
args = parser.parse_args()


def test_rolling_window(args):
    dataframe = pd.read_csv(args.data_path)
    X_train, X_test, y_train, y_test = rolling_sample(
        dataframe,
        window_num=5, sample_num=52
    )


if __name__ == '__main__':
    test_rolling_window(args)
