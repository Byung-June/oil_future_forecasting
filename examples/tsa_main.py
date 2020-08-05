# %%
import os
import sys
import time

import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from oil_forecastor.tsa_forecastor.tsa_models import *
from oil_forecastor.model_selection._utility import rolling_sample, get_features, adf_test


# %%

class GenModel:
    def __init__(self, data, window_num, sample_num, forecast_period, feature_num):
        self._data = data
        self._window_num = window_num
        self._sample_num = sample_num
        self._forecast_period = forecast_period
        self._feature_num = feature_num
        self._model = None

    def gen_model(self, model, process_num=4):
        self._model = model
        # list_test = range(self._window_num + self._sample_num - 2, len(self._data) - 1)
        list_test = range(self._window_num + self._sample_num - 2, self._window_num + self._sample_num + 2)
        pool = Pool(processes=process_num)
        result = pool.map(self.pool_func, tqdm(list_test))
        pool.close()
        pool.join()
        return result

    def pool_func(self, t_):
        x_train_, x_test_, y_train_, y_test_ = rolling_sample(
            self._data,
            window_num=self._window_num, sample_num=self._sample_num, time=t_
        )
        if self._model == 'arima':
            return arima(
                x_train_, x_test_, y_train_, y_test_,
                self._data.index[t_+1], self._forecast_period, self._feature_num
            )
        else:
            return t_


# %%
if __name__ == '__main__':
    data = pd.read_csv('../data/data_input_normalized_stationary.csv', index_col=0)
    # print(data)

    # arima test
    g = GenModel(data, window_num=5, sample_num=52, forecast_period=1, feature_num=2)
    arma = g.gen_model('arima')
    print('result', arma)
    arma2 = g.gen_model('arima', 2)
    print('result', arma2)
