# %%
from tqdm import tqdm
from multiprocessing import Pool

from oil_forecastor.tsa_forecastor.tsa_models import *
from oil_forecastor.model_selection._utility import rolling_train_test_split, denoising_func


# %%

class GenModel:
    def __init__(self, data, window_num, sample_num, forecast_period, feature_num, denoise=None):
        """
        :param data:
        :param window_num:
        :param sample_num:
        :param forecast_period:
        :param feature_num: feature selection number
        :param denoise:
             'wavelet_db1',
             'wavelet_db2',
             'bilateral',
             'moving_average'
        """
        self._data = denoising_func(data, filter=denoise)
        self._window_num = window_num
        self._sample_num = sample_num
        self._forecast_period = forecast_period
        self._feature_num = feature_num
        self._model = None

    def gen_model(self, model, process_num=4):
        """
        :param model: model that you want to run
        :param process_num: number of process
        :return:
        """
        self._model = model
        list_test = range(self._window_num + self._sample_num - 2, len(self._data) - 1)
        # list_test = range(self._window_num + self._sample_num - 2, self._window_num + self._sample_num + 6)
        pool = Pool(processes=process_num)
        result = pool.map(self.pool_func, tqdm(list_test))
        pool.close()
        pool.join()

        return result

    def pool_func(self, t_):
        """
        :param t_: time spot for each analysis
        :return: single time-series analysis result,
            including forecasting, real value, params of the model
        """
        garch_models = ['arx-garch', 'arx-gjr-garch', 'arx-tgarch']

        x_train_, x_test_, y_train_, y_test_ = rolling_train_test_split(
            self._data,
            window_num=self._window_num, sample_num=self._sample_num, time=t_
        )
        if self._model == 'arima':
            return arima(
                x_train_, x_test_, y_train_, y_test_,
                self._data.index[t_+1], self._forecast_period, self._feature_num
            )

        elif self._model in garch_models:
            return garch(
                x_train_, x_test_, y_train_, y_test_,
                self._data.index[t_ + 1], self._forecast_period, self._feature_num,
                model=self._model
            )

        else:
            return t_


# %%
if __name__ == '__main__':
    data = pd.read_csv('../data/data_input_normalized_stationary.csv', index_col=0)
    # print(data)
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # arima test
    input_w = 5
    input_s = 15
    input_f = 10
    g = GenModel(data, window_num=input_w, sample_num=input_s, forecast_period=1, feature_num=input_f,
                 denoise='wavelet_db1')
    arma = pd.DataFrame(g.gen_model('arima', process_num=4))
    arma.to_csv('db1_arima_%s_%s_%s.csv' % (input_w, input_s, input_f))
    print('result', arma)

    # arima test
    input_w = 22
    input_s = 45
    input_f = 10
    g = GenModel(data, window_num=input_w, sample_num=input_s, forecast_period=1, feature_num=input_f,
                 denoise='wavelet_db1')
    arma = pd.DataFrame(g.gen_model('arima', process_num=4))
    arma.to_csv('db1_arima_%s_%s_%s.csv' % (input_w, input_s, input_f))
    print('result', arma)

    # arima test
    input_w = 5
    input_s = 15
    input_f = 10
    g = GenModel(data, window_num=input_w, sample_num=input_s, forecast_period=1, feature_num=input_f,
                 denoise='moving_average')
    arma = pd.DataFrame(g.gen_model('arima', process_num=4))
    arma.to_csv('ma5_arima_%s_%s_%s.csv' % (input_w, input_s, input_f))
    print('result', arma)

    # arima test
    input_w = 22
    input_s = 45
    input_f = 10
    g = GenModel(data, window_num=input_w, sample_num=input_s, forecast_period=1, feature_num=input_f,
                 denoise='moving_average')
    arma = pd.DataFrame(g.gen_model('arima', process_num=4))
    arma.to_csv('ma5_arima_%s_%s_%s.csv' % (input_w, input_s, input_f))
    print('result', arma)