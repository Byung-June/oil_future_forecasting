import pandas as pd


def r2_oos_func(data_):
    data_ = pd.read_csv(data_, index_col=1)
    data_ = data_.drop(columns=['Unnamed: 0'])
    pred = data_.iloc[:, 0]
    y = data_.iloc[:, 1]
    r2_oos = 1 - sum(pow(pred-y, 2)) / sum(pow(y, 2))
    # print(sum(pow(pred-y, 2)))
    # print(sum(pow(y-data_.iloc[:, 1].mean(), 2)))
    return r2_oos, data_


if __name__=='__main__':
    # data = pd.read_csv('arima_5_52_10.csv', index_col=1)
    # data = data.drop(columns=['Unnamed: 0'])
    # print(data)

    r2_oos, data = r2_oos_func('arima_5_52_0.csv')
    print(r2_oos)