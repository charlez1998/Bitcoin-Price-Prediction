# grid search lstm for airline passengers
from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import pandas as pd

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# transform list into supervised learning format
def series_to_supervised(data, n_in=1, n_out=1):
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    agg.dropna(inplace=True)
    return agg.values


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# difference dataset
def difference(data, order):
    return [data[i] - data[i - order] for i in range(order, len(data))]


# fit a model
def model_fit(train, config):
    # unpack config
    n_input, n_nodes, n_epochs, n_batch, n_diff = config
    # prepare data
    if n_diff > 0:
        train = difference(train, n_diff)
    # transform series into supervised format
    data = series_to_supervised(train, n_in=n_input)
    # separate inputs and outputs
    train_x, train_y = data[:, :-1], data[:, -1]
    # reshape input data into [samples, timesteps, features]
    n_features = 1
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit model
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# forecast with the fit model
def model_predict(model, history, config):
    # unpack config
    n_input, _, _, _, n_diff = config
    # prepare data
    correction = 0.0
    if n_diff > 0:
        correction = history[-n_diff]
        history = difference(history, n_diff)
    # reshape sample into [samples, timesteps, features]
    x_input = array(history[-n_input:]).reshape((1, n_input, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return correction + yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # fit model
    model = model_fit(train, cfg)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = model_predict(model, history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    print(' > %.3f' % error)
    return error


# score a model, return None on failure
def repeat_evaluate(data, config, n_test, n_repeats=10):
    # convert config to a key
    key = str(config)
    # fit and evaluate the model n times
    scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
    # summarize score
    result = mean(scores)
    print('> Model[%s] %.3f' % (key, result))
    return (key, result)


# grid search configs
def grid_search(data, cfg_list, n_test):
    # evaluate configs
    scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


# create a list of configs to try
def model_configs():
    # define scope of configs
    n_input = [12]
    n_nodes = [100]
    n_epochs = [50]
    n_batch = [150]
    n_diff = [12]
    # create configs
    configs = list()
    for i in n_input:
        for j in n_nodes:
            for k in n_epochs:
                for l in n_batch:
                    for m in n_diff:
                        cfg = [i, j, k, l, m]
                        configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs

# define dataset
dataframe = pd.read_csv('cleaned_data2.csv')
dataframe = dataframe.loc[:, 'Date': 'Open']
dataframe.set_index("Date", inplace = True)
data = dataframe.values
# data split
n_test = 724
# model configs
cfg_list = model_configs()
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 10 configs
for cfg, error in scores[:3]:
    print(cfg, error)

#Some Results from our grid search:
#For n_test = 724
# > Model[[12, 100, 50, 1, 12]] 3020.067
# > Model[[12, 100, 50, 150, 12]] 2067.875
# done
# [12, 100, 50, 150, 12] 2067.8748901339122
# [12, 100, 50, 1, 12] 3020.066932045372

# > Model[[12, 250, 50, 150, 12]] 2181.680
# > Model[[12, 500, 50, 150, 12]] 2505.534
# > Model[[12, 1500, 50, 150, 12]] 2750.448

#For n_test = 372
# > Model[[12, 50, 50, 1, 12]] 2713.215
# > Model[[12, 50, 50, 150, 12]] 2725.830
# > Model[[12, 50, 100, 1, 12]] 3473.338
# > Model[[12, 50, 100, 150, 12]] 2706.546
# > Model[[12, 100, 50, 1, 12]] 3618.940
# > Model[[12, 100, 50, 150, 12]] 2682.452
# > Model[[12, 100, 100, 1, 12]] 4493.516
# > Model[[12, 100, 100, 150, 12]] 2739.054
# done
# [12, 100, 50, 150, 12] 2682.451918210313
# [12, 50, 100, 150, 12] 2706.545733230764
# [12, 50, 50, 1, 12] 2713.2154904638023

#For n_test = 877
# > Model[[12, 100, 50, 150, 12]] 2211.491

#Our optimal hyperparameters are hence:
# n_input: 12
# n_nodes: 100
# n_epochs: 50
# n_batch: 150
# n_diff: 12