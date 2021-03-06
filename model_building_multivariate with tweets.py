from math import sqrt
import pandas as pd
from numpy import concatenate
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
	# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
	    cols.append(df.shift(i))
	    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
	    cols.append(df.shift(-i))
	    if i == 0:
		    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
	    else:
		    names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
	# drop rows with NaN values
    if dropnan:
	    agg.dropna(inplace=True)
    return agg

# load dataset
df = pd.read_csv('cleaned_data2.csv', index_col = 1)
df.drop(columns = ['Unnamed: 0', "Volume", "Marketcap", "Proportion Traded"], inplace = True)
values = df.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[7,8,9,10,11]], axis=1, inplace=True)

values = reframed.values
train = values[:-372, :]
test = values[-372:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(1500, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

#For Visualization:

index_reset = df.reset_index(level=0)
grab_dates = index_reset['Date'][-372:]

dates = pd.DataFrame(grab_dates).reset_index().drop(columns = ['index'])
price_actual = pd.DataFrame({"Open Price": inv_y})
price_prediction = pd.DataFrame({'Open Price Prediction' : inv_yhat})

actual_df = dates.join(price_actual)
actual_df['Date'] = pd.to_datetime(actual_df['Date'])
prediction_df = dates.join(price_prediction)
prediction_df['Date'] = pd.to_datetime(prediction_df['Date'])

sns.lineplot(actual_df['Date'], actual_df['Open Price'])
sns.lineplot(prediction_df['Date'], prediction_df['Open Price Prediction'])
plt.show()