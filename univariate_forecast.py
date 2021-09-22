import pandas as pd
import numpy as np
from numpy import array

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Total COVID confirmed cases
df = pd.read_csv('cleaned_data2.csv')
df = df.loc[:, 'Date': 'Open']

#Use data until 12 days before as training
x = len(df)-1
train=df.iloc[:x]
train.Date = pd.to_datetime(train.Date)
train.set_index("Date", inplace = True)
test = df.iloc[x:]
test.Date = pd.to_datetime(test.Date)
test.set_index("Date", inplace = True)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# define input sequence
raw_seq = train["Open"].tolist()
# choose a number of time steps
n_steps_in, n_steps_out =1, 31
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
# model = Sequential()
# model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, 1)))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(n_steps_out))
# model.compile(loss='mse', optimizer='adam')

model = Sequential()
model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(n_steps_in, n_features)))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit
model.fit(X, y, epochs=50, batch_size=72, verbose=0)
# demonstrate prediction
x_input = test['Open'].to_numpy()
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

#For Visualization:
original_test = test['Open'].to_numpy()
yhat_final = yhat.reshape(n_steps_out,)
time_series_array = test.index  #Get dates for test data

#Add new dates for the forecast period
for k in range(0, n_steps_out):
    time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

# Create a dataframe to capture the forecast data
df_forecast = pd.DataFrame(columns=["Predicted Price (USD)"], index=time_series_array)
df_forecast["Predicted Price (USD)"] = np.concatenate((original_test, yhat_final), axis = None)

#Plot
original = df[['Date', 'Open']]
original['Date']=pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '2021-6-1']

plt.title("Predicted Bitcoin Price for the next month", fontsize =15)
#sns.lineplot(x = df_forecast.index, y= df_forecast['Predicted Price (USD)'])
sns.lineplot(x =original['Date'], y= original['Open'])
#plt.show()

#Forecast Evaluation:
# true_values = {'Date':["2021-08-22", "2021-08-23", "2021-08-24", "2021-08-25", "2021-08-26", "2021-08-27" ,"2021-08-28"],
#                'Open':[48869.10547, 49291.675781, 49562.347656, 47727.257813, 49002.640625, 46894.554688, 49072.585938]}
#
# future_true = pd.DataFrame(true_values)
# future_true.Date = pd.to_datetime(future_true.Date)
# future_true.set_index('Date', inplace = True)
#
# only_last_seven = df_forecast[-7:]
#
# rmse = np.sqrt(mean_squared_error(future_true, only_last_seven))
# mae = mean_absolute_error(future_true, only_last_seven)
# print(rmse)
# print(mae)

df_future_month = pd.read_csv("august_to_september.csv")
df_future_month = df_future_month.loc[:, 'Date': 'Open']
df_future_month.Date = pd.to_datetime(df_future_month.Date)
df_future_month.set_index('Date', inplace = True)

only_last_31 = df_forecast[-31:]

sns.lineplot(x =df_future_month.index, y= df_future_month['Open'])
sns.lineplot(x =only_last_31.index, y= only_last_31['Predicted Price (USD)'])
plt.show()

rmse = np.sqrt(mean_squared_error(df_future_month, only_last_31))
mae = mean_absolute_error(df_future_month, only_last_31)
print(rmse)
print(mae)



#test out ins and outs above


