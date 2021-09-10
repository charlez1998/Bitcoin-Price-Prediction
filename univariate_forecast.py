import pandas as pd
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt

#Total COVID confirmed cases
df = pd.read_csv('cleaned_data2.csv')
df = df.loc[:, 'Date': 'Open']
#df.set_index("Date", inplace = True)

#Use data until 14 days before as training
x = len(df)-12

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
n_steps_in, n_steps_out = 12, 7
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, 1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(loss='mse', optimizer='adam')
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
original_time_array = test.index

#Add new dates for the forecast period
for k in range(0, n_steps_out):
    time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

# Create a dataframe to capture the forecast data
df_forecast = pd.DataFrame(columns=["Predicted"], index=time_series_array)
df_forecast["Predicted"] = np.concatenate((original_test, yhat_final), axis = None)

df_true = pd.DataFrame(columns=["Actual Price"], index=original_time_array)
df_true["True Price"] = test.values

#Plot
#df_forecast.plot(title="Predictions for next 7 days")

import seaborn as sns
sns.lineplot(x = df_forecast.index, y= df_forecast['Predicted'])
sns.lineplot(x =df_true.index, y= df_true['True Price'])
plt.show()