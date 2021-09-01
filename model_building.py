import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_data2.csv")
#df = df.groupby([pd.Grouper(key='Date', freq='D')]).first().reset_index()
df = df.set_index('Date')
df = df[['Adj Close']]

# split data
split_date = '2020-09-27'
df_train = df.loc[df.index <= split_date].copy()
df_test = df.loc[df.index > split_date].copy()

# Data preprocess
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))

color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
price_plot = df.plot(style='', figsize=(15,5), color=color_pal[0], title='BTC Weighted_Price Price (USD) by Hours')
plt.show()

# _ = df_test \
#     .rename(columns={'Weighted_Price': 'Test Set'}) \
#     .join(df_train.rename(columns={'Weighted_Price': 'Training Set'}), how='outer') \
#     .plot(figsize=(15,5), title='BTC Weighted_Price Price (USD) by Hours', style='')
# plt.show()

combined_plot = df_test.rename(columns={'Adj Close': 'Test Set'}).join(df_train.rename(columns={'Adj Close': 'Training Set'}), how='outer').plot(figsize=(15,5), title='BTC Weighted_Price Price (USD) by Hours', style='')
plt.show()
