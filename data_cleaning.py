# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 18:59:59 2021

@author: charlie
"""

import pandas as pd
import numpy as np
from data_collection import scrape

price_df = pd.read_csv("btc_3yrs.csv")
tweet_df = scrape("https://bitinfocharts.com/comparison/bitcoin-tweets.html")
mc_df = scrape("https://bitinfocharts.com/comparison/bitcoin-marketcap.html#3y")

#check for all null values
print(price_df.isnull().sum())
print(tweet_df.isnull().sum())
print(mc_df.isnull().sum())

print(tweet_df[pd.isnull(tweet_df['Tweets'])])
print(mc_df[pd.isnull(mc_df['Marketcap'])])

#Finding indices of missing rows:
np.where(pd.isnull(price_df))

#By filling out the data for the dates of these indexes using coinmarketcap.com:


price_df.iloc[605, price_df.columns.get_loc('Open')] = 7116.55
price_df.iloc[605, price_df.columns.get_loc('High')] = 7167.18
price_df.iloc[605, price_df.columns.get_loc('Low')] = 7050.33
price_df.iloc[605, price_df.columns.get_loc('Close')] = 7096.18
price_df.iloc[605, price_df.columns.get_loc('Adj Close')] = 7096.18
price_df.iloc[605, price_df.columns.get_loc('Volume')] = 32513423567

price_df.iloc[780, price_df.columns.get_loc('Open')] = 10927.91
price_df.iloc[780, price_df.columns.get_loc('High')] = 11102.67
price_df.iloc[780, price_df.columns.get_loc('Low')] = 10846.85
price_df.iloc[780, price_df.columns.get_loc('Close')] = 11064.46
price_df.iloc[780, price_df.columns.get_loc('Adj Close')] = 11064.46
price_df.iloc[780, price_df.columns.get_loc('Volume')] = 22799117613

price_df.iloc[783, price_df.columns.get_loc('Open')] = 11392.64
price_df.iloc[783, price_df.columns.get_loc('High')] = 11698.47
price_df.iloc[783, price_df.columns.get_loc('Low')] = 11240.69
price_df.iloc[783, price_df.columns.get_loc('Close')] = 11555.36
price_df.iloc[783, price_df.columns.get_loc('Adj Close')] = 11555.36
price_df.iloc[783, price_df.columns.get_loc('Volume')] = 26163972642

price_df.iloc[784, price_df.columns.get_loc('Open')] = 11548.72
price_df.iloc[784, price_df.columns.get_loc('High')] = 11548.98
price_df.iloc[784, price_df.columns.get_loc('Low')] = 11321.22
price_df.iloc[784, price_df.columns.get_loc('Close')] = 11425.90
price_df.iloc[784, price_df.columns.get_loc('Adj Close')] = 11425.90
price_df.iloc[784, price_df.columns.get_loc('Volume')] = 24241420521

#convert dataframe dates to consistent type
price_df['Date'] = pd.to_datetime(price_df['Date'])
tweet_df['Date'] = pd.to_datetime(tweet_df['Date'])
mc_df['Date'] = pd.to_datetime(mc_df['Date'])

#get min and max dates for all three dataframes
price_df.set_index('Date', inplace = True)
print(price_df.index.min())
print(price_df.index.max())

tweet_df.set_index('Date', inplace = True)
print(tweet_df.index.min())
print(tweet_df.index.max())

mc_df.set_index('Date', inplace = True)
print(mc_df.index.min())
print(mc_df.index.max())

#filter dataframes with starting date 2019/08/20 and ending date 2021/08/20 (two years)
start_date = '2018-08-21'
end_date = '2021-08-21'

price_mask = (price_df.index >= start_date) & (price_df.index <= end_date)
tweet_mask = (tweet_df.index >= start_date) & (tweet_df.index <= end_date)
mc_mask = (mc_df.index >= start_date) & (mc_df.index <= end_date)
final_price = price_df.loc[price_mask]
final_tweet = tweet_df.loc[tweet_mask]
final_mc = mc_df.loc[mc_mask]

#merge into one dataframe
merge1 = pd.merge(final_price, final_tweet, left_index=True, right_index=True)
df = pd.merge(merge1, final_mc, left_index=True, right_index=True)

#get volume/marketcap = percent traded column 
df["Marketcap"] = pd.to_numeric(df["Marketcap"], downcast="float")
df['Proportion Traded'] = df['Volume']/df['Marketcap']

df.to_csv('cleaned_data.csv')