# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 18:59:59 2021

@author: charlie
"""

import pandas as pd 
from data_collection import scrape

price_df = pd.read_csv("btc_2years.csv")
tweet_df = scrape("https://bitinfocharts.com/comparison/bitcoin-tweets.html")
mc_df = scrape("https://bitinfocharts.com/comparison/bitcoin-marketcap.html#3y")




#check for all null values
print(price_df.isnull().sum())
print(tweet_df[pd.isnull(tweet_df['Tweets'])])
print(mc_df[pd.isnull(mc_df['Marketcap'])])

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
start_date = '2019-08-20'
end_date = '2021-08-20'

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

#check kaggle for more data cleaning 