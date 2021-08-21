# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 18:59:59 2021

@author: charlie
"""

import pandas as pd 
import numpy as np
from data_collection import scrape

price_df = pd.read_csv("btc_2years.csv")
tweet_df = scrape("https://bitinfocharts.com/comparison/bitcoin-tweets.html")
mc_df = scrape("https://bitinfocharts.com/comparison/bitcoin-marketcap.html#3y")

print(price_df)