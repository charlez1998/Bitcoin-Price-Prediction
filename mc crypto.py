import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import investpy 

stocks = ['BTC', 'ETH','DC]
          
begin_date = "01/01/2019"
end_date = "31/07/2021"
          
def generate_stock_returns(stocks_list, begin_date, end_date):
    
    prices = pd.DataFrame()
    
    for stock in stocks_list:
        df_ = investpy.get_stock_historical_data(stock=stock,country='Philippines',from_date=begin_date,to_date=end_date).Close
        df_.rename(stock, inplace=True)                                             
        df_.columns = [stock]
        prices = pd.concat([prices, df_],axis=1)
        prices.index.name = "Date"
   return prices

prices = generate_stock_returns(stocks, begin_date, end_date)