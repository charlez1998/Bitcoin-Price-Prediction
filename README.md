# Bitcoin Price Prediction: Project Overview
* Created a tool that estimates Bitcoin Price (RMSE ~ $ 200-300 USD) for every day leading up to seven days. 

## Code and Resources Used
Python Version: 3.8
Packages: pandas, numpy, sklearn, matplotlib, seaborn, keras
For Web Framework Requirements: pip install -r requirements.txt
Bitcoin Price Dataset Used: https://finance.yahoo.com/quote/BTC-USD/history?p=BTC-USD
Referenced Sites: 
* https://coinmarketcap.com/currencies/bitcoin/historical-data/
* https://bitinfocharts.com/comparison/bitcoin-tweets.html#3y
Consolidated Articles: 
* https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
* https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
* https://machinelearningmastery.com/how-to-grid-search-deep-learning-models-for-time-series-forecasting/
## Web Scraping
Using Beautifulsoup, I implemented a scraper for the bitinfocharts site in order to get the following:
* Bitcoin related tweets for the past three years from the current day.
* Market Cap of Bitcoin for the past three years from the current day.
## Data Cleaning

## EDA

## Model Building
* Scraped Bitcoin related Tweets and Market Cap for the current day all the way to 3 years ago which may aid in price estimation. 
* Optimized the Stacked LSTM Model using a grid search to reach the best model.
## Model Performance 
All models described below only pertain to the univariate case:


## Productionization
The goal is to have a local webserver regularly forecast these prices every single week.
