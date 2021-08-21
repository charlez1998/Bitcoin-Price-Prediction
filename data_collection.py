# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 18:53:40 2021

@author: charlie
"""

import requests 
from bs4 import BeautifulSoup
import pandas as pd
import re

def parse_strlist(sl):
    clean = re.sub("[\[\],\s]","",sl)
    splitted = re.split("[\'\"]",clean)
    values_only = [s for s in splitted if s != '']
    return values_only

def get_variable(url):
    i = 0
    while url[i] != "-":
        i += 1 
    return url[i+1:].split(".")[0].capitalize()

def scrape(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    scripts = soup.find_all('script')
    for script in scripts:
        if script.string == None:
            pass 
        elif 'd = new Dygraph(document.getElementById("container")' in script.string:
            StrList = script.string
            StrList = '[[' + StrList.split('[[')[-1]
            StrList = StrList.split(']]')[0] +']]'
            StrList = StrList.replace("new Date(", '').replace(')','')
            dataList = parse_strlist(StrList)
    date = []
    variable = []
    for each in dataList:
        if (dataList.index(each) % 2) == 0:
            date.append(each)
        else:
            variable.append(each)

    df = pd.DataFrame(list(zip(date, variable)), columns=["Date", get_variable(url)])
    return df

#TEST
# df1 = scrape("https://bitinfocharts.com/comparison/bitcoin-tweets.html")
# df2 = scrape('https://bitinfocharts.com/comparison/bitcoin-marketcap.html#3y')
# df3 = scrape("https://bitinfocharts.com/comparison/google_trends-btc.html#3y")