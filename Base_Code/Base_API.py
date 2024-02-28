from alpha_vantage.timeseries import TimeSeries
# import pandas, keras, tensorflow, sklearn, numpy
# import pandas, keras, sklearn, numpy
import pandas, numpy

import requests

'''
Resources: 

https://github.com/RomelTorres/alpha_vantage
https://patrickalphac.medium.com/stock-api-landscape-5c6e054ee631
https://alpha-vantage.readthedocs.io/en/latest/#
https://www.alphavantage.co/documentation/
https://github.com/RomelTorres/alpha_vantage/blob/develop/alpha_vantage/timeseries.py
'''



class price_data_json_daily(object):

    def __init__(self,stock_symbol, api_key = 'WP1ERVVZOQQIWUCJ'):
        self.api_key = api_key
        self.stock_symbol = stock_symbol
        # self.ts_pandas = TimeSeries(key=self.api_key, output_format='pandas', indexing_type='integer')
        self.url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&apikey={}'.format(self.stock_symbol, self.api_key)

    def daily_stock_data(self):

        r = requests.get(self.url)
        return r.json()

class price_data_json_intraday(object):

    def __init__(self, stock_symbol, api_key = 'WP1ERVVZOQQIWUCJ', interval = '60min'):

        self.api_key = api_key
        self.stock_symbol = stock_symbol
        self.interval = interval
        # self.url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={}&interval={}&apikey={}.format(self.stock_symbol, self.interval, self.api_key)'
        self.url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={}&interval={}&outputsize=full&apikey={}'.format(self.stock_symbol,self.interval,self.api_key)

        self.r = requests.get(self.url)
        self.data = self.r.json()
        self.keys = list(self.data.keys())
        self.values = list(self.data.values())

    def trading_data(self):

        # print(self.data)
        # print(self.keys)
        # print(self.values)

        return self.values[1]

    def meta_data(self):

        return self.values[0]


class price_data_pandas(object):

    def __init__(self, stock_symbol, api_key = 'WP1ERVVZOQQIWUCJ'):

        '''
        https://github.com/RomelTorres/alpha_vantage/blob/develop/alpha_vantage/timeseries.py
        :param stock_symbol:
        :param api_key:
        '''

        self.api_key = api_key
        self.stock_symbol = stock_symbol
        self.ts_pandas = TimeSeries(key = self.api_key, output_format='pandas', indexing_type='integer')

    def daily_stock_data(self):

        data, meta_data = self.ts_pandas.get_daily(self.stock_symbol)
        return data, meta_data

    def intra_day(self, interval = "60min"):
        data, meta_data = self.ts_pandas.get_intraday(symbol=self.stock_symbol, interval=interval, outputsize='full')
        return data, meta_data

    def adjusted_data(self):

        data, meta_data = self.ts_pandas.get_daily_adjusted(symbol=self.stock_symbol)
        return data, meta_data













