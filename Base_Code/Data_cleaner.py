from vhagar_public.Base_Code.Base_API import *
from vhagar_public.Base_Code.Technical_analysis import *
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


class array_process(object):

    def __init__(self, array):

        self.array = array
        self.shape = array.shape

    def reshape(self):

        array = self.array
        return array.reshape(1, -1)

    def array_first_term(self):

        reshape = self.reshape()
        return reshape[0]

class data_split(object):

    def __init__(self, dataframe, split = 0.80):

        self.dataframe = dataframe
        self.split = split


        self.train_data = pd.DataFrame(self.dataframe[0:int(len(self.dataframe) * self.split)])

        self.test_data = pd.DataFrame(self.dataframe[int(len(self.dataframe) * self.split):
                                                   int(len(self.dataframe))])

        self.train_data_shape = self.train_data.shape
        self.test_data_shape = self.test_data.shape


class LTSM_training_data(object):

    def __init__(self, stock_data):

        '''
        Designed to be used with Indicators_data class object
        However if you have stock data in pandas format with the columns:

        Index(['index', '1. open', '2. high', '3. low', '4. close',
       '5. adjusted close', '6. volume', '7. dividend amount',
       '8. split coefficient', 'average price'],
        dtype='object')

        please feel free to use it
        '''

        self.stock_data = stock_data
        self.training_set_open = self.stock_data[['1. open']].values
        self.training_set_high = self.stock_data[['2. high']].values
        self.training_set_low = self.stock_data[['3. low']].values
        self.training_set_close = self.stock_data[['4. close']].values
        self.training_set_avg_price = self.stock_data[['average price']].values

class numpy_pandas_convertor(object):
    '''
    converts numpy arrays to pandas columns and dataframes
    '''

    def __init__(self,arr, name = "Data"):

        self.arr = arr
        self.modified_array = self.arr.reshape(1,-1)[0]
        self.name = name

    def pandas_convertor(self):

        return pd.DataFrame(dict(zip([self.name], [self.modified_array])), dtype="object")


class pandas_formatter(object):

    def __init__(self, dataframe):

        '''
        https://pandas.pydata.org/docs/user_guide/merging.html
        :param dataframe:
        '''

        self.dataframe = dataframe.copy()

    def index_modification(self, modificaiton_amount = 1):

        self.dataframe.index += modificaiton_amount

        return self.dataframe
    def index_modification_nan(self, modificaiton_amount = 1):

        self.dataframe.index += modificaiton_amount

        '''
        add 0 indexes above this to gain
        '''

        zero_arr = np.array([0] * modificaiton_amount)
        nan_arr = np.array([np.nan] * modificaiton_amount)

        corrective_values = dict(zip(list(self.dataframe.columns), [nan_arr, nan_arr]))
        corrective_idx = pd.DataFrame(corrective_values)
        self.dataframe = pd.concat([corrective_idx,self.dataframe], axis= 0)

        return self.dataframe


class dataframe_join(object):

    def __init__(self, dataframes=[]):
        self.dataframes = dataframes

    def join_index(self):

        new_df = pd.concat(self.dataframes)
        return new_df

    # def value_insertion_merge(self, dataframe = None):
    #     '''
    #
    #     :param dataframe: insert dataframe values to be inserted into the colummns
    #     :return:
    #     '''
    #
    #     if dataframe == None:
    #
    #         return self.index_modification(modificaiton_amount=0)
    #
    #     dataframe_vals = dataframe.values()
    #
    #     print(dataframe_vals)
    #
    #     return