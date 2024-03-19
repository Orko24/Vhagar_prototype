from vhagar_private.Base_Code.Base_API import *
from vhagar_private.Base_Code.Technical_analysis import *

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
