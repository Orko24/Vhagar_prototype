import numpy as np
import pandas as pd

from vhagar_public.Base_Code.Base_API import *
import matplotlib.pyplot as plt
import numpy, pandas

stock = "MFST"
'''
design a system with the following indicators:

i) SMA ii) EMA iii) MACD
iv) OBV v) MFI vi) RS
vii) Bollinger Bands 
viii) Momentum ix) Williams % R
x) FSTO xi) KST xii) Fibonacci Levels

https://www.kaggle.com/code/lusfernandotorres/data-science-for-financial-markets
'''


# stock = 'GOOG'
# api_key = 'N88J2EN2MD3E978B'

class Stock_data(object):

    def __init__(self, stock_symbol, api_key = 'WP1ERVVZOQQIWUCJ', outputsize='full'):

        '''
        Just utilize pandas for now, no need to overcomplicate the code using Json
        '''

        '''
        output format = "full" add that in 
        '''

        self.stock_symbol = stock_symbol
        self.api_key = api_key
        self.outputsize = outputsize

        self.pandas_price_data = price_data_pandas(self.stock_symbol, api_key = self.api_key,
                                                   outputsize= self.outputsize)

        self.price_data_intraday, self.metadata = self.pandas_price_data.adjusted_data()

        self.data_cols = self.price_data_intraday.columns
        self.price_data_intraday['average price'] = self.price_data_intraday[list(self.data_cols)[1:5]].sum(axis=1, numeric_only=True) / 4


    def avg_price(self):

        avg_price = self.price_data_intraday['average price']

        return avg_price

    def data_set(self):

        data = self.price_data_intraday

        return data

class Indicators(object):

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

    def SMA(self, timeframe = 15):
        self.stock_data['SMA{}'.format(timeframe)] = self.stock_data['4. close'].rolling(timeframe).mean()
        sma_data = self.stock_data['SMA{}'.format(timeframe)]
        return sma_data

    def common_sma(self, sma_timeframes=None):

        # sma_timeframes = sma_timeframes
        if sma_timeframes is None:
            sma_timeframes = [5, 10, 15, 30,50]

        sma_col_names = []
        for timeframe in sma_timeframes:
            sma_col_name = 'SMA{}'.format(timeframe)
            sma_col_names.append(sma_col_name)
            self.stock_data[sma_col_name] = self.SMA(timeframe=timeframe)

        return self.stock_data[sma_col_names]


    def EWMA(self, timeframe = 15):

        self.stock_data['EWMA{}'.format(timeframe)] = self.stock_data['4. close'].ewm(span=timeframe).mean()
        ema_data = self.stock_data['EWMA{}'.format(timeframe)]
        return ema_data

    def common_ewma(self, ewma_timeframes=None):

        # ewma_timeframes = [5, 10, 15, 30]
        if ewma_timeframes is None:
            ewma_timeframes = [5, 10, 15, 30,50]

        ewma_col_names = []
        for timeframe in ewma_timeframes:
            ewma_col_name = 'EWMA{}'.format(timeframe)
            ewma_col_names.append(ewma_col_name)
            self.stock_data[ewma_col_name] = self.SMA(timeframe=timeframe)

        return self.stock_data[ewma_col_names]

    def MACD(self):

        '''
        https://www.quantifiedstrategies.com/python-and-macd-trading-strategy/#:~:text=Calculating%20the%20MACD%20indicator%20in%20Python,-The%20MACD%20is&text=Pandas%20provides%20a%20function%20to,period%20EMA%20of%20the%20MACD.
        :return: macd, macd_signal, macd_histogram
        '''

        self.stock_data['EWMA_12'] = self.stock_data['average price'].ewm(span=12).mean()
        self.stock_data['EWMA_26'] = self.stock_data['average price'].ewm(span=26).mean()

        self.stock_data['macd'] = self.stock_data['EWMA_12'] - self.stock_data['EWMA_26']
        self.stock_data['macd_signal'] = self.stock_data['macd'].ewm(span = 9).mean()
        self.stock_data['macd_histogram'] = self.stock_data['macd'] - self.stock_data['macd_signal']

        return self.stock_data[['macd', 'macd_signal' , 'macd_histogram']]

    def OBV(self):
        '''
        https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/obv#:~:text=On%20Balance%20Volume%20is%20calculated,the%20security's%20price%20closes%20down.
        https://github.com/Magnifique-d/stock-trading-strategy

        :return:
        '''

        close_data = list(self.stock_data['4. close'])
        volume_data = list(self.stock_data['6. volume'])

        n = len(close_data)

        OBV_data = [0] * n

        for i in range(1,n):

            if close_data[i] > close_data[i-1]:
                OBV_data[i] = OBV_data[i-1] + volume_data[i]

            elif close_data[i] < close_data[i-1]:
                OBV_data[i] = OBV_data[i-1] - volume_data[i]
            else:
                OBV_data[i] = OBV_data[i-1]

        OBV_dataframe = pd.DataFrame({'OBV': OBV_data})
        self.stock_data = pd.concat([self.stock_data, OBV_dataframe], axis=1)

        return self.stock_data['OBV']

    def MFI(self):

        '''
        https://www.investopedia.com/terms/m/mfi.asp
        https://www.kaggle.com/code/sureshmecad/money-flow-index-mfi
        :return:
        '''

        '''
        '2. high', '3. low', '4. close'
        '''

        typical_price = (self.stock_data['2. high'] + self.stock_data['3. low'] + self.stock_data['4. close'])/3

        n = len(typical_price)
        period = 14
        money_flow = typical_price * self.stock_data['6. volume']
        positive_flow = [0] * n
        negative_flow = [0] * n

        for i in range(1, n):

            if typical_price[i] > typical_price[i-1]:
                positive_flow[i] = money_flow[i-1]
            elif typical_price[i] < typical_price[i-1]:
                negative_flow[i] = money_flow[i-1]
            else:
                positive_flow[i] = 0
                negative_flow[i] = 0

        positive_money_flow = []
        negative_money_flow = []

        for i in range(period - 1, len(positive_flow)):
            positive_money_flow.append(sum(positive_flow[i + 1 - period: i + 1]))

        for i in range(period - 1, len(negative_flow)):
            negative_money_flow.append(sum(negative_flow[i + 1 - period: i + 1]))

        MFI = 100 * (np.array(negative_money_flow) / (np.array(positive_money_flow) + np.array(negative_money_flow)))
        filler_zeros = np.array([0] * (period - 1))
        MFI = np.concatenate((filler_zeros, MFI), axis = None)

        MFI_dataframe = pd.DataFrame({"MFI": MFI})
        self.stock_data = pd.concat([self.stock_data, MFI_dataframe], axis= 1)

        return self.stock_data['MFI']

    def RSI(self, period = 14):

        '''
        https://www.kaggle.com/code/lusfernandotorres/data-science-for-financial-markets
        https://github.com/mtamer/python-rsi/blob/master/src/stock.py
        :return:
        '''

        prices = self.stock_data['4. close']
        profits = [0] * len(prices)

        for i in range(1,len(prices)):
            profits[i] = prices[i] - prices[i-1]

        curr_arr = profits[:period]
        rsi_arr = [0] * len(prices)
        for i in range(period+1, len(prices)):

            avg_profit = (sum(map(lambda x: x >= 0, curr_arr)))/period
            avg_loss = (sum(map(lambda x: x < 0, curr_arr)))/period
            rs = avg_profit/avg_loss
            rsi_arr[i] = 100 - (100/(1 + rs))
            curr_arr.pop(0)
            curr_arr.append(profits[i])


        RSI_dataframe = pd.DataFrame({'RSI': rsi_arr})
        self.stock_data = pd.concat([self.stock_data, RSI_dataframe], axis=1)

        return self.stock_data['RSI']

    def Bollinger_Bands(self, timeframe = 20):

        SMA = self.stock_data['4. close'].rolling(timeframe).mean()
        SD = self.stock_data['4. close'].rolling(timeframe).std()
        UB = SMA + (2 * SD)
        LB = SMA - (2 * SD)
        MB = SMA

        self.stock_data['Upper Bollinger Band'] = UB
        self.stock_data['Middle Bollinger Band'] = MB
        self.stock_data['Lower Bollinger Band'] = LB

        return self.stock_data[['Upper Bollinger Band', 'Middle Bollinger Band', 'Lower Bollinger Band']]

    def Momentum(self, timeframe = 10):

        '''
        https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html
        https://www.investopedia.com/articles/technical/081501.asp
        https://www.wallstreetmojo.com/momentum-indicator/#:~:text=The%20momentum%20indicator%20is%20calculated,(Cn)%20multiplied%20by%20100.&text=C%20is%20the%20latest%20closing%20price%20of%20a%20particular%20stock.

        :return:
        '''

        close_data = self.stock_data['4. close']

        momentum_data = [0] * len(close_data)
        momemntum_indicator_data = [0] * len(close_data)

        for i in range(timeframe-1, len(close_data)):

            momentum_data[i] = close_data[i] - close_data[i - (timeframe-1)]
            momemntum_indicator_data[i] = (close_data[i]/close_data[i - (timeframe-1)]) * 100

        momentum_dataframe = pd.DataFrame({'Momentum': momentum_data, 'Momentum Indicator': momemntum_indicator_data})

        self.stock_data = pd.concat([self.stock_data, momentum_dataframe], axis=1)

        return self.stock_data[['Momentum', 'Momentum Indicator']]

    def Williams_R(self, timeframe = 14):

        '''
        https://www.investopedia.com/terms/w/williamsr.asp

        WR = highest_high - close(dataframe) / (highest_high - lowest_low)

        :return:
        '''

        high_data, low_data, close_data = self.stock_data['2. high'], self.stock_data['3. low'], self.stock_data['4. close']
        n = len(high_data)

        curr_low_data = low_data[:timeframe]
        curr_high_data = high_data[:timeframe]

        R_williams_data = [0] * n

        for i in range(timeframe+1, len(high_data)):

            max_high = max(curr_high_data)
            min_low = min(curr_low_data)
            R_williams_data[i] = (max_high - close_data[i-1])/(max_high - min_low)
            curr_low_data = low_data[i-timeframe:i]
            curr_high_data = high_data[i-timeframe:i]

        R_Williams_dataframe = pd.DataFrame({"Williams % R": R_williams_data})
        self.stock_data = pd.concat([self.stock_data, R_Williams_dataframe], axis=1)
        return self.stock_data["Williams % R"]

    def Stochasitic_Indicators(self, timeframe = 14):

        '''
        https://www.technicalindicators.net/indicators-technical-analysis/86-stochastic-oscillator-ssto-fsto
        https://www.investopedia.com/terms/s/stochasticoscillator.asp
        :return:
        '''

        high_data, low_data, close_data = self.stock_data['2. high'], self.stock_data['3. low'], self.stock_data['4. close']
        n = len(high_data)

        curr_low_data = low_data[:timeframe]
        curr_high_data = high_data[:timeframe]

        K_percentage = [0] * n

        for i in range(timeframe+1, len(high_data)):

            max_high = max(curr_high_data)
            min_low = min(curr_low_data)

            K_percentage[i] = ((close_data[i-1] - min_low)/(max_high - min_low)) * 100

            curr_low_data = low_data[i-timeframe:i]
            curr_high_data = high_data[i-timeframe:i]

        Stochastic_DataFrame = pd.DataFrame({"Slow Stochastic Oscillator": K_percentage})
        Fast_delay_timeframe = 3
        FSTO = Stochastic_DataFrame["Slow Stochastic Oscillator (SSTO)"].rolling(Fast_delay_timeframe).mean()
        Stochastic_DataFrame["Fast Stochastic Oscillator (FSTO)"] = FSTO

        self.stock_data = pd.concat([self.stock_data, Stochastic_DataFrame], axis=1)

        return self.stock_data[["Slow Stochastic Oscillator (SSTO)", "Fast Stochastic Oscillator (FSTO)"]]

    def KST(self):

        '''
        https://www.investopedia.com/terms/k/know-sure-thing-kst.asp
        https://medium.com/codex/algorithmic-trading-with-the-know-sure-thing-indicator-in-python-68860a38a879

        :return:
        '''

        high_data, low_data, close_data = self.stock_data['2. high'], self.stock_data['3. low'], self.stock_data['4. close']
        n = len(high_data)

        def ROC_calcs(ROC_period):

            ROC = [0] * n
            for i in range(ROC_period, n):
                ROC[i] = ((close_data[i] - close_data[i - ROC_period]) / (close_data[i - ROC_period])) * 100

            return ROC

        def ROC_dataframe(ROC_periods):

            data_col_names = list(map(lambda x: "ROC period {}". format(x), ROC_periods))
            ROC_data = list(map(lambda x: ROC_calcs(x), ROC_periods))
            ROC_dic = dict(zip(data_col_names, ROC_data))
            ROC_dataframe = pd.DataFrame(ROC_dic)

            return ROC_dataframe

        def RCMA_data(ROC_data, period):

            RMCA = ROC_data.rolling(window=period).mean()
            return RMCA

        def RCMA_compact_data(ROC_data_and_period):
            '''

            :param ROC_data_and_period: takes ROC_data and period as a tuple
            :return: RCMA_data
            '''

            ROC_data, period = ROC_data_and_period
            RCMA = RCMA_data(ROC_data,period)

            return RCMA


        def KST_line(ROC_data, sma_periods):

            '''
            :param ROC_data: pandas data file
            :return: KST line as a pandas data column

            algorithm take the following

            KL = ROC_dataframe[ROC period {}].specified_sma in the KL formula
            Design the algorithm 1 step at a time
            '''

            ROC_data_cols = ROC_data.columns
            data_cols = list(zip(ROC_data_cols,sma_periods))
            RCMAs = []

            for a, b in data_cols:
                RCMA = list(ROC_data[a].rolling(window= b).mean())
                RCMAs.append(RCMA)

            RCMAs = np.array(list(map(np.array, RCMAs)))

            '''
            (ROCSMA1 * 1) + (ROCSMA2 * 2) + (ROCSMA3 * 3) + (ROCSMA4 * 4)
            '''

            KST_line = (RCMAs[0] * 1) + (RCMAs[1] * 2) + (RCMAs[2] * 3) + (RCMAs[3] * 4)
            KST_line_dataframe = pd.DataFrame({"KST Line": KST_line})

            return KST_line_dataframe

        def KST_signal_line(KST_dataframe):

            KST_line = KST_dataframe["KST Line"]
            KST_dataframe['Signal Line'] = KST_line.rolling(window = 9).mean()

            return KST_dataframe

        ROC_periods = [10, 15, 20, 30]
        ROC_dataframe_ = ROC_dataframe(ROC_periods)
        sma_periods = [10,10,10,15]
        KST_line_ = KST_line(ROC_data = ROC_dataframe_, sma_periods = sma_periods)
        KST_data = KST_signal_line(KST_line_)

        self.stock_data = pd.concat([self.stock_data, KST_data], axis = 1)

        return self.stock_data[["KST Line", 'Signal Line']]

    def Fibonnaci_Levels(self):

        '''
        https://eodhd.com/financial-academy/technical-analysis-examples/fibonacci-sequence-in-trading-with-python/#:~:text=Fibonacci%20Retracement%20and%20Extensions%20for%20Trading%20in%20Python&text=If%20we%20take%20the%20Fibonacci,89%20we%20get%200.236%2C%20etc.
        https://blog.quantinsti.com/fibonacci-retracement-trading-strategy-python/

        :return:
        '''

        def fibonacci_sequence(n):

            fib_seq = [0,1]
            # result = [0] * n

            for i in range(2,n):

                result = fib_seq[i-2] + fib_seq[i-1]
                fib_seq.append(result)

            return fib_seq

        def fibannci_retracement_sequences(n):

            const = 9
            fib_seq = fibonacci_sequence(const + n)
            result = [0] * (n-1)

            for i in range(const+1, len(fib_seq)):

                x_n = fib_seq[const]
                ratio = x_n/fib_seq[i]
                result[(i -1) - const] = ratio

            fib_seq_prime = [(result[0])**(1/2)]
            result = fib_seq_prime + result

            return result[::-1]


        '''
        Use close prices and generate n number of columns based on the retracement golden ratio's,
        Then get the prices based on the formula given here: 
        
        https://eodhd.com/financial-academy/technical-analysis-examples/fibonacci-sequence-in-trading-with-python/#:~:text=Fibonacci%20Retracement%20and%20Extensions%20for%20Trading%20in%20Python&text=If%20we%20take%20the%20Fibonacci,89%20we%20get%200.236%2C%20etc.
        
        create an automated system to create price levels based on the formula for both 
        downward and upward levels        
        
        '''

        def upward_retracement(max_high, min_low, fib_retracement_percentage):

            Uptrend_Retracement = max_high - ((max_high - min_low) * fib_retracement_percentage)
            Uptrend_Extension = max_high + ((max_high - min_low) * fib_retracement_percentage)

            dataset_names = ['Upward Retracement ({})%'.format(round((fib_retracement_percentage * 100)),3),
                             'Upward Extension ({})%'.format(round((fib_retracement_percentage * 100)),3)]

            retracement_value = [[Uptrend_Retracement], [Uptrend_Extension]]
            pandas_data = dict(zip(dataset_names, retracement_value))
            upward_data = pd.DataFrame(pandas_data)

            return upward_data

        def downward_retracement(max_high, min_low, fib_retracement_percentage):

            Downward_Retracement = min_low + ((max_high - min_low) * fib_retracement_percentage)
            Downward_Extension = min_low - ((max_high - min_low) * fib_retracement_percentage)

            dataset_names = ['Downward Retracement ({})%'.format(round((fib_retracement_percentage * 100)),3),
                             'Downward Extension ({})%'.format(round((fib_retracement_percentage * 100)),3)]

            retracement_value = [[Downward_Retracement], [Downward_Extension]]
            pandas_data = dict(zip(dataset_names, retracement_value))
            downward_data = pd.DataFrame(pandas_data)

            return downward_data

        def fib_retracements(max_high, min_low, fib_ratio):

            upward_test_data = upward_retracement(max_high, min_low, fib_ratio)
            downward_test_data = downward_retracement(max_high, min_low, fib_ratio)
            fib_retracements = pd.concat([upward_test_data, downward_test_data], axis = 1)

            return fib_retracements

        def full_fibonnaci_data(high_data, low_data, n = 4):

            max_high = max(high_data)
            min_low = min(low_data)
            fib_ratios = fibannci_retracement_sequences(n)
            retracements = []

            for i in range(len(fib_ratios)):
                retracements.append(fib_retracements(max_high=max_high, min_low=min_low, fib_ratio=fib_ratios[i]))
            results_dataframe = pd.concat(retracements, axis=1)

            return results_dataframe, results_dataframe.columns

        high_data, low_data, close_data = self.stock_data['2. high'], self.stock_data['3. low'], self.stock_data['4. close']
        # fib_ratios = fibannci_retracement_sequences(n = 4)
        results_dataframe, columns = full_fibonnaci_data(high_data, low_data)

        return results_dataframe, columns