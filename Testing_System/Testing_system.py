from vhagar_public.Base_Code.Base_API import *
from vhagar_public.Base_Code.Technical_analysis import *
from vhagar_public.Base_Code.Predictive_Analytics import *

def testing_system():


    stock = 'GOOG'
    api_key = 'N88J2EN2MD3E978B'
    pandas_price_data = price_data_pandas(stock_symbol= stock, api_key= api_key)
    Stock_data_class = Stock_data(stock_symbol=stock, api_key=api_key)

    stock_data =  Stock_data_class.data_set()
    stock_meta_data = Stock_data_class.metadata
    stock_data_cols = stock_data.columns

    Indicators_data = Indicators(stock_data = stock_data)

    LTSM_data = LTSM_training_data(stock_data = stock_data)



    def OBV_data():

        Obv_data = Indicators_data.OBV()
        return Obv_data


    def MFI_data():

        MFI_data = Indicators_data.MFI()
        return MFI_data

    def RSI_data():

        RSI_data = Indicators_data.RSI()
        return RSI_data

    def Momentum_data():

        Momentum_data = Indicators_data.Momentum()
        return Momentum_data

    def R_Williams_data():

        Williams_R = Indicators_data.Williams_R()
        return Williams_R

    def KST():

        KST = Indicators_data.KST()
        return KST

    def Fibonnaci():

        Fibonnaci = Indicators_data.Fibonnaci_Levels()
        return Fibonnaci

    def LTSM_():

        '''
        LTSM Tests
        '''

        def LTSM_training_set():
            stock_data = LTSM_data.stock_data
            # training_set = LTSM_data.training_set()
            training_set_open = LTSM_data.training_set_open
            training_set_high = LTSM_data.training_set_high
            training_set_low = LTSM_data.training_set_low
            training_set_close = LTSM_data.training_set_close
            training_set_avgprice = LTSM_data.training_set_avg_price


            print(stock_data)
            print(training_set_open)
            print(training_set_high)
            print(training_set_low)
            print(training_set_close)
            print(training_set_avgprice)


            return



        LTSM_training_set()


        return



    def all_tests():

        LTSM_()

        return





    print('')
    print('')

    print('------')
    print('')
    print('')

    all_tests()

    print('')
    print('')
    print('------')


    return

def main():
    testing_system()
    return

main()

