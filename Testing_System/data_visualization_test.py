import matplotlib.pyplot as plt
from vhagar_public.Base_Code.Base_API import *
from vhagar_public.Base_Code.Technical_analysis import *
from vhagar_public.Base_Code.Predictive_Analytics import *

def testing_data_vis():
    '''

    Build out a testing system for the machine learning thing
    :return:
    '''

    stock = 'GOOG'
    api_key = 'N88J2EN2MD3E978B'
    outputsize = 'full'

    Stock_data_class = Stock_data(stock_symbol=stock, api_key=api_key, outputsize= outputsize)

    stock_data = Stock_data_class.data_set()
    stock_meta_data = Stock_data_class.metadata
    stock_data_cols = stock_data.columns

    Indicators_data = Indicators(stock_data=stock_data)

    LTSM_data = LTSM_training_data(stock_data=stock_data)

    '''
    Algo is as follows:

    1) Get the training data
    2) Get the testing data

    Note try to use linear regression or a polynomial fit to produce future testing data and use 
    the machine learning model to optimize future prices, run indicator code to get buy/sell signal
    '''


    '''error in training data, use LTSM_training_data to generate training data from stock data'''
    training_data = LTSM_training_data(stock_data = stock_data).training_set_close
    testing_data = LTSM_training_data(stock_data = stock_data).training_set_close

    sequential_model = model_generation_optimized(training_data = training_data, testing_data= testing_data,
                                                  timestamp = len(training_data) - 1)

    '''
    visualized data
    '''
    print(" ")
    print(" ")
    print("------------------------")

    print(stock_data)
    print(" ")
    print(" ")
    print("------------------------")

    print(training_data)
    print(len(training_data))

    print(" ")
    print(" ")
    print("------------------------")

    print(" ")
    print(" ")
    print("------------------------")

    print(testing_data)
    print(len(testing_data))

    print(" ")
    print(" ")
    print("------------------------")

    chart_data = sequential_model.sequential_prediction_dataframe()

    dataframe_visualation(chart_data).dataframe_plot()


    return


if __name__ == "__main__":
    testing_data_vis()
