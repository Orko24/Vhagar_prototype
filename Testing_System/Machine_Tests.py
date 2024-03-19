from vhagar_private.Base_Code.Base_API import *
from vhagar_private.Base_Code.Technical_analysis import *
from vhagar_private.Base_Code.Predictive_Analytics import *

def testing_machine_learning():

    '''

    Build out a testing system for the machine learning thing
    :return:
    '''

    stock = 'GOOG'
    api_key = 'INSERT_API_KEY'

    Stock_data_class = Stock_data(stock_symbol=stock, api_key=api_key)

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
                                                  timestamp = 99)

    '''
    visualized data
    '''

    chart_data = sequential_model.prediction_dataframe()

    print(chart_data)

    plot_function = dataframe_visualation(dataframe = chart_data)
    plot_function.plot()


    return




if __name__ == "__main__":
    testing_machine_learning()
