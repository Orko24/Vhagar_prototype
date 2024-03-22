import matplotlib.pyplot as plt
from vhagar_public.Base_Code.Base_API import *
from vhagar_public.Base_Code.Technical_analysis import *
from vhagar_public.Base_Code.Predictive_Analytics import *
from vhagar_public.Base_Code.Advanced_Predictive_Analytics import *
from vhagar_public.Base_Code.Data_cleaner import *


def testing_rectilinear():
    '''

    Build out a testing system for the machine learning thing
    :return:
    '''

    stock = 'GOOG'
    api_key = 'INSERT_API_KEY'
    outputsize = 'full'

    Stock_data_class = Stock_data(stock_symbol=stock, api_key=api_key, outputsize=outputsize)

    stock_data = Stock_data_class.data_set()
    stock_meta_data = Stock_data_class.metadata
    stock_data_cols = stock_data.columns

    Indicators_data = Indicators(stock_data=stock_data)

    LTSM_data = LTSM_training_data(stock_data=stock_data)

    print(stock_data)

    training_data = LTSM_training_data(stock_data=stock_data).training_set_close

    rectified_data_object = rectified_predictions(dataset= training_data, train_test_split= 0.80)
    # print(training_data[:70], len(training_data), len(training_data[:70]))

    training_data_rectified = rectified_data_object.train_data
    testing_data_rectified = rectified_data_object.test_data

    pred_data_frame = rectified_data_object.predictions_data_frame()
    timeframe = rectified_data_object.timestamp

    print(pred_data_frame)
    print(timeframe)

    print(testing_data_rectified)
    print(training_data_rectified)


    return

if __name__ == "__main__":
    testing_rectilinear()