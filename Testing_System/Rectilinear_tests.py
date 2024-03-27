import matplotlib.pyplot as plt
import pandas as pd

from vhagar_public.Base_Code.Base_API import *
from vhagar_public.Base_Code.Technical_analysis import *
from vhagar_public.Base_Code.Predictive_Analytics import *
from vhagar_public.Base_Code.Advanced_Predictive_Analytics import *
from vhagar_public.Base_Code.Data_cleaner import *
from vhagar_public.Base_Code.python_sql_postgress import *

def testing_rectilinear():
    '''

    Build out a testing system for the machine learning thing
    :return:
    '''

    stock = 'GOOG'
    api_key = 'N88J2EN2MD3E978B'
    outputsize = 'full'

    Stock_data_class = Stock_data(stock_symbol=stock, api_key=api_key, outputsize=outputsize)

    stock_data = Stock_data_class.data_set()
    stock_meta_data = Stock_data_class.metadata
    stock_data_cols = stock_data.columns

    Indicators_data = Indicators(stock_data=stock_data)

    LTSM_data = LTSM_training_data(stock_data=stock_data)

    print(stock_data)

    training_data = LTSM_training_data(stock_data=stock_data).training_set_close
    rectified_data_object = rectified_predictions(dataset= training_data,
                                                  train_test_split= 0.80,
                                                  timestamp= 100, model_name= "keras_3", epochs = 10)

    predictions_data_frame, testing_dataframe = rectified_data_object.predictions_refined()





    print(predictions_data_frame)
    print(" ")
    print(" ")
    print(testing_dataframe)
    '''
    try to get the training data out as well to keep everything consistent
    '''

    machine_set_data = LTSM_data.training_set_close

    data_split_object = data_split(dataframe=machine_set_data)
    train_data = data_split_object.train_data
    test_data = data_split_object.test_data

    customized_relu_object = customized_rectified_predictions(training_data=train_data, testing_data=train_data,
                                                              model_name="keras_4", testing_data_timeframe=250,
                                                              epochs=10)

    predictions_test_dataframe, training_data_frame = customized_relu_object.predictions_refined()

    print(predictions_test_dataframe)
    print(training_data_frame)




    return

if __name__ == "__main__":

    testing_rectilinear()