import matplotlib.pyplot as plt
import pandas as pd

from vhagar_public.Base_Code.Base_API import *
from vhagar_public.Base_Code.Technical_analysis import *
from vhagar_public.Base_Code.Predictive_Analytics import *
from vhagar_public.Base_Code.Advanced_Predictive_Analytics import *
from vhagar_public.Base_Code.Data_cleaner import *
from vhagar_public.Base_Code.python_sql_postgress import *
import os


def rectilinear():
    '''

    Build out a testing system for the machine learning thing
    :return:
    '''

    def predictive_framework(train_data, test_data, model_name = "keras_6", testing_data_timeframe = 100, epochs = 100):


        customized_relu_object = customized_rectified_predictions(training_data=train_data, testing_data=test_data,
                                                                  model_name= model_name,
                                                                  testing_data_timeframe= testing_data_timeframe,
                                                                  epochs=epochs)

        predictions_test_dataframe, training_data_frame = customized_relu_object.predictions_data_frame()

        return predictions_test_dataframe, training_data_frame


    def LTSM_close_data(stock, api_key = 'N88J2EN2MD3E978B', outputsize = 'full'):

        Stock_data_class = Stock_data(stock_symbol=stock, api_key=api_key, outputsize=outputsize)
        stock_data = Stock_data_class.data_set()
        LTSM_data = LTSM_training_data(stock_data=stock_data)
        machine_set_data = LTSM_data.training_set_close
        full_data = numpy_pandas_convertor(arr=machine_set_data, name="Full Data").pandas_convertor()

        return full_data

    def data_visization_func(train_data, test_data , stockname = None, save_location = None,
                             model_name = "keras_6", testing_data_timeframe = 100, epochs = 100):

        results = predictive_framework(train_data=train_data, test_data=test_data,
                                       model_name= model_name, testing_data_timeframe= testing_data_timeframe,
                                       epochs=epochs)

        predictions_test_dataframe, training_data_frame = results

        '''
        Data visualization modification
        '''


        dataframe_vis_class = dataframe_visualation(dataframe=predictions_test_dataframe,
                                                    stock_name= stockname, save_location= save_location)
        # graph_vals = dataframe_vis_class.dataframe_dic()
        #
        # print(graph_vals)

        dataframe_vis_class.dataframe_plot()


        return predictions_test_dataframe, training_data_frame


    class training_test_visualization(object):

        def __init__(self, stock, save_location):

            self.stock = stock
            self.save_location = save_location

            self.full_data = LTSM_close_data(stock = self.stock)
            self.data_split_object = data_split(dataframe= self.full_data)
            self.train_data = self.data_split_object.train_data
            self.test_data = self.data_split_object.test_data

            '''
            Data visualization modification
            '''

            self.results = data_visization_func(train_data= self.train_data,
                                                test_data= self.test_data, stockname= self.stock,
                                                save_location = self.save_location)


            self.predictions_test_dataframe, self.training_data_frame = self.results

    def graph_generator(stocks, save_directory = "generated_data/vhagar_graphs/production_graphs"):

        for stock in stocks:
            print(stock)
            pure_save_location = "{}/{}.jpg".format(save_directory, stock)
            print(pure_save_location)
            # print(save_location)
            train_test_vis = training_test_visualization(stock=stock, save_location = pure_save_location)
            full_data = train_test_vis.full_data
            train_data = train_test_vis.train_data
            test_data = train_test_vis.test_data
            predictions_test_dataframe, training_data_frame = train_test_vis.results

        return



    stocks = ["IBM", "GOOG", "MSFT", "NVDA", "TSLA", "AAPL", "AMD", "RKLB"]
    # stocks = ["IBM"]
    graph_generator(stocks = stocks)

    return

# def testing_unit():
#     rectilinear()
#     return

if __name__ == "__main__":

    rectilinear()