import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class dataframe_visualation(object):

    def __init__(self, dataframe, stock_name = None, save_location = None):

        self.dataframe = dataframe
        self.save_location = save_location
        self.stock_name = "The plot for the {} stock price".format(stock_name)
        self.colname = list(self.dataframe.columns.values)
        self.index = self.dataframe.index

    def dataframe_plot(self):

        '''
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
        https://pandas.pydata.org/docs/dev/reference/api/pandas.DataFrame.plot.line.html
        https://stackoverflow.com/questions/54842391/plotting-two-dataframe-columns-with-different-colors-in-python
        :return:
        '''
        # sns.set_theme(style="darkgrid")
        ax = self.dataframe.plot(figsize = (12,12), legend = True, use_index = True, title = self.stock_name,
                            xlabel = "Stock Index", ylabel = "Stock Price", color=['green', 'red'])

        # save plot to a location in the drive
        # print(self.save_location)

        plt.savefig(self.save_location)
        # plt.show()

        return

    def dataframe_dic(self):

        dataframe_dict = self.dataframe.to_dict()

        col_name = list(dataframe_dict.keys())
        y_values = list(dataframe_dict.values())
        x_values = list(self.dataframe.index)

        graph_vals = dict(zip(col_name, (np.array(x_values), np.array(y_values))))


        print(graph_vals)

        return graph_vals

    def detail_plot(self):

        '''
        complete later for very specific pieces that produce detailed graphs

        :return:
        '''
        graph_vals = self.dataframe_dic()

        return


    # def plot(self):
    #
    #     plt.figure(figsize=(14,5))
    #     sns.set_theme(style="darkgrid")
    #
    #     for name in self.colname:
    #         sns.lineplot(data=self.dataframe, x=self.index, y= name)
    #         sns.despine()
    #
    #
    #
    #     return




# class data_visualization(object):
#
#     def __init__(self, training_data, testing_data,
#                  testing_data_color = 'black', training_data_color = "green",
#                  training_data_name = "Training Data", testing_data_name = "Testing Data",
#                  plot_title = "Data Prediction Visualization", xlabel = "X axis",
#                  ylabel = "Y axis"):
#
#         self.training_data = training_data
#         self.testing_data = testing_data
#         self.testing_data_color = testing_data_color
#         self.training_data_name = training_data_name
#         self.training_data_color = training_data_color
#         self.testing_data_name = testing_data_name
#         self.plot_title = plot_title
#         self.xlabel = xlabel
#         self.ylabel = ylabel
#
#     def plot(self):
#         plt.plot(self.testing_data, color=self.testing_data, label= self.testing_data_name)
#         plt.plot(self.training_data, color= self.training_data_color, label= self.training_data_name)
#         plt.title(self.plot_title)
#         plt.xlabel(self.xlabel)
#         plt.ylabel(self.ylabel)
#         plt.legend()
#         plt.show()
#
#         return


# '''-------------------------------------------------'''
        #
        # print(y_pred)
        #
        # print(" ")
        # print(" ")
        # print(" -------------------------------- ")
        #
        # print(y_test)
        #
        # print(" ")
        # print(" ")
        # print(" -------------------------------- ")
        #
        # plt.figure(figsize=(12, 6))
        # plt.plot(y_test, 'b', label="Original Price")
        # plt.plot(y_pred, 'r', label="Predicted Price")
        # plt.xlabel('Time')
        # plt.ylabel('Price')
        # plt.legend()
        # plt.grid(True)
        # plt.show()