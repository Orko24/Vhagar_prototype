import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class dataframe_visualation(object):

    def __init__(self, dataframe):

        self.dataframe = dataframe
        self.colname = list(self.dataframe.columns.values)
        self.index = self.dataframe.index

    def dataframe_plot(self):

        '''
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html

        :return:
        '''
        # sns.set_theme(style="darkgrid")
        self.dataframe.plot(figsize = (12,12), style = "darkgrid", legend = True)
        plt.show()

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
