from vhagar_private.Base_Code.Base_API import *
from vhagar_private.Base_Code.Technical_analysis import *
from vhagar_private.Base_Code.Data_cleaner import *


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
import seaborn as sns


'''
Proof of Concept
'''
class model_generation_optimized(object):

    def __init__(self, training_data, testing_data, timestamp = 25):

        '''


        :param training_data: numpy array (1,X)
        :param testing_data: numpy array (1,X)
        :param timestamp: (time stamp for data iterations)
        '''

        '''MODFIY FUNCTION SO IT TAKES PANDAS DATA FRAME AS INPUT AND DISTILLS IT INTO REQUIRED 
        TRAINING SETS SO ALGO DEVELOPMENT IS STREAMLINED
        '''

        self.training_data = training_data
        self.testing_data = testing_data
        self.timestamp = timestamp

    def sequential_prediction(self):

        '''
        generate model than do the prediction
        :return:
        '''

        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(self.training_data)

        X_train = []
        y_train = []

        '''modify code: range(self.timestamp, self.timestamp + 1975) --> range(self.timestamp, len(self.training_data))'''

        # for i in range(self.timestamp, self.timestamp + 1975):

        for i in range(self.timestamp, len(self.training_data)):
            X_train.append(training_set_scaled[i - self.timestamp:i, 0])
            y_train.append(training_set_scaled[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        '''use more complicated machine learning model to increase neural fidelity'''

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))

        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))

        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=100, batch_size=32)


        # training_data_1_D = self.training_data.reshape(1,-1)[0]
        # testing_data_1_D = self.testing_data.reshape(1,-1)[0]
        training_data_name = "Training Data"
        testing_data_name = "Testing Data"

        data_names = [training_data_name,testing_data_name]
        data = [self.training_data.reshape(1,-1)[0], self.testing_data.reshape(1,-1)[0]]
        dataset_total = pd.DataFrame(dict(zip(data_names,data)))

        inputs = dataset_total[len(dataset_total) - len(self.testing_data.reshape(1,-1)[0]) - self.timestamp:].values
        inputs = inputs.reshape(-1,1)
        X_test = []

        for i in range(self.timestamp, len(inputs)):

            X_test.append((inputs[i-self.timestamp: i, 0]))

        X_test = np.array(X_test)
        X_test_tensor = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_data = model.predict(X_test_tensor)
        predicted_data_result = sc.inverse_transform(predicted_data)

        return predicted_data_result

    def sequential_prediction_dataframe(self):

        prediction = self.sequential_prediction()
        training_data = self.training_data

        def format_array(training_data, array):

            n = len(training_data) - len(array)
            resultant_array = np.array((n * [np.nan]) + list(array))
            return resultant_array

        data = [prediction, training_data]
        first_array_terms = list(map(lambda x: array_process(array = x).array_first_term(), data))
        prediction_data, training_data_modified = first_array_terms[0], first_array_terms[1]
        formatted_arrays = [format_array(training_data = training_data_modified, array = x) for x in first_array_terms]
        formated_array_names = ['Prediction', 'Training Data']

        pandas_data = dict(zip(formated_array_names, formatted_arrays))
        prediction_dataframe = pd.DataFrame.from_dict(pandas_data)


        return prediction_dataframe


class dataframe_visualation(object):

    def __init__(self, dataframe):

        self.dataframe = dataframe
        self.colname = list(self.dataframe.columns.values)
        self.index = self.dataframe.index

    def dataframe_plot(self):

        self.dataframe.plot(figsize = (12,12))
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


