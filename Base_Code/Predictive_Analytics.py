from vhagar_public.Base_Code.Base_API import *
from vhagar_public.Base_Code.Technical_analysis import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

class LTSM_training_data(object):

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
        self.training_set_open = self.stock_data[['1. open']].values
        self.training_set_high = self.stock_data[['2. high']].values
        self.training_set_low = self.stock_data[['3. low']].values
        self.training_set_close = self.stock_data[['4. close']].values
        self.training_set_avg_price = self.stock_data[['average price']].values

class model_generation_optimized(object):

    def __init__(self, training_data, testing_data, timestamp = 25):

        '''


        :param training_data: numpy array (1,X)
        :param testing_data: numpy array (1,X)
        :param timestamp: (time stamp for data iterations)
        '''

        self.training_data = training_data
        self.testing_data = testing_data
        self.timestamp = timestamp

    def prediction(self):

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


        training_data_1_D = self.training_data.reshape(1,-1)[0]
        testing_data_1_D = self.testing_data.reshape(1,-1)[0]
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

    def prediction_charts(self):

        prediction = self.prediction()
        training_data = self.training_data


        def char_format(prediction, training_data):

            prediction_shape = prediction.shape
            training_data_shape = training_data.shape

            print(prediction_shape)
            print(training_data_shape)

            print(prediction)
            print(training_data)

            print(prediction.reshape(1,-1))
            print(training_data.reshape(1,-1))

            print(prediction.reshape(1, -1)[0].shape)
            print(training_data.reshape(1, -1)[0].shape)


            return

        char_format(prediction, training_data)

        return


class data_visualization(object):

    def __init__(self, training_data, testing_data,
                 testing_data_color = 'black', training_data_color = "green",
                 training_data_name = "Training Data", testing_data_name = "Testing Data",
                 plot_title = "Data Prediction Visualization", xlabel = "X axis",
                 ylabel = "Y axis"):

        self.training_data = training_data
        self.testing_data = testing_data
        self.testing_data_color = testing_data_color
        self.training_data_name = training_data_name
        self.training_data_color = training_data_color
        self.testing_data_name = testing_data_name
        self.plot_title = plot_title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self):
        plt.plot(self.testing_data, color=self.testing_data, label= self.testing_data_name)
        plt.plot(self.training_data, color= self.training_data_color, label= self.training_data_name)
        plt.title(self.plot_title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.show()

        return


