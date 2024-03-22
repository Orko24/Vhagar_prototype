from vhagar_public.Base_Code.Base_API import *
from vhagar_public.Base_Code.Technical_analysis import *
from vhagar_public.Base_Code.Data_cleaner import *
from vhagar_public.Base_Code.Data_vis_API import *


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
import os.path



'''
API to add advanced artificial intelligence models for price prediction systems 

'''

"""
Auxilary Class
"""


class rectified_predictions(object):

    def __init__(self, dataset, train_test_split=0.70, timestamp = 100, model_name = "keras"):

        '''
        https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
        :param dataset: enter one type column of data. i.e dataset will take pandas data with 1 column as parameter.
        dataset format must be (1,X) use LTSM_training_data if necessary.

        :param train_test_split: modify the train test split up to the users discretion
        '''


        self.dataset = dataset
        self.train_test_split = train_test_split
        self.timestamp = timestamp

        '''
        train = pd.DataFrame(data[0:int(len(data)*0.70)])
        test = pd.DataFrame(data[int(len(data)*0.70): int(len(data))])
        
        print(train.shape)
        print(test.shape)
        
        training FORMAT MUST BE IN (1,X) find a way to split them down the middle
        '''

        self.train_data = pd.DataFrame(self.dataset[0:int(len(self.dataset)*self.train_test_split)])
        self.test_data = pd.DataFrame(self.dataset[int(len(self.dataset)*self.train_test_split):
                                                   int(len(self.dataset))])

        self.train_data_shape = self.train_data.shape
        self.test_data_shape = self.test_data.shape
        self.model_name = model_name

    def scalar_vals(self):

        scaler = MinMaxScaler(feature_range=(0, 1))

        return scaler

    def training_values(self):

        scaler = self.scalar_vals()
        train_values = self.train_data
        # test_values = self.test_data
        data_training_array = scaler.fit_transform(train_values)
        x_train = []
        y_train = []
        for i in range(100, data_training_array.shape[0]):
            x_train.append(data_training_array[i - 100: i])
            y_train.append(data_training_array[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        return x_train, y_train



    def create_model_1(self):

        x_train, y_train = self.training_values()

        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True
                       , input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.5))
        # model.add(LSTM(units=80, activation='relu', return_sequences=True))
        # model.add(Dropout(0.4))
        # model.add(LSTM(units=60, activation='relu', return_sequences=True))
        # model.add(Dropout(0.3))
        # model.add(LSTM(units=50, activation='relu', return_sequences=True))
        # model.add(Dropout(0.2))

        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])

        return model


    def prediction_model(self):
        '''
        add rectified linear model
        :return:
        '''

        # if not keras.models not present, then run the model else use the previous model

        train_values = self.train_data
        test_values = self.test_data
        scaler = self.scalar_vals()
        x_train, y_train = self.training_values()
        # print(x_train.shape)

        if os.path.isfile('Training_models/{}.keras'.format(self.model_name)):
            model = tf.keras.models.load_model('Training_models/{}.keras'.format(self.model_name))
            # print(model.summary())
        else:
            model = self.create_model_1()
            model.summary()
            x_train, y_train = self.training_values()
            model.fit(x_train, y_train, epochs=100)
            model.save('Training_models/{}.keras'.format(self.model_name))



        past_days = pd.DataFrame(train_values[-self.timestamp:])
        test_df = pd.DataFrame(test_values)
        final_df = past_days._append(test_df, ignore_index=True)
        input_data = scaler.fit_transform(final_df)


        x_test = []
        y_test = []
        for i in range(self.timestamp, input_data.shape[0]):
            x_test.append(input_data[i - self.timestamp: i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        y_pred = model.predict(x_test)

        scale_factor = 1 / scaler.scale_[0]
        y_pred = y_pred * scale_factor
        y_test = y_test * scale_factor

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


        return y_pred, y_test

    def predictions_data_frame(self):

        '''
        take all the data presented and make a dataframe

        :return:
        '''

        y_pred, y_test = self.prediction_model()
        y_pred = y_pred.reshape(1,-1)[0]

        names = ["Predicted Data", "Original Data"]
        dict_res = dict(zip(names, [y_pred,y_test]))
        # print(pd.DataFrame(dict_res))

        return pd.DataFrame(dict_res)








