from vhagar_private.Base_Code.Base_API import *
from vhagar_private.Base_Code.Technical_analysis import *
from vhagar_private.Base_Code.Data_cleaner import *


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential




'''
API to add advanced artificial intelligence models for price prediction systems 

'''

"""
Auxilary Class
"""


class rectified_predictions(object):

    def __init__(self, dataset, train_test_split=0.70):

        '''
        https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
        :param dataset: enter one type column of data. i.e dataset will take pandas data with 1 column as parameter.
        dataset format must be (1,X) use LTSM_training_data if necessary.

        :param train_test_split: modify the train test split up to the users discretion
        '''


        self.dataset = dataset
        self.train_test_split = train_test_split

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


    def prediction_model(self):
        '''
        add rectified linear model
        :return:
        '''

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_values = self.train_data
        test_values = self.test_data

        data_training_array = scaler.fit_transform(train_values)
        x_train = []
        y_train = []

        for i in range(100, data_training_array.shape[0]):
            x_train.append(data_training_array[i - 100: i])
            y_train.append(data_training_array[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        print(x_train.shape)

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

        model.add(Dense(units=1))

        model.summary()

        print(model.summary())






        return





