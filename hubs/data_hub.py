import os
import sys

import pandas as pd
import numpy as np

import sklearn.preprocessing
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts

import tensorflow
from tensorflow import keras

class Data:
    """
    Attributes:


    Methods:
    
    """

    def __init__(self):
        self.scaler = StandardScaler()##Create an object of this library in particular
        self.max_value = 0
        self.array_size = 0
        self.control_vector_size= 0

    def data_process(self, file, test_split, norm, neurons, avoid_col):
        ##Lets define the absolute path for this folder
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        ##Find the complete excel file route
        excel_path = os.path.join(data_dir, file)

        ##lets load the raw excel file
        data_raw = pd.read_excel(excel_path, sheet_name = 0)

        #print(f'Data_Raw {data_raw}')

        ##lets store the original features
        columns = data_raw.shape[1]
        original_features = data_raw[data_raw.columns[:columns-neurons]]
        original_labels = data_raw[data_raw.columns[columns-neurons:columns]]
        print(f'Original_features {original_features}')
        print(f'Original_labels {original_labels}')

        ##original_features.to_excel('output.xlsx', index=False, engine='openpyxl')

        ##Lets convert the raw data to an array
        data_arr = np.array(data_raw)

        ##Lets label encode any text in the data

        ##Lets create a boolean array with the size of the columns of the original data array
        str_cols = np.empty(data_arr.shape[1], dtype=bool)

        ##lets read columns data type
        for i in range(0,data_arr.shape[1]):
            str_cols[i] = np.issubdtype(type(data_arr[0,i]), np.str_)

        for i in range(0, data_arr.shape[1]):
            if str_cols[i]:
                le = LabelEncoder()
                data_arr[:, i] = le.fit_transform(data_arr[:,i]) + 1


        ##Lets split the data into features and labels
        data_features = data_arr[:,avoid_col:-neurons]
        data_labels = data_arr[:, -neurons:]

        print(f'DATA_FEATURES: {data_features}')
        print(f'DATA_LABELS: {data_labels}')
        
        if neurons == 1:
            data_labels = data_labels.reshape(-1,1)

        ##lets check the dimensions of the arrays
        #print(f'Dimensions: {data_labels.shape}')

        if norm == True:
            ##Lets normalize the data
            data_features_norm = self.scaler.fit_transform(data_features)
            data_labels_norm = self.scaler.fit_transform(data_labels)    
        else:
            data_features_norm = data_features
            data_labels_norm = data_labels

        ##lets split the data into training and testing
        ##input (train, test) output (train, test)

        if test_split != 0:
            train_features, test_features, train_labels, test_labels = tts(data_features_norm, data_labels_norm, test_size=test_split)
        else:
            test_features = 0
            test_labels = 0
            train_features = data_features_norm
            train_labels = data_labels_norm
            print(f'Features: {train_features}')
            print(f'labels: {train_labels}')


        return train_features, test_features, train_labels, test_labels, original_features, original_labels
        

    def denormalize(self, data):

        data_denorm = self.scaler.inverse_transform(data)

        return data_denorm


    def download_database(self, database):
        if database == 'MNIST':
            (train_images, train_labels),(test_images, test_labels) = keras.datasets.mnist.load_data()

        elif database == 'CIFAR10':
            pass
        elif database == 'CIFAR100':
            pass

        return train_images, test_images, train_labels, test_labels 
    
    def timeseries_process(self, window_size, horizon_size, file, test_split, norm):
        ##Lets define the absolute path for this folder
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        ##Find the complete excel file route
        excel_path = os.path.join(data_dir, file)

        ##lets load the raw excel file
        data_raw = pd.read_excel(excel_path, sheet_name = 0)

        ##lets turn it into an array
        array_raw = np.array(data_raw)

        ## Lets get the number of sample times for the whole data (# of sample times)
        data_length = array_raw.shape[1]

        print(f'Sample Times for time series: {data_length}')

        ##Lets create the data base array for storing the data the proper way
        ##Rows = (#Sample Times - Window - horizon)
        ##Columns = (window + horizon)
        time_series_arr = np.zeros((data_length - window_size - horizon_size + 1, window_size*2 + horizon_size + 1))

        print(f'Time_Series_Arr rows: {time_series_arr.shape[0]}')
        print(f'Data Array Raw Shape: {array_raw.shape}')
        print(f'Time_Series_Arr Shape: {time_series_arr.shape}')

        ##we have to look trough all the raw data, and take the correct data points and store them as window and horizon
        for i in range(data_length - window_size - horizon_size):
            vector = np.concatenate((array_raw[1, i:i+window_size+horizon_size], array_raw[0, i:i+window_size+horizon_size])) ## DIRECTA  SE TROCAN PARA PID
            #vector = np.concatenate((array_raw[0, i:i+window_size+horizon_size], array_raw[1, i:i+window_size+horizon_size])) ## INVERSA
            time_series_arr[i] = vector

        ##lets print this time_series_arr
        print('Time Series')
        print(time_series_arr)

        
        ##lets store the original features
        columns = time_series_arr.shape[1]
        original_features = data_raw[data_raw.columns[0:-horizon_size]]
        original_labels = data_raw[data_raw.columns[-horizon_size:]]
        print(f'Original_features {original_features}')
        print(f'Original_labels {original_labels}')

        ##now that we have the time series as a normal database for neural networks we have to split it on features and labels
        data_features = time_series_arr[:, 0:-horizon_size]
        data_labels = time_series_arr[:, -horizon_size:]

        ##lets take the max value of the data
        self.max_value = max(data_labels)

        ##lets take the number of features
        self.array_size = data_features.shape[1]

        ##lets normalize the data
        
        if norm == True:
            ##Lets normalize the data
            if horizon_size == 1:
                data_labels = data_labels.reshape(-1,1)
            sc = StandardScaler()
            data_features_norm = self.scaler.fit_transform(data_features)
            data_labels_norm = self.scaler.fit_transform(data_labels)    
        else:
            data_features_norm = data_features
            data_labels_norm = data_labels

        ##lets split the data into training and testing
        ##input (train, test) output (train, test)

        if test_split != 0:
            train_features, test_features, train_labels, test_labels = tts(data_features_norm, data_labels_norm, test_size=test_split)
        else:
            test_features = 0
            test_labels = 0
            train_features = data_features_norm
            train_labels = data_labels_norm
            #print(f'Features: {train_features}')
            #print(f'labels: {train_labels}')

        data = np.hstack((train_features, train_labels))
        #print(data)
        
        # Create a DataFrame from the data
        df = pd.DataFrame(data)
        
        # Save the DataFrame to an Excel file
        excel_filename = 'DATA_PID_ORGANIZADA_ADATP.xlsx'
        df.to_excel(excel_filename, index=False)


        return train_features, test_features, train_labels, test_labels, original_features, original_labels


    
    def timeseries_process_adapt(self, window_size, horizon_size, file, test_split, norm):
        ##Lets define the absolute path for this folder
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        ##Find the complete excel file route
        excel_path = os.path.join(data_dir, file)

        ##lets load the raw excel file
        data_raw = pd.read_excel(excel_path, sheet_name = 0)

        ##lets turn it into an array
        array_raw = np.array(data_raw)

        ## Lets get the number of sample times for the whole data (# of sample times)
        data_length = array_raw.shape[1]

        print(f'Sample Times for time series: {data_length}')

        ##Lets create the data base array for storing the data the proper way
        ##Rows = (#Sample Times - Window - horizon)
        ##Columns = (window + horizon)
        time_series_arr = np.zeros((data_length - window_size - horizon_size + 1, window_size*4 + horizon_size + 3))

        print(f'Time_Series_Arr rows: {time_series_arr.shape[0]}')
        print(f'Data Array Raw Shape: {array_raw.shape}')
        print(f'Time_Series_Arr Shape: {time_series_arr.shape}')

        ##we have to look trough all the raw data, and take the correct data points and store them as window and horizon
        for i in range(data_length - window_size - horizon_size):
            vector = np.concatenate((array_raw[0, i:i+window_size+horizon_size], array_raw[3, i:i+window_size+horizon_size], array_raw[2, i:i+window_size+horizon_size], array_raw[1, i:i+window_size+horizon_size])) #DIRECTA para PID
            ## para PID siempre U queda al final
            ## el resto conserve el orden
            time_series_arr[i] = vector

        ##lets print this time_series_arr
        print('Time Series')
        print(time_series_arr)

        
        ##lets store the original features
        columns = time_series_arr.shape[1]
        original_features = data_raw[data_raw.columns[0:-horizon_size]]
        original_labels = data_raw[data_raw.columns[-horizon_size:]]
        print(f'Original_features {original_features}')
        print(f'Original_labels {original_labels}')

        ##now that we have the time series as a normal database for neural networks we have to split it on features and labels
        data_features = time_series_arr[:, 0:-horizon_size]
        data_labels = time_series_arr[:, -horizon_size:]

        ##lets take the max value of the data
        self.max_value = max(data_labels)

        ##lets take the number of features
        self.array_size = data_features.shape[1]

        ##lets normalize the data
        
        if norm == True:
            ##Lets normalize the data
            if horizon_size == 1:
                data_labels = data_labels.reshape(-1,1)
            sc = StandardScaler()
            data_features_norm = self.scaler.fit_transform(data_features)
            data_labels_norm = self.scaler.fit_transform(data_labels)    
        else:
            data_features_norm = data_features
            data_labels_norm = data_labels

        ##lets split the data into training and testing
        ##input (train, test) output (train, test)

        if test_split != 0:
            train_features, test_features, train_labels, test_labels = tts(data_features_norm, data_labels_norm, test_size=test_split)
        else:
            test_features = 0
            test_labels = 0
            train_features = data_features_norm
            train_labels = data_labels_norm
            #print(f'Features: {train_features}')
            #print(f'labels: {train_labels}')

        data = np.hstack((train_features, train_labels))
        #print(data)
        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Save the DataFrame to an Excel file
        excel_filename = 'DATA_PID_ORGANIZADA_ADAPT.xlsx'
        df.to_excel(excel_filename, index=False)

        return train_features, test_features, train_labels, test_labels, original_features, original_labels


    def get_max_value(self):
        return self.max_value
    
    def get_array_size(self):
        return self.array_size

##Lets run this method of time_series just for testing
##T = Data()
##T.timeseries_process(3,1,'DATA_SENO_DIRECTO.xlsx')
            

    