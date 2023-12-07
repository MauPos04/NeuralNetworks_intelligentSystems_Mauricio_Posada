
from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron
from hubs.models.perceptron_multi import PerceptronMulti
from hubs.models.ffm_tf import ffm_tf
from hubs.models.xgboost import xgb
from hubs.models.conv_tf import conv_tf


class Neural:
    def __init__(self):
        pass

    def run_model(self, model, file_name, iter, alfa, test_split, norm, stop_condition, neurons, avoid_col, chk_name, train, data_type):
        data = Data()
        if model == 'conv_tf':
            train_images, test_images, train_labels, test_labels = data.download_database('MNIST')
        else:
            if data_type == 'time_series':
                window_size = 7
                horizon_size = 1
                train_features, test_features, train_labels, test_labels, original_features, original_labels = data.timeseries_process(window_size, horizon_size, file_name, test_split, norm)
            elif data_type == 'time_series_adapt':
                window_size = 1
                horizon_size = 1
                train_features, test_features, train_labels, test_labels, original_features, original_labels = data.timeseries_process_adapt(window_size, horizon_size, file_name, test_split, norm)
        
            elif data_type == 'data':
                train_features, test_features, train_labels, test_labels, original_features, original_labels = data.data_process(file_name, test_split,norm, neurons, avoid_col)
        
        
        if model == 'perceptron':
            print('Running Perceptron Model')
            P = Perceptron()
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)

        elif model == 'ffm_tf':
            print('Running FFM Model')
            P = ffm_tf()
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)

        elif model == 'perceptron_multi':
            print('Running perceptron Multi Model')
            P = PerceptronMulti()
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)

        elif model == 'xgb':
            print('Running XGBoost model')
            P = xgb(depth = 10)
            P.run(train_features, test_features, train_labels, test_labels, original_features, original_labels, iter, alfa, stop_condition, chk_name, train, neurons)
        
        elif model == 'conv_tf':
            print("Running Convolutional TF Model")
            P = conv_tf()
            P.run(train_images, test_images, train_labels, test_labels, iter)