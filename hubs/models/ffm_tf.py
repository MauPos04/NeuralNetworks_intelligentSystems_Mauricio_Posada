##lets import libraries   ##MPP
import tensorflow as tf 
from tensorflow import keras 
from keras.layers import Dense

#common modules
import numpy as np
import pandas as pd
import os
import sys

##import tools for visualization
import matplotlib.pyplot as plt

##Import the metrics libraries 
from sklearn.metrics import accuracy_score as acs

##For denormalizing the data
from sklearn.preprocessing import StandardScaler

class ffm_tf:

    def _init_(self):
        pass

    def run(self,train_features,test_features,train_labels,test_labels,original_features,original_labels,iter,alfa,stop_condition,chk_name,train,neurons):
        model = self.build_model(train_features.shape[1]+1, train_labels.shape[1],alfa)

        ##let's create a stop funcion
        early_stop = keras.callbacks.EarlyStopping(monitor='mse', patience=stop_condition)

        #let's train the model
        if train:
            history = model.fit(train_features,train_labels,epochs=iter, verbose=1, callbacks=[early_stop],validation_split = 0)

            training_data = pd.DataFrame(history.history)
            print(training_data)

            plt.figure()
            plt.plot(training_data['mse'],label='mse')
            plt.show()

            ##let's validate the trained model
            pred_out = model.predict(test_features)

            plt.figure()
            plt.plot(pred_out,'r',label = 'Prediction Output')
            plt.plot(test_labels, 'b', label='Real Output')
            plt.show()

            ##SKLEARN for accuracy metrics
            accuracy = acs(test_labels.astype(int), pred_out.astype(int)) * 100
            print(f'Accuracy: {accuracy:.2f}%')

            ##Let's ask if the user wants to store the model
            r = input("Save model? (Y-N)")
            if r == 'Y':
                model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'ffm_tf'))
                chekpoint_file = os.path.join(model_dir, f'{chk_name}.h5')
                print(f'Checkpoint path: {chekpoint_file}')
                model.save_weights(chekpoint_file)
                print('Model Saved')

            elif r == 'N':
                print('Model NOT Saved')

            else:
                print('Command not recognized')
            
        else:
            ##We are not training a model here, just using an already existing model
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'ffm_tf'))
            chekpoint_file = os.path.join(model_dir, f'{chk_name}.h5')

            ##Let's load the model
            model.load_weights(chekpoint_file)

            ##Predicition Output
            pred_out = model.predict(train_features)

            ##Let's denormalize the data
            SC = StandardScaler()

            original_labels_norm = SC.fit_transform(original_labels)

            if neurons == 1:
                pred_out = pred_out.reshape(-1,1)

            pred_out_denorm = SC.inverse_transform(pred_out)

            pred_df = pd.DataFrame(pred_out_denorm)

            result_data = pd.concat([original_features,pred_df],axis=1)

            print(f'Dataframe : {result_data}')
            
            results_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'data','results'))

            results_file = os.path.join(results_dir, f'{chk_name}_RESULTS_FFM.xlsx')

            ##Let's store the dataframe as excel file
            result_data.to_excel(results_file, index = False, engine = 'openpyxl')

            plt.figure()
            plt.plot(pred_out, 'r', label='Model Output')
            plt.plot(train_labels,'b', label = 'Real Output')
            plt.xlabel('data points')
            plt.ylabel('Normalized value')
            plt.title(f'Prediction Output of model {chk_name}')
            plt.legend()
            plt.show()

            ##Let's show the accuracy value for this training batch
            accuracy = acs(train_labels.astype(int), pred_out.astype(int)) * 100
            print(f'Prediction Accuracy: {accuracy:.2f}%')


    def build_model(self, hidden_neurons,output, alfa):
        model = keras.Sequential([
            Dense(hidden_neurons, activation=tf.nn.sigmoid,input_shape=[hidden_neurons-1]),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(output)
        ])
    
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=alfa), loss = 'mse', metrics=['mse'])

        return model