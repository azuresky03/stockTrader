from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import tensorflow as tf


class MultiVariantLSTM(keras.Model):
    def __init__(self, n_features):
        self.layers = [
            LSTM(units=128, activation="relu", return_sequences=True),
            LSTM(units=64, activation="relu", return_sequences=False),
            Dense(units=25),
            Dense(units=1)
        ]
        
    

    def call(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        
        return outputs
