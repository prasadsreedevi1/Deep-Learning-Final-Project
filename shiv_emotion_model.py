import tensorflow as tf
import numpy as np
import random
import math

class LSTMRegressor(tf.keras.Model):
    def __init__(self, hidden_size=128):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size

        self.pre_dense = tf.keras.layers.Dense(256, activation='relu')  
        self.lstm = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(hidden_size),
            return_sequences=False
        )
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(2, activation='linear')  
        
    def call(self, inputs, is_testing=False):
        x = self.pre_dense(inputs)   
        x = self.lstm(x)
        x = self.dense1(x)
        output = self.output_layer(x)
        return output

    def loss(self, predictions, targets):
        return tf.reduce_mean(tf.keras.losses.mean_squared_error(targets, predictions))

    def accuracy(self, predictions, targets):
        # For regression, "accuracy" could be interpreted as inverse of mean absolute error
        mae = tf.reduce_mean(tf.abs(targets - predictions))
        return 1.0 / (1.0 + mae)  # pseudo-accuracy metric