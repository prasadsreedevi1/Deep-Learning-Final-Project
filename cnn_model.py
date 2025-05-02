from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math
import pandas as pd
from shiv_preprocessing import get_data

from tensorflow.keras.regularizers import l2

from matplotlib import pyplot as plt
from tensorflow.keras.layers import GlobalAveragePooling2D

import os
import tensorflow as tf
import numpy as np
import random
import math




class CNN(tf.keras.Model):
    def __init__(self, classes):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(CNN, self).__init__()

        # Initialize all hyperparameters
        self.loss_list = []
        self.batch_size = 64
        self.input_width = 256
        self.input_height = 256
        self.image_channels = 20
        self.accuracy_list = []
        self.va_mae_list = []
        self.num_classes = classes

        self.hidden_layer_size = 128

        self.epsilon = 1e-3  # this is used for batch normalization only!
        self.layer_1_1 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), 
                                              strides=(1, 1),
            padding='SAME',
            activation='relu',
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=l2(1e-4),

            bias_initializer='zeros',
        )
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
        self.batch_norm_4 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
        self.batch_norm_5 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
        self.layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),padding='SAME', activation='relu', use_bias=True, kernel_initializer='he_normal')

        self.layer_3 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu',
            padding='SAME', use_bias=True, kernel_initializer='he_normal'
        )

        self.layer_3_1 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), strides=(2, 2), activation='relu',
            padding='SAME',  use_bias=True, kernel_initializer='he_normal'
        )
        self.layer_3_2 = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), strides=(2, 2), activation='relu',
            padding='SAME', use_bias=True, kernel_initializer='he_normal'
        )

        self.layer_4 = tf.keras.layers.Dense(units=self.hidden_layer_size, activation='relu', kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')

        self.layer_5 = tf.keras.layers.Dense(units=self.hidden_layer_size, activation='relu', kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')
      
        self.augment_1 = tf.keras.layers.RandomZoom(0.1)
        self.augment_2 = tf.keras.layers.RandomTranslation(0.1, 0.1)
        self.augment_3 = tf.keras.layers.RandomContrast(0.1)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dropout1 = tf.keras.layers.Dropout(0.5)


        self.output_layer = tf.keras.layers.Dense(units=self.num_classes, activation='softmax', kernel_initializer='he_normal')
        self.regression_output_layer = tf.keras.layers.Dense(units=2, activation='sigmoid', kernel_initializer='he_normal')

    def call(self, inputs, is_testing=False):
   
        if not is_testing:
            inputs = self.augment_1(inputs)
            inputs = self.augment_2(inputs)
            inputs = self.augment_3(inputs)
        
        x = self.layer_1_1(inputs)
        x = self.batch_norm_1(x)
        x = tf.nn.relu(x)
        
        x = tf.nn.max_pool(
            x,
            ksize=[1, 2, 2, 1],
            strides=[1,2,2,1],
            padding='SAME'
        )
        if not is_testing:
            x = tf.image.random_flip_left_right(x)
        x = self.layer_2(x)
        x = self.batch_norm_2(x)
        x = tf.nn.relu(x)
       
        x = tf.nn.max_pool(
            x,
            ksize=[1, 2, 2, 1],
            strides=[1,2,2,1],
            padding='SAME'
        )

        x = self.layer_3(x)
        x = self.batch_norm_3(x)
        x = tf.nn.relu(x)
        
        x = tf.nn.max_pool(
            x,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )
        x = self.layer_3_1(x)
        x = self.batch_norm_4(x)
        x = tf.nn.relu(x)
        
        x = tf.nn.max_pool(
            x,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )
        # x = self.layer_3_2(x)
        # x = tf.nn.relu(x)
        # x = self.batch_norm_5(x)
        # x = tf.nn.max_pool(
        #     x,
        #     ksize=[1, 2, 2, 1],
        #     strides=[1, 2, 2, 1],
        #     padding='SAME'
        # )
        # if not is_testing:
        #     x = tf.image.random_flip_left_right(x)

        x = GlobalAveragePooling2D()(x)

        # x = tf.reshape(x, [tf.shape(x)[0], -1])
        x = self.layer_4(x)
        x = self.dropout(x)



        # x = self.layer_5(x)
        # x = self.dropout1(x)

        output = self.output_layer(x)

        valence_arousal_preds = self.regression_output_layer(x)
        return output, valence_arousal_preds
    
    def loss(self, genre_logits, genre_labels, valence_arousal_preds, valence_arousal_labels):

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        genre_loss = loss_fn(genre_labels, genre_logits)
    
        if valence_arousal_preds is not None and valence_arousal_labels is not None:
            regression_loss_fn = tf.keras.losses.MeanSquaredError()
            va_loss = regression_loss_fn(valence_arousal_labels, valence_arousal_preds)
            total_loss = genre_loss + 5*va_loss
            return total_loss
        return genre_loss

    def accuracy(self,logits, labels):
        predictions = tf.argmax(logits, axis=1, output_type=tf.int32) 
        labels = tf.cast(labels, tf.int32) 
        equal_values = tf.equal(predictions, labels)
        accuracy = tf.reduce_mean(tf.cast(equal_values, tf.float32))
        return accuracy

    def valence_arousal_accuracy(self, preds, labels):
        labels = tf.cast(labels, tf.float32)
        accuracy = tf.abs(preds - labels)
        return tf.reduce_mean(tf.cast(accuracy, tf.float32))


if __name__ == '__main__':
    main()