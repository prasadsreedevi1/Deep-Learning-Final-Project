from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math
import pandas as pd
from preprocessing import get_data

    
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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
        self.input_width = 128
        self.input_height = 128
        self.image_channels = 3

       
        self.num_classes = len(classes)

        self.hidden_layer_size = 320

        self.epsilon = 1e-3  # this is used for batch normalization only!
        self.layer_1_1 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), 
                                              strides=(1, 1),
            padding='SAME',
            activation='relu',
            use_bias=True,
            kernel_initializer='he_normal',
            bias_initializer='zeros',
        )
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
        self.layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding='SAME', activation='relu', use_bias=True, kernel_initializer='he_normal')

        self.layer_3 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu',
            padding='SAME', use_bias=True, kernel_initializer='he_normal'
        )
        self.layer_4 = tf.keras.layers.Dense(units=self.hidden_layer_size, activation='relu', kernel_initializer='he_normal')
        self.dropout_1 = tf.keras.layers.Dropout(0.5)

        self.layer_5 = tf.keras.layers.Dense(units=self.hidden_layer_size, activation='relu', kernel_initializer='he_normal')
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
        self.dropout_3 = tf.keras.layers.Dropout(0.5)
        self.dropout_4 = tf.keras.layers.Dropout(0.5)
        self.dropout_5 = tf.keras.layers.Dropout(0.5)




        self.output_layer = tf.keras.layers.Dense(units=self.num_classes, activation='softmax', kernel_initializer='he_normal')


    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        if is_testing:
            conv_weights = self.layer_1_1.get_weights()  
            self.layer_1.set_weights(conv_weights[0], conv_weights[1])
            x = self.layer_1(inputs)
        else:
            x = self.layer_1_1(inputs)
        x = tf.nn.relu(x)
        x = self.batch_norm_1(x)
        x = tf.nn.max_pool(
            x,
            ksize=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

        x = self.layer_2(x)
        x = tf.nn.relu(x)
        x = self.batch_norm_2(x)
        x = tf.nn.max_pool(
            x,
            ksize=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

        x = self.layer_3(x)
        x = tf.nn.relu(x)
        x = self.batch_norm_3(x)
        x = tf.nn.max_pool(
            x,
            ksize=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

        x = tf.reshape(x, [tf.shape(x)[0], -1])

        x = self.layer_4(x)
        if not is_testing:
            x = self.dropout_1(x)

        x = self.layer_5(x)
        if not is_testing:
            x = self.dropout_2(x)

        
        output = self.output_layer(x)
        return output
    
    def loss(self, logits, labels):

        categorical_entropy = tf.keras.losses.CategoricalCrossentropy()
        
        categorical_loss = categorical_entropy(labels, logits)
        
        return categorical_loss

    def accuracy(self, logits, labels):
	
        predictions = tf.argmax(logits, axis=1)
        argmax_labels = tf.argmax(labels, axis=1)
        equal_values = tf.equal(predictions, argmax_labels)
        accuracy = tf.reduce_mean(tf.cast(equal_values, tf.float32))
        return accuracy


    
def main():
    
    #get train and test data
    #split_train_test()
    
    test_imgs, test_labels = get_data('data/deam/test_data.csv')
    train_imgs, train_labels = get_data('data/deam/train_data.csv')

    # # TODO: assignment.main() pt 1
    # # Load your testing and training data using the get_data function
    # train_inputs, train_labels = get_data(AUTOGRADER_TRAIN_FILE, classes = [3,5])
    # test_inputs, test_labels = get_data(AUTOGRADER_TEST_FILE, classes = [3,5])
    

    # # TODO: assignment.main() pt 2
    # # Initialize your model and optimizer
    # #model = MLP(classes= [3,5])
    # model = CNN(classes = [3,5])
    # adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.001) # use legacy bc warning
    
    

    # # TODO: assignment.main() pt 3
    # # Train your model
    # for epoch in range(10):
    #     train(model = model, optimizer=adam, train_inputs = train_inputs, train_labels = train_labels)

    # # TODO: assignment.main() pt 4
    # # Test your model
    # accuracy = test(model = model, test_inputs = test_inputs, test_labels = test_labels)
    # #print("test_accuracy: ", accuracy)

    return


if __name__ == '__main__':
    main()
