from __future__ import absolute_import

from cathy_model import CNN
import os

import tensorflow as tf
import numpy as np
import random
import math
import pandas as pd
from preprocessing import get_data


def train(model, optimizer, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: None
    '''
    indicies = tf.range(tf.shape(train_inputs)[0])
    
    shuffled_indicies = tf.random.shuffle(indicies)
    shuffled_inputs = tf.gather(train_inputs, shuffled_indicies)
    shuffled_labels = tf.gather(train_labels, shuffled_indicies)
    #no flip bc x direction is time which is important
    #flipped_inputs = tf.image.random_flip_left_right(shuffled_inputs)
    sample_size = shuffled_inputs.shape[0]
    batch_size = 64
    for i in range(0, sample_size, batch_size):
        inputs_batch = shuffled_inputs[i:i+batch_size]
        labels_batch = shuffled_labels[i:i+batch_size]
        with tf.GradientTape() as tape:
            output = model.call(inputs_batch, False)
            loss = model.loss(output, labels_batch)
            accuracy = model.accuracy(output, labels_batch)
        gradients = tape.gradient(loss,model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print("current accuracy", accuracy)



def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    indicies = tf.range(tf.shape(test_inputs)[0])
    shuffled_indicies = tf.random.shuffle(indicies)
    shuffled_inputs = tf.gather(test_inputs, shuffled_indicies)
    shuffled_labels = tf.gather(test_labels, shuffled_indicies)
    sample_size = shuffled_inputs.shape[0]
    batch_size = 64
    batches = 0
    total_accuracy = 0
    for i in range(0, sample_size, batch_size):
        inputs_batch = shuffled_inputs[i:i+batch_size]
        labels_batch = shuffled_labels[i:i+batch_size]
        output = model.call(inputs_batch, False)
        loss = model.loss(output, labels_batch)
        accuracy = model.accuracy(output, labels_batch)
        total_accuracy+= accuracy
        batches+=1
    print(total_accuracy/batches)
    return total_accuracy/batches

def main():
    
    #get train and test data
    #split_train_test()
    
    # test_imgs, test_labels = get_data('data/deam/test_data.csv')
    # train_imgs, train_labels = get_data('data/deam/train_data.csv')
    # cnn_model = CNN(classes)
    # optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
 
    # train(cnn_model, optimizer, train_imgs, train_labels)
    # test_accuracy = test(cnn_model, train_imgs, test_labels)

    test_imgs, test_labels = get_data('data/test_data_shiv.csv')
    train_imgs, train_labels = get_data('data/train_data_shiv.csv')

    df = pd.read_csv('data/train_data.csv')
    df["genre_id"] = df["Genre"].astype('category').cat.codes
    num_classes = df["genre_id"].nunique()
    
    test_labels = tf.convert_to_tensor([tf.one_hot(label, num_classes) for label in test_labels])
    train_labels = tf.convert_to_tensor([tf.one_hot(label, num_classes) for label in train_labels])
    classes = np.arange(num_classes) + 1

    cnn_model = CNN(classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    for epoch in range(10):
        train(cnn_model, optimizer, train_imgs, train_labels)
    test_accuracy = test(cnn_model, test_imgs, test_labels)
    print("test accuracy", test_accuracy)

    return


    


if __name__ == '__main__':
    main()