
from __future__ import absolute_import

from cathy_model import CNN
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau

import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

import numpy as np
import random
import math
import pandas as pd
from shiv_preprocessing import get_data


def train(model, optimizer, train_inputs, train_genre_labels, train_va_labels):
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
    shuffled_train_genre_labels = tf.gather(train_genre_labels, shuffled_indicies)
    shuffled_train_va_labels = tf.gather(train_va_labels, shuffled_indicies)
    # flipped_inputs = tf.image.random_flip_left_right(shuffled_inputs)
    sample_size = shuffled_inputs.shape[0]
    batch_size = 64
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    #     from_logits=False,
    #     reduction=tf.keras.losses.Reduction.NONE
    # )
    for i in range(0, sample_size, batch_size):
            inputs_batch = shuffled_inputs[i:i+batch_size]
            genre_labels_batch = shuffled_train_genre_labels[i:i+batch_size]
            va_labels_batch = shuffled_train_va_labels[i:i+batch_size]
            with tf.GradientTape() as tape:
                output, valence_arousal_preds = model.call(inputs_batch, False)
                # output = model.call(inputs_batch, False)
                # loss = model.loss(output, labels_batch)
                loss = model.loss(output, genre_labels_batch, valence_arousal_preds, va_labels_batch)
                accuracy = model.accuracy(output, genre_labels_batch)
                va_mae = model.valence_arousal_accuracy(valence_arousal_preds, va_labels_batch)
                # print("Training accuracy" + str(accuracy))
                # print("Training loss" + str(loss))
            gradients = tape.gradient(loss,model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
       


    # raise NotImplementedError


def test(model, test_inputs, test_genre_labels, test_va_labels):
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
    shuffled_test_genre_labels = tf.gather(test_genre_labels, shuffled_indicies)
    shuffled_test_va_labels = tf.gather(test_va_labels, shuffled_indicies)
    sample_size = shuffled_inputs.shape[0]
    batch_size = 64
    batches = 0
    total_accuracy = 0
    va_mae_total = 0
    for i in range(0, sample_size, batch_size):
        inputs_batch = shuffled_inputs[i:i+batch_size]
        batch_test_genre_labels = shuffled_test_genre_labels[i:i+batch_size]
        batch_test_va_labels = shuffled_test_va_labels[i:i+batch_size]
        output, valence_arousal_preds = model.call(inputs_batch, True)
        # output = model.call(inputs_batch, True)
        loss = model.loss(output, batch_test_genre_labels, valence_arousal_preds, batch_test_va_labels)
        accuracy = model.accuracy(output, batch_test_genre_labels)
        total_accuracy+= accuracy
        batches+=1
        va_mae = model.valence_arousal_accuracy(valence_arousal_preds, batch_test_va_labels)
        va_mae_total += va_mae
        # print("Testing accuracy" + str(accuracy))
        # print("Testing loss" + str(loss))
    print("Testing genre accuracy" + str(total_accuracy/batches))
    print("Valence-Arousal MAE:", va_mae_total / batches)
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

    train_imgs, train_genre_labels, train_va_labels = get_data('data/train_data_shiv.csv')
    test_imgs, test_genre_labels, test_va_labels = get_data('data/test_data_shiv.csv')
    df = pd.read_csv('data/train_data_shiv.csv')
    df["genre_id"] = df["Genre"].astype('category').cat.codes
    num_classes = df["genre_id"].nunique()  
    train_genre_labels = tf.convert_to_tensor(train_genre_labels, dtype=tf.int32)
    test_genre_labels = tf.convert_to_tensor(test_genre_labels, dtype=tf.int32)
    train_va_labels = tf.convert_to_tensor(train_va_labels, dtype=tf.float32)
    test_va_labels = tf.convert_to_tensor(test_va_labels, dtype=tf.float32)

    # test_labels = tf.convert_to_tensor([tf.one_hot(label, num_classes) for label in test_labels])
    # train_labels = tf.convert_to_tensor([tf.one_hot(label, num_classes) for label in train_labels])
    # classes = np.arange(num_classes) + 1
    # cnn model = CNN(classes)
   
   
    cnn_model = CNN(num_classes)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
    cnn_model.optimizer = optimizer 
    

    for epoch in range(1000):
        train(cnn_model, optimizer, train_imgs, train_genre_labels, train_va_labels)
        test_accuracy = test(cnn_model, test_imgs, test_genre_labels, test_va_labels)
       
        print(f"Epoch {epoch:3d} — Test Accuracy={test_accuracy:.4f}")
      
    return


    


if __name__ == '__main__':
    main()