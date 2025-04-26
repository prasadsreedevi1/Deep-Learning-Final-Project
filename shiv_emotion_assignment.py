from shiv_emotion_model import LSTMRegressor
import os

import tensorflow as tf
import numpy as np
import random
import math
import pandas as pd
from shiv_preprocessing import get_data_emotion

def train(model, optimizer, train_inputs, train_labels):
    '''
    Trains the model for one epoch on sequential input data.
    
    :param model: the initialized model for forward/backward pass
    :param train_inputs: shape (num_samples, timesteps, input_features)
    :param train_labels: shape (num_samples, 2) – valence and arousal
    '''
    indices = tf.range(tf.shape(train_inputs)[0])
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_inputs = tf.gather(train_inputs, shuffled_indices)
    shuffled_labels = tf.gather(train_labels, shuffled_indices)

    sample_size = tf.shape(train_inputs)[0]
    batch_size = 64

    for i in range(0, sample_size, batch_size):
        inputs_batch = shuffled_inputs[i:i+batch_size]
        labels_batch = shuffled_labels[i:i+batch_size]

        with tf.GradientTape() as tape:
            predictions = model(inputs_batch, training=True)
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(labels_batch, predictions))
            mae = tf.reduce_mean(tf.keras.losses.mean_absolute_error(labels_batch, predictions))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Batch {i//batch_size + 1}: Loss = {loss.numpy():.4f}, MAE = {mae.numpy():.4f}")

def test(model, test_inputs, test_labels):
    """
    Evaluates the model on test data. Reports average loss and MAE over all batches.
    
    :param model: trained model
    :param test_inputs: shape (num_samples, timesteps, input_features)
    :param test_labels: shape (num_samples, 2) – valence and arousal
    :return: tuple (avg_mse, avg_mae)
    """
    indices = tf.range(tf.shape(test_inputs)[0])
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_inputs = tf.gather(test_inputs, shuffled_indices)
    shuffled_labels = tf.gather(test_labels, shuffled_indices)

    sample_size = tf.shape(test_inputs)[0]
    batch_size = 64
    total_mse = 0.0
    batches = 0

    for i in range(0, sample_size, batch_size):
        inputs_batch = shuffled_inputs[i:i+batch_size]
        labels_batch = shuffled_labels[i:i+batch_size]

        predictions = model(inputs_batch, training=False)
        mse = tf.reduce_mean(tf.keras.losses.mean_squared_error(labels_batch, predictions))

        total_mse += mse
        batches += 1

    avg_mse = total_mse / batches

    print(f"Test MSE: {avg_mse:.4f}")
    return avg_mse

def main():
    test_imgs, test_labels = get_data_emotion('data/test_data_shiv.csv')
    train_imgs, train_labels = get_data_emotion('data/train_data_shiv.csv')
    
    train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
    test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)
    
    model = LSTMRegressor(hidden_size=128)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    for epoch in range(10):
        print(f"\nEpoch {epoch+1}")
        train(model, optimizer, train_imgs, train_labels)

    # Test model
    mse = test(model, test_imgs, test_labels)
    print(f"Final Test MSE: {mse:.4f}")


    
if __name__ == '__main__':
    main()