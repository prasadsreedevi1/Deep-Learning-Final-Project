import os
import tensorflow as tf
import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from lstm_cnn_model import LSTMMultiTask
from preprocessing import get_data_emotion   
from preprocessing import get_data


def train(model, optimizer, inputs, train_genre_labels, train_va_labels, batch_size=64):
    # shuffle
    indicies = tf.range(tf.shape(inputs)[0])
    
    shuffled_indicies = tf.random.shuffle(indicies)
    shuffled_inputs = tf.gather(inputs, shuffled_indicies)
    shuffled_train_genre_labels = tf.gather(train_genre_labels, shuffled_indicies)
    shuffled_train_va_labels = tf.gather(train_va_labels, shuffled_indicies)
    sample_size = shuffled_inputs.shape[0]
    batch_size = 64
    for i in range(0, sample_size, batch_size):
       
        inputs_batch = shuffled_inputs[i:i+batch_size]
        genre_labels_batch = shuffled_train_genre_labels[i:i+batch_size]
        va_labels_batch = shuffled_train_va_labels[i:i+batch_size]
        with tf.GradientTape() as tape:
            preds = model(inputs_batch, is_testing=False)
            targets = {"genre": genre_labels_batch, "valence_arousal": va_labels_batch}
            loss = model.compute_loss(preds, targets)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        metrics = model.compute_metrics(preds, targets)


def test(model, inputs, genre_labels, va_labels, batch_size=64):

    total_loss = 0.0
    total_genre_acc = 0.0
    total_va_mae = 0.0
    sample_size = inputs.shape[0]
    batch_size = 64
    num_batches = (sample_size + batch_size - 1) // batch_size
    for i in range(0, sample_size, batch_size):
        inputs_batch = inputs[i:i+batch_size]
        genre_labels_batch = genre_labels[i:i+batch_size]
        va_labels_batch = va_labels[i:i+batch_size]

        preds = model(inputs_batch, is_testing=True)
        targets = {"genre": genre_labels_batch, "valence_arousal": va_labels_batch}
        loss = model.compute_loss(preds, targets)
        metrics = model.compute_metrics(preds, targets)

        total_loss += loss
        total_genre_acc += metrics["genre_accuracy"]
        total_va_mae += metrics["valence_arousal_mae"]

    avg_loss = total_loss/ tf.cast(num_batches, tf.float32)
    avg_genre_acc = total_genre_acc/tf.cast(num_batches, tf.float32)
    avg_va_mae = total_va_mae/ tf.cast(num_batches, tf.float32)

    print("Eval:", "loss=", avg_loss, "genre_acc=", avg_genre_acc,"va_mae=", avg_va_mae)

    return avg_loss, avg_genre_acc, avg_va_mae


def plot_loss_and_accuracy(losses, genre_accuracies):
    epochs = list(range(len(losses)))

    fig, ax1 = plt.subplots(figsize=(8,5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', color=color, fontsize=14)
    ax1.plot(epochs, losses, color=color, linewidth=2, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color, fontsize=14)
    ax2.plot(epochs, genre_accuracies, color=color, linewidth=2, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.05)

    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.title('LSTM Loss and Genre Accuracy over Epochs', fontsize=16)
    fig.tight_layout()
    plt.savefig('data/lstm_loss_and_accuracy.png')
    plt.show()

def plot_loss(losses):
    epochs = list(range(len(losses)))
    plt.figure(figsize=(8,5))
    plt.plot(epochs, losses, color='red', linewidth=2)
    plt.title('LSTM Loss per Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('data/lstm_loss.png')
    plt.show()

def plot_genre_accuracy(genre_accuracies):
    epochs = list(range(len(genre_accuracies)))
    plt.figure(figsize=(8,5))
    plt.plot(epochs, genre_accuracies, color='blue', linewidth=2)
    plt.title('LSTM Genre Classification Accuracy over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', linewidth=0.5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('data/lstm_genre_accuracy.png')
    plt.show()

def plot_valence_mae(valence_maes):
    epochs = list(range(len(valence_maes)))
    plt.figure(figsize=(8,5))
    plt.plot(epochs, valence_maes, color='green', linewidth=2)
    plt.title('LSTM Valence-Arousal MAE over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('MAE', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('data/lstm_va_mae.png')
    plt.show()

def main():
    train_imgs, train_genre_labels, train_va_labels = get_data_emotion('data/train_data_shiv.csv')
    test_imgs, test_genre_labels, test_va_labels = get_data_emotion('data/test_data_shiv.csv')

    train_genre_labels = tf.convert_to_tensor(train_genre_labels, dtype=tf.int32)
    test_genre_labels = tf.convert_to_tensor(test_genre_labels, dtype=tf.int32)
    train_va_labels = tf.convert_to_tensor(train_va_labels, dtype=tf.float32)
    test_va_labels = tf.convert_to_tensor(test_va_labels, dtype=tf.float32)

    model = LSTMMultiTask(num_genres=4, hidden_size=128, reg_weight=10.0)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_per_epoch = []
    genre_acc_per_epoch = []
    va_mae_per_epoch = []
    epochs = 100
    for epoch in range(1, epochs+1):
        tf.print(f"\nEpoch {epoch}/{epochs}")
        train(model, optimizer, train_imgs, train_genre_labels, train_va_labels, batch_size=64)
        avg_loss, avg_genre_acc, avg_va_mae = test(model, test_imgs, test_genre_labels, test_va_labels, batch_size=64)
        loss_per_epoch.append(avg_loss.numpy())
        genre_acc_per_epoch.append(avg_genre_acc.numpy())
        va_mae_per_epoch.append(avg_va_mae.numpy())
    plot_loss(loss_per_epoch)
    plot_genre_accuracy(genre_acc_per_epoch)
    plot_valence_mae(va_mae_per_epoch)
    plot_loss_and_accuracy(loss_per_epoch, genre_acc_per_epoch)

if __name__ == '__main__':
    main()