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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def train(model, optimizer, inputs, train_genre_labels, train_va_labels, batch_size=64):
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


def plot_loss_and_accuracy(losses, accuracies):
    epochs = list(range(len(losses)))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', color='red', fontsize=14)
    ax1.plot(epochs, losses, color='red', label='Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('Accuracy', color='blue', fontsize=14)
    ax2.plot(epochs, accuracies, color='blue', label='Accuracy', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 1.05)  

    plt.title('LSTM Loss and Genre Accuracy over Epochs', fontsize=16)
    fig.tight_layout()
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.savefig('data/lstm_loss_accuracy_combined.png')
    plt.show()



def plot_valence_mae(valence_maes):
    epochs = list(range(len(valence_maes)))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, valence_maes, label='Valence-Arousal MAE', color='green', linewidth=2)
    plt.title('LSTM Valence-Arousal MAE over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('MAE', fontsize=14)
    plt.ylim(0, max(valence_maes)*1.2)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/lstm_va_mae.png')
    plt.show()

def main():
    train_imgs, train_genre_labels, train_va_labels = get_data_emotion('data/train_data.csv')
    test_imgs, test_genre_labels, test_va_labels = get_data_emotion('data/test_data.csv')

    train_genre_labels = tf.convert_to_tensor(train_genre_labels, dtype=tf.int32)
    test_genre_labels = tf.convert_to_tensor(test_genre_labels, dtype=tf.int32)
    train_va_labels = tf.convert_to_tensor(train_va_labels, dtype=tf.float32)
    test_va_labels = tf.convert_to_tensor(test_va_labels, dtype=tf.float32)

    model = LSTMMultiTask(num_genres=4, hidden_size=128, reg_weight=10.0)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_per_epoch = []
    genre_acc_per_epoch = []
    va_mae_per_epoch = []
    epochs = 75
    for epoch in range(1, epochs+1):
        tf.print(f"\nEpoch {epoch}/{epochs}")
        train(model, optimizer, train_imgs, train_genre_labels, train_va_labels, batch_size=64)
        avg_loss, avg_genre_acc, avg_va_mae = test(model, test_imgs, test_genre_labels, test_va_labels, batch_size=64)
        loss_per_epoch.append(avg_loss.numpy())
        genre_acc_per_epoch.append(avg_genre_acc.numpy())
        va_mae_per_epoch.append(avg_va_mae.numpy())
    plot_valence_mae(va_mae_per_epoch)
    plot_loss_and_accuracy(loss_per_epoch, genre_acc_per_epoch)

    all_preds = []
    batch_size = 64
    for i in range(0, test_imgs.shape[0], batch_size):
        inputs_batch = test_imgs[i:i+batch_size]
        preds = model(inputs_batch, is_testing=True)
        genre_logits = preds["genre"]
        predicted_batch = tf.argmax(genre_logits, axis=1, output_type=tf.int32).numpy()
        all_preds.extend(predicted_batch)

    predicted_labels = np.array(all_preds)
    true_labels = test_genre_labels.numpy()

    cm = confusion_matrix(true_labels, predicted_labels)
    genre_names = ["classical", "country", "blues", "electronic"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genre_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
    plt.title('LSTM Confusion Matrix: Genre Classification')
    plt.tight_layout()
    plt.savefig('data/lstm_confusion_matrix.png')
    plt.show()
    

if __name__ == '__main__':
    main()