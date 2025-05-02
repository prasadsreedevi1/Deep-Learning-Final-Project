
from __future__ import absolute_import

from cathy_model import CNN
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf

import numpy as np
import random
import math
import pandas as pd
from shiv_preprocessing import get_data
import matplotlib.pyplot as plt



def train(model, optimizer, train_inputs, train_genre_labels, train_va_labels):
    indicies = tf.range(tf.shape(train_inputs)[0])
    shuffled_indicies = tf.random.shuffle(indicies)
    shuffled_inputs = tf.gather(train_inputs, shuffled_indicies)
    shuffled_train_genre_labels = tf.gather(train_genre_labels, shuffled_indicies)
    shuffled_train_va_labels = tf.gather(train_va_labels, shuffled_indicies)
    sample_size = shuffled_inputs.shape[0]
    batch_size = 64
   
    for i in range(0, sample_size, batch_size):
            inputs_batch = shuffled_inputs[i:i+batch_size]
            genre_labels_batch = shuffled_train_genre_labels[i:i+batch_size]
            va_labels_batch = shuffled_train_va_labels[i:i+batch_size]
            with tf.GradientTape() as tape:
                output, valence_arousal_preds = model.call(inputs_batch, False)
               
                loss = model.loss(output, genre_labels_batch, valence_arousal_preds, va_labels_batch)
                accuracy = model.accuracy(output, genre_labels_batch)
                va_mae = model.valence_arousal_accuracy(valence_arousal_preds, va_labels_batch)
                
              
            gradients = tape.gradient(loss,model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
       


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

    plt.title('CNN Loss and Genre Accuracy over Epochs', fontsize=16)
    fig.tight_layout()
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.savefig('data/cnn_loss_accuracy_combined.png')
    plt.show()

def plot_genre_accuracy(genre_accuracies):
    epochs = list(range(len(genre_accuracies)))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, genre_accuracies, label='Genre Accuracy', color='blue', linewidth=2)
    plt.title('CNN Genre Classification Accuracy over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/cnn_genre_accuracy.png')
    plt.show()

def plot_valence_mae(valence_maes):
    epochs = list(range(len(valence_maes)))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, valence_maes, label='Valence-Arousal MAE', color='green', linewidth=2)
    plt.title('CNN Valence-Arousal MAE over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('MAE', fontsize=14)
    plt.ylim(0, max(valence_maes)*1.2)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/cnn_va_mae.png')
    plt.show()

def test(model, test_inputs, test_genre_labels, test_va_labels):

    sample_size = test_inputs.shape[0]
    batch_size = 64
    batches = 0
    total_accuracy = 0
    total_loss = 0
    va_mae_total = 0
    for i in range(0, sample_size, batch_size):
        inputs_batch = test_inputs[i:i+batch_size]
        batch_test_genre_labels = test_genre_labels[i:i+batch_size]
        batch_test_va_labels = test_va_labels[i:i+batch_size]
        output, valence_arousal_preds = model.call(inputs_batch, True)
        loss = model.loss(output, batch_test_genre_labels, valence_arousal_preds, batch_test_va_labels)
        accuracy = model.accuracy(output, batch_test_genre_labels)
        total_accuracy+= accuracy
        batches+=1
        total_loss += loss
        va_mae = model.valence_arousal_accuracy(valence_arousal_preds, batch_test_va_labels)
        va_mae_total += va_mae

    avg_accuracy = total_accuracy / batches
    avg_va_mae = va_mae_total / batches
    avg_loss = total_loss / batches

    model.accuracy_list.append(avg_accuracy.numpy())
    model.va_mae_list.append(avg_va_mae.numpy())
    model.loss_list.append(avg_loss.numpy())
    print("Testing genre accuracy" + str(total_accuracy/batches))
    print("Valence-Arousal MAE:", va_mae_total / batches)
    return total_accuracy/batches

def visualize_loss(losses):
   
    epochs = list(range(len(losses)))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, color='red', linewidth=2)
    plt.title('CNN Loss per Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('data/cnn_loss.png')
    plt.show()

def visualize_accuracy(accuracies):
    x = [i for i in range(len(accuracies))]
    plt.plot(x, accuracies)
    plt.title('Training Accuracy per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    
    plt.show()


def main():
    

    train_imgs, train_genre_labels, train_va_labels = get_data('data/train_data_shiv.csv')
    test_imgs, test_genre_labels, test_va_labels = get_data('data/test_data_shiv.csv')

    num_classes = 4
    train_genre_labels = tf.convert_to_tensor(train_genre_labels, dtype=tf.int32)
    test_genre_labels = tf.convert_to_tensor(test_genre_labels, dtype=tf.int32)
    train_va_labels = tf.convert_to_tensor(train_va_labels, dtype=tf.float32)
    test_va_labels = tf.convert_to_tensor(test_va_labels, dtype=tf.float32)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',   
        factor=0.5,            
        patience=10,              
        verbose=1,
        min_lr=1e-6
    )
   
    cnn_model = CNN(num_classes)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
    cnn_model.optimizer = optimizer 

    reduce_lr.set_model(cnn_model)
    reduce_lr.on_train_begin() 



    for epoch in range(150):
        train(cnn_model, optimizer, train_imgs, train_genre_labels, train_va_labels)
        test_accuracy = test(cnn_model, test_imgs, test_genre_labels, test_va_labels)
       
        print(f"Epoch {epoch:3d} — Test Accuracy={test_accuracy:.4f}")
        reduce_lr.on_epoch_end(epoch, logs={'val_accuracy': test_accuracy})
       
    genre_logits, _ = cnn_model.call(test_imgs, is_testing=True)
    predicted_labels = tf.argmax(genre_logits, axis=1, output_type=tf.int32).numpy()
    true_labels = test_genre_labels.numpy()

    cm = confusion_matrix(true_labels, predicted_labels)
    genre_names = ["classical", "country", "blues", "electronic"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genre_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
    plt.title('CNN Confusion Matrix: Genre Classification')
    plt.tight_layout()
    plt.savefig('data/cnn_confusion_matrix.png')
    plt.show()
    visualize_loss(cnn_model.loss_list)
    plot_genre_accuracy(cnn_model.accuracy_list)
    plot_valence_mae(cnn_model.va_mae_list)
    plot_loss_and_accuracy(cnn_model.loss_list, cnn_model.accuracy_list)


    


if __name__ == '__main__':
    main()