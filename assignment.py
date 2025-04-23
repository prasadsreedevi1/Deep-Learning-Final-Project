import os
import tensorflow as tf
import numpy as np
import random
import math
import pandas as pd
from preprocessing import get_data

def split_train_test():
    
    #get list of song_ids
    df = pd.read_csv('data/deam/final_song_labels.csv')
    song_ids = df['song_id'].to_numpy()

    #print(song_ids[0:10])
    #print(song_ids[-10:])
    
    #random shuffle
    shuffled_ids = np.random.permutation(song_ids)
    
    #split 80-20ish
    split = int(len(shuffled_ids) * 0.8)
    train_ids = shuffled_ids[:split]
    test_ids = shuffled_ids[split:]
    
    train_df = df[df['song_id'].isin(train_ids)].reset_index(drop=True)
    test_df = df[df['song_id'].isin(test_ids)].reset_index(drop=True)
    
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    
    #save as new csv
    train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    
    
    
def main():
    
    #get train and test data
    split_train_test()
    
    # test_imgs, test_labels = get_data('data/deam/test_data.csv')
    # train_imgs, train_labels = get_data('data/deam/train_data.csv')

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
