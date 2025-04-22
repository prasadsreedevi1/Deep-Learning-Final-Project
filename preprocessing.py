import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from PIL import Image

def generate_spectrograms():
    audio_directory = "data/deam/DEAM_audio/MEMD_audio"          
    spectrogram_directory = "data/deam/DEAM_spectrograms"     
    os.makedirs(spectrogram_directory, exist_ok=True)

    for file in os.listdir(audio_directory):
        #getting the song id
        song_id = file.split(".")[0]
        audio_path = os.path.join(audio_directory, file)

        y, sr = librosa.load(audio_path, sr=44100)
        #we are skipping the first 15 seconds because of instructions of the authors. they said the first 15 minutes are not stable
        y = y[15 * sr:] 

        spec = librosa.feature.melspectrogram(y=y, sr=sr)

        spec = librosa.power_to_db(spec, ref=np.max)
        np.save(os.path.join(spectrogram_directory, f"{song_id}.npy"), spec)
        plt.figure(figsize=(10, 4))
        plt.imshow(spec, aspect='auto', origin='lower', cmap='magma')

        plt.axis('off')
        plt.savefig(os.path.join(spectrogram_directory, f"{song_id}.png"))
        plt.close()



    #getting data (average valence and average arousal) for each song
    va_df = pd.read_csv("/Users/cathyzhao/Desktop/cs1470/Deep-Learning-Final-Project/data/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv")

    # va_df2 = pd.read_csv("/Users/cathyzhao/Desktop/cs1470/Deep-Learning-Final-Project/data/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv")

    # va_df = pd.concat([va_df1, va_df2], ignore_index=True)
    va_df.columns = va_df.columns.str.strip() 

    va_df["valence"] = va_df["valence_mean"] / 9
    va_df["arousal"] = va_df["arousal_mean"] / 9


    # #normalizing values
    # va_df["valence"] = va_df["valence_mean"] / 9
    # va_df["arousal"] = va_df["arousal_mean"] / 9


    va_df["spec_path"] = va_df["song_id"].astype(str).apply(lambda x: f"data/deam/DEAM_spectrograms/{x}.npy")

    #metadata for songs 2-1000
    meta1 = pd.read_csv("data/metadata/metadata_2013.csv")


    #metadata for songs 1001-2000
    # print("meta2")
    # meta2 = pd.read_csv("data/metadata/metadata_2014.csv")
    # print(meta2.head(2))
    # print("Columns:", meta2.columns.tolist())
    meta2_rows = []
    with open("data/metadata/metadata_2014.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 6 and row[0].isdigit():
                song_id = int(row[0])
                genre = row[4].strip()
                meta2_rows.append((song_id, genre))

    meta2 = pd.DataFrame(meta2_rows, columns=["song_id", "Genre"])



    print("looping through meta")
    for df in [meta1, meta2]:
        print("strip columns")
        df.columns = df.columns.str.strip()
        if "Id" in df.columns or "id" in df.columns:
            df.rename(columns={"Id": "song_id"}, inplace=True)
        print("song_id through df")
        print(df["song_id"])
        df["song_id"] = df["song_id"].astype(int)
        print("genre")
        df["Genre"] = df["Genre"].str.strip()

    #concatenating them because they are separate
    genre_df = pd.concat([meta1, meta2], ignore_index=True)

    #merging valence and arousal data with genre and song id
    df = pd.merge(va_df, genre_df[["song_id", "Genre"]], on="song_id", how="inner")

    #making genre category
    df["genre_id"] = df["Genre"].astype('category')

    #making each category of genre a number
    genres = sorted(df["Genre"].unique())
    genre_to_idx = {}

    for i in range(len(genres)):
        genre_to_idx[genres[i]] = i

    df["genre_id"] = df["Genre"].map(genre_to_idx)

    df.to_csv("data/deam/final_song_labels.csv", index=False)
    
def get_data(path):
    df = pd.read_csv(path)

    img_array = []
    labels = []
    print(f"Total images to process: {len(df)}")

    #loop through the CSV to load each image and label
    for i, row in df.iterrows():
        print(f"Processing image {i + 1}/{len(df)}")
        image_path_png = row['spec_path'].replace('.npy', '.png')
        
        try:
            img = Image.open(image_path_png).convert('RGB')
            
            print(f"Loaded image: {image_path_png}, size: {img.size}")
            
            #normalize
            img_as_array = np.array(img) / 255.0 
            
            img_array.append(img_as_array)
            labels.append(row['genre_id'])
        except Exception as e:
            print(f"Error loading {image_path_png}: {e}")
            continue

    #convert lists to numpy arrays
    img_array = np.array(img_array)
    labels = np.array(labels)

    print("Final shape of img_array:", img_array.shape)
    print("Final shape of labels:", labels.shape)
    
    return img_array, labels


def main():
    generate_spectrograms()
    return

if __name__ == '__main__':
    main()