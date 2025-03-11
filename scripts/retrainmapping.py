import os
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split

# --- 📂 Config ---
DATA_DIR = "data_retrain/training"
MAPPING_CSV = "data_retrain/mapping.csv"
MODEL_PATH = "models/fusion.h5"
NEW_MODEL_PATH = "models/image_audio_fusion_model_retrained.h5"

# --- 📌 Vérification du dossier models ---
os.makedirs("models", exist_ok=True)

# --- 📌 Chargement des données ---
def load_data():
    df = pd.read_csv(MAPPING_CSV)
    X_images, X_audio, y_labels = [], [], []
    for _, row in df.iterrows():
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64)) / 255.0
            X_images.append(img.reshape(64, 64, 1))
        
        y, sr = librosa.load(row["audio_path"], sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = cv2.resize(S_db, (64, 64))
        X_audio.append(S_db.reshape(64, 64, 1))
        
        y_labels.append(row["label"])
    
    print(f"✅ Données chargées : {len(X_images)} échantillons valides")
    return np.array(X_images), np.array(X_audio), np.array(y_labels)

# --- 📌 Chargement et modification du modèle ---
def load_and_modify_model():
    old_model = load_model(MODEL_PATH)
    for layer in old_model.layers:
        layer.trainable = False

    last_layer_output = old_model.layers[-2].output
    hidden = Dense(128, activation="relu")(last_layer_output)
    hidden = Dropout(0.2)(hidden)
    new_output = Dense(3, activation="softmax")(hidden)

    new_model = Model(inputs=old_model.input, outputs=new_output)
    new_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return new_model

# --- 📌 Entraînement ---
def main():
    X_images, X_audio, y_labels = load_data()
    new_model = load_and_modify_model()
    X_train, X_val, y_train, y_val = train_test_split(X_images, y_labels, test_size=0.2, random_state=42)

    new_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)
    new_model.save(NEW_MODEL_PATH)
    print(f"✅ Modèle réentraîné et sauvegardé sous {NEW_MODEL_PATH}")

if __name__ == "__main__":
    main()
