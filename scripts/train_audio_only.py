#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import librosa
import librosa.display
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, TensorBoard
from tqdm.keras import TqdmCallback 
# Répertoires sources pour les données audio training
audio_train_dir_cats = "data/audio/cleaned/train/cats"
audio_train_dir_dogs = "data/audio/cleaned/train/dogs"

X_audio, y = [], []

# Fonction pour générer un spectrogramme à partir d'un fichier audio
def process_audio_file(audio_path):
    try:
        y_audio, sr = librosa.load(audio_path, sr=22050, duration=2)
    except Exception as e:
        print(f"❌ Erreur de chargement {audio_path} : {e}")
        return None
    # Calcul du spectrogramme et conversion en échelle logarithmique
    spectrogram = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Création explicite d'une figure et d'un axe pour éviter les erreurs
    fig, ax = plt.subplots(figsize=(3, 3))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.axis("off")
    
    temp_img_path = "temp_spectrogram.png"
    fig.savefig(temp_img_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
    # Chargement de l'image générée avec OpenCV
    spec_img = cv2.imread(temp_img_path, cv2.IMREAD_GRAYSCALE)
    if spec_img is None:
        print(f"❌ Erreur de lecture du spectrogramme généré pour {audio_path}")
        return None
    spec_img = cv2.resize(spec_img, (64, 64)) / 255.0
    return spec_img

# Traitement des fichiers audio pour les chats (label 0)
if os.path.exists(audio_train_dir_cats):
    for file in os.listdir(audio_train_dir_cats):
        if file.lower().endswith(".wav"):
            audio_path = os.path.join(audio_train_dir_cats, file)
            spec_img = process_audio_file(audio_path)
            if spec_img is not None:
                X_audio.append(spec_img)
                y.append(0)
else:
    print(f"❌ Répertoire introuvable : {audio_train_dir_cats}")

# Traitement des fichiers audio pour les chiens (label 1)
if os.path.exists(audio_train_dir_dogs):
    for file in os.listdir(audio_train_dir_dogs):
        if file.lower().endswith(".wav"):
            audio_path = os.path.join(audio_train_dir_dogs, file)
            spec_img = process_audio_file(audio_path)
            if spec_img is not None:
                X_audio.append(spec_img)
                y.append(1)
else:
    print(f"❌ Répertoire introuvable : {audio_train_dir_dogs}")

# Conversion en tenseurs et mise en forme
X_audio = np.array(X_audio).reshape(-1, 64, 64, 1)
y = np.array(y)

print(f"Total audio traités : {len(y)}")

# Construction du modèle CNN pour les spectrogrammes audio
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
class LoggingCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"----- Epoch {epoch+1} started -----")
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        acc = logs.get("accuracy")
        val_loss = logs.get("val_loss")
        val_acc = logs.get("val_accuracy")
        print(f"----- Epoch {epoch+1} ended: loss={loss:.4f}, accuracy={acc:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_acc:.4f} -----")

# Création d'un TensorBoard callback
tensorboard_callback = TensorBoard(log_dir='./logs/audio', histogram_freq=1)

# Ajout de ce callback dans la liste
callbacks = [LoggingCallback(), TqdmCallback(verbose=1), tensorboard_callback]

# Entraînement du modèle
model.fit(X_audio, y, epochs=10, validation_split=0.2, batch_size=16, callbacks=callbacks)

# Sauvegarde du modèle entraîné
os.makedirs("models", exist_ok=True)
model.save("models/audio.keras")
print("✅ Modèle audio sauvegardé !")
