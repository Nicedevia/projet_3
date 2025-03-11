#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split

# --- 📚 Configuration et chemins ---
MAPPING_CSV = "data/data_fusion_model/fusion_mapping.csv"
MODEL_PATH = "models/fusion.h5"  # Nouveau modèle sauvegardé
IMAGE_MODEL_PATH = "models/image.h5"
AUDIO_MODEL_PATH = "models/audio.h5"

# --- 📌 Fonctions de prétraitement ---
def preprocess_image(image_path):
    """Charge et pré-traite une image en niveaux de gris."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(64, 64, 1)

def preprocess_audio(audio_path):
    """Charge un spectrogramme pré-généré et le pré-traite."""
    spec_path = audio_path.replace("cleaned", "spectrograms").replace(".wav", ".png")
    if not os.path.exists(spec_path):
        print(f"❌ Spectrogramme introuvable pour {audio_path} -> {spec_path}")
        return None
    spec_img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    if spec_img is None:
        return None
    spec_img = cv2.resize(spec_img, (64, 64)) / 255.0
    return spec_img.reshape(64, 64, 1)

# --- 📌 Chargement des données ---
def load_data():
    df = pd.read_csv(MAPPING_CSV)
    print(f"🔍 Nombre d'exemples dans le mapping : {len(df)}")

    X_images, X_audio, y_labels = [], [], []
    for _, row in df.iterrows():
        img = preprocess_image(row["image_path"])
        aud = preprocess_audio(row["audio_path"])
        if img is None or aud is None:
            continue
        X_images.append(img)
        X_audio.append(aud)
        y_labels.append(row["label"])
        
    return np.array(X_images), np.array(X_audio), np.array(y_labels)

# --- 📌 Chargement des modèles pré-entraînés ---
def load_pretrained_models():
    print("🔍 Chargement des modèles individuels pré-entraînés...")
    
    image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH, compile=False)
    audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, compile=False)
    
    # Vérification et extraction des features
    if isinstance(image_model, tf.keras.Sequential):
        image_feature_model = Model(inputs=image_model.layers[0].input, outputs=image_model.layers[-2].output, name="image_feature_extractor")
    else:
        image_feature_model = Model(inputs=image_model.input, outputs=image_model.layers[-2].output, name="image_feature_extractor")
    
    if isinstance(audio_model, tf.keras.Sequential):
        audio_feature_model = Model(inputs=audio_model.layers[0].input, outputs=audio_model.layers[-2].output, name="audio_feature_extractor")
    else:
        audio_feature_model = Model(inputs=audio_model.input, outputs=audio_model.layers[-2].output, name="audio_feature_extractor")
    
    image_feature_model.trainable = False
    audio_feature_model.trainable = False

    return image_feature_model, audio_feature_model

# --- 📌 Création du modèle fusionné ---
def build_fusion_model(image_feature_model, audio_feature_model):
    image_input = Input(shape=(64, 64, 1), name="image_input")
    audio_input = Input(shape=(64, 64, 1), name="audio_input")

    image_features = image_feature_model(image_input)
    audio_features = audio_feature_model(audio_input)

    combined_features = concatenate([image_features, audio_features], name="fusion_layer")
    fc = Dense(128, activation="relu")(combined_features)
    fc = Dropout(0.3)(fc)
    fc = Dense(64, activation="relu")(fc)
    final_output = Dense(3, activation="softmax", name="output_layer")(fc)

    fusion_model = Model(inputs=[image_input, audio_input], outputs=final_output, name="fusion_model")
    fusion_model.compile(optimizer="adam", 
                         loss="sparse_categorical_crossentropy", 
                         metrics=["accuracy"])

    return fusion_model

# --- 📌 Entraînement du modèle fusionné ---
def train_fusion_model(fusion_model, X_images, X_audio, y_labels):
    X_train_img, X_val_img, X_train_audio, X_val_audio, y_train, y_val = train_test_split(
        X_images, X_audio, y_labels, test_size=0.2, random_state=42
    )

    class_weights = {0: 1.0, 1: 1.0, 2: 2.0}

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        TqdmCallback()
    ]

    print("🚀 Entraînement du modèle fusionné...")
    fusion_model.fit([X_train_img, X_train_audio], y_train,
                     validation_data=([X_val_img, X_val_audio], y_val),
                     epochs=10, batch_size=16, callbacks=callbacks,
                     class_weight=class_weights)  

    return fusion_model

# --- 📌 Sauvegarde du modèle ---
def save_model_h5(model, filename=MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    model.save(filename)
    print(f"✅ Modèle sauvegardé en {filename}")

# --- 📌 Programme principal ---
def main():
    X_images, X_audio, y_labels = load_data()
    image_feature_model, audio_feature_model = load_pretrained_models()
    fusion_model = build_fusion_model(image_feature_model, audio_feature_model)
    fusion_model.summary()
    trained_model = train_fusion_model(fusion_model, X_images, X_audio, y_labels)
    save_model_h5(trained_model)

if __name__ == "__main__":
    main()
