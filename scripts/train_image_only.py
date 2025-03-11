#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm.keras import TqdmCallback

# Répertoire pour les images d'entraînement
TRAIN_IMAGE_DIR = "data/images/cleaned/training_set"

# Paramètres
IMG_SIZE = (64, 64)
BATCH_SIZE = 128
EPOCHS = 20

# Création d'un générateur d'images avec augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Création des générateurs pour chaque catégorie (Chat et Chien)
train_generator = train_datagen.flow_from_directory(
    TRAIN_IMAGE_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",  # Conserve le format grayscale
    batch_size=BATCH_SIZE,
    class_mode="binary",  # 0 pour Chat, 1 pour Chien
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_IMAGE_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset='validation'
)

# Construction d'un modèle CNN amélioré pour les images
model = tf.keras.Sequential([
    # Bloc 1
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    
    # Bloc 2
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    
    # Bloc 3
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    
    # Classification
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Callbacks pour l'entraînement
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    TqdmCallback(verbose=1)
]

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Sauvegarde du modèle
os.makedirs("models", exist_ok=True)
model.save("models/image.h5")
print("✅ Modèle image sauvegardé !")

