import sys
import os

import pytest
import numpy as np
# Ajoute le chemin du projet dans les variables d'environnement Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.train_image_audio_fusion import preprocess_image, preprocess_audio  # Import de tes fonctions




def test_preprocess_image():
    img = preprocess_image("data/images/test/cat.jpg")
    assert img is not None, "Erreur: Image non chargée"
    assert img.shape == (64, 64, 1), "Erreur: Format d'image incorrect"

def test_preprocess_audio():
    spec = preprocess_audio("data/audio/test/cat.wav")
    assert spec is not None, "Erreur: Audio non chargé"
    assert spec.shape == (64, 64, 1), "Erreur: Format audio incorrect"
