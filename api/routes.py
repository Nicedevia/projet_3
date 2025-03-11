from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import numpy as np
import io
import cv2
import librosa
import tensorflow as tf
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from .model_loader import load_image_model, load_audio_model, load_fusion_model

DEFAULT_THRESHOLD = 0.5
router = APIRouter()

# Charger les modèles
image_model = load_image_model()
audio_model = load_audio_model()
fusion_model = load_fusion_model()

# ✅ Forcer l'initialisation des modèles pour éviter les erreurs
fake_image = np.random.rand(1, 64, 64, 1).astype(np.float32)
fake_audio = np.random.rand(1, 64, 64, 1).astype(np.float32)

_ = image_model.predict(fake_image)  # 🔥 Initialisation du modèle image
_ = audio_model.predict(fake_audio)  # 🔥 Initialisation du modèle audio

# ✅ Création des extracteurs de caractéristiques après initialisation
image_extractor = tf.keras.Model(inputs=image_model.input, outputs=image_model.layers[-2].output)
audio_extractor = tf.keras.Model(inputs=audio_model.input, outputs=audio_model.layers[-2].output)

# ---------------------------
# Définition des métriques Prometheus
# ---------------------------
request_counter = Counter("http_requests_total", "Nombre total de requêtes reçues")
prediction_duration = Histogram("model_prediction_duration_seconds", "Durée des prédictions du modèle en secondes")
prediction_errors = Counter("model_prediction_errors_total", "Nombre total d'erreurs lors des prédictions")

@router.get("/metrics", tags=["Monitoring"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ---------------------------
# Fonctions de Prétraitement
# ---------------------------
def preprocess_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Image invalide")
    img = cv2.resize(img, (64, 64)) / 255.0
    img = img.reshape(1, 64, 64, 1)
    features = image_extractor.predict(img)
    return features

def preprocess_audio_from_bytes(audio_bytes: bytes) -> np.ndarray:
    try:
        audio_stream = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_stream, sr=22050, duration=2)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Audio invalide") from e

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    spec_img = cv2.resize(S_db, (64, 64))
    spec_img = (spec_img - spec_img.min()) / (spec_img.max() - spec_img.min())
    spec_img = spec_img.reshape(1, 64, 64, 1)
    features = audio_extractor.predict(spec_img)
    return features

# ---------------------------
# Endpoints de Prédiction
# ---------------------------

@router.post("/predict/multimodal", tags=["Prediction"])
async def predict_multimodal(
    image_file: UploadFile = File(..., description="Fichier image (JPEG ou PNG)"),
    audio_file: UploadFile = File(..., description="Fichier audio (WAV)"),
    threshold: float = Query(DEFAULT_THRESHOLD, ge=0, le=1, description="Seuil pour la classification (0 à 1)")
):
    if image_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Format d'image non supporté")
    if audio_file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Format audio non supporté")
    
    image_bytes = await image_file.read()
    audio_bytes = await audio_file.read()
    
    try:
        features_image = preprocess_image_from_bytes(image_bytes)
        features_audio = preprocess_audio_from_bytes(audio_bytes)
    except Exception as e:
        prediction_errors.inc()
        raise e

    start_time = time.time()
    prediction = fusion_model.predict([features_image, features_audio])
    duration = time.time() - start_time
    prediction_duration.observe(duration)
    
    label = "Chien" if prediction[0][0] > threshold else "Chat"
    confidence = float(prediction[0][0])
    return {"prediction": label, "confidence": confidence, "used_threshold": threshold}
