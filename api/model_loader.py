import tensorflow as tf

def load_image_model():
    model = tf.keras.models.load_model("models/image_classifier.h5", compile=True)
    print("✅ Modèle IMAGE chargé avec succès !")
    return model

def load_audio_model():
    model = tf.keras.models.load_model("models/audio_classifier.h5", compile=True)
    print("✅ Modèle AUDIO chargé avec succès !")
    return model

def load_fusion_model():
    model = tf.keras.models.load_model("models/image_audio_fusion_new_model.h5", compile=True)
    print("✅ Modèle FUSION chargé avec succès !")
    return model
