# Utiliser Python 3.9 comme base
FROM python:3.9

# Désactiver l'interaction pour éviter les interruptions
ENV DEBIAN_FRONTEND=noninteractive

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier uniquement le fichier requirements.txt en premier (optimisation du cache)
COPY requirements.txt /app/requirements.txt

# Vérifier si le fichier requirements.txt a bien été copié (debug)
RUN ls -l /app/requirements.txt || echo "❌ ERREUR: Le fichier requirements.txt n'a pas été copié !"

# Mettre à jour pip et installer les dépendances requises
RUN python -m pip install --upgrade pip certifi
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copier l'ensemble du projet dans le conteneur après l'installation des dépendances
COPY . /app/

# Exposer les ports nécessaires
EXPOSE 8000  
EXPOSE 9090  

# Assurer que le dossier des modèles existe
RUN mkdir -p /app/models

# Copier les modèles existants dans le conteneur
COPY models /app/models

# 🔥 Attendre 5 secondes avant de démarrer Uvicorn pour éviter des crashs
CMD ["sh", "-c", "sleep 5 && uvicorn api.api:app --host 0.0.0.0 --port 8000"]
