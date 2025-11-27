# Dockerfile

# Image de base
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les dépendances dans le conteneur
COPY requirements.txt /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers dans le conteneur
COPY . /app/

# Création des fichiers models et mlruns
RUN mkdir -p /app/models
RUN mkdir -p /app/mlruns

# Commande de démarrage 
CMD ["python", "run_pipeline.py", "train_and_predict"]

