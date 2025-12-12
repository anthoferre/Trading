import datetime
import traceback
from io import BytesIO

import joblib
import matplotlib
import pandas as pd

# Force Matplotlib à utiliser le backend 'Agg' (non interactif, pour l'écriture de fichiers)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Le reste de vos imports...
import mplfinance as mpf
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from src.prediction import get_prediction, load_latest_model

# IMPORTANT : Utilisation des chemins absolus DOCKER
DF_PATH = "/app/models/df_labeled.pkl"
MODEL_PATH = "/app/models/XGBoost_Trading_Model.pkl"
FEATURES_LIST_PATH = "/app/models/XGBoost_Trading_Model_features.pkl"
CONFIG_PATH = "/app/config/config.yaml"

# Variables globales chargées une seule fois
model = None
FEATURES_LIST = []
CONFIG = {}
df_global = None

# --- CHARGEMENT INITIAL ---
try:
    # 1. Chargement de la configuration (cruciale pour les paramètres Ticker, Interval, etc.)
    with open(CONFIG_PATH, 'r') as file:
        CONFIG = yaml.safe_load(file)
    print("✅ Configuration chargée avec succès.")

    # 2. Chargement des features entraînées
    FEATURES_LIST = joblib.load(FEATURES_LIST_PATH)
    print("✅ Liste des features chargée avec succès.")

    # 3. Chargement du modèle
    model = load_latest_model(MODEL_PATH)
    print("✅ Modèle chargé avec succès.")

    # 4. Chargement du dataframe
    df_global = joblib.load(DF_PATH)
    print("✅ Dataframe chargé avec succès.")

except Exception as e:
    print(f"❌ Erreur critique lors du démarrage de l'API: {e}")
    # Ne pas lancer l'API si le modèle ou la config manque


# Initialisation de l'API
app = FastAPI(title="Trading Prediction API", version="1.0")

# --- ENDPOINTS ---


@app.get("/health")
def check_health():
    """Endpoint pour vérifier si l'API est active et si le modèle est chargé."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.datetime.now().isoformat()
    }

# Fichier : app.py (Modification de l'endpoint /chart)


@app.get("/chart", response_class=Response)
async def generate_chart(num_bougies: int = 100):
    """
    Génère un graphique en chandeliers à partir du DataFrame pré-calculé
    et le renvoie en tant qu'image PNG.
    """
    global df_global

    if df_global is None:
        raise HTTPException(status_code=503, detail="Données de graphique non disponibles. Vérifiez les logs de démarrage.")

    try:
        # 1. Utilisation et filtration du DataFrame global
        df_candle = df_global.tail(num_bougies)

        print(f"{isinstance(df_candle.index, pd.DatetimeIndex)}")

        # Vérification si le DataFrame filtré est vide
        if df_candle.empty:
            raise HTTPException(status_code=404, detail="Pas assez de données historiques pour afficher les bougies demandées.")

        # 2. Définition des addplots (Utilise les colonnes de df_candle)
        apds = [
            mpf.make_addplot(df_candle['rsi'], panel=2, ylabel='RSI'),
            mpf.make_addplot(df_candle['lower_band'], panel=0, color='green', label='Lower Band'),
            mpf.make_addplot(df_candle['upper_band'], panel=0, color='red', label='Upper Band'),
            mpf.make_addplot(df_candle['middle_band'], panel=0, color='blue', label='Middle Band'),
            mpf.make_addplot(df_candle['macd'], panel=3, color='blue', label='MACD'),
            mpf.make_addplot(df_candle['signal'], panel=3, color='green', label='Signal'),
        ]

        # 3. Enregistrement de la figure en mémoire
        buf = BytesIO()

        mpf.plot(
            data=df_candle,
            addplot=apds,
            type='candle',
            ylabel='Prix',
            style='yahoo',
            title=f"Analyse {CONFIG.get('data', {}).get('ticker', 'N/A')} - {num_bougies} bougies",
            savefig=dict(fname=buf, format='png'),
            figsize=(12, 8),
            volume='Volume' in df_candle.columns  # S'affiche seulement si la colonne existe
        )

        # 4. Renvoyer l'image
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")

    except KeyError as e:
        # Attrape si une colonne d'indicateur est manquante dans le DataFrame chargé
        raise HTTPException(status_code=500, detail=f"Colonne d'indicateur manquante dans le DataFrame: {e}.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur de génération du graphique: {str(e)}")


@app.get("/predict")
def predict_realtime_candle():
    """
    Appelle la fonction get_prediction() centralisée pour exécuter l'ingestion,
    le feature engineering et la prédiction sur la dernière bougie.
    """
    if model is None:
        return {"error": "Modèle non disponible. Vérifiez les logs de démarrage."}

    if not FEATURES_LIST or not CONFIG:
        return {"error": "Configuration ou Features manquantes."}

    try:
        # APPEL DIRECT À LA LOGIQUE CENTRALISÉE DANS src/prediction.py
        result = get_prediction(
            final_model=model,
            ticker=CONFIG['data']['ticker'],
            features_train_cols=FEATURES_LIST
        )

        # Le résultat est déjà formaté comme un dictionnaire par get_prediction()
        return result

    except Exception as e:
        # Si une erreur se produit pendant l'exécution (ex: pas de connexion yfinance)
        return {"status": "error", "message": f"Erreur lors de la prédiction automatique: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    # Le port est 80 dans le conteneur, mappé à 8000 par Docker-compose
    uvicorn.run(app, host="0.0.0.0", port=80)
