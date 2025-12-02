from fastapi import FastAPI
import joblib
import datetime
import yaml
from src.prediction import get_prediction, load_latest_model 

# IMPORTANT : Utilisation des chemins absolus DOCKER
MODEL_PATH = "/app/models/XGBoost_Trading_Model.pkl"
FEATURES_LIST_PATH = "/app/models/XGBoost_Trading_Model_features.pkl"
CONFIG_PATH = "/app/config/config.yaml"

# Variables globales chargées une seule fois
model = None
FEATURES_LIST = []
CONFIG = {}

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
            interval=CONFIG['data']['interval'],
            period=CONFIG['data']['period'],
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