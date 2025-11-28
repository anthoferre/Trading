from fastapi import FastAPI
from pydantic import BaseModel, create_model
import joblib
import pandas as pd
import uvicorn
import datetime

MODEL_PATH = "/app/models/XGBoost_Trading_Model.pkl"
FEATURES_LIST_PATH = "/app/models/XGBoost_Trading_Model_features.pkl"

model = None
FEATURES_LIST = []

try:
    FEATURES_LIST = joblib.load(FEATURES_LIST_PATH)
    print("✅ Liste des features chargée avec succès.")

    model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé avec succès.")

    fields = {name: (float, ...) for name in FEATURES_LIST}
    FeatureInput = create_model('FeatureInput', **fields)
    print("✅ Schéma Pydantic créé dynamiquement.")

except FileNotFoundError as e:
    print(f"❌ Erreur: Fichier non trouvé: {e}")
    FeatureInput = BaseModel
except Exception as e:
    print(f"❌ Erreur inattendue lors du chargement: {e}")
    FeatureInput = BaseModel

# Initialisation de l'API
app = FastAPI(title="Trading Prediction API", version="1.0")

# Endpoints de l'API

@app.get("/health")
def check_health():
    """Endpoint pour vérifier si l'API est active et si le modèle est chargé."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/predict")
def predict_last_candle(data: FeatureInput):
    """Effectue une prédiction d'action sur la dernière bougie (Achat/Vente/Ne rien faire) à partir des features fournis."""
    if model is None:
        return {"error": "Modèle non disponible. Entraînez le Pipeline d'abord."}
    try:
        # Convertir les données Pydantic en DataFrame pour le Pipeline
        input_data_df = pd.DataFrame([data.model_dump()])

        prediction_int = model.predict(input_data_df)[0]
        prediction_proba = model.predict_proba(input_data_df)[0].tolist()

        label_map = {0: "Vente", 1: "Achat", 2: "Neutre"}
        action_predite = label_map.get(prediction_int, "Inconnu")

        return {
            "prediction_action": action_predite,
            "probabilities": {
                "Ne rien faire": prediction_proba[2],
                "Achat": prediction_proba[1],
                "Vente": prediction_proba[0]
            },
            "model_version": "V1.0"
        }
    
    except Exception as e:
        return {"error": f"Erreur lors de la prédiction: {str(e)}"}
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)