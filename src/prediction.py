# src/prediction.py

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from src.data_ingestion import fetch_data
from src.feature_engineering import generate_features_and_labels

def load_latest_model(model_path: str = "models/XGBoost_Trading_Model.pkl") -> Pipeline:
    """
    Charge le modèle (Pipeline) à partir du disque.
    Args:
        model_path: Chemin où se situe le modèle
    Returns:
        Pipeline: Modèle (Pipeline) chargé.
    """
    try:
        final_model = joblib.load(model_path)
        print(f"Modèle chargé avec succès depuis : {model_path}")
        return final_model
    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier modèle n'a pas été trouvé à {model_path}. Assurez vous d'avoir exécuter l'entraînement.")
    

def get_prediction(final_model: Pipeline, ticker: str, interval: str, period: str, features_train_cols: list) -> dict:
    """
    Prédire la dernière bougie en direct.
    Args:
        final_model : Le meilleur modèle (Pipeline)
        ticker : le symbole boursier
        interval : l'intervalle entre les bougies
        period: la période complète des données
        features_train_cols: la liste des colonnes du df de l'entraînement
    """
    df_raw = fetch_data(ticker=ticker, interval=interval, period=period)

    if df_raw.empty:
        return {"status": "error", "message": "Impossible de charger les données récentes."}

    # Calcul des features
    df_labeled = generate_features_and_labels(df=df_raw)

    # Dernière bougie pour la prédiction
    df_last_candle = df_labeled.iloc[[-1]].copy()

    list_barriers = ['tp_long', 'sl_long', 'tp_short', 'sl_short']
    latest_features = df_last_candle.drop(
        columns=list_barriers + ['label', 'Close', 'Open', 'High', 'Low', 'Volume', 'ema_12', 'ema_26', 'ema_50', 'ema_200'],
        errors='ignore'
    )

    latest_features = df_last_candle.loc[:, features_train_cols]

    # Prédiction
    prediction_prob = final_model.predict_proba(latest_features)
    prob_vente = prediction_prob[0][0]
    prob_achat = prediction_prob[0][1]
    prob_neutre = prediction_prob[0][2]

    # Détermination des niveaux TP/SL
    if prob_achat > prob_vente and prob_achat > prob_neutre:
        action = "ACHAT (LONG)"
        tp_level = df_last_candle['tp_long'].iloc[0]
        sl_level = df_last_candle['sl_long'].iloc[0]
    elif prob_vente > prob_achat and prob_vente > prob_neutre:
        action = "VENTE (SHORT)"
        tp_level = df_last_candle['tp_short'].iloc[0]
        sl_level = df_last_candle['sl_short'].iloc[0]
    else:
        action = "NE RIEN FAIRE"
        tp_level = None
        sl_level = None

    # Résultats

    return {
        "status": "success",
        "date_observation": latest_features.index[0],
        "action_predite": action,
        "probabilites": {
            "vente": f"{prob_vente:.4f}",
            "achat": f"{prob_achat:.4f}",
            "neutre": f"{prob_neutre:.4f}"
        },
        "niveaux_trading": {
            "take_profit": tp_level,
            "stop_loss": sl_level
        }
    }