# run_pipeline.py

import pandas as pd
import numpy as np
import mplfinance as mpf
from src.data_ingestion import fetch_data
from src.feature_engineering import generate_features_and_labels
from src.preprocessing import get_prepocessor, create_training_pipeline
from src.prediction import get_prediction, load_latest_model
from src.training import run_wfv_training
import yaml

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Charge la configuration à partir du fichier YAML."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(mode: str = "train_and_predict"):
    """Orchestre le Pipeline"""
    config = load_config()

    df_raw = fetch_data(
        ticker=config['data']['ticker'],
        interval=config['data']['interval'],
        period=config['data']['period']
    )

    df_labeled = generate_features_and_labels(
        df=df_raw,
        tp_mult=config['strategy']['tp_mult'],
        sl_mult=config['strategy']['sl_mult']
    )

    df_to_train = df_labeled.copy().dropna(axis='index', how='any')
    target = df_to_train['label']
    features = df_to_train.drop(columns=['label', 'tp_long', 'sl_long', 'tp_short', 'sl_short'], errors='ignore')

    # Enregistrer le nom des colonnes pour la prédiction aussi (les mêmes colonnes exactement)
    features_train_cols = features.columns.tolist()

    # Préprocesseur pour l'encodage des variables catégorielles
    preprocessor = get_prepocessor(features)

    TRAIN_SIZE = int(len(features) * config['training']['train_size_ratio'])
    TEST_SIZE = int(len(features) * config['training']['test_size_ratio'])
    STEP_SIZE = TEST_SIZE 

    if mode == "train_and_predict" or mode == "train":
        print("Entraînement du modèle")
        final_model, wfc_accuracy = run_wfv_training(
            features=features,
            target=target,
            TRAIN_SIZE=TRAIN_SIZE,
            TEST_SIZE=TEST_SIZE,
            STEP_SIZE=STEP_SIZE,
            preprocessor=preprocessor,
            config=config
        )
        print(f"Modèle sauvegardé avec précision WFV: {wfc_accuracy:.4f}")

    if mode == "train_and_predict" or mode == "predict":
        print("Prédiction en temps réel")
        try:
            if mode == 'predict':
                final_model = load_latest_model(config['training']['model_path'])
        
            prediction_result = get_prediction(
                final_model=final_model,
                ticker=config['data']['ticker'],
                interval=config['data']['interval'],
                period=config['data']['period'],
                features_train_cols=features_train_cols
            )

            print("\n" + "="*70)
            print(f"RÉSULTAT DE LA PRÉDICTION EN PRODUCTION POUR {config['data']['ticker']}")
            print("="*70)
            print(f"Date de l'observation : {prediction_result.get('date_observation')}")
            print(f"Action prédite : **{prediction_result.get('action_predite')}**")
            print(f"Prob. Achat/Vente/Neutre : {prediction_result['probabilites']['achat']} / {prediction_result['probabilites']['vente']} / {prediction_result['probabilites']['neutre']}")
            print(f"Niveaux de trading : SL={prediction_result['niveaux_trading']['stop_loss']:.4f}, TP={prediction_result['niveaux_trading']['take_profit']:.4f}")
            print("="*70)

        except FileNotFoundError as e:
            print(f"Erreur de prédiction : {e}")


if __name__ == "__main__":
    main(mode='train_and_predict')