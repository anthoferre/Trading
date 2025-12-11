# run_pipeline.py (Mode Forcé : Entraînement puis Prédiction)

import mlflow
import numpy as np
import yaml

from src.data_ingestion import fetch_data
from src.feature_engineering import generate_features_and_labels
from src.prediction import get_prediction, load_latest_model
from src.preprocessing import drop_multicol, get_prepocessor
from src.training import run_tscv_training

MLFLOW_TRACKING_URI = "file:///app/mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "Trading_Model_TSCV_Experiment"


def load_config(*, config_path: str = "config/config.yaml") -> dict:
    """Charge la configuration à partir du fichier YAML."""
    # Le fichier doit exister, sinon l'exécution échoue ici.
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def prepare_data(config: dict) -> tuple:
    """Orchestre le Pipeline d'ingestion et de feature engineering."""

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

    # Nettoyage des NaT et des colonnes de label/barrières
    df_to_train = df_labeled.copy().dropna(axis='index', how='any')
    target = df_to_train['label']
    features = df_to_train.drop(columns=['label', 'tp_long', 'sl_long', 'tp_short', 'sl_short'],
                                errors='ignore')

    cols_to_drop = features.select_dtypes(include=np.number).columns[features.select_dtypes(include=np.number).median() > 1000]
    features_for_model = features.drop(labels=cols_to_drop, errors='ignore')
    features_all_cols = features.columns.tolist()

    preprocessor = get_prepocessor(features_for_model)

    return df_labeled, features_for_model, target, preprocessor, features_all_cols


def execute_train_mode(features, target, preprocessor, config):
    """Exécute l'entraînement TSCV complet et sauvegarde le modèle."""

    N_SPLITS = int(config['training']['n_splits'])

    print("🚀 Démarrage de l'entraînement du modèle ...")
    final_model, tscv_accuracy = run_tscv_training(
        features=features,
        target=target,
        N_SPLITS=N_SPLITS,
        preprocessor=preprocessor,
        config=config
    )
    print(f"✅ Modèle sauvegardé avec précision TSCV: {tscv_accuracy:.4f}")


def execute_predict_mode(features_all_cols, config):
    """Exécute la prédiction sur la dernière bougie."""

    print("🎯 Chargement du modèle et Prédiction en temps réel")
    try:
        # Le modèle est chargé depuis le chemin de sauvegarde
        final_model = load_latest_model(config['training']['model_path'])

        prediction_result = get_prediction(
            final_model=final_model,
            ticker=config['data']['ticker'],
            interval=config['data']['interval'],
            period=config['data']['period'],
            features_train_cols=features_all_cols
        )

        print("\n" + "=" * 70)
        print(f"RÉSULTAT DE LA PRÉDICTION EN PRODUCTION POUR {config['data']['ticker']} {config['data']['interval']}")
        print("=" * 70)
        print(f"Date de l'observation : {prediction_result.get('date_observation')}")
        print(f"Action prédite : **{prediction_result.get('action_predite')}**")
        print(f"Prob. Achat/Vente/Neutre : {prediction_result['probabilites']['achat']} /  \
               {prediction_result['probabilites']['vente']} / {prediction_result['probabilites']['neutre']}")

        if prediction_result.get('action_predite') in ['ACHAT', 'VENTE']:
            sl = prediction_result['niveaux_trading']['stop_loss']
            tp = prediction_result['niveaux_trading']['take_profit']
            print(f"Niveaux de trading : SL={sl:.4f}, TP={tp:.4f}")
        else:
            print("Niveaux de trading : Non applicable (Action NEUTRE)")
        # -----------------------------------------------

        print("=" * 70)

        return prediction_result

    except FileNotFoundError as e:
        print(f"❌ Erreur de prédiction : Modèle non trouvé. Assurez-vous d'avoir exécuté l'entraînement auparavant.  \
               Détails: {e}")
        return None


def execute_train_and_predict_forced(features_train, target, preprocessor, features_all_cols, config):
    """Exécute l'entraînement WFV puis la prédiction immédiate."""

    # 1. Entraînement
    execute_train_mode(features_train, target, preprocessor, config)

    # 2. Prédiction (utilise le modèle fraîchement sauvegardé)
    prediction_result = execute_predict_mode(features_all_cols, config)

    return prediction_result


def main():
    """
    Exécute le Pipeline de trading en mode 'Entraînement et Prédiction' (forcé).
    """

    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        config = load_config()
    except FileNotFoundError:
        mlflow.create_experiment(EXPERIMENT_NAME)
        print("❌ Erreur critique : Le fichier 'config/config.yaml' est introuvable. Arrêt du pipeline.")
        return

    print("⚡Entraînement et Prédiction Séquentiels")

    df_labeled, features_for_models, target, preprocessor, features_all_cols = prepare_data(config)

    features_without_multicol = drop_multicol(features=features_for_models, threshold=0.85)

    # 2. Exécution séquentielle
    execute_train_and_predict_forced(features_without_multicol, target, preprocessor, features_all_cols, config)


if __name__ == "__main__":
    main()
