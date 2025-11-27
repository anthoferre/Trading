# run_pipeline.py

from src.data_ingestion import fetch_data
from src.feature_engineering import generate_features_and_labels
from src.preprocessing import get_prepocessor
from src.prediction import get_prediction, load_latest_model
from src.training import run_wfv_training
import yaml
import argparse

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Charge la configuration à partir du fichier YAML."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def prepare_data(config: dict) -> tuple:
    """Orchestre le Pipeline"""
    
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

    return df_labeled, features, target, preprocessor, features_train_cols

def execute_train_mode(features, target, preprocessor, config):
    """Exécute l'entraînement WFV complet et sauvegarde le modèle"""

    TRAIN_SIZE = int(len(features) * config['training']['train_size_ratio'])
    TEST_SIZE = int(len(features) * config['training']['test_size_ratio'])
    STEP_SIZE = TEST_SIZE 

    
    print("🚀 Démarrage de l'entraînement du modèle ...")
    final_model, wfc_accuracy = run_wfv_training(
        features=features,
        target=target,
        TRAIN_SIZE=TRAIN_SIZE,
        TEST_SIZE=TEST_SIZE,
        STEP_SIZE=STEP_SIZE,
        preprocessor=preprocessor,
        config=config
    )
    print(f"✅ Modèle sauvegardé avec précision WFV: {wfc_accuracy:.4f}")

def execute_predict_mode(features_train_cols, config):
    """Exécute la prédiction sur la dernière bougie"""

    print("🎯 Chargement du modèle et Prédiction en temps réel")
    try:
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

        return prediction_result

    except FileNotFoundError as e:
        print(f"Erreur de prédiction : {e}")
        return None

def main(mode: str):
    """Orchestre le Pipeline selon le mode sélectionné"""
    config = load_config()

    if mode == 'train':
        _, features, target, preprocessor, _ = prepare_data(config)
        execute_train_mode(features, target, preprocessor, config)
    
    elif mode == 'predict':
        df_labeled, features, target, preprocessor, features_train_cols = prepare_data(config)
        execute_predict_mode(features_train_cols, config)
    else:
        print(f"Mode {mode} non reconnu. Veuillez choisir 'train' ou 'predict'.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Lance le Pipeline de trading en mode 'train' ou 'predict'"
    )

    parser.add_argument(
        "mode",
        type=str,
        choices=['train', 'predict', 'train_and_predict'],
        default="train_and_predict",
        nargs='?',
        help="Le mode d'exécution désiré: 'train' ou 'predict'."
    )

    args = parser.parse_args()
    
    main(args.mode)