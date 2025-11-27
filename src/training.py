# src/training.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
from src.preprocessing import create_training_pipeline
import mlflow
import joblib


param_grid = {
    'feature_selection__k': [5,10,20],
    'classifier__n_estimators' : [100,200],
}

def hyperparameter_optimization(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> dict:
    """
    Effectue une recherche par grille pour trouver les meilleurs hyperparamètres (KBest, n_estimators).
    Args:
        X_train: Features d'entrainement
        y_train: Labels d'entraînement
        preprocessor: Le ColumnTransformer à utiliser dans la Pipeline
    Returns:
        dict: Les meilleurs paramètres trouvés.
    """

    temp_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest()),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    ])


    grid_search = GridSearchCV(
            estimator=temp_pipeline,
            param_grid=param_grid,
            cv=3,
            scoring='f1_macro',
            verbose=0,
            n_jobs=-1
        )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

def run_wfv_training(features: pd.DataFrame, target: pd.Series, TRAIN_SIZE: int, TEST_SIZE: int, STEP_SIZE: int, preprocessor: ColumnTransformer, config: dict) -> tuple[Pipeline, float]:
    """
    Exécute la Walk Forward Validation (WFV), loggue les métriques avec MLFlow et enregistre le modèle.
    Args:
        features: caractéristiques
        target : variable cible (label)
        TRAIN_SIZE, TEST_SIZE, STEP_SIZE : Paramètres de découpage WFV.
        preprocessor: ColumnTransformer 
        config: Dictionnaire de configuration(pour logguer les métadonnées) 
    Returns:
        tuple[Pipeline, float]: Le modèle final entraîné et la précision WFV cumulée.
    """
    # Initialisation MLFLOW
    mlflow.set_tracking_uri("http://mlflow-tracking-server:5000")
    mlflow.set_experiment("Trading_Model_Training")
    with mlflow.start_run():

        # Logguer les paramètres de configuration/environnement
        mlflow.log_params(config)
        mlflow.log_params({"TRAIN_SIZE": TRAIN_SIZE, "TEST_SIZE": TEST_SIZE, "STEP_SIZE": STEP_SIZE})


        n_cycles = int((len(features) - TRAIN_SIZE) / STEP_SIZE)
        if n_cycles < 1: 
            print("Pas assez de données pour le WFV. Utilisation de la séparation standard.")
            n_cycles = 1 # Force un cycle
        print(f"\n--- DÉMARRAGE WFV avec {n_cycles} cycles---")

        # Recherche des hyperparamètres (k pour le KBest et n_estimators pour le xgboost)
        best_params = hyperparameter_optimization(
            X_train=features.iloc[:TRAIN_SIZE],
            y_train=target.iloc[:TRAIN_SIZE],
            preprocessor=preprocessor
        )
        print(f"Meilleurs hyperparamètres initiaux : {best_params}")
        mlflow.log_params(best_params)

        all_predictions = pd.Series(dtype=int)
        all_test_targets = pd.Series(dtype=int)
        final_model = None
            
        # BOUCLE WFV
        for cycle in range(n_cycles):
            # Découpage temporel des données
            start_train = cycle * STEP_SIZE
            end_train = start_train + TRAIN_SIZE
            start_test = end_train
            end_test = start_test + TEST_SIZE
            
            X_train = features.iloc[start_train:end_train]
            y_train = target.iloc[start_train:end_train]
            X_test = features.iloc[start_test:end_test]
            y_test = target.iloc[start_test:end_test]
            
            if X_test.empty or X_train.empty: break

            sample_weights = compute_sample_weight(
                class_weight='balanced', 
                y=y_train
            )

            # Création du Pipeline avec les meilleurs hyperparamètres
            current_pipeline = create_training_pipeline(
                k_features=best_params['feature_selection__k'],
                preprocessor=preprocessor,
                n_estimators=best_params['classifier__n_estimators']
            )

            # Entraînement et Prédiction
            current_pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
            y_pred_cycle = current_pipeline.predict(X_test)

            # Stockage des Résultats
            all_predictions = pd.concat([all_predictions, pd.Series(y_pred_cycle, index=X_test.index)])
            all_test_targets = pd.concat([all_test_targets, y_test])

            # Le modèle du dernier cycle est le "meilleur" car le plus récent.
            final_model = current_pipeline

        # Evaluation finale et logging
        final_accuracy = accuracy_score(all_test_targets, all_predictions)
        mlflow.log_metric("wfv_cumulative_accuracy", final_accuracy)
        print(f"Précision WFV cumulée : {final_accuracy:.4f}")

        model_name = "XGBoost_Trading_Model"


        joblib.dump(final_model, f"models/{model_name}.pkl")
        mlflow.log_artifact(f"models/{model_name}.pkl")

        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            registered_model_name=model_name
        )

        print(f"Modèle enregistré dans le registre MLFlow sous le nom : {model_name}")

        return final_model, final_accuracy