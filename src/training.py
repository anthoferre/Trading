import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.preprocessing import create_training_pipeline

param_grid = {
    'feature_selection__threshold': ['median', 'mean', 0.01],
    'classifier__n_estimators': [100, 200],
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
        ('feature_selection', SelectFromModel(XGBClassifier(objective='multi:softprob',
                                                            eval_metric='mlogloss', random_state=42))),
        ('classifier', XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42))
    ])

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        estimator=temp_pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1_macro',
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    feature_selector = best_pipeline.named_steps['feature_selection'].get_support()
    print(f"Les features gardés pour le modèle sont : {X_train.columns[feature_selector]}")
    return grid_search.best_params_ , grid_search.best_score_, grid_search.best_estimator_


def run_tscv_training(features: pd.DataFrame, target: pd.Series, N_SPLITS: int,
                      preprocessor: ColumnTransformer, config: dict) -> tuple[Pipeline, float]:
    """
    Exécute la Time Serie Split (TSCV), loggue les métriques avec MLFlow et enregistre le modèle.

    Args:
        features: caractéristiques
        target : variable cible (label)
        N_SPLITS: nb de plis pour le Time Series Splits
        preprocessor: ColumnTransformer
        config: Dictionnaire de configuration(pour logguer les métadonnées)
    Returns:
        tuple[Pipeline, float]: Le modèle final entraîné et la précision WFV cumulée.
    """
    # Logguer les paramètres de configuration/environnement
    # Les logs sont envoyés au run MLflow ACTIF
    mlflow.log_params(config)
    mlflow.log_params({"N_SPLITS": N_SPLITS})

    features_train_cols = features.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

    print(f"\n--- Démarrage Time Series Split avec {N_SPLITS} cycles---")

    # Recherche des hyperparamètres (k pour le KBest et n_estimators pour le xgboost)
    best_params, best_score, best_estimator = hyperparameter_optimization(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor
    )
    print(f"Meilleurs hyperparamètres : {best_params}")
    mlflow.log_params(best_params)

    y_pred = best_estimator.predict(X_test)

    final_accuracy = best_estimator.score(X_test, y_test)

    mlflow.log_metric("tsvv_balanced_accuracy", final_accuracy)
    print(f"Précision : {final_accuracy:.4f}")

    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion")
    print(conf_matrix)

    # Rapport de classification
    report = classification_report(y_true=y_test, y_pred=y_pred)
    print("Rapport de classification")
    print(report)

    model_name = "XGBoost_Trading_Model"

    # Sauvegarde de la liste de features utilisées pour l'entraînement (pour l'API)
    joblib.dump(features_train_cols, f"models/{model_name}_features.pkl")

    # Sauvegarde du modèle (Pipeline)
    joblib.dump(best_estimator, f"models/{model_name}.pkl")

    # Log des artefacts (fichiers locaux)
    mlflow.log_artifact(f"models/{model_name}_features.pkl")
    mlflow.log_artifact(f"models/{model_name}.pkl")

    # Enregistrement dans le Registre MLflow
    mlflow.sklearn.log_model(
        sk_model=best_estimator,
        artifact_path="model",
        registered_model_name=model_name
    )

    print(f"Modèle enregistré dans le registre MLFlow sous le nom : {model_name}")

    return best_estimator, final_accuracy
