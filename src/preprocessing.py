# src/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


def get_prepocessor(features: pd.DataFrame) -> ColumnTransformer:
    """
    Définit le préprocesseur pour l'encodage des variables catégorielles.
    Args:
        features: les caractéristiques du dataframe
    Returns:
        ColumnTransformer: Le transformateur de colonnes Scikit-learn.
    """
    # Encodage des variables catégorielles
    list_cat_col = features.select_dtypes(exclude=np.number).columns.tolist()

    ordre_cat = [
        ['Survente', 'Normal', 'Surachat'],
        ['Survente', 'Normal', 'Surachat'],
        ['Survente', 'Normal', 'Surachat']
    ]

    ordinal_encoder = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(categories=ordre_cat, handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('ord_cat', ordinal_encoder, list_cat_col)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    return preprocessor


def create_training_pipeline(threshold: float, preprocessor: ColumnTransformer, n_estimators: int) -> Pipeline:
    """
    Créer le Pipeline complet comprenant le préprocessing, la sélection des features et le modèle XGBoost.
    Args:
        k_features: Le nombre de features à sélectionner (issu de l'optimisation)
        preprocessor: Le ColumnTransformer défini par get_preprocessor
        n_estimators: Le nombre d'estimateurs xgboost (issu de l'optimisation)
    Returns:
        Pipeline: Le Pipeline Scikit-learn prêt à être entrainé.
    """

    selector = SelectFromModel(XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42),
                               threshold=threshold)
    modele = XGBClassifier(
        objective='multi:softprob',
        n_estimators=n_estimators,
        eval_metric='mlogloss',
        random_state=42
    )

    # Pipeline final
    pipeline_final = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', selector),
        ('classifier', modele)
    ])

    return pipeline_final


def drop_multicol(features, *, threshold: float = 0.85):
    """
    Docstring for drop_multicol

    :param features: Description
    :param threshold: Description
    :type threshold: float
    """
    corr = features.select_dtypes(include=np.number).corr()
    triangle_sup = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in triangle_sup.columns if any(triangle_sup[col] > threshold)]
    return features.drop(columns=to_drop, errors='ignore')
