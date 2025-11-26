# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
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

    ordre_cat =[
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

def create_training_pipeline(k_features:int, preprocessor: ColumnTransformer, n_estimators: int ) -> Pipeline:
    """
    Créer le Pipeline complet comprenant le préprocessing, la sélection des features et le modèle XGBoost.
    Args:
        k_features: Le nombre de features à sélectionner (issu de l'optimisation)
        preprocessor: Le ColumnTransformer défini par get_preprocessor
        n_estimators: Le nombre d'estimateurs xgboost (issu de l'optimisation)
    Returns:
        Pipeline: Le Pipeline Scikit-learn prêt à être entrainé.
    """

    selector = SelectKBest(k=k_features)
    modele = XGBClassifier(
        n_estimators= n_estimators,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
        )

    # Pipeline final
    pipeline_final = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', selector),
        ('classifier', modele)
    ])

    return pipeline_final