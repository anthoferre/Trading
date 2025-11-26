import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import mplfinance as mpf
from sklearn.utils.class_weight import compute_sample_weight
from src.data_ingestion import fetch_data
from src.feature_engineering import generate_features_and_labels
from src.preprocessing import get_prepocessor, create_training_pipeline

TICKER = 'GC=F'

if __name__ == "__main__":
    print(f"Démarrage de l'acquisition des données pour {TICKER}... ")
    df_raw = fetch_data()

    if df_raw.empty:
        print("Arrêt du Pipeline : Données non disponibles")
    else:

        # Création des indicateurs clés et de la variable cible (label)
        df_label = generate_features_and_labels(df=df_raw)


        # 3. Préparation des données finales
        df_plot_temp = df_raw.iloc[-200:].copy()
        df_plot = df_plot_temp.loc[:,['High','Close','Open','Low','ema_12','ema_26','ema_50','ema_200','Volume']].copy()
        df_last_candle = df_raw.iloc[[-1]].copy()

        # Suppression des colonnes de prix bruts (High, Low, Open, Close, Volume) et des colonnes de barrière
        list_col_to_drop = df_raw.select_dtypes(include=np.number).columns[(df_raw.select_dtypes(include=np.number).mean() > 1000)].tolist()
        list_barriers = ['tp_long', 'sl_long', 'tp_short', 'sl_short']

        # --- FILTRAGE ET NETTOYAGE ---
        list_col_na = df_raw.columns[df_raw.isna().sum()>100].tolist()

        # Suppression des valeurs brutes qui ne sont pas des features utiles pour XGBoost
        df_raw.drop(columns=list_col_to_drop + list_barriers + list_col_na, inplace=True, errors='ignore')

        df_raw.dropna(axis='index', how='any', inplace=True)

        target = df_raw['label']
        features = df_raw.drop(columns=['label']) # On garde uniquement les indicateurs

        # Préprocesseur pour l'encodage des variables catégorielles
        preprocessor = get_prepocessor(features)

        # Création du Pipeline final prêt à être entrainé
        pipeline_final = create_training_pipeline(k_features=20, preprocessor=preprocessor, n_estimators=200)

        # --- VALIDATION CROISÉE SÉQUENTIELLE (WALK-FORWARD VALIDATION - WFV) ---

        # PARAMÈTRES WFV
        TRAIN_SIZE = int(len(features) * 0.75) 
        TEST_SIZE = int(len(features) * 0.05) 
        STEP_SIZE = TEST_SIZE 

        n_cycles = int((len(features) - TRAIN_SIZE) / STEP_SIZE)
        if n_cycles < 1: 
            print("Pas assez de données pour le WFV. Utilisation de la séparation standard.")
            n_cycles = 1 # Force un cycle

        all_predictions = pd.Series(dtype=int)
        all_test_targets = pd.Series(dtype=int)

        print(f"\n--- DÉMARRAGE WFV ---")
        print(f"Nombre de cycles WFV (splits) : {n_cycles}")

        # Recherche d'hyperparamètres initiale (sur la première fenêtre d'entraînement)
        X_train_initial = features.iloc[:TRAIN_SIZE]
        y_train_initial = target.iloc[:TRAIN_SIZE]

        param_grid = {
            'feature_selection__k': [5,10,20],
            'classifier__n_estimators' : [100,200],
        }

        grid_search_initial = GridSearchCV(
            estimator=pipeline_final,
            param_grid=param_grid,
            cv=3,
            scoring='f1_macro',
            verbose=0,
            n_jobs=-1
        )
        grid_search_initial.fit(X_train_initial, y_train_initial)
        best_params = grid_search_initial.best_params_
        print(f"Meilleurs hyperparamètres initiaux : {best_params}")

        # BOUCLE WFV
        for cycle in range(n_cycles):
            
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

            # Reconstruction du Pipeline avec les meilleurs hyperparamètres
            best_pipeline_cycle = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('feature_selection', SelectKBest(k=best_params['feature_selection__k'])),
                ('classifier', XGBClassifier(n_estimators=best_params['classifier__n_estimators'], 
                                            use_label_encoder=False, 
                                            eval_metric='logloss',
                                            random_state=42))
            ])
            
            # Entraînement et Prédiction
            best_pipeline_cycle.fit(X_train, y_train, classifier__sample_weight=sample_weights)
            y_pred_cycle = best_pipeline_cycle.predict(X_test)
            y_pred_series = pd.Series(y_pred_cycle, index=X_test.index)

            # Stockage des Résultats
            all_predictions = pd.concat([all_predictions, y_pred_series])
            all_test_targets = pd.concat([all_test_targets, y_test])

            acc = accuracy_score(y_test, y_pred_cycle)
            # print(f"Cycle {cycle + 1}: Précision {acc:.4f}")

        # ÉVALUATION GLOBALE WFV
        if not all_test_targets.empty:
            print("\n" + "="*70)
            print("RÉSULTATS DE LA VALIDATION CROISÉE SÉQUENTIELLE (WFV)")
            print("="*70)

            final_accuracy = accuracy_score(all_test_targets, all_predictions)
            print(f"Précision WFV cumulée : {final_accuracy:.4f}")

            cm_wfv = confusion_matrix(all_test_targets, all_predictions)
            print("\nMatrice de Confusion WFV :")
            print(cm_wfv)

            cr_wfv = classification_report(y_true=all_test_targets, y_pred=all_predictions)
            print("\nClassification Report WFV :")
            print(cr_wfv)
            
            final_live_model = best_pipeline_cycle # Le modèle du dernier cycle est le plus récent
        else:
            print("\nAucun résultat WFV à afficher (données insuffisantes après nettoyage).")
            exit()

        # --- PRÉDICTION EN TEMPS RÉEL ET AFFICHAGE ---

        # 1. Réaligner la dernière bougie et supprimer les colonnes de prix/barrière
        df_last_candle_features = df_last_candle.drop(columns=list_col_to_drop + list_barriers + ['label'], errors='ignore')
        latest_features = df_last_candle_features.loc[:, X_train.columns].copy()

        # 2. Prédiction
        prediction_prob = final_live_model.predict_proba(latest_features)
        prob_vente = prediction_prob[0][0]
        prob_achat = prediction_prob[0][1]
        prob_ne_rien_faire = prediction_prob[0][2]

        # 3. Récupération des TP/SL (les colonnes de barrière n'ont pas été supprimées de df_last_candle)
        if prob_vente > prob_achat:
            # Vente (Short)
            tp_last_candle = df_last_candle['tp_short'].iloc[0]
            sl_last_candle = df_last_candle['sl_short'].iloc[0]
        else:
            # Achat (Long)
            tp_last_candle = df_last_candle['tp_long'].iloc[0]
            sl_last_candle = df_last_candle['sl_long'].iloc[0]

        print(f"\n--- PRÉDICTION DE LA DERNIÈRE BOUGIE ---")
        print(f"Date observation : {latest_features.index[0]}")
        print(f"Probabilité de vente :{prob_vente:.4f}")
        print(f"Probabilité d'achat :{prob_achat:.4f}")
        print(f"Probabilité de ne rien faire :{prob_ne_rien_faire:.4f}")
        print(f"SL : {sl_last_candle:.6f}")
        print(f"TP : {tp_last_candle:.6f}")


        # 4. Affichage du graphique
        tp_series = pd.Series(tp_last_candle, index=df_plot.index)
        sl_series = pd.Series(sl_last_candle, index=df_plot.index)

        apds = [
            mpf.make_addplot(df_plot['ema_12'], color='cyan', label='EMA 12'),
            mpf.make_addplot(df_plot['ema_26'], color='steelblue', label='EMA 26'),
            mpf.make_addplot(df_plot['ema_50'], color='blue', label='EMA 50'),
            mpf.make_addplot(df_plot['ema_200'], color='navy', label='EMA 200'),
            mpf.make_addplot(sl_series, color = 'red', linestyle='--', label='Stop Loss'),
            mpf.make_addplot(tp_series, color = 'green', linestyle='--', label='Take Profit')
        ]

        mpf.plot(df_plot, type='candle', volume=False, style ='yahoo', 
                addplot=apds, title="Graphique avec EMAs et Niveaux TP/SL Prédits")