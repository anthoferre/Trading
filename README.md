# 📖 README : Modèle de Trading

## 🌟 Vue d'Ensemble

Ce projet déploie une application complète de prédiction de trading basée sur des conteneurs. Il permet l'entraînement isolé d'un modèle (via le service pipeline), le suivi des expériences avec MLflow, et l'exposition du modèle via une API REST pour les prédictions en temps réel.

L'architecture est entièrement gérée par Docker Compose pour assurer l'isolation de l'environnement et la reproductibilité.

## 🚀 Démarrage et Installation
### 1. Prérequis

Assurez-vous d'avoir installé les outils suivants sur votre machine hôte :

    Docker Desktop

    Docker Compose (compatible avec la version 3.8+ de docker-compose.yaml)

    Git

### 2. Clonage et Lancement

Clônez le projet et lancez l'intégralité de la stack Docker.
```
Bash
# Cloner le dépôt (si applicable)
git clone https://github.com/anthoferre/Trading.git
cd Trading
```

Construire et lancer tous les services en mode détaché. Ceci inclut l'API, le Pipeline d'entraînement et MLflow.
```
docker-compose up -d --build
```

### 3. Accès aux Interfaces
| Service spécique et lien URL | Fonction |
| :--- | :--- |
| [Trading API](http://localhost:8000) | Point d'accès pour les requêtes de prédiction. |
| [MLflow Tracking](http://localhost:5000)	| Interface pour visualiser l'historique des modèles. |

## ⚙️ Workflow d'Utilisation

L'utilisation du projet se fait en deux étapes : l'entraînement du modèle, puis le redémarrage du service API pour son déploiement.
### Étape 1 : Entraînement du Modèle

Le service pipeline est utilisé pour exécuter votre script d'entraînement (run_pipeline.py). Ce service s'exécute en une seule fois.

 ```
Bash
docker-compose run --rm pipeline python /app/trading_project/run_pipeline.py
 ```

Le script s'exécute, entraîne le modèle, et enregistre l'artefact dans le serveur MLflow.

### Étape 2 : Déploiement du Nouveau Modèle

Pour que l'API utilise le modèle fraîchement entraîné, elle doit être redémarrée.
 ```
Bash
docker-compose restart api
 ```
Le conteneur api redémarre et exécute son script de démarrage, qui est configuré pour charger le dernier modèle de production ou le modèle le plus récent enregistré dans MLflow.


## ❌ Problèmes de Configuration ou de Débogage
| Tâche	| Commande | Description |
| :--- | :--- | :--- |
| Vérifier les Logs de l'API | ```docker-compose logs -f api``` | Affiche la sortie en temps réel du service API. Utile pour voir les erreurs de chargement du modèle. |
| Accès Interactif à l'API |  ```docker exec -it api bash ```	| Permet d'entrer dans le conteneur de l'API pour déboguer les fichiers ou les environnements. |
| Mettre à jour la Configuration |  ```docker-compose restart api ```	| À exécuter après toute modification du fichier de configuration local. |
| Forcer la Reconstruction |	 ```docker-compose up -d --build --force-recreate ```	| À utiliser si un changement dans le Dockerfile ou les dépendances ne semble pas être pris en compte. |

## 🗑️ Nettoyage Complet

Pour arrêter et supprimer tous les conteneurs, réseaux, et volumes associés au projet :
```
Bash

docker-compose down -v
```

## 📁 4. Structure du Projet

Les fichiers principaux du projet et leurs rôles sont les suivants :

| Fichier| Rôle | Description
| :--- | :--- | :--- |
| `run_pipeline.py` | Orchestrateur | Coordonne les étapes : ingestion, feature engineering, pré-traitement, entraînement WFV, et prédiction. |
| `src/data_ingestion.py` | Ingestion | Gère la récupération des données historiques (ex: via yfinance). |
| `src/feature_engineering.py` | Features & Label | Crée tous les indicateurs techniques et labellise les données via la Double Barrière. |
| `src/preprocessing.py` | Pré-traitement | Définit le ColumnTransformer pour l'encodage ordinal et la structure de la Pipeline finale. |
| `src/training.py` | Entraînement WFV | Contient la boucle de Validation Walk-Forward, l'optimisation des hyperparamètres (GridSearch) et le logging MLflow. |
| `src/prediction.py` | Prédiction | Gère le chargement du modèle enregistré (joblib) et la logique de prédiction temps réel sur la dernière bougie. |
| `config/config.yaml` | Configuration | Contient les paramètres de la stratégie (ticker, intervalle, multiplicateurs TP/SL, tailles WFV). |
| `models/` | Stockage Local | Répertoire de sauvegarde des artefacts du modèle (.pkl) et des features. |
| `mlruns/` | MLflow | Stockage des logs, paramètres et modèles enregistrés par MLflow. |
| `api/main.py` | API REST | Point d'entrée pour le service web (ex: FastAPI) servant la fonction get_prediction. |

## 📈 5. Méthodologie de Modélisation

Le pipeline suit un flux strict : Ingestion -> Feature Engineering -> Pré-traitement -> Entraînement WFV.

### 5.1. 🧹 Pré-traitement des Données (src/preprocessing.py)

Gestion des Catégories : Utilise un ColumnTransformer pour appliquer un encodage ordinal aux features catégorielles (ex: niveau_rsi).

Les catégories ordonnées sont : ['Survente', 'Normal', 'Surachat'].

Les valeurs manquantes sont imputées avec une valeur constante avant l'encodage.

Pipeline Globale : Le create_training_pipeline assemble le préprocesseur, la sélection des features (SelectKBest) et le classifieur (XGBClassifier) dans une seule Pipeline pour garantir l'application cohérente des transformations lors de l'entraînement et de la prédiction.

### 5.2. 🔬 Feature Engineering (src/feature_engineering.py)

Un riche ensemble de features techniques est calculé à partir des données OHLCV.

| Catégorie | Indicateurs Clés | Note |
| :--- | :--- | :--- |
| Tendance/Momentum | MACD, EMA, Momentum à Long Terme | Les distances Prix-EMA sont normalisées par l'ATR. |
| Volatilité | ATR (Average True Range), Bandes de Bollinger (Position, Bandwidth) | L'ATR est essentiel pour la labellisation et la normalisation. |
| Oscillateurs | RSI, Stochastique K/D | Mesures de surachat/survente. |
| Volume | OBV (Momentum), VWAP | Relatif au prix typique. |
| Cible (Label) | Double Barrière | Le label est binaire (Achat=1, Vente=0, Neutre=2) basé sur l'atteinte d'un niveau de Take Profit (tp_mult) ou Stop Loss (sl_mult). |

### 5.3. 🧠 Entraînement (src/training.py)

Le modèle est entraîné via la *Walk-Forward Validation* essentielle pour du trading où la composante temporelle est importante.

**Optimisation Initiale** : Une *GridSearchCV* initiale est lancée sur le premier fold pour déterminer les meilleurs hyperparamètres (*nombre de features k* et *n_estimators* pour XGBoost).

**Boucle WFV** : Le modèle est entraîné séquentiellement sur des fenêtres glissantes (*TRAIN_SIZE, TEST_SIZE, STEP_SIZE*).

**Gestion du Déséquilibre** : Des poids d'échantillons `(compute_sample_weight(class_weight='balanced'))` sont utilisés pour pallier le déséquilibre entre les classes Achat, Vente et Neutre.

**Logging** : Tous les paramètres, métriques (précision WFV cumulée) et artefacts (modèle .pkl, liste des features) sont enregistrés dans *MLflow*.
