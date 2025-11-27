import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import generate_features_and_labels
from src.utils import get_double_barrier_levels

@pytest.fixture
def sample_ohlcv_data():
    """Créé un DataFrame pour tester les calculs."""
    data_points = 100
    df = pd.DataFrame({
        'Open': np.random.uniform(90, 110, data_points),
        'High': np.random.uniform(110, 120, data_points),
        'Low': np.random.uniform(80, 90, data_points),
        'Close': np.linspace(100,120, data_points) + np.random.randn(data_points),
        'Volume': np.random.randint(500,2000,data_points)
    })
    df.index = pd.to_datetime(pd.date_range(start='2025-01-01', periods=data_points, freq='h'))
    return df


def test_feature_complet(sample_ohlcv_data):
    """Vérifie que toutes les colonnes ont été créées"""
    df = generate_features_and_labels(df=sample_ohlcv_data)
    expected_cols = [
        'ema_12', 'ema_200', 'macd', 'rsi', 'upper_band', 'lower_band', 'bandwidth', 'obv', 'stoch_k', 'stoch_d', 'atr', 'label'
    ]

    for col in expected_cols:
        assert col in df.columns, f"La colonne {col} est manquante lors de l'étape de feature_engineering"

def test_ema(sample_ohlcv_data):
    """Vérifie les sorties des calculs d'EMA"""
    df = generate_features_and_labels(sample_ohlcv_data)

    assert not np.isnan(df['ema_50'].iloc[50]), "L'EMA 50 ne devrait pas être nulle à ce niveau"

def test_double_barrier(sample_ohlcv_data):
    """Vérifie que les niveaux TP/SL sont calculés correctement en fonction de l'ATR"""

    sample_ohlcv_data['atr'] = 1.0

    tp_mult = 3.0
    sl_mult = 1.5

    df = get_double_barrier_levels(
        df=sample_ohlcv_data,
        atr_col='atr',
        profit_mult=tp_mult,
        stop_mult=sl_mult, 
    )

    test_index = 50
    close_price = df['Close'].iloc[test_index]

    expected_tp_long = close_price + 3.0
    expected_sl_long = close_price - 1.5

    assert np.isclose(df['tp_long'].iloc[test_index], expected_tp_long)
    assert np.isclose(df['sl_long'].iloc[test_index], expected_sl_long)


def test_nan(sample_ohlcv_data):
    """Vérifie que les premières valeurs de certaines colonnes sont bien des NaNs."""
    df = generate_features_and_labels(sample_ohlcv_data)
    
    assert np.isnan(df['upper_band'].iloc[18]), "La valeur de la variable 'upper_band' devrait être nulle"
    assert not np.isnan(df['lower_band'].iloc[19]), "La valeur de la variable 'lower_band' ne devrait pas être nulle"

# Continuer dans tests/test_feature_engineering.py


def test_rsi_level():
    """Teste explicitement la logique np.where pour les niveaux RSI aux seuils 30 et 70."""
    
    # Créer un DataFrame avec uniquement la colonne 'rsi' contrôlée
    data_rsi = {
        'rsi': [np.nan] * 5 + [29.9, 30.0, 30.1, 69.9, 70.0, 70.1, 50.0]
    }
    df_test = pd.DataFrame(data_rsi)
    
    df_test['niveau_rsi'] = np.where(
        df_test['rsi'] > 70, 
        "Surachat", 
        np.where(
            df_test['rsi'] < 30, 
            "Survente", 
            "Normal"
        )
    )
     
    # Test à la limite 30
    assert df_test['niveau_rsi'].iloc[5] == "Survente"  # 29.9 < 30
    assert df_test['niveau_rsi'].iloc[6] == "Normal"    # 30.0 >= 30 (Normal)
    assert df_test['niveau_rsi'].iloc[7] == "Normal"    # 30.1 >= 30
    
    # Test à la limite 70
    assert df_test['niveau_rsi'].iloc[8] == "Normal"    # 69.9 <= 70
    assert df_test['niveau_rsi'].iloc[9] == "Normal"    # 70.0 <= 70
    assert df_test['niveau_rsi'].iloc[10] == "Surachat" # 70.1 > 70
    

def test_ema_croisement():
    """Vérifie que le flag binaire de croisement EMA change correctement
    lorsque l'EMA courte franchit l'EMA longue à la hausse."""
    
    # --- PRÉPARATION DU SCÉNARIO DE CROISEMENT ---
    # Créer 50 points de données
    data_points = 50
    data = {
        # Les prix sont constants et bas pour que EMA12 < EMA26 au début
        'Open': [10.0] * data_points,
        'High': [12.0] * data_points,
        'Low': [8.0] * data_points,
        'Volume': [1000] * data_points
    }
    df_test = pd.DataFrame(data)
    
    # Simuler le prix : stable (bas), puis augmentation pour forcer le croisement
    df_test['Close'] = np.concatenate([
        np.full(10, 10.0),                  # Jours 0-9 : Prix stable à 10.0
        np.linspace(10.0, 15.0, 10),         # Jours 10-19 : Hausse progressive
        np.full(30, 15.0)                   # Jours 20-49 : Prix stable à 15.0
    ])
    
    df_result = generate_features_and_labels(df_test)
   
    # 1. Situation de départ (Index 9) : EMA12 < EMA26 (flag doit être 0)
    # L'EMA courte (12) devrait toujours être sous l'EMA longue (26) au début.
    assert df_result['croisement_ema_ct'].iloc[9] == 0, "Le flag devrait être 0 avant le croisement."
    
    # 2. Trouver le point de croisement
    # Le croisement se produit lorsque EMA12 > EMA26. 
    # Nous trouvons le premier index où cette condition est remplie.
    croisement_idx = df_result[df_result['ema_12'] > df_result['ema_26']].index.min()
    
    # Convertir l'index Datetime en index numérique pour la vérification
    idx_croisement = df_result.index.get_loc(croisement_idx)
    
    # 3. Assertion : Au moment du croisement, le flag doit passer à 1
    assert df_result['croisement_ema_ct'].iloc[idx_croisement] == 1, "Le flag doit passer à 1 exactement au moment du croisement."
    
    # 4. Assertion : La ligne précédente (idx_croisement - 1) doit être 0
    assert df_result['croisement_ema_ct'].iloc[idx_croisement - 1] == 0, "Le flag devrait être 0 juste avant le croisement."