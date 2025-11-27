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