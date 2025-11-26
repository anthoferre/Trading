# src/feature_engineering.py

import pandas as pd
import numpy as np
from src.preprocessing import calculate_atr_series, calculate_rsi, get_double_barrier_levels, label_double_barrier

def generate_features_and_labels(df: pd.DataFrame, tp_mult: float = 2.0, sl_mult: float = 1.0) -> pd.DataFrame:
    """
    Calcule toutes les features (indicateurs techniques) et la variable cible (label).
    Args:
        df: Dataframe contenant les données brutes de bougies issues de Yahoo Finance.
        tp_mult: Multiplicateur pour le Take Profit
        sl_mult : Multiplicateur pour le Stop Loss
    Returns:
        pd.Dataframe: Dataframe enrichi avec les features et la variable cible.
    """

    df_copy = df.copy()

    # Différence avec la cloture de la bougie précédente
    df_copy['diff_close'] = df_copy['Close'].diff()

    # Taille de la bougie et mèches
    df_copy['taille_bougie'] = df_copy['High'] - df_copy['Low']
    df_copy['taille_corps'] = np.maximum(df_copy['Close'], df_copy['Open']) - np.minimum(df_copy['Close'], df_copy['Open'])
    df_copy['ratio_corps_bougie'] = df_copy['taille_corps'] / df_copy['taille_bougie']
    df_copy['taille_meche_sup'] = df_copy['High'] - np.maximum(df_copy['Close'], df_copy['Open'])
    df_copy['taille_meche_inf'] = np.minimum(df_copy['Close'], df_copy['Open']) - df_copy['Low']

    # Moyenne Mobile Exponentielle (EMA)
    for span in [12, 26, 50, 200]:
        col_name = f'ema_{span}'
        df_copy[col_name] = df_copy['Close'].ewm(span=span, adjust=False).mean()

    # Prix de Cloture - EMA & Momentum
    for span in [12, 26, 50, 200]:
        df_copy[f'close_ema_{span}'] = df_copy['Close'] - df_copy[f'ema_{span}']

    df_copy['momentum_court_terme'] = df_copy['ema_12'] - df_copy['ema_26']
    df_copy['momentum_moyen_terme'] = df_copy['ema_12'] - df_copy['ema_50']
    df_copy['momentum_long_terme'] = df_copy['ema_50'] - df_copy['ema_200']

    for col in ['court_terme', 'moyen_terme', 'long_terme']:
        df_copy[f'diff_momentum_{col}'] = df_copy[f'momentum_{col}'].diff()

    # Croisement EMA
    df_copy['croisement_ema_ct'] = (df_copy['ema_12'] > df_copy['ema_26']).astype(int)
    df_copy['croisement_ema_mt'] = (df_copy['ema_12'] > df_copy['ema_50']).astype(int)
    df_copy['croisement_ema_lt'] = (df_copy['ema_50'] > df_copy['ema_200']).astype(int)

    # MACD
    df_copy['macd'] = df_copy['ema_12'] - df_copy['ema_26']
    df_copy['signal'] = df_copy['macd'].ewm(span=9, adjust=False).mean()
    df_copy['histogramme'] = df_copy['macd'] - df_copy['signal']

    # ROC EMA
    for span in [12, 26, 50, 200]:
        df_copy[f'roc_ema_{span}'] = df_copy[f'ema_{span}'].diff(periods=5) / df_copy[f'ema_{span}'].shift(5)

    # RSI
    df_copy['rsi'] = calculate_rsi(df_copy, close='Close', N=14).astype(float) # Change to float before np.where
    df_copy['niveau_rsi'] = np.where(df_copy['rsi'] > 70, "Surachat", np.where(df_copy['rsi'] < 30, "Survente", "Normal"))
    df_copy['midligne_rsi'] = (df_copy['rsi'] >= 50).astype(int)
    df_copy['roc_rsi_1'] = df_copy['rsi'].diff(1) / df_copy['rsi'].shift(1)
    df_copy['roc_rsi_5'] = df_copy['rsi'].diff(5) / df_copy['rsi'].shift(5) 

    # Bandes de Bollinger
    df_copy['middle_band'] = df_copy['Close'].rolling(window=20).mean()
    std_20 = df_copy['Close'].rolling(window=20).std()
    df_copy['upper_band'] = df_copy['middle_band'] + 2 * std_20
    df_copy['lower_band'] = df_copy['middle_band'] - 2 * std_20
    df_copy['bandwidth'] = (df_copy['upper_band'] - df_copy['lower_band']) / df_copy['middle_band']
    df_copy['close_bollinger'] = (df_copy['Close'] - df_copy['lower_band']) / (df_copy['upper_band'] - df_copy['lower_band'])
    df_copy['niveau_close_bb'] = np.where(df_copy['close_bollinger'] > 1, "Surachat", np.where(df_copy['close_bollinger'] < 0, "Survente", "Normal"))
    df_copy['croisement_bb'] = df_copy['Close'] - df_copy['middle_band']

    # Volume (Simplified OBV, Price-Volume)
    price_direction = np.sign(df_copy['Close'].diff())
    obv_change = price_direction * df_copy['Volume']
    df_copy['obv'] = obv_change.cumsum()
    df_copy['obv_sign'] = ((df_copy['obv'].diff()) > 0).astype(int)
    df_copy['roc_obv'] = df_copy['obv'].diff(5) / df_copy['obv'].shift(5)
    df_copy['typical_price'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3
    df_copy['price_volume'] = df_copy['typical_price'] * df_copy['Volume']
    df_copy['VWA_14'] = df_copy['price_volume'].rolling(14).sum() / df_copy['Volume'].rolling(14).sum()
    df_copy['relation_prix_vwap'] = (df_copy['Close'] > df_copy['VWA_14']).astype(int)
    df_copy['dist_prix_vwap'] = df_copy['Close'] - df_copy['VWA_14']

    # Stochastique
    max_high = df_copy['High'].rolling(14).max()
    min_low = df_copy['Low'].rolling(14).min()
    # Gérer la division par zéro
    denominator = (max_high - min_low)
    df_copy['stoch_k'] = np.where(denominator != 0, (df_copy['Close'] - min_low) / denominator, 0)
    df_copy['stoch_d'] = df_copy['stoch_k'].rolling(3).mean()
    df_copy['niveau_stoch'] = np.where(df_copy['stoch_k'] > 0.8, "Surachat", np.where(df_copy['stoch_k'] < 0.2, "Survente", "Normal"))
    df_copy['croisement_stoch'] = (df_copy['stoch_k'] > df_copy['stoch_d']).astype(int)

    # ATR et Normalisation
    df_copy['atr'] = calculate_atr_series(df_copy, period=14)
    for span in [12, 26, 50, 200]:
        df_copy[f'close_ema_{span}_normalisee'] = df_copy[f'close_ema_{span}'] / df_copy['atr']

    # --- LABELLISATION CIBLE ---

    # 1. Calcul des barrières
    df_copy = get_double_barrier_levels(df_copy, atr_col='atr', profit_mult=tp_mult, stop_mult=sl_mult)

    # 2. Calcul du label (retire les NaN)
    df_copy['label'] = label_double_barrier(df_copy)

    return df_copy
