import numpy as np
import pandas as pd


def calculate_rsi(df, close, N):
    """

    """
    # 1. Calculer la différence de prix (changement de jour en jour)
    delta = df[close].diff()

    # 2. Créer les colonnes de Gain (Up) et de Perte (Down)
    # Remplacer les NaN (première ligne) par 0 avant le calcul
    up = delta.mask(delta < 0, 0) # Gain si delta > 0, sinon 0
    down = -delta.mask(delta > 0, 0) # Perte si delta < 0, sinon 0

    # Remplacer les valeurs initiales NaN créées par .diff() par 0
    up = up.fillna(0)
    down = down.fillna(0)

    # 3. Calculer les moyennes mobiles exponentielles des gains et des pertes
    # N.B. : Le 'adjust=False' est crucial pour reproduire la formule de Wilder.
    avg_gain = up.ewm(com=N-1, adjust=False).mean()
    avg_loss = down.ewm(com=N-1, adjust=False).mean()

    # 4. Calculer la Force Relative (RS)
    # np.where est utilisé pour gérer la division par zéro (cas où avg_loss est 0)
    RS = np.where(avg_loss == 0, 0, avg_gain / avg_loss)

    # 5. Calculer le RSI
    rsi = 100 - (100 / (1 + RS))

    return rsi


def calculate_atr_series(
    df: pd.DataFrame,
    period: int = 14,
    col_high: str = 'high',
    col_low: str = 'low',
    col_close: str = 'close'
) -> pd.Series:
    """
    Calcule l'Average True Range (ATR) et retourne la série de l'ATR.

    Args:
        df (pd.DataFrame): DataFrame contenant les prix.
        period (int): Période de lissage de l'ATR (par défaut 14).

    Returns:
        pd.Series: Une nouvelle Série Pandas contenant les valeurs de l'ATR.
    """

    if len(df) < period:
        print(f"Attention: Le DataFrame ({len(df)} lignes) est trop court pour calculer l'ATR sur {period} périodes.")
        return pd.Series([np.nan] * len(df), index=df.index)

    # Créer les séries des prix
    high = df[col_high]
    low = df[col_low]
    close = df[col_close]
    prev_close = close.shift(1)

    # --- Calcul du True Range (TR) ---
    range_high_low = high - low
    range_high_prev_close = abs(high - prev_close)
    range_low_prev_close = abs(low - prev_close)

    # Le TR est le maximum des trois
    tr_series = np.maximum.reduce([
        range_high_low,
        range_high_prev_close,
        range_low_prev_close
    ])

    tr_series = pd.Series(tr_series, index=df.index)

    # --- Lissage du True Range pour obtenir l'ATR ---
    # com=period-1 pour le lissage de Wilder (alpha = 1/period)
    atr_series = tr_series.ewm(com=period - 1, adjust=False).mean()

    return atr_series


def get_double_barrier_levels(df, atr_col='ATR', profit_mult=2.0, stop_mult=1.0):
    """Calcule les niveaux de prix des barrières pour chaque période."""

    atr_threshold = df[atr_col]

    # Cas d'achat (long)
    df['tp_long'] = df['close'] + (atr_threshold * profit_mult)
    df['sl_long'] = df['close'] - (atr_threshold * stop_mult)

    # Cas de vente (short)
    df['tp_short'] = df['close'] - (atr_threshold * profit_mult)
    df['sl_short'] = df['close'] + (atr_threshold * stop_mult)

    return df


def label_double_barrier(df: pd.DataFrame) -> pd.Series:
    """
    Détermine la classe cible (1: Long, 0: Short) en fonction du premier contact
    avec les barrières de profit/perte, sans limite de temps (jusqu'à la fin du DF).

    AVERTISSEMENT : Peut être très lent sur les grands jeux de données !
    """

    labels = pd.Series(index=df.index, dtype=float)
    N = len(df)

    for i in range(N):

        # ----------------------------------------------------
        # CORRECTION : Fenêtre de recherche jusqu'à la fin (N)
        # ----------------------------------------------------
        future_window = df.iloc[i + 1:N].copy()

        if future_window.empty:
            labels.iloc[i] = 2.0  # Non résolu / Fin de série
            continue

        # Niveaux de barrières pour l'observation actuelle
        tp_long = df.iloc[i]['tp_long']
        sl_long = df.iloc[i]['sl_long']
        tp_short = df.iloc[i]['tp_short']
        sl_short = df.iloc[i]['sl_short']

        # --- 1. Déterminer les indices du PREMIER contact pour chaque barrière ---

        # Long side
        hit_tp_long_idx = future_window[future_window['high'] >= tp_long].index.min()
        hit_sl_long_idx = future_window[future_window['low'] <= sl_long].index.min()

        # Short side
        hit_tp_short_idx = future_window[future_window['low'] <= tp_short].index.min()
        hit_sl_short_idx = future_window[future_window['high'] >= sl_short].index.min()

        # --- 2. Trouver le premier événement global ---

        first_hits = pd.Series({
            'TP_Long': hit_tp_long_idx,
            'SL_Long': hit_sl_long_idx,
            'TP_Short': hit_tp_short_idx,
            'SL_Short': hit_sl_short_idx
        }).dropna()

        if first_hits.empty:
            labels.iloc[i] = 2.0  # Non résolu dans la fenêtre restante
            continue

        first_contact_time = first_hits.min()
        first_contacts = first_hits[first_hits == first_contact_time].index.tolist()

        # --- 3. Déterminer la Classe Cible (Label) ---

        if 'TP_Long' in first_contacts:
            labels.iloc[i] = 1.0
        elif 'TP_Short' in first_contacts:
            labels.iloc[i] = 0.0
        else:  # Si SL Long ou SL Short est touché en premier
            labels.iloc[i] = 2.0

    return labels.astype(int)


def calculate_adx_dmi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calcule l'Average Directional Index (ADX), le Positive Directional Indicator (+DI)
    et le Negative Directional Indicator (-DI) pour un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes 'high', 'low', 'close'.
        period (int): Période de lissage (par défaut 14).

    Returns:
        pd.DataFrame: DataFrame original augmenté des colonnes 'ADX', 'DI+', 'DI-'.
    """

    # Assurez-vous d'avoir des colonnes de travail sans modifier l'original (bonne pratique)
    df_result = df.copy()

    # --- 1. Mouvements Directionnels Bruts (UpMove, DownMove) ---
    df_result['UpMove'] = df_result['high'].diff()
    df_result['DownMove'] = df_result['low'].diff() * -1  # Rendre le mouvement vers le bas positif

    # --- 2. True Range (TR) ---
    high_low = df_result['high'] - df_result['low']
    high_close_prev = abs(df_result['high'] - df_result['close'].shift(1))
    low_close_prev = abs(df_result['low'] - df_result['close'].shift(1))

    # Le TR est le maximum de ces trois valeurs
    df_result['TR'] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

    # --- 3. Détermination des +DM et -DM ---
    df_result['+DM'] = np.where((df_result['UpMove'] > df_result['DownMove']) & (df_result['UpMove'] > 0),
                                df_result['UpMove'],
                                0)

    df_result['-DM'] = np.where((df_result['DownMove'] > df_result['UpMove']) & (df_result['DownMove'] > 0),
                                df_result['DownMove'],
                                0)

    # --- 4. Lissage (Wilder's Smoothing ou EMA) ---
    # Nous utilisons ewm (Exponentially Weighted Moving) avec adjust=False pour le lissage de Wilder

    alpha = 1 / period  # Facteur de lissage de Wilder

    df_result['Smoothed_+DM'] = df_result['+DM'].ewm(alpha=alpha, adjust=False).mean()
    df_result['Smoothed_-DM'] = df_result['-DM'].ewm(alpha=alpha, adjust=False).mean()
    df_result['Smoothed_TR'] = df_result['TR'].ewm(alpha=alpha, adjust=False).mean()

    # --- 5. Calcul des Indicateurs Directionnels (+DI et -DI) ---
    df_result['DI+'] = (df_result['Smoothed_+DM'] / df_result['Smoothed_TR']) * 100
    df_result['DI-'] = (df_result['Smoothed_-DM'] / df_result['Smoothed_TR']) * 100

    # --- 6. Calcul du DX et de l'ADX ---
    df_result['DX'] = (abs(df_result['DI+'] - df_result['DI-']) / (df_result['DI+'] + df_result['DI-'])) * 100

    # ADX est le lissage du DX
    df_result['ADX'] = df_result['DX'].ewm(alpha=alpha, adjust=False).mean()

    # --- Nettoyage et Retour ---
    # Suppression des colonnes intermédiaires pour garder seulement ADX, DI+, DI-
    cols_to_keep = df.columns.tolist() + ['ADX', 'DI+', 'DI-']

    return df_result[cols_to_keep]
