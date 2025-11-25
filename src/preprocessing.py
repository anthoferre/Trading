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
    col_high: str = 'High', 
    col_low: str = 'Low', 
    col_close: str = 'Close'
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
    
    # Barrière Supérieure (Take-Profit)
    df['Barrier_Up'] = df['Close'] + (atr_threshold * profit_mult)
    
    # Barrière Inférieure (Stop-Loss)
    df['Barrier_Down'] = df['Close'] - (atr_threshold * stop_mult)
    
    return df

def label_double_barrier(df: pd.DataFrame) -> pd.Series:
    """
    Détermine la classe cible (1 ou -1) en fonction du premier contact 
    avec la barrière de profit ou de perte sur les données futures disponibles.
    
    Retourne 0 si aucune barrière n'est touchée (cas rare, fin de série).
    """
    
    labels = pd.Series(index=df.index, dtype=float)
    
    # Itération sur toutes les périodes
    for i in range(len(df)):
        
        current_index = df.index[i]
        
        barrier_up = df.loc[current_index, 'Barrier_Up']
        barrier_down = df.loc[current_index, 'Barrier_Down']
        
        # La fenêtre de recherche future est TOUT le reste du DataFrame
        # Attention : C'est ce qui rend l'itération lente
        future_window = df.iloc[i+1:]
        
        # --- 1. Vérification des contacts ---
        
        # Le prix Haut a-t-il touché le niveau de profit ?
        hit_up = future_window[future_window['High'] >= barrier_up]
        
        # Le prix Bas a-t-il touché le niveau de stop ?
        hit_down = future_window[future_window['Low'] <= barrier_down]
        
        # --- 2. Détermination du premier contact ---
        
        if hit_up.empty and hit_down.empty:
            # Cas 0: Aucune barrière touchée (se produit uniquement vers la fin de la série)
            labels.loc[current_index] = 2
            
        elif not hit_up.empty and (hit_down.empty or hit_up.index[0] < hit_down.index[0]):
            # Classe 1: Barrière Supérieure touchée en premier
            labels.loc[current_index] = 1
            
        elif not hit_down.empty and (hit_up.empty or hit_down.index[0] < hit_up.index[0]):
            # Classe -1: Barrière Inférieure touchée en premier
            labels.loc[current_index] = 0
            
        elif not hit_up.empty and not hit_down.empty and hit_up.index[0] == hit_down.index[0]:
            # Cas d'un contact le même jour (gestion du conflit - souvent arbitraire)
            # Ici, nous favorisons le signal le plus fort si les deux sont touchés.
            # Pour simplifier, nous attribuons 0 ou -1/+1 en fonction du plus grand mouvement.
            # Pour l'instant, laissons 0 pour indiquer le conflit/ambiguïté.
            labels.loc[current_index] = 2
            
    return labels
