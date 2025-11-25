import pandas as pd
import numpy as np
import yfinance as yf
from src.preprocessing import calculate_atr_series, calculate_rsi, get_double_barrier_levels, label_double_barrier

try:
    df_raw = yf.download(tickers=['GC=F'], multi_level_index=False, interval='15m', period= '15d')
except:
    print("Il y a un problème dans les paramètres rentrés dans la fonction yf.download")

# Différence avec la cloture de la bougie précédente
df_raw['diff_close'] = df_raw['Close'].diff()

# Taille de la bougie
df_raw['taille_bougie'] = df_raw['High'] - df_raw['Low']
df_raw['taille_corps'] = np.maximum(df_raw['Close'], df_raw['Open']) - np.minimum(df_raw['Close'], df_raw['Open'])
df_raw['ratio_corps_bougie'] = df_raw['taille_corps'] / df_raw['taille_bougie']
df_raw['taille_meche_sup'] = df_raw['High'] - np.maximum(df_raw['Close'], df_raw['Open'])
df_raw['taille_meche_inf'] = np.minimum(df_raw['Close'], df_raw['Open']) - df_raw['Low']

# Moyenne Mobile Exponentielle
df_raw['ema_12'] = df_raw['Close'].ewm(span=12, adjust=False).mean()
df_raw['ema_26'] = df_raw['Close'].ewm(span=26, adjust=False).mean()
df_raw['ema_50'] = df_raw['Close'].ewm(span=50, adjust=False).mean()
df_raw['ema_200'] = df_raw['Close'].ewm(span=200, adjust=False).mean()

# Prix de Cloture - EMA
df_raw['close_ema_12'] = df_raw['Close'] - df_raw['ema_12']
df_raw['close_ema_26'] = df_raw['Close'] - df_raw['ema_26']
df_raw['close_ema_50'] = df_raw['Close'] - df_raw['ema_50']
df_raw['close_ema_200'] = df_raw['Close'] - df_raw['ema_200']

# Momentum
df_raw['momentum_court_terme'] = df_raw['ema_12'] - df_raw['ema_26']
df_raw['momentum_moyen_terme'] = df_raw['ema_12'] - df_raw['ema_50']
df_raw['momentum_long_terme'] = df_raw['ema_50'] - df_raw['ema_200']
df_raw['diff_momentum_ct'] = df_raw['momentum_court_terme'].diff()
df_raw['diff_momentum_mt'] = df_raw['momentum_moyen_terme'].diff()
df_raw['diff_momentum_lt'] = df_raw['momentum_long_terme'].diff()

# Croisement EMA
df_raw['croisement_ema_ct'] = (df_raw['ema_12'] > df_raw['ema_26']).astype(int)
df_raw['croisement_ema_mt'] = (df_raw['ema_12'] > df_raw['ema_50']).astype(int)
df_raw['croisement_ema_lt'] = (df_raw['ema_50'] > df_raw['ema_200']).astype(int)

# MACD
df_raw['macd'] = df_raw['ema_12'] - df_raw['ema_26']
df_raw['signal'] = df_raw['macd'].ewm(span=9, adjust=False).mean()
df_raw['histogramme'] = df_raw['macd'] - df_raw['signal']

# ROC EMA
df_raw['roc_ema_12'] = df_raw['ema_12'].diff(periods=5) / df_raw['ema_12'].shift(5)
df_raw['roc_ema_26'] = df_raw['ema_26'].diff(periods=5) / df_raw['ema_26'].shift(5)
df_raw['roc_ema_50'] = df_raw['ema_50'].diff(periods=5) / df_raw['ema_50'].shift(5)
df_raw['roc_ema_200'] = df_raw['ema_200'].diff(periods=5) / df_raw['ema_200'].shift(5)

# RSI
df_raw['rsi'] = calculate_rsi(df_raw, close='Close', N=14).astype(int)
df_raw['niveau_rsi'] = np.where(df_raw['rsi'] > 70, "Surachat", np.where((df_raw['rsi'] < 30) & (df_raw['rsi'] != 0), "Survente", "Normal"))
df_raw['midligne_rsi'] = (df_raw['rsi'] >= 50).astype(int)
df_raw['roc_rsi_1'] = df_raw['rsi'].diff(1) / df_raw['rsi'].shift(1)
df_raw['roc_rsi_5'] = df_raw['rsi'].diff(5) / df_raw['rsi'].shift(5) 

# Bandes de Bollinger
df_raw['middle_band'] = df_raw['Close'].rolling(window=20).mean()
std_20 = df_raw['Close'].rolling(window=20).std()
df_raw['upper_band'] = df_raw['middle_band'] + 2 * std_20
df_raw['lower_band'] = df_raw['middle_band'] - 2 * std_20
df_raw['bandwidth'] = (df_raw['upper_band'] - df_raw['lower_band']) / df_raw['middle_band']
df_raw['close_bollinger'] = (df_raw['Close'] - df_raw['lower_band']) / (df_raw['upper_band'] - df_raw['lower_band'])
df_raw['niveau_close_bb'] = np.where(df_raw['close_bollinger'] > 1, "Surachat", np.where(df_raw['close_bollinger'] < 0, "Survente", "Normal"))
df_raw['croisement_bb'] = df_raw['Close'] - df_raw['middle_band']

# Volume
price_direction = np.sign(df_raw['Close'].diff())
obv_change = price_direction * df_raw['Volume']
df_raw['obv'] = obv_change.cumsum()
df_raw['obv_sign'] = ((df_raw['obv'].diff()) > 0).astype(int)
df_raw['roc_obv'] = df_raw['obv'].diff(5) / df_raw['obv'].shift(5)
df_raw['typical_price'] = (df_raw['High'] + df_raw['Low'] + df_raw['Close']) / 3
df_raw['price_volume'] = df_raw['typical_price'] * df_raw['Volume']
df_raw['VWA_14'] = df_raw['price_volume'].rolling(14).sum() / df_raw['Volume'].rolling(14).sum()
df_raw['relation_prix_vwap'] = (df_raw['Close'] > df_raw['VWA_14']).astype(int)
df_raw['dist_prix_vwap'] = df_raw['Close'] - df_raw['VWA_14']

# Stochastique
max_high = df_raw['High'].rolling(14).max()
min_low = df_raw['Low'].rolling(14).min()
df_raw['stoch_k'] = (df_raw['Close'] - min_low) / (max_high - min_low)
df_raw['stoch_d'] = df_raw['stoch_k'].rolling(3).mean()
df_raw['niveau_stoch'] = np.where(df_raw['stoch_k'] > 0.8, "Surachat", np.where(df_raw['stoch_k'] < 0.2, "Survente", "Normal"))
df_raw['croisement_stoch'] = (df_raw['stoch_k'] > df_raw['stoch_d']).astype(int)

# ATR
df_raw['atr'] = calculate_atr_series(df_raw, period=14)
df_raw['close_ema_12_normalisee'] = df_raw['close_ema_12'] / df_raw['atr']
df_raw['close_ema_26_normalisee'] = df_raw['close_ema_26'] / df_raw['atr']
df_raw['close_ema_50_normalisee'] = df_raw['close_ema_50'] / df_raw['atr']
df_raw['close_ema_200_normalisee'] = df_raw['close_ema_200'] / df_raw['atr']

# Variables cibles
df_raw = get_double_barrier_levels(df_raw, atr_col='atr')

df_raw['label'] = label_double_barrier(df_raw)

df_raw = df_raw[df_raw['label'] != 2.0]

list_col_to_drop = df_raw.select_dtypes(include=np.number).columns[(df_raw.select_dtypes(include=np.number).mean() > 1000)].tolist()

df_raw.drop(columns=list_col_to_drop, inplace=True)


