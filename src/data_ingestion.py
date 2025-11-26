# src/data_ingestion.py

import yfinance as yf
import pandas as pd

def fetch_data(ticker: str = 'GC=F', interval: str = '1h', period: str = '15d') -> pd.DataFrame:
    """
    Télecharge les données historiques à partir de Yahoo Finance.
    Args:
        ticker: Symbole boursier
        interval: Intervalle des bougies
        period: Période à télécharger

    Returns:
        pd.Dataframe: Dataframe contenant les données.
    """
    try:
        df_raw = yf.download(
            tickers=ticker,
            multi_level_index=False,
            interval=interval,
            period= period)
        if df_raw.empty:
            raise ValueError(f"Aucune donnée trouvée pour le symbole {ticker} dans la période {period}")
        return df_raw
    except Exception as e:
        print(f"Erreur lors du téléchargement des données : {e}")
        return pd.DataFrame()