import sqlite3
from typing import List, Tuple

import pandas as pd

DB_NAME = 'trading_data.db'
TABLE_NAME = 'prix_ohlcv'


def create_connection(db_file: str = DB_NAME) -> sqlite3.Connection:
    """Créer une connexion à la base de donnée SQLite"""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(f"Erreur de connexion à la base de donnée {e}")
        if conn:
            conn.close()
        return None


def create_table(conn : sqlite3.Connection):
    """Créer la table des prix si elle n'existe pas"""
    sql_create_table = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        symbol TEXT NOT NULL,
        datetime TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        PRIMARY KEY (symbol, datetime)
    );
    """

    try:
        cursor = conn.cursor()
        cursor.execute(sql_create_table)
        conn.commit()
        print(f"Table {TABLE_NAME} vérifiée / créée avec succès")
    except sqlite3.Error as e:
        print(f"Erreur lors de la création de la table {e}")


def insert_data(conn: sqlite3.Connection, df: pd.DataFrame):
    """Insère de nouvelles données ou met à jour les données existantes."""
    data_to_insert: List[Tuple] = df.to_records(index=False).tolist()

    sql_insert = f"""
    INSERT OR REPLACE INTO {TABLE_NAME}
    (symbol, datetime, open, high, low, close, volume)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    try:
        cursor = conn.cursor()
        cursor.executemany(sql_insert, data_to_insert)
        conn.commit()
        print(f"Insertion/mise à jour de {len(data_to_insert)} lignes réussie.")
    except sqlite3.Error as e:
        print(f"Erreur lors de l'insertion des données {e}")


def load_data(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    """Charge les données d'un actif dans un dataframe pandas"""
    try:
        query = f"""
        SELECT datetime, open, high, low, close, volume
        FROM {TABLE_NAME}
        WHERE symbol = ?
        ORDER BY datetime ASC
        """
        params = (symbol,)

        df = pd.read_sql(
            sql=query,
            con=conn,
            params=params,
            index_col='datetime',
            parse_dates=True
        )
        print(f"Chargement réussi de {len(df)} lignes pour {symbol}.")
        return df
    except sqlite3.Error as e:
        print(f"Erreur lors du chargement des données {e}")
        return pd.DataFrame()


def get_last_datetime(conn: sqlite3.Connection, symbol: str)-> str:
    """Retourne la date la plus récente pour un actif donné."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT MAX(datetime) FROM {TABLE_NAME} WHERE symbol = ?", (symbol,))
        last_dt = cursor.fetchone()[0]
        return last_dt if last_dt else '1970-01-01 00:00:00'
    except sqlite3.Error as e:
        print(f"Erreur lors de la récupération de la dernière date: {e}")
        return '1970-01-01 00:00:00'