"""Funciones reutilizables para ingeniería de features (lags, etc.)."""

import pandas as pd


def add_lag(df: pd.DataFrame, lags: list[int], col: str) -> pd.DataFrame:
    """
    Agrega columnas rezagadas (lags) de `col` por (date_block_num, shop_id, item_id).

    Parámetros
    ----------
    df : pd.DataFrame
        Debe incluir: date_block_num, shop_id, item_id, y `col`.
    lags : list[int]
        Lags a crear (ej. [1,2,3,6,12]).
    col : str
        Nombre de la columna base (ej. "item_cnt_month").

    Regresa
    -------
    pd.DataFrame
        DataFrame con columnas nuevas: f"{col}_lag_{k}".
    """
    df = df.copy()
    tmp = df[["date_block_num", "shop_id", "item_id", col]]

    for lag in lags:
        shifted = tmp.copy()
        shifted["date_block_num"] += lag
        shifted = shifted.rename(columns={col: f"{col}_lag_{lag}"})

        df = df.merge(shifted, on=["date_block_num", "shop_id", "item_id"], how="left")

    return df
