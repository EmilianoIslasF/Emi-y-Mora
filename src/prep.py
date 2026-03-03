"""
Preparación de datos (raw -> features -> splits).

Qué hace:
1) Lee CSVs en data/raw
2) Agrega ventas diarias a mensuales (target: item_cnt_month) + clipping
3) Construye una "matriz" (mes, tienda, producto) para incluir ceros
4) Crea features: categoría, estacionalidad, lags y agregados
5) Genera x_train, y_train, x_valid, y_valid y x_test (para inference)
6) Guarda CSVs en data/prep y data/inference
"""

import logging
import time

import numpy as np
import pandas as pd

import argparse
from pathlib import Path

from src.utils.features import add_lag
from src.utils.logging_config import setup_logger
from src.utils.paths import INFERENCE_DIR, PREP_DIR, RAW_DIR

pd.set_option("display.max_columns", 100)

# =========================
# Constantes del pipeline
# =========================
COLUMNA_TARGET = "item_cnt_month"
LAGS = [1, 2, 3, 6, 12]
CLIP_MIN, CLIP_MAX = 0, 20


def _parse_lags(lags_str: str) -> list[int]:
    """
    Parsea lags desde string tipo "1,2,3,6,12" o "1 2 3 6 12".
    Si viene vacío, regresa los defaults actuales.
    """
    s = (lags_str or "").strip()
    if not s:
        return LAGS
    # soporta coma o espacios
    parts = [p.strip() for p in s.replace(",", " ").split()]
    return [int(p) for p in parts if p]


def parse_args() -> argparse.Namespace:
    """
    CLI para permitir rutas/hiperparámetros desde Docker.
    Importante: con defaults se conserva el comportamiento actual.
    """
    p = argparse.ArgumentParser(description="Preprocessing step (raw -> prep/inference).")

    p.add_argument(
        "--raw-dir",
        type=str,
        default=str(RAW_DIR),
        help="Directorio con CSVs raw (sales_train.csv, items.csv, test.csv).",
    )
    p.add_argument(
        "--prep-dir",
        type=str,
        default=str(PREP_DIR),
        help="Directorio de salida para splits de entrenamiento/validación (data/prep).",
    )
    p.add_argument(
        "--inference-dir",
        type=str,
        default=str(INFERENCE_DIR),
        help="Directorio de salida para X_test de inferencia (data/inference).",
    )

    p.add_argument(
        "--clip-min",
        type=float,
        default=float(CLIP_MIN),
        help="Valor mínimo para clipping del target item_cnt_month.",
    )
    p.add_argument(
        "--clip-max",
        type=float,
        default=float(CLIP_MAX),
        help="Valor máximo para clipping del target item_cnt_month.",
    )
    p.add_argument(
        "--lags",
        type=str,
        default=",".join(map(str, LAGS)),
        help='Lista de lags, por ejemplo: "1,2,3,6,12".',
    )

    return p.parse_args()


def _aplicar_args(args: argparse.Namespace) -> None:
    """
    Aplica args a las variables globales del módulo.
    Con defaults, no cambia nada vs el comportamiento actual.
    """
    global RAW_DIR, PREP_DIR, INFERENCE_DIR, CLIP_MIN, CLIP_MAX, LAGS

    RAW_DIR = Path(args.raw_dir)
    PREP_DIR = Path(args.prep_dir)
    INFERENCE_DIR = Path(args.inference_dir)

    CLIP_MIN = int(args.clip_min) if float(args.clip_min).is_integer() else float(args.clip_min)
    CLIP_MAX = int(args.clip_max) if float(args.clip_max).is_integer() else float(args.clip_max)

    LAGS = _parse_lags(args.lags)


def cargar_datos_raw(logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los CSVs base desde data/raw."""
    logger.info("Cargando datos raw...")
    sales = pd.read_csv(RAW_DIR / "sales_train.csv")
    items = pd.read_csv(RAW_DIR / "items.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")

    logger.info("sales_train: %s filas", format(len(sales), ","))
    logger.info("items:       %s filas", format(len(items), ","))
    logger.info("test:        %s filas", format(len(test), ","))
    return sales, items, test


def crear_date_features(sales: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Convierte date a datetime y crea year, month y date_block_num (0..33)."""
    logger.info("Creando date features (year, month, date_block_num)...")
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"], format="%d.%m.%Y")
    sales["year"] = sales["date"].dt.year
    sales["month"] = sales["date"].dt.month
    sales["date_block_num"] = sales["date"].dt.to_period("M").factorize()[0]
    logger.info("date_block_num max en train: %d", int(sales["date_block_num"].max()))
    return sales


def agregar_mensual_target(sales: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Agrega ventas diarias a mensuales por (mes, tienda, producto) + clipping."""
    logger.info("Agregando diario -> mensual (target)...")
    monthly = sales.groupby(
        ["date_block_num", "shop_id", "item_id"], as_index=False
    ).agg(item_cnt_month=("item_cnt_day", "sum"))
    monthly[COLUMNA_TARGET] = monthly[COLUMNA_TARGET].clip(CLIP_MIN, CLIP_MAX)
    logger.info("monthly: %s filas (mes, tienda, item)", format(len(monthly), ","))
    logger.info("Clipping aplicado a %s: [%d, %d]", COLUMNA_TARGET, CLIP_MIN, CLIP_MAX)
    return monthly


def construir_grid_completo(monthly: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Crea filas (mes, tienda, producto) para aprender también los ceros:
    si un item no se vendió en el mes, debe existir una fila con 0.
    """
    logger.info("Construyendo grid completo (para incluir ceros)...")
    grid: list[pd.DataFrame] = []
    blocks = monthly["date_block_num"].unique()

    for block in blocks:
        cur = monthly[monthly["date_block_num"] == block]
        shops_in_month = cur["shop_id"].unique()
        items_in_month = cur["item_id"].unique()
        grid.append(
            pd.DataFrame(
                [(block, s, i) for s in shops_in_month for i in items_in_month],
                columns=["date_block_num", "shop_id", "item_id"],
            )
        )

    matrix = pd.concat(grid, ignore_index=True)
    matrix = matrix.merge(monthly, on=["date_block_num", "shop_id", "item_id"], how="left")
    matrix[COLUMNA_TARGET] = matrix[COLUMNA_TARGET].fillna(0)

    logger.info("matrix (con ceros): %s filas", format(len(matrix), ","))
    return matrix


def agregar_categoria(matrix: pd.DataFrame, items: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Agrega item_category_id a partir de items.csv."""
    logger.info("Agregando categoría (item_category_id)...")
    items_small = items[["item_id", "item_category_id"]]
    return matrix.merge(items_small, on="item_id", how="left")


def agregar_estacionalidad(matrix: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Crea variables de estacionalidad: month y year a partir de date_block_num."""
    logger.info("Agregando estacionalidad (month, year)...")
    matrix = matrix.copy()
    matrix["month"] = (matrix["date_block_num"] % 12).astype(np.int8)
    matrix["year"] = (matrix["date_block_num"] // 12).astype(np.int8)
    return matrix


def agregar_lags_y_agregados(matrix: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Crea lags del target y agregados (tienda y producto) con sus lags."""
    logger.info("Agregando lags del target: %s", LAGS)
    matrix = add_lag(matrix, LAGS, COLUMNA_TARGET)
    lag_cols = [c for c in matrix.columns if "lag_" in c]
    matrix[lag_cols] = matrix[lag_cols].fillna(0)

    logger.info("Agregado por tienda (mean mensual) + lags")
    shop_month = matrix.groupby(["date_block_num", "shop_id"], as_index=False).agg(
        shop_cnt_month=(COLUMNA_TARGET, "mean")
    )
    matrix = matrix.merge(shop_month, on=["date_block_num", "shop_id"], how="left")
    matrix = add_lag(matrix, LAGS, "shop_cnt_month")
    matrix = matrix.drop(columns=["shop_cnt_month"])

    logger.info("Agregado por item (mean mensual) + lags")
    item_month = matrix.groupby(["date_block_num", "item_id"], as_index=False).agg(
        item_cnt_month_mean=(COLUMNA_TARGET, "mean")
    )
    matrix = matrix.merge(item_month, on=["date_block_num", "item_id"], how="left")
    matrix = add_lag(matrix, LAGS, "item_cnt_month_mean")
    matrix = matrix.drop(columns=["item_cnt_month_mean"])

    matrix = matrix.fillna(0)
    logger.info("Lags y agregados listos")
    return matrix


def preparar_test_matrix(test: pd.DataFrame, items: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Construye dataset del mes 34 con columnas compatibles para inference."""
    logger.info("Preparando test_matrix (mes 34)...")
    items_small = items[["item_id", "item_category_id"]]

    test_matrix = test.copy()
    test_matrix["date_block_num"] = 34
    test_matrix = test_matrix.merge(items_small, on="item_id", how="left")

    test_matrix["month"] = (test_matrix["date_block_num"] % 12).astype(np.int8)
    test_matrix["year"] = (test_matrix["date_block_num"] // 12).astype(np.int8)

    test_matrix[COLUMNA_TARGET] = 0
    logger.info("test_matrix: %s filas", format(len(test_matrix), ","))
    return test_matrix


def split_train_valid_test(
    matrix: pd.DataFrame, test_matrix: pd.DataFrame, logger: logging.Logger
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Une train+test para asegurar mismas columnas/estructura y luego hace split temporal:
    - train: meses < 33
    - valid: mes 33
    - test: mes 34
    """
    logger.info("Construyendo all_data (train+test) para que tengan mismas features...")
    common_cols = [
        "date_block_num",
        "shop_id",
        "item_id",
        "item_category_id",
        "month",
        "year",
        COLUMNA_TARGET,
    ]
    all_data = pd.concat([matrix[common_cols], test_matrix[common_cols]], ignore_index=True)

    logger.info("Agregando lags a all_data (para que el test también tenga lags)...")
    all_data = add_lag(all_data, LAGS, COLUMNA_TARGET)
    lag_cols = [c for c in all_data.columns if "lag_" in c]
    all_data[lag_cols] = all_data[lag_cols].fillna(0)

    train_data = all_data[all_data["date_block_num"] <= 33].copy()
    test_data = all_data[all_data["date_block_num"] == 34].copy()

    x_train = train_data[train_data["date_block_num"] < 33].drop(columns=[COLUMNA_TARGET])
    y_train = train_data[train_data["date_block_num"] < 33][COLUMNA_TARGET]

    x_valid = train_data[train_data["date_block_num"] == 33].drop(columns=[COLUMNA_TARGET])
    y_valid = train_data[train_data["date_block_num"] == 33][COLUMNA_TARGET]

    x_test = test_data.drop(columns=[COLUMNA_TARGET])

    logger.info("Split temporal listo")
    logger.info("x_train: %s | y_train: %s", x_train.shape, y_train.shape)
    logger.info("x_valid: %s | y_valid: %s", x_valid.shape, y_valid.shape)
    logger.info("x_test:  %s", x_test.shape)
    return x_train, y_train, x_valid, y_valid, x_test


def guardar_outputs(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    x_test: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    """Guarda outputs en data/prep y data/inference."""
    logger.info("Guardando outputs (data/prep y data/inference)...")
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    x_train.to_csv(PREP_DIR / "X_train.csv", index=False)
    y_train.to_frame(COLUMNA_TARGET).to_csv(PREP_DIR / "y_train.csv", index=False)

    x_valid.to_csv(PREP_DIR / "X_valid.csv", index=False)
    y_valid.to_frame(COLUMNA_TARGET).to_csv(PREP_DIR / "y_valid.csv", index=False)

    x_test.to_csv(INFERENCE_DIR / "X_test.csv", index=False)

    logger.info("Listo: CSV creados en data/prep/ y data/inference/")


def main() -> None:
    """Ejecuta el pipeline completo de preparación de datos."""
    args = parse_args()
    _aplicar_args(args)

    logger = setup_logger("prep")
    start_time = time.time()
    logger.info("Inicio del script prep.py")

    sales, items, test = cargar_datos_raw(logger)
    sales = crear_date_features(sales, logger)

    monthly = agregar_mensual_target(sales, logger)

    matrix = construir_grid_completo(monthly, logger)
    matrix = agregar_categoria(matrix, items, logger)
    matrix = agregar_estacionalidad(matrix, logger)
    matrix = agregar_lags_y_agregados(matrix, logger)

    test_matrix = preparar_test_matrix(test, items, logger)

    x_train, y_train, x_valid, y_valid, x_test = split_train_valid_test(matrix, test_matrix, logger)
    guardar_outputs(x_train, y_train, x_valid, y_valid, x_test, logger)

    duration = time.time() - start_time
    logger.info("prep.py terminado correctamente en %.2f segundos", duration)


if __name__ == "__main__":
    main()