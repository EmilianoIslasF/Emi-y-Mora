"""
Preparación de datos para SageMaker Processing.

Qué hace:
1) Lee CSVs en input-dir
2) Agrega ventas diarias a mensuales (target: item_cnt_month) + clipping
3) Construye una matriz (mes, tienda, producto) para incluir ceros
4) Crea features ligeras: categoría, estacionalidad y lags del target
5) Genera 3 splits temporales etiquetados:
   - train: meses < 32
   - validation: mes 32
   - test: mes 33
6) Guarda:
   - train/train.csv
   - validation/validation.csv
   - test/test.csv

Notas:
- Este script deja listo el ProcessingStep del pipeline.
- Para batch transform del mes 34 (Kaggle test) lo preparamos después.
"""

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

pd.set_option("display.max_columns", 100)

# =========================
# Constantes del pipeline
# =========================
COLUMNA_TARGET = "item_cnt_month"
LAGS = [1, 2, 3, 6, 12]
CLIP_MIN, CLIP_MAX = 0, 20

# Defaults locales / SageMaker
LOCAL_INPUT_DIR = Path("data/raw")
LOCAL_TRAIN_OUTPUT_DIR = Path("data/processed/train")
LOCAL_VALIDATION_OUTPUT_DIR = Path("data/processed/validation")
LOCAL_TEST_OUTPUT_DIR = Path("data/processed/test")

SM_INPUT_DIR = Path("/opt/ml/processing/input")
SM_TRAIN_OUTPUT_DIR = Path("/opt/ml/processing/output/train")
SM_VALIDATION_OUTPUT_DIR = Path("/opt/ml/processing/output/validation")
SM_TEST_OUTPUT_DIR = Path("/opt/ml/processing/output/test")


def _default_input_dir() -> Path:
    return SM_INPUT_DIR if SM_INPUT_DIR.exists() else LOCAL_INPUT_DIR


def _default_train_output_dir() -> Path:
    return SM_TRAIN_OUTPUT_DIR if Path("/opt/ml/processing").exists() else LOCAL_TRAIN_OUTPUT_DIR


def _default_validation_output_dir() -> Path:
    return (
        SM_VALIDATION_OUTPUT_DIR
        if Path("/opt/ml/processing").exists()
        else LOCAL_VALIDATION_OUTPUT_DIR
    )


def _default_test_output_dir() -> Path:
    return SM_TEST_OUTPUT_DIR if Path("/opt/ml/processing").exists() else LOCAL_TEST_OUTPUT_DIR


def _parse_lags(lags_str: str) -> list[int]:
    """
    Parsea lags desde string tipo '1,2,3,6,12' o '1 2 3 6 12'.
    """
    s = (lags_str or "").strip()
    if not s:
        return LAGS
    parts = [p.strip() for p in s.replace(",", " ").split()]
    return [int(p) for p in parts if p]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocessing step para SageMaker (raw -> train/validation/test)."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(_default_input_dir()),
        help="Directorio con CSVs raw (sales_train.csv, items.csv).",
    )
    parser.add_argument(
        "--train-output-dir",
        type=str,
        default=str(_default_train_output_dir()),
        help="Directorio de salida para train.csv.",
    )
    parser.add_argument(
        "--validation-output-dir",
        type=str,
        default=str(_default_validation_output_dir()),
        help="Directorio de salida para validation.csv.",
    )
    parser.add_argument(
        "--test-output-dir",
        type=str,
        default=str(_default_test_output_dir()),
        help="Directorio de salida para test.csv.",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=float(CLIP_MIN),
        help="Valor mínimo para clipping del target item_cnt_month.",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=float(CLIP_MAX),
        help="Valor máximo para clipping del target item_cnt_month.",
    )
    parser.add_argument(
        "--lags",
        type=str,
        default=",".join(map(str, LAGS)),
        help='Lista de lags, por ejemplo: "1,2,3,6,12".',
    )

    return parser.parse_args()


def _aplicar_args(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, float, float, list[int]]:
    input_dir = Path(args.input_dir)
    train_output_dir = Path(args.train_output_dir)
    validation_output_dir = Path(args.validation_output_dir)
    test_output_dir = Path(args.test_output_dir)

    clip_min = int(args.clip_min) if float(args.clip_min).is_integer() else float(args.clip_min)
    clip_max = int(args.clip_max) if float(args.clip_max).is_integer() else float(args.clip_max)
    lags = _parse_lags(args.lags)

    return (
        input_dir,
        train_output_dir,
        validation_output_dir,
        test_output_dir,
        clip_min,
        clip_max,
        lags,
    )


def cargar_datos_raw(
    input_dir: Path, logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga los CSVs base desde input_dir.
    Espera al menos:
    - sales_train.csv
    - items.csv
    """
    logger.info("Cargando datos raw desde %s ...", input_dir)

    sales = pd.read_csv(input_dir / "sales_train.csv")
    items = pd.read_csv(input_dir / "items.csv")

    logger.info("sales_train: %s filas", format(len(sales), ","))
    logger.info("items:       %s filas", format(len(items), ","))

    return sales, items


def crear_date_features(sales: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Convierte date a datetime y crea year, month y date_block_num (0..33).
    """
    logger.info("Creando date features (year, month, date_block_num)...")
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"], format="%d.%m.%Y")
    sales["year"] = sales["date"].dt.year.astype(np.int16)
    sales["month"] = sales["date"].dt.month.astype(np.int8)
    sales["date_block_num"] = sales["date"].dt.to_period("M").factorize()[0].astype(np.int8)

    logger.info(
        "date_block_num min/max en train: %d / %d",
        int(sales["date_block_num"].min()),
        int(sales["date_block_num"].max()),
    )
    return sales


def agregar_mensual_target(
    sales: pd.DataFrame, clip_min: float, clip_max: float, logger: logging.Logger
) -> pd.DataFrame:
    """
    Agrega ventas diarias a mensuales por (mes, tienda, producto) + clipping.
    """
    logger.info("Agregando diario -> mensual (target)...")

    monthly = sales.groupby(
        ["date_block_num", "shop_id", "item_id"], as_index=False
    ).agg(item_cnt_month=("item_cnt_day", "sum"))

    monthly[COLUMNA_TARGET] = monthly[COLUMNA_TARGET].clip(clip_min, clip_max).astype(np.float32)
    monthly["date_block_num"] = monthly["date_block_num"].astype(np.int8)
    monthly["shop_id"] = monthly["shop_id"].astype(np.int16)
    monthly["item_id"] = monthly["item_id"].astype(np.int32)

    logger.info("monthly: %s filas (mes, tienda, item)", format(len(monthly), ","))
    logger.info("Clipping aplicado a %s: [%s, %s]", COLUMNA_TARGET, clip_min, clip_max)

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

        block_df = pd.DataFrame(
            [(block, shop_id, item_id) for shop_id in shops_in_month for item_id in items_in_month],
            columns=["date_block_num", "shop_id", "item_id"],
        )

        block_df["date_block_num"] = block_df["date_block_num"].astype(np.int8)
        block_df["shop_id"] = block_df["shop_id"].astype(np.int16)
        block_df["item_id"] = block_df["item_id"].astype(np.int32)
        grid.append(block_df)

    matrix = pd.concat(grid, ignore_index=True)
    matrix = matrix.merge(monthly, on=["date_block_num", "shop_id", "item_id"], how="left")
    matrix[COLUMNA_TARGET] = matrix[COLUMNA_TARGET].fillna(0).astype(np.float32)

    logger.info("matrix (con ceros): %s filas", format(len(matrix), ","))
    return matrix


def agregar_categoria(matrix: pd.DataFrame, items: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Agrega item_category_id a partir de items.csv.
    """
    logger.info("Agregando categoría (item_category_id)...")

    items_small = items[["item_id", "item_category_id"]].copy()
    items_small["item_id"] = items_small["item_id"].astype(np.int32)
    items_small["item_category_id"] = items_small["item_category_id"].astype(np.int16)

    matrix = matrix.merge(items_small, on="item_id", how="left")
    matrix["item_category_id"] = matrix["item_category_id"].fillna(-1).astype(np.int16)

    return matrix


def agregar_estacionalidad(matrix: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Crea variables de estacionalidad: month y year a partir de date_block_num.
    """
    logger.info("Agregando estacionalidad (month, year)...")
    matrix["month"] = (matrix["date_block_num"] % 12).astype(np.int8)
    matrix["year"] = (matrix["date_block_num"] // 12).astype(np.int8)
    return matrix


def agregar_target_lags_ligeros(
    data: pd.DataFrame, lags: list[int], logger: logging.Logger
) -> pd.DataFrame:
    """
    Agrega lags del target usando groupby + shift.
    """
    logger.info("Agregando lags ligeros del target: %s", lags)

    data = data.sort_values(["shop_id", "item_id", "date_block_num"]).copy()
    grp = data.groupby(["shop_id", "item_id"], sort=False)[COLUMNA_TARGET]

    for lag in lags:
        col = f"{COLUMNA_TARGET}_lag_{lag}"
        data[col] = grp.shift(lag).fillna(0).astype(np.float32)

    logger.info("Lags ligeros creados correctamente")
    return data


def split_y_guardar_outputs(
    matrix: pd.DataFrame,
    lags: list[int],
    train_output_dir: Path,
    validation_output_dir: Path,
    test_output_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Prepara el dataset final, hace split temporal y guarda cada split
    directamente a disco para evitar copias gigantes en memoria.

    Split:
    - train: meses < 32
    - validation: mes 32
    - test: mes 33

    Train se guarda fragmentado por mes para evitar un train.csv gigante.
    """
    logger.info("Preparando dataset final con lags y split temporal...")

    base_cols = [
        "date_block_num",
        "shop_id",
        "item_id",
        "item_category_id",
        "month",
        "year",
        COLUMNA_TARGET,
    ]

    data = matrix.loc[:, base_cols].copy()

    del matrix
    gc.collect()

    data["date_block_num"] = data["date_block_num"].astype(np.int8)
    data["shop_id"] = data["shop_id"].astype(np.int16)
    data["item_id"] = data["item_id"].astype(np.int32)
    data["item_category_id"] = data["item_category_id"].astype(np.int16)
    data["month"] = data["month"].astype(np.int8)
    data["year"] = data["year"].astype(np.int8)
    data[COLUMNA_TARGET] = data[COLUMNA_TARGET].astype(np.float32)

    data = agregar_target_lags_ligeros(data, lags, logger)

    lag_cols = [f"{COLUMNA_TARGET}_lag_{lag}" for lag in lags]
    ordered_cols = [
        "date_block_num",
        "shop_id",
        "item_id",
        "item_category_id",
        "month",
        "year",
        *lag_cols,
        COLUMNA_TARGET,
    ]
    data = data[ordered_cols]

    train_output_dir.mkdir(parents=True, exist_ok=True)
    validation_output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)

    validation_path = validation_output_dir / "validation.csv"
    test_path = test_output_dir / "test.csv"

    months = sorted(data["date_block_num"].unique().tolist())
    train_months = [m for m in months if 20 <= m < 32]

    logger.info("Meses train: %s", train_months)

    train_rows = int((data["date_block_num"] < 32).sum())
    validation_rows = int((data["date_block_num"] == 32).sum())
    test_rows = int((data["date_block_num"] == 33).sum())

    logger.info(
        "Filas por split -> train: %s | validation: %s | test: %s",
        train_rows,
        validation_rows,
        test_rows,
    )

    logger.info("Guardando train fragmentado por mes...")
    for month in train_months:
        month_path = train_output_dir / f"train_{int(month):02d}.csv"
        month_df = data.loc[data["date_block_num"] == month, ordered_cols]
        logger.info(
            "Guardando mes %s en %s (%s filas)",
            int(month),
            month_path,
            len(month_df),
        )
        month_df.to_csv(month_path, index=False)
        del month_df
        gc.collect()
        
    logger.info("Guardando validation.csv ...")
    validation_df = data.loc[data["date_block_num"] == 32, ordered_cols]
    validation_df.to_csv(validation_path, index=False)
    del validation_df
    gc.collect()

    logger.info("Guardando test.csv ...")
    test_df = data.loc[data["date_block_num"] == 33, ordered_cols]
    test_df.to_csv(test_path, index=False)
    del test_df
    gc.collect()

    logger.info("Guardando train fragmentado por mes...")
    for month in train_months:
        month_path = train_output_dir / f"train_{int(month):02d}.csv"
        month_df = data.loc[data["date_block_num"] == month, ordered_cols]
        logger.info(
            "Guardando mes %s en %s (%s filas)",
            int(month),
            month_path,
            len(month_df),
        )
        month_df.to_csv(month_path, index=False)
        del month_df
        gc.collect()

    logger.info("Archivos train guardados en: %s", train_output_dir)
    logger.info("Archivo validation guardado en: %s", validation_path)
    logger.info("Archivo test guardado en: %s", test_path)


def main() -> None:
    args = parse_args()
    (
        input_dir,
        train_output_dir,
        validation_output_dir,
        test_output_dir,
        clip_min,
        clip_max,
        lags,
    ) = _aplicar_args(args)

    logger = setup_logger("prep")
    start_time = time.time()

    logger.info("Inicio del script prep.py")
    logger.info("input_dir: %s", input_dir)
    logger.info("train_output_dir: %s", train_output_dir)
    logger.info("validation_output_dir: %s", validation_output_dir)
    logger.info("test_output_dir: %s", test_output_dir)
    logger.info("lags: %s", lags)

    sales, items = cargar_datos_raw(input_dir, logger)
    sales = crear_date_features(sales, logger)

    monthly = agregar_mensual_target(sales, clip_min, clip_max, logger)

    matrix = construir_grid_completo(monthly, logger)
    del monthly
    gc.collect()

    matrix = agregar_categoria(matrix, items, logger)
    del items
    gc.collect()

    matrix = agregar_estacionalidad(matrix, logger)

    logger.info("Saltando features pesadas para evitar OOM en SageMaker Studio")

    split_y_guardar_outputs(
        matrix=matrix,
        lags=lags,
        train_output_dir=train_output_dir,
        validation_output_dir=validation_output_dir,
        test_output_dir=test_output_dir,
        logger=logger,
    )

    duration = time.time() - start_time
    logger.info("prep.py terminado correctamente en %.2f segundos", duration)


if __name__ == "__main__":
    main()