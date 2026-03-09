"""
Preparación de datos (raw -> features -> splits).

Qué hace:
1) Lee CSVs en data/raw
2) Agrega ventas diarias a mensuales (target: item_cnt_month) + clipping
3) Construye una "matriz" (mes, tienda, producto) para incluir ceros
4) Crea features ligeras: categoría, estacionalidad y lags del target
5) Genera x_train, y_train, x_valid, y_valid y x_test (para inference)
6) Guarda CSVs en data/prep y data/inference
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
    """
    s = (lags_str or "").strip()
    if not s:
        return LAGS
    parts = [p.strip() for p in s.replace(",", " ").split()]
    return [int(p) for p in parts if p]


def parse_args() -> argparse.Namespace:
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


def _aplicar_args(args: argparse.Namespace) -> tuple[Path, Path, Path, float, float, list[int]]:
    raw_dir = Path(args.raw_dir)
    prep_dir = Path(args.prep_dir)
    inference_dir = Path(args.inference_dir)

    clip_min = int(args.clip_min) if float(args.clip_min).is_integer() else float(args.clip_min)
    clip_max = int(args.clip_max) if float(args.clip_max).is_integer() else float(args.clip_max)
    lags = _parse_lags(args.lags)

    return raw_dir, prep_dir, inference_dir, clip_min, clip_max, lags


def cargar_datos_raw(
    raw_dir: Path, logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los CSVs base desde data/raw."""
    logger.info("Cargando datos raw...")
    sales = pd.read_csv(raw_dir / "sales_train.csv")
    items = pd.read_csv(raw_dir / "items.csv")
    test = pd.read_csv(raw_dir / "test.csv")

    logger.info("sales_train: %s filas", format(len(sales), ","))
    logger.info("items:       %s filas", format(len(items), ","))
    logger.info("test:        %s filas", format(len(test), ","))

    return sales, items, test


def crear_date_features(sales: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Convierte date a datetime y crea year, month y date_block_num (0..33)."""
    logger.info("Creando date features (year, month, date_block_num)...")
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"], format="%d.%m.%Y")
    sales["year"] = sales["date"].dt.year.astype(np.int16)
    sales["month"] = sales["date"].dt.month.astype(np.int8)
    sales["date_block_num"] = sales["date"].dt.to_period("M").factorize()[0].astype(np.int8)
    logger.info("date_block_num max en train: %d", int(sales["date_block_num"].max()))
    return sales


def agregar_mensual_target(
    sales: pd.DataFrame, clip_min: float, clip_max: float, logger: logging.Logger
) -> pd.DataFrame:
    """Agrega ventas diarias a mensuales por (mes, tienda, producto) + clipping."""
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
            [(block, s, i) for s in shops_in_month for i in items_in_month],
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
    """Agrega item_category_id a partir de items.csv."""
    logger.info("Agregando categoría (item_category_id)...")
    items_small = items[["item_id", "item_category_id"]].copy()
    items_small["item_id"] = items_small["item_id"].astype(np.int32)
    items_small["item_category_id"] = items_small["item_category_id"].astype(np.int16)

    matrix = matrix.merge(items_small, on="item_id", how="left")
    matrix["item_category_id"] = matrix["item_category_id"].fillna(-1).astype(np.int16)
    return matrix


def agregar_estacionalidad(matrix: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Crea variables de estacionalidad: month y year a partir de date_block_num."""
    logger.info("Agregando estacionalidad (month, year)...")
    matrix = matrix.copy()
    matrix["month"] = (matrix["date_block_num"] % 12).astype(np.int8)
    matrix["year"] = (matrix["date_block_num"] // 12).astype(np.int8)
    return matrix


def preparar_test_matrix(test: pd.DataFrame, items: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Construye dataset del mes 34 con columnas compatibles para inference."""
    logger.info("Preparando test_matrix (mes 34)...")
    items_small = items[["item_id", "item_category_id"]].copy()
    items_small["item_id"] = items_small["item_id"].astype(np.int32)
    items_small["item_category_id"] = items_small["item_category_id"].astype(np.int16)

    test_matrix = test.copy()
    test_matrix["date_block_num"] = 34
    test_matrix = test_matrix.merge(items_small, on="item_id", how="left")

    test_matrix["shop_id"] = test_matrix["shop_id"].astype(np.int16)
    test_matrix["item_id"] = test_matrix["item_id"].astype(np.int32)
    test_matrix["item_category_id"] = test_matrix["item_category_id"].fillna(-1).astype(np.int16)
    test_matrix["date_block_num"] = test_matrix["date_block_num"].astype(np.int8)
    test_matrix["month"] = (test_matrix["date_block_num"] % 12).astype(np.int8)
    test_matrix["year"] = (test_matrix["date_block_num"] // 12).astype(np.int8)
    test_matrix[COLUMNA_TARGET] = np.float32(0)

    logger.info("test_matrix: %s filas", format(len(test_matrix), ","))
    return test_matrix


def agregar_target_lags_ligeros(
    all_data: pd.DataFrame, lags: list[int], logger: logging.Logger
) -> pd.DataFrame:
    """
    Agrega lags del target de forma más ligera usando groupby + shift.
    """
    logger.info("Agregando lags ligeros del target: %s", lags)

    all_data = all_data.sort_values(["shop_id", "item_id", "date_block_num"]).copy()
    grp = all_data.groupby(["shop_id", "item_id"], sort=False)[COLUMNA_TARGET]

    for lag in lags:
        col = f"{COLUMNA_TARGET}_lag_{lag}"
        all_data[col] = grp.shift(lag).fillna(0).astype(np.float32)

    logger.info("Lags ligeros creados correctamente")
    return all_data


def split_train_valid_test(
    matrix: pd.DataFrame,
    test_matrix: pd.DataFrame,
    lags: list[int],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Une train+test para asegurar mismas columnas/estructura y luego hace split temporal:
    - train: meses < 33
    - valid: mes 33
    - test: mes 34

    Aquí mantenemos lags del target, pero con una implementación ligera.
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

    all_data["date_block_num"] = all_data["date_block_num"].astype(np.int8)
    all_data["shop_id"] = all_data["shop_id"].astype(np.int16)
    all_data["item_id"] = all_data["item_id"].astype(np.int32)
    all_data["item_category_id"] = all_data["item_category_id"].astype(np.int16)
    all_data["month"] = all_data["month"].astype(np.int8)
    all_data["year"] = all_data["year"].astype(np.int8)
    all_data[COLUMNA_TARGET] = all_data[COLUMNA_TARGET].astype(np.float32)

    all_data = agregar_target_lags_ligeros(all_data, lags, logger)

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
    prep_dir: Path,
    inference_dir: Path,
    logger: logging.Logger,
) -> None:
    """Guarda outputs en data/prep y data/inference."""
    logger.info("Guardando outputs (data/prep y data/inference)...")
    prep_dir.mkdir(parents=True, exist_ok=True)
    inference_dir.mkdir(parents=True, exist_ok=True)

    x_train.to_csv(prep_dir / "X_train.csv", index=False)
    y_train.to_frame(COLUMNA_TARGET).to_csv(prep_dir / "y_train.csv", index=False)

    x_valid.to_csv(prep_dir / "X_valid.csv", index=False)
    y_valid.to_frame(COLUMNA_TARGET).to_csv(prep_dir / "y_valid.csv", index=False)

    x_test.to_csv(inference_dir / "X_test.csv", index=False)

    logger.info("Listo: CSV creados en data/prep/ y data/inference/")


def main() -> None:
    args = parse_args()
    raw_dir, prep_dir, inference_dir, clip_min, clip_max, lags = _aplicar_args(args)

    logger = setup_logger("prep")
    start_time = time.time()
    logger.info("Inicio del script prep.py")

    sales, items, test = cargar_datos_raw(raw_dir, logger)
    sales = crear_date_features(sales, logger)

    monthly = agregar_mensual_target(sales, clip_min, clip_max, logger)

    matrix = construir_grid_completo(monthly, logger)
    matrix = agregar_categoria(matrix, items, logger)
    matrix = agregar_estacionalidad(matrix, logger)

    # Quitamos esta parte pesada porque explota RAM en Studio
    logger.info("Saltando agregar_lags_y_agregados(matrix) para evitar OOM")

    test_matrix = preparar_test_matrix(test, items, logger)

    x_train, y_train, x_valid, y_valid, x_test = split_train_valid_test(
        matrix, test_matrix, lags, logger
    )
    guardar_outputs(x_train, y_train, x_valid, y_valid, x_test, prep_dir, inference_dir, logger)

    duration = time.time() - start_time
    logger.info("prep.py terminado correctamente en %.2f segundos", duration)


if __name__ == "__main__":
    main()
