"""
Entrenamiento del modelo (Processing outputs -> model artifacts).

Compatible con:
- ejecución local
- Amazon SageMaker BYOC TrainingStep

Soporta 2 layouts de entrada:

1) Layout nuevo (recomendado)
   - train/train.csv
   - validation/validation.csv
   Cada CSV incluye la columna target: item_cnt_month

2) Layout legacy
   - X_train.csv / y_train.csv
   - X_valid.csv / y_valid.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logging_config import setup_logger

TARGET_COL = "item_cnt_month"
CLIP_MIN, CLIP_MAX = 0, 20
SEED = 42

# =========================
# Rutas locales
# =========================
LOCAL_TRAIN_DIR = Path("data/processed/train")
LOCAL_VALIDATION_DIR = Path("data/processed/validation")
LOCAL_ARTIFACTS_DIR = Path("artifacts")
LEGACY_PREP_DIR = Path("data/prep")

# =========================
# Rutas estándar SageMaker
# =========================
SM_INPUT_CONFIG = Path("/opt/ml/input/config/hyperparameters.json")
SM_INPUT_TRAIN = Path("/opt/ml/input/data/train")
SM_INPUT_VALIDATION = Path("/opt/ml/input/data/validation")
SM_MODEL_DIR = Path("/opt/ml/model")


def running_in_sagemaker() -> bool:
    """
    Detecta si estamos dentro de un SageMaker Training Job real.
    No basta con que exista /opt/ml, porque en Studio también existe.
    """
    return (
        "SM_TRAINING_ENV" in os.environ
        or SM_INPUT_CONFIG.exists()
        or SM_INPUT_TRAIN.exists()
        or SM_INPUT_VALIDATION.exists()
    )


def load_sagemaker_hyperparameters() -> dict[str, Any]:
    if not SM_INPUT_CONFIG.exists():
        return {}

    with open(SM_INPUT_CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)


def hp_get(hp: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in hp:
            return hp[key]
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training step para SageMaker (train/validation -> model)."
    )

    parser.add_argument(
        "--train-dir",
        type=str,
        default=str(LOCAL_TRAIN_DIR),
        help="Directorio con train.csv o layout legacy.",
    )
    parser.add_argument(
        "--validation-dir",
        type=str,
        default=str(LOCAL_VALIDATION_DIR),
        help="Directorio con validation.csv o layout legacy.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=str(LOCAL_ARTIFACTS_DIR),
        help="Directorio donde se guardan modelo y métricas.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="model.joblib",
        help="Nombre del archivo del modelo.",
    )
    parser.add_argument(
        "--metrics-name",
        type=str,
        default="metrics.json",
        help="Nombre del archivo de métricas.",
    )

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--clip-min", type=float, default=float(CLIP_MIN))
    parser.add_argument("--clip-max", type=float, default=float(CLIP_MAX))

    args, _ = parser.parse_known_args()

    cli_tokens = sys.argv[1:]

    user_set_train_dir = any(
        t == "--train-dir" or t.startswith("--train-dir=") for t in cli_tokens
    )
    user_set_validation_dir = any(
        t == "--validation-dir" or t.startswith("--validation-dir=") for t in cli_tokens
    )
    user_set_artifacts_dir = any(
        t == "--artifacts-dir" or t.startswith("--artifacts-dir=") for t in cli_tokens
    )
    user_set_model_name = any(
        t == "--model-name" or t.startswith("--model-name=") for t in cli_tokens
    )
    user_set_metrics_name = any(
        t == "--metrics-name" or t.startswith("--metrics-name=") for t in cli_tokens
    )
    user_set_alpha = any(t == "--alpha" or t.startswith("--alpha=") for t in cli_tokens)
    user_set_n_estimators = any(
        t == "--n-estimators" or t.startswith("--n-estimators=") for t in cli_tokens
    )
    user_set_learning_rate = any(
        t == "--learning-rate" or t.startswith("--learning-rate=") for t in cli_tokens
    )
    user_set_max_depth = any(
        t == "--max-depth" or t.startswith("--max-depth=") for t in cli_tokens
    )
    user_set_seed = any(t == "--seed" or t.startswith("--seed=") for t in cli_tokens)
    user_set_clip_min = any(
        t == "--clip-min" or t.startswith("--clip-min=") for t in cli_tokens
    )
    user_set_clip_max = any(
        t == "--clip-max" or t.startswith("--clip-max=") for t in cli_tokens
    )

    if running_in_sagemaker():
        hp = load_sagemaker_hyperparameters()

        if not user_set_train_dir:
            args.train_dir = str(
                hp_get(hp, "train-dir", "train_dir", default=str(SM_INPUT_TRAIN))
            )
        if not user_set_validation_dir:
            args.validation_dir = str(
                hp_get(
                    hp,
                    "validation-dir",
                    "validation_dir",
                    default=str(SM_INPUT_VALIDATION),
                )
            )
        if not user_set_artifacts_dir:
            args.artifacts_dir = str(
                hp_get(hp, "artifacts-dir", "artifacts_dir", default=str(SM_MODEL_DIR))
            )
        if not user_set_model_name:
            args.model_name = str(
                hp_get(hp, "model-name", "model_name", default="model.joblib")
            )
        if not user_set_metrics_name:
            args.metrics_name = str(
                hp_get(hp, "metrics-name", "metrics_name", default="metrics.json")
            )

        if not user_set_alpha:
            args.alpha = float(hp_get(hp, "alpha", default=args.alpha))
        if not user_set_n_estimators:
            args.n_estimators = int(
                hp_get(hp, "n-estimators", "n_estimators", default=args.n_estimators)
            )
        if not user_set_learning_rate:
            args.learning_rate = float(
                hp_get(
                    hp,
                    "learning-rate",
                    "learning_rate",
                    default=args.learning_rate,
                )
            )
        if not user_set_max_depth:
            args.max_depth = int(
                hp_get(hp, "max-depth", "max_depth", default=args.max_depth)
            )
        if not user_set_seed:
            args.seed = int(hp_get(hp, "seed", default=args.seed))
        if not user_set_clip_min:
            args.clip_min = float(
                hp_get(hp, "clip-min", "clip_min", default=args.clip_min)
            )
        if not user_set_clip_max:
            args.clip_max = float(
                hp_get(hp, "clip-max", "clip_max", default=args.clip_max)
            )

    return args


@dataclass
class EvalResult:
    rmse: float
    mae: float
    r2: float
    mape: float | None


def _recursive_find(directory: Path, pattern: str) -> list[Path]:
    if not directory.exists():
        return []
    return sorted([p for p in directory.rglob(pattern) if p.is_file()])


def _candidate_dirs(split_dir: Path, split_name: str) -> list[Path]:
    """
    Genera carpetas candidatas para encontrar el split.
    """
    candidates: list[Path] = [split_dir]

    # Si el usuario apuntó a data/processed/train o validation pero no existe,
    # intentamos layout legacy en data/prep.
    candidates.append(LEGACY_PREP_DIR)

    # Extra: si pasa data/processed y no data/processed/train
    if split_name == "train":
        candidates.append(Path("data/processed"))
    else:
        candidates.append(Path("data/processed"))

    # Quitamos duplicados preservando orden
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    return unique_candidates


def resolver_csv(split_dir: Path, expected_name: str, split_name: str) -> Path:
    """
    Busca el CSV esperado dentro del directorio del canal, de forma robusta:
    - exacto en split_dir/expected_name
    - recursivo en split_dir/**/expected_name
    - único CSV recursivo dentro del split_dir
    """
    for candidate_dir in _candidate_dirs(split_dir, split_name):
        exact_path = candidate_dir / expected_name
        if exact_path.exists():
            return exact_path

        exact_recursive = _recursive_find(candidate_dir, expected_name)
        if len(exact_recursive) == 1:
            return exact_recursive[0]

        csvs = _recursive_find(candidate_dir, "*.csv")
        if len(csvs) == 1:
            return csvs[0]

    raise FileNotFoundError(
        f"No encontré {expected_name} en {split_dir} ni en rutas candidatas."
    )


def _load_labeled_csv(
    split_dir: Path,
    expected_name: str,
    split_name: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.Series] | None:
    """
    Intenta cargar layout nuevo:
    - un CSV etiquetado exacto (train.csv / validation.csv)
    - o varios CSVs etiquetados dentro del directorio del split
      (por ejemplo train_00.csv, train_01.csv, ...).
    """
    candidate_dirs = _candidate_dirs(split_dir, split_name)

    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            continue

        exact_path = candidate_dir / expected_name
        if exact_path.exists():
            logger.info("Cargando %s desde %s", split_name, exact_path)
            df = pd.read_csv(exact_path)

            if TARGET_COL not in df.columns:
                logger.info(
                    "Encontré %s pero no contiene target '%s'; probaré layout legacy.",
                    exact_path,
                    TARGET_COL,
                )
                return None

            if df.empty:
                raise ValueError(f"El archivo {exact_path} está vacío.")

            x = df.drop(columns=[TARGET_COL]).copy()
            y = df[TARGET_COL].copy()

            logger.info("%s -> X: %s | y: %s", split_name, x.shape, y.shape)
            return x, y

        csvs = sorted([p for p in candidate_dir.rglob("*.csv") if p.is_file()])
        if not csvs:
            continue

        dfs = []
        all_labeled = True

        for csv_path in csvs:
            df = pd.read_csv(csv_path)

            if TARGET_COL not in df.columns:
                all_labeled = False
                break

            if not df.empty:
                dfs.append(df)

        if all_labeled and dfs:
            logger.info(
                "Cargando %s desde %s archivos CSV en %s",
                split_name,
                len(dfs),
                candidate_dir,
            )
            full_df = pd.concat(dfs, ignore_index=True)

            x = full_df.drop(columns=[TARGET_COL]).copy()
            y = full_df[TARGET_COL].copy()

            logger.info("%s -> X: %s | y: %s", split_name, x.shape, y.shape)
            return x, y

    return None

def _resolve_legacy_paths(split_name: str, split_dir: Path) -> tuple[Path, Path]:
    """
    Soporta layout legacy:
    - data/prep/X_train.csv, y_train.csv
    - data/prep/X_valid.csv, y_valid.csv
    """
    if split_name == "train":
        x_name = "X_train.csv"
        y_name = "y_train.csv"
    elif split_name == "validation":
        x_name = "X_valid.csv"
        y_name = "y_valid.csv"
    else:
        raise ValueError(f"split_name no soportado para layout legacy: {split_name}")

    candidates = _candidate_dirs(split_dir, split_name)

    for candidate_dir in candidates:
        x_path = candidate_dir / x_name
        y_path = candidate_dir / y_name
        if x_path.exists() and y_path.exists():
            return x_path, y_path

        x_recursive = _recursive_find(candidate_dir, x_name)
        y_recursive = _recursive_find(candidate_dir, y_name)
        if len(x_recursive) == 1 and len(y_recursive) == 1:
            return x_recursive[0], y_recursive[0]

    raise FileNotFoundError(
        f"No encontré archivos legacy {x_name}/{y_name} para split={split_name}."
    )


def _load_legacy_split(
    split_dir: Path,
    split_name: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.Series]:
    x_path, y_path = _resolve_legacy_paths(split_name, split_dir)

    logger.info("Cargando %s legacy X desde %s", split_name, x_path)
    logger.info("Cargando %s legacy y desde %s", split_name, y_path)

    x = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)

    if x.empty:
        raise ValueError(f"El archivo {x_path} está vacío.")
    if y_df.empty:
        raise ValueError(f"El archivo {y_path} está vacío.")

    if TARGET_COL in y_df.columns:
        y = y_df[TARGET_COL].copy()
    elif y_df.shape[1] == 1:
        y = y_df.iloc[:, 0].copy()
        y.name = TARGET_COL
    else:
        raise ValueError(
            f"No pude inferir la columna target en {y_path}. "
            f"Columnas encontradas: {list(y_df.columns)}"
        )

    if len(x) != len(y):
        raise ValueError(
            f"Layout legacy inconsistente en {split_name}: "
            f"X tiene {len(x)} filas y y tiene {len(y)}."
        )

    logger.info("%s legacy -> X: %s | y: %s", split_name, x.shape, y.shape)
    return x, y


def cargar_split(
    split_dir: Path,
    expected_name: str,
    split_name: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carga un split desde:
    1) CSV etiquetado (layout nuevo)
    2) X/y separados (layout legacy)
    """
    labeled = _load_labeled_csv(split_dir, expected_name, split_name, logger)
    if labeled is not None:
        return labeled

    logger.info("Intentando layout legacy para %s...", split_name)
    return _load_legacy_split(split_dir, split_name, logger)


def cargar_datos(
    logger: logging.Logger,
    train_dir: Path,
    validation_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    x_train, y_train = cargar_split(train_dir, "train.csv", "train", logger)
    x_valid, y_valid = cargar_split(validation_dir, "validation.csv", "validation", logger)

    if list(x_train.columns) != list(x_valid.columns):
        raise ValueError(
            "Las columnas de train y validation no coinciden. "
            "Revisa el preprocessing."
        )

    logger.info("Número de features: %d", len(x_train.columns))
    return x_train, y_train, x_valid, y_valid


def entrenar_ridge(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 1.0,
) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    return model


def entrenar_gbr(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    random_state: int = SEED,
) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model


def evaluar(
    y_true: pd.Series,
    y_pred: np.ndarray,
    clip_min: float,
    clip_max: float,
    clip: bool = True,
) -> EvalResult:
    y_true_arr = np.asarray(y_true)

    if clip:
        y_pred = np.clip(y_pred, clip_min, clip_max)

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred)))
    mae = float(mean_absolute_error(y_true_arr, y_pred))
    r2 = float(r2_score(y_true_arr, y_pred))

    mask = y_true_arr != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true_arr[mask] - y_pred[mask]) / y_true_arr[mask])))
    else:
        mape = None

    return EvalResult(rmse=rmse, mae=mae, r2=r2, mape=mape)


def guardar_artefactos(
    logger: logging.Logger,
    model: Any,
    metrics: dict[str, Any],
    artifacts_dir: Path,
    model_name: str,
    metrics_name: str,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / model_name
    metrics_path = artifacts_dir / metrics_name

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Modelo guardado en: %s", model_path)
    logger.info("Métricas guardadas en: %s", metrics_path)


def main() -> None:
    args = parse_args()

    train_dir = Path(args.train_dir)
    validation_dir = Path(args.validation_dir)
    artifacts_dir = Path(args.artifacts_dir)

    logger = setup_logger("train")
    start_time = time.time()

    logger.info("Inicio train.py")
    logger.info("train_dir: %s", train_dir)
    logger.info("validation_dir: %s", validation_dir)
    logger.info("artifacts_dir: %s", artifacts_dir)
    logger.info(
        "Hiperparámetros -> alpha=%s, n_estimators=%s, learning_rate=%s, max_depth=%s, seed=%s",
        args.alpha,
        args.n_estimators,
        args.learning_rate,
        args.max_depth,
        args.seed,
    )

    x_train, y_train, x_valid, y_valid = cargar_datos(
        logger=logger,
        train_dir=train_dir,
        validation_dir=validation_dir,
    )

    logger.info("Entrenando baseline Ridge...")
    ridge = entrenar_ridge(x_train, y_train, alpha=args.alpha)
    pred_ridge = ridge.predict(x_valid)
    eval_ridge = evaluar(
        y_true=y_valid,
        y_pred=pred_ridge,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        clip=True,
    )
    logger.info("Ridge RMSE (valid): %.4f", eval_ridge.rmse)

    logger.info("Entrenando GradientBoostingRegressor...")
    gbr_params: dict[str, Any] = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "random_state": args.seed,
    }
    gbr = entrenar_gbr(x_train, y_train, **gbr_params)

    pred_gbr = gbr.predict(x_valid)
    eval_gbr = evaluar(
        y_true=y_valid,
        y_pred=pred_gbr,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        clip=True,
    )

    logger.info("GBR valid RMSE: %.4f", eval_gbr.rmse)
    logger.info("GBR valid MAE : %.4f", eval_gbr.mae)
    logger.info("GBR valid R2  : %.4f", eval_gbr.r2)

    logger.info("Reentrenando modelo final con train + validation...")
    x_all = pd.concat([x_train, x_valid], ignore_index=True)
    y_all = pd.concat([y_train, y_valid], ignore_index=True)
    gbr.fit(x_all, y_all)

    metrics: dict[str, Any] = {
        "baseline_valid": {
            "model": "Ridge",
            **asdict(eval_ridge),
        },
        "gbr_valid_clipped": asdict(eval_gbr),
        "target_col": TARGET_COL,
        "target_clip": [args.clip_min, args.clip_max],
        "feature_columns": list(x_train.columns),
        "n_features": len(x_train.columns),
        "model_name": "GradientBoostingRegressor",
        "model_params": gbr_params,
        "seed": args.seed,
    }

    guardar_artefactos(
        logger=logger,
        model=gbr,
        metrics=metrics,
        artifacts_dir=artifacts_dir,
        model_name=args.model_name,
        metrics_name=args.metrics_name,
    )

    duration = time.time() - start_time
    logger.info("train.py terminado correctamente en %.2f segundos", duration)


if __name__ == "__main__":
    main()