"""
Evaluación del modelo para SageMaker Processing.

Lee:
- modelo desde /opt/ml/processing/input/model/
- test.csv desde /opt/ml/processing/input/test/

Escribe:
- /opt/ml/processing/output/evaluation/evaluation.json

Métrica principal:
- RMSE
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
from pathlib import Path

EXTRA_PATHS = [
    Path("/opt/ml/processing/code"),
    Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) > 1 else None,
    Path.cwd(),
]

for p in EXTRA_PATHS:
    if p is not None and p.exists():
        sys.path.insert(0, str(p))

from src.utils.logging_config import setup_logger

TARGET_COL = "item_cnt_month"
CLIP_MIN, CLIP_MAX = 0, 20

# Local
LOCAL_MODEL_DIR = Path("artifacts")
LOCAL_TEST_DIR = Path("data/processed/test")
LOCAL_EVAL_DIR = Path("artifacts/evaluation")

# SageMaker Processing
SM_MODEL_DIR = Path("/opt/ml/processing/input/model")
SM_TEST_DIR = Path("/opt/ml/processing/input/test")
SM_EVAL_DIR = Path("/opt/ml/processing/output/evaluation")


def running_in_sagemaker() -> bool:
    return Path("/opt/ml/processing").exists()


def default_model_dir() -> Path:
    return SM_MODEL_DIR if running_in_sagemaker() else LOCAL_MODEL_DIR


def default_test_dir() -> Path:
    return SM_TEST_DIR if running_in_sagemaker() else LOCAL_TEST_DIR


def default_evaluation_dir() -> Path:
    return SM_EVAL_DIR if running_in_sagemaker() else LOCAL_EVAL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model on test split.")

    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(default_model_dir()),
        help="Directorio que contiene el modelo entrenado.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=str(default_test_dir()),
        help="Directorio que contiene test.csv.",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=str,
        default=str(default_evaluation_dir()),
        help="Directorio de salida para evaluation.json.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="model.joblib",
        help="Nombre del archivo del modelo.",
    )
    parser.add_argument(
        "--test-name",
        type=str,
        default="test.csv",
        help="Nombre del archivo test.",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=float(CLIP_MIN),
        help="Límite inferior de clipping para predicciones.",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=float(CLIP_MAX),
        help="Límite superior de clipping para predicciones.",
    )

    args, _ = parser.parse_known_args()
    return args


def resolve_file(directory: Path, expected_name: str, suffix: str) -> Path:
    expected = directory / expected_name
    if expected.exists():
        return expected

    files = sorted(directory.glob(f"*{suffix}"))
    if len(files) == 1:
        return files[0]

    raise FileNotFoundError(
        f"No encontré {expected_name} en {directory} y tampoco pude inferir un único archivo {suffix}."
    )


def load_model(model_dir: Path, model_name: str, logger: logging.Logger) -> Any:
    model_path = resolve_file(model_dir, model_name, ".joblib")
    logger.info("Cargando modelo desde %s", model_path)
    return joblib.load(model_path)


def load_test_data(
    test_dir: Path,
    test_name: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.Series]:
    test_path = resolve_file(test_dir, test_name, ".csv")
    logger.info("Cargando test desde %s", test_path)

    df = pd.read_csv(test_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"El archivo {test_path} no contiene la columna target '{TARGET_COL}'.")

    if df.empty:
        raise ValueError(f"El archivo {test_path} está vacío.")

    x_test = df.drop(columns=[TARGET_COL]).copy()
    y_test = df[TARGET_COL].copy()

    logger.info("x_test: %s | y_test: %s", x_test.shape, y_test.shape)
    return x_test, y_test


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    clip_min: float,
    clip_max: float,
) -> dict[str, float | None]:
    y_true_arr = np.asarray(y_true)
    y_pred = np.clip(y_pred, clip_min, clip_max)

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred)))
    mae = float(mean_absolute_error(y_true_arr, y_pred))
    r2 = float(r2_score(y_true_arr, y_pred))

    mask = y_true_arr != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true_arr[mask] - y_pred[mask]) / y_true_arr[mask])))
    else:
        mape = None

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
    }


def write_evaluation(
    metrics: dict[str, float | None],
    evaluation_dir: Path,
    logger: logging.Logger,
) -> None:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    output_path = evaluation_dir / "evaluation.json"

    payload = {
        "regression_metrics": {
            "rmse": {
                "value": metrics["rmse"],
                "standard_deviation": "NaN",
            },
            "mae": {
                "value": metrics["mae"],
                "standard_deviation": "NaN",
            },
            "r2": {
                "value": metrics["r2"],
                "standard_deviation": "NaN",
            },
            "mape": {
                "value": metrics["mape"],
                "standard_deviation": "NaN",
            },
        }
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("evaluation.json guardado en %s", output_path)


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    test_dir = Path(args.test_dir)
    evaluation_dir = Path(args.evaluation_dir)

    logger = setup_logger("evaluate")
    start_time = time.time()

    logger.info("Inicio evaluate.py")
    logger.info("model_dir: %s", model_dir)
    logger.info("test_dir: %s", test_dir)
    logger.info("evaluation_dir: %s", evaluation_dir)

    model = load_model(model_dir, args.model_name, logger)
    x_test, y_test = load_test_data(test_dir, args.test_name, logger)

    logger.info("Generando predicciones...")
    y_pred = model.predict(x_test)

    metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )

    logger.info("RMSE test: %.4f", metrics["rmse"])
    logger.info("MAE test : %.4f", metrics["mae"])
    logger.info("R2 test  : %.4f", metrics["r2"])

    write_evaluation(metrics, evaluation_dir, logger)

    duration = time.time() - start_time
    logger.info("evaluate.py terminado correctamente en %.2f segundos", duration)


if __name__ == "__main__":
    main()