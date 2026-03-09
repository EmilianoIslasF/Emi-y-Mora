"""
Entrenamiento y evaluación del modelo (prep -> train).
Compatible con ejecución local y con Amazon SageMaker BYOC.
"""

from __future__ import annotations

import argparse
import json
import logging
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
from src.utils.paths import ARTIFACTS_DIR, PREP_DIR

TARGET_COL = "item_cnt_month"
CLIP_MIN, CLIP_MAX = 0, 20
SEED = 42

# Rutas estándar del contrato SageMaker
SM_INPUT_CONFIG = Path("/opt/ml/input/config/hyperparameters.json")
SM_INPUT_TRAIN = Path("/opt/ml/input/data/train")
SM_MODEL_DIR = Path("/opt/ml/model")


def running_in_sagemaker() -> bool:
    return SM_INPUT_TRAIN.exists() or SM_INPUT_CONFIG.exists()


def load_sagemaker_hyperparameters() -> dict[str, Any]:
    if not SM_INPUT_CONFIG.exists():
        return {}

    with open(SM_INPUT_CONFIG, "r", encoding="utf-8") as f:
        hp = json.load(f)

    # SageMaker suele mandar todo como string
    return hp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training step (prep -> model).")

    # Defaults locales
    p.add_argument("--prep-dir", type=str, default=str(PREP_DIR))
    p.add_argument("--artifacts-dir", type=str, default=str(ARTIFACTS_DIR))
    p.add_argument("--model-name", type=str, default="model.joblib")
    p.add_argument("--metrics-name", type=str, default="metrics.json")

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=SEED)

    p.add_argument("--clip-min", type=float, default=float(CLIP_MIN))
    p.add_argument("--clip-max", type=float, default=float(CLIP_MAX))

    # Ignora args extra tipo "train"
    args, _ = p.parse_known_args()

    # Si corre en SageMaker, sobreescribe con contrato SageMaker
    if running_in_sagemaker():
        hp = load_sagemaker_hyperparameters()

        # Datos: si subiste data/prep al canal train, los csv quedan directo en /opt/ml/input/data/train
        args.prep_dir = hp.get("prep-dir", str(SM_INPUT_TRAIN))
        args.artifacts_dir = str(SM_MODEL_DIR)

        if "model-name" in hp:
            args.model_name = hp["model-name"]
        if "metrics-name" in hp:
            args.metrics_name = hp["metrics-name"]

        if "alpha" in hp:
            args.alpha = float(hp["alpha"])
        if "n-estimators" in hp:
            args.n_estimators = int(hp["n-estimators"])
        if "learning-rate" in hp:
            args.learning_rate = float(hp["learning-rate"])
        if "max-depth" in hp:
            args.max_depth = int(hp["max-depth"])
        if "seed" in hp:
            args.seed = int(hp["seed"])
        if "clip-min" in hp:
            args.clip_min = float(hp["clip-min"])
        if "clip-max" in hp:
            args.clip_max = float(hp["clip-max"])

    return args


@dataclass
class EvalResult:
    rmse: float
    mae: float
    r2: float
    mape: float | None


def cargar_datos_prep(
    logger: logging.Logger,
    prep_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    logger.info("Cargando datasets desde %s", prep_dir)

    x_train = pd.read_csv(prep_dir / "X_train.csv")
    y_train = pd.read_csv(prep_dir / "y_train.csv")[TARGET_COL]

    x_valid = pd.read_csv(prep_dir / "X_valid.csv")
    y_valid = pd.read_csv(prep_dir / "y_valid.csv")[TARGET_COL]

    logger.info("x_train: %s | y_train: %s", x_train.shape, y_train.shape)
    logger.info("x_valid: %s | y_valid: %s", x_valid.shape, y_valid.shape)

    return x_train, y_train, x_valid, y_valid


def entrenar_ridge(x_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Ridge:
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


def evaluar(y_true: pd.Series, y_pred: np.ndarray, clip: bool = True) -> EvalResult:
    if clip:
        y_pred = np.clip(y_pred, CLIP_MIN, CLIP_MAX)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    mask = y_true != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
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
    joblib.dump(model, model_path)

    metrics_path = artifacts_dir / metrics_name
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Modelo guardado en: %s", model_path)
    logger.info("Métricas guardadas en: %s", metrics_path)


def main() -> None:
    args = parse_args()

    global CLIP_MIN, CLIP_MAX
    CLIP_MIN = args.clip_min
    CLIP_MAX = args.clip_max

    prep_dir = Path(args.prep_dir)
    artifacts_dir = Path(args.artifacts_dir)

    logger = setup_logger("train")
    start_time = time.time()

    logger.info("Inicio train.py")
    logger.info("prep_dir: %s", prep_dir)
    logger.info("artifacts_dir: %s", artifacts_dir)

    x_train, y_train, x_valid, y_valid = cargar_datos_prep(logger, prep_dir)

    logger.info("Entrenando baseline Ridge")
    ridge = entrenar_ridge(x_train, y_train, alpha=args.alpha)
    pred_ridge = ridge.predict(x_valid)
    rmse_ridge = float(np.sqrt(mean_squared_error(y_valid, pred_ridge)))
    logger.info("Ridge RMSE (valid): %.4f", rmse_ridge)

    logger.info("Entrenando GradientBoostingRegressor")
    gbr_params: dict[str, Any] = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "random_state": args.seed,
    }
    gbr = entrenar_gbr(x_train, y_train, **gbr_params)

    pred_gbr = gbr.predict(x_valid)
    eval_gbr = evaluar(y_valid, pred_gbr, clip=True)

    logger.info("RMSE: %.4f", eval_gbr.rmse)
    logger.info("MAE : %.4f", eval_gbr.mae)
    logger.info("R2  : %.4f", eval_gbr.r2)

    logger.info("Reentrenando con train + valid")
    x_all = pd.concat([x_train, x_valid], ignore_index=True)
    y_all = pd.concat([y_train, y_valid], ignore_index=True)
    gbr.fit(x_all, y_all)

    metrics: dict[str, Any] = {
        "baseline": {"model": "Ridge", "rmse_valid": rmse_ridge},
        "gbr_valid_clipped": asdict(eval_gbr),
        "target_clip": [CLIP_MIN, CLIP_MAX],
        "features": list(x_train.columns),
        "model_name": "GradientBoostingRegressor",
        "model_params": gbr_params,
        "seed": args.seed,
    }

    guardar_artefactos(
        logger,
        gbr,
        metrics,
        artifacts_dir,
        args.model_name,
        args.metrics_name,
    )

    duration = time.time() - start_time
    logger.info("Entrenamiento terminado en %.2f segundos", duration)


if __name__ == "__main__":
    main()
