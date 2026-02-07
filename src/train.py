"""
Entrenamiento y evaluación del modelo (prep -> train).

Qué hace:
1) Carga x_train, y_train, x_valid, y_valid desde data/prep
2) Entrena un baseline (Ridge) y un modelo principal (GradientBoostingRegressor)
3) Evalúa en valid con métricas (RMSE, MAE, R2, MAPE) usando clipping 0..20
4) Re-entrena el mejor modelo con (train + valid)
5) Guarda artefactos en artifacts/: model.joblib y metrics.json
"""

from __future__ import annotations

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


@dataclass
class EvalResult:
    """Contenedor para métricas de evaluación."""

    rmse: float
    mae: float
    r2: float
    mape: float | None


def cargar_datos_prep(
    logger: logging.Logger, prep_dir: Path = PREP_DIR
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Carga los datasets preprocesados: x_train, y_train, x_valid, y_valid."""
    logger.info("Cargando datasets desde data/prep...")
    x_train = pd.read_csv(prep_dir / "X_train.csv")
    y_train = pd.read_csv(prep_dir / "y_train.csv")[TARGET_COL]

    x_valid = pd.read_csv(prep_dir / "X_valid.csv")
    y_valid = pd.read_csv(prep_dir / "y_valid.csv")[TARGET_COL]

    logger.info("x_train: %s | y_train: %s", x_train.shape, y_train.shape)
    logger.info("x_valid: %s | y_valid: %s", x_valid.shape, y_valid.shape)
    return x_train, y_train, x_valid, y_valid


def entrenar_ridge(
    x_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0
) -> Ridge:
    """Entrena baseline Ridge."""
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
    """Entrena Gradient Boosting Regressor."""
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model


def evaluar(y_true: pd.Series, y_pred: np.ndarray, clip: bool = True) -> EvalResult:
    """Calcula métricas. Opcionalmente aplica clipping 0..20."""
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
    artifacts_dir: Path = ARTIFACTS_DIR,
    model_name: str = "model.joblib",
    metrics_name: str = "metrics.json",
) -> None:
    """Guarda el modelo y métricas en artifacts/."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / model_name
    joblib.dump(model, model_path)

    metrics_path = artifacts_dir / metrics_name
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Modelo guardado en: %s", model_path)
    logger.info("Métricas guardadas en: %s", metrics_path)


def main() -> None:
    """Ejecuta el pipeline de entrenamiento y guarda artefactos."""
    logger = setup_logger("train")
    start_time = time.time()
    logger.info("Inicio del script train.py")

    # 1) Cargar datos
    x_train, y_train, x_valid, y_valid = cargar_datos_prep(logger)

    # 2) Baseline Ridge
    logger.info("Entrenando baseline Ridge...")
    ridge = entrenar_ridge(x_train, y_train, alpha=1.0)
    pred_ridge = ridge.predict(x_valid)
    rmse_ridge = float(np.sqrt(mean_squared_error(y_valid, pred_ridge)))
    logger.info("Ridge RMSE (valid, sin clip): %.4f", rmse_ridge)

    # 3) Modelo principal
    logger.info("Entrenando modelo principal (GradientBoostingRegressor)...")
    gbr_params: dict[str, Any] = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 4,
        "random_state": SEED,
    }
    gbr = entrenar_gbr(x_train, y_train, **gbr_params)

    # 4) Evaluación (clipped)
    pred_gbr = gbr.predict(x_valid)
    eval_gbr = evaluar(y_valid, pred_gbr, clip=True)

    logger.info("Métricas GBR (valid, con clip 0..20):")
    logger.info("  RMSE: %.4f", eval_gbr.rmse)
    logger.info("  MAE : %.4f", eval_gbr.mae)
    logger.info("  R2  : %.4f", eval_gbr.r2)
    if eval_gbr.mape is None:
        logger.info("  MAPE: None")
    else:
        logger.info("  MAPE: %.4f", eval_gbr.mape)

    # 5) Reentrenar con train + valid
    logger.info("Reentrenando con train + valid (x_all, y_all)...")
    x_all = pd.concat([x_train, x_valid], ignore_index=True)
    y_all = pd.concat([y_train, y_valid], ignore_index=True)
    gbr.fit(x_all, y_all)
    logger.info("Reentrenado listo. x_all: %s | y_all: %s", x_all.shape, y_all.shape)

    # 6) Guardar artefactos
    metrics: dict[str, Any] = {
        "baseline": {"model": "Ridge", "rmse_valid": rmse_ridge},
        "gbr_valid_clipped": asdict(eval_gbr),
        "target_clip": [CLIP_MIN, CLIP_MAX],
        "features": list(x_train.columns),
        "model_name": "GradientBoostingRegressor",
        "model_params": gbr_params,
        "seed": SEED,
    }
    guardar_artefactos(logger, gbr, metrics)

    duration = time.time() - start_time
    logger.info("train.py terminado correctamente en %.2f segundos", duration)


if __name__ == "__main__":
    main()
