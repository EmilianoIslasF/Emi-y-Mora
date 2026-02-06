# src/train.py
"""
Entrenamiento y evaluación del modelo (prep -> train).

Qué hace:
1) Carga X_train, y_train, X_valid, y_valid desde data/prep
2) Entrena un baseline (Ridge) y un modelo principal (GradientBoostingRegressor)
3) Evalúa en valid con métricas (RMSE, MAE, R2, MAPE) usando clipping 0..20
4) Re-entrena el mejor modelo con (train + valid)
5) Guarda artefactos en artifacts/: model.joblib y metrics.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.logging_config import setup_logger
from src.utils.paths import PREP_DIR, ARTIFACTS_DIR


TARGET_COL = "item_cnt_month"
CLIP_MIN, CLIP_MAX = 0, 20
SEED = 42


@dataclass
class EvalResult:
    """Contenedor simple para guardar métricas de evaluación."""
    rmse: float
    mae: float
    r2: float
    mape: float | None


def cargar_datos_prep(
    logger,
    prep_dir: Path = PREP_DIR
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Carga los datasets preprocesados: X_train, y_train, X_valid, y_valid."""
    logger.info("Cargando datasets desde data/prep...")
    X_train = pd.read_csv(prep_dir / "X_train.csv")
    y_train = pd.read_csv(prep_dir / "y_train.csv")[TARGET_COL]

    X_valid = pd.read_csv(prep_dir / "X_valid.csv")
    y_valid = pd.read_csv(prep_dir / "y_valid.csv")[TARGET_COL]

    logger.info(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    logger.info(f"X_valid: {X_valid.shape} | y_valid: {y_valid.shape}")
    return X_train, y_train, X_valid, y_valid


def entrenar_ridge(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Ridge:
    """Entrena baseline Ridge."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def entrenar_gbr(
    X_train: pd.DataFrame,
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
    model.fit(X_train, y_train)
    return model


def evaluar(y_true: pd.Series, y_pred: np.ndarray, clip: bool = True) -> EvalResult:
    """Calcula métricas en valid. Opcionalmente aplica clipping 0..20."""
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
    logger,
    model,
    metrics: dict,
    artifacts_dir: Path = ARTIFACTS_DIR,
    model_name: str = "model.joblib",
    metrics_name: str = "metrics.json",
) -> None:
    """Guarda el modelo y métricas en artifacts/."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / model_name
    joblib.dump(model, model_path)
    (artifacts_dir / metrics_name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info(f"Modelo guardado en: {model_path}")
    logger.info(f"Métricas guardadas en: {artifacts_dir / metrics_name}")


def main() -> None:
    logger = setup_logger("train")
    start_time = time.time()
    logger.info("Inicio del script train.py")

    # 1) Cargar datos ya preparados
    X_train, y_train, X_valid, y_valid = cargar_datos_prep(logger)

    # 2) Baseline Ridge
    logger.info("Entrenando baseline Ridge...")
    ridge = entrenar_ridge(X_train, y_train, alpha=1.0)
    pred_ridge = ridge.predict(X_valid)
    rmse_ridge = float(np.sqrt(mean_squared_error(y_valid, pred_ridge)))
    logger.info(f"Ridge RMSE (valid, sin clip): {rmse_ridge:.4f}")

    # 3) Modelo principal (GBR)
    logger.info("Entrenando modelo principal (GradientBoostingRegressor)...")
    gbr_params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 4,
        "random_state": SEED,
    }
    gbr = entrenar_gbr(X_train, y_train, **gbr_params)

    # 4) Evaluación (clipped)
    pred_gbr = gbr.predict(X_valid)
    eval_gbr = evaluar(y_valid, pred_gbr, clip=True)

    logger.info("Métricas GBR (valid, con clip 0..20):")
    logger.info(f"  RMSE: {eval_gbr.rmse:.4f}")
    logger.info(f"  MAE : {eval_gbr.mae:.4f}")
    logger.info(f"  R2  : {eval_gbr.r2:.4f}")
    logger.info(f"  MAPE: {eval_gbr.mape:.4f}" if eval_gbr.mape is not None else "  MAPE: None")

    # 5) Reentrenar con train + valid
    logger.info("Reentrenando con train + valid (X_all, y_all)...")
    X_all = pd.concat([X_train, X_valid], ignore_index=True)
    y_all = pd.concat([y_train, y_valid], ignore_index=True)
    gbr.fit(X_all, y_all)
    logger.info(f"Reentrenado listo. X_all: {X_all.shape} | y_all: {y_all.shape}")

    # 6) Guardar artefactos
    metrics = {
        "rmse_ridge_valid": rmse_ridge,
        "rmse_gbr_valid_clipped": eval_gbr.rmse,
        "mae_gbr_valid_clipped": eval_gbr.mae,
        "r2_gbr_valid_clipped": eval_gbr.r2,
        "mape_gbr_valid_clipped": eval_gbr.mape,
        "target_clip": [CLIP_MIN, CLIP_MAX],
        "features": list(X_train.columns),
        "model_name": "GradientBoostingRegressor",
        "model_params": gbr_params,
        "seed": SEED,
    }
    guardar_artefactos(logger, gbr, metrics)

    duration = time.time() - start_time
    logger.info(f"train.py terminado correctamente en {duration:.2f} segundos")


if __name__ == "__main__":
    main()
