"""
Inferencia / predicción en batch.

Qué hace:
1) Carga X_test (features) desde data/inference
2) Carga IDs desde data/raw/test.csv
3) Carga el modelo entrenado desde artifacts
4) Genera predicciones (con clipping)
5) Guarda submission.csv en data/predictions
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.utils.paths import RAW_DIR, INFERENCE_DIR, PREDICTIONS_DIR, ARTIFACTS_DIR


# -------------------------------
# Logging básico
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


CLIP_MIN, CLIP_MAX = 0, 20


def main() -> None:
    logger.info("Inicio del script inference.py")

    # Crear carpeta de predicciones
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    x_test_path = INFERENCE_DIR / "X_test.csv"
    raw_test_path = RAW_DIR / "test.csv"
    model_path = ARTIFACTS_DIR / "model.joblib"

    # -------------------------------
    # Validaciones
    # -------------------------------
    if not x_test_path.exists():
        raise FileNotFoundError(f"No existe {x_test_path}")

    if not raw_test_path.exists():
        raise FileNotFoundError(f"No existe {raw_test_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"No existe {model_path}")

    # -------------------------------
    # Cargar datos
    # -------------------------------
    logger.info(f"Cargando X_test desde {x_test_path}")
    X_test = pd.read_csv(x_test_path)
    logger.info(f"X_test shape: {X_test.shape}")

    logger.info(f"Cargando IDs desde {raw_test_path}")
    raw_test = pd.read_csv(raw_test_path)

    if "ID" not in raw_test.columns:
        raise ValueError("test.csv debe incluir columna 'ID'")

    ids = raw_test["ID"].copy()

    # -------------------------------
    # Cargar modelo
    # -------------------------------
    logger.info(f"Cargando modelo desde {model_path}")
    model = joblib.load(model_path)

    # -------------------------------
    # Predicción
    # -------------------------------
    logger.info("Generando predicciones")
    preds = model.predict(X_test)
    preds = np.clip(preds, CLIP_MIN, CLIP_MAX)

    # -------------------------------
    # Crear submission
    # -------------------------------
    submission = pd.DataFrame({
        "ID": ids,
        "item_cnt_month": preds
    })

    out_path = PREDICTIONS_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)

    logger.info("Inference finalizado correctamente")
    logger.info(f"Submission guardado en: {out_path}")
    logger.info(f"Total filas: {len(submission)}")


if __name__ == "__main__":
    main()

