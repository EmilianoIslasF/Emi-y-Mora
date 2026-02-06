"""Rutas del proyecto para mantener paths consistentes y reproducibles."""

from pathlib import Path

# Raíz del repo: .../Emi-y-Mora/
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PREP_DIR = DATA_DIR / "prep"
INFERENCE_DIR = DATA_DIR / "inference"
PREDICTIONS_DIR = DATA_DIR / "predictions"

ARTIFACTS_DIR = ROOT / "artifacts"
