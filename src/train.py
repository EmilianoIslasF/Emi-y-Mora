from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    ROOT = Path(__file__).resolve().parents[1]

    prep_dir = ROOT / "data" / "prep"
    art_dir  = ROOT / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(prep_dir / "X_train.csv")
    y_train = pd.read_csv(prep_dir / "y_train.csv")["item_cnt_month"]

    X_valid = pd.read_csv(prep_dir / "X_valid.csv")
    y_valid = pd.read_csv(prep_dir / "y_valid.csv")["item_cnt_month"]
    print("Starting training...")
    # 1) Baseline Ridge (como en tu notebook)
    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X_train, y_train)
    pred_ridge = model_ridge.predict(X_valid)
    rmse_ridge = float(np.sqrt(mean_squared_error(y_valid, pred_ridge)))

    # 2) Modelo final: Gradient Boosting (como en tu notebook)
    gbr = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    gbr.fit(X_train, y_train)
    pred_gbr = gbr.predict(X_valid)
    rmse_gbr = float(np.sqrt(mean_squared_error(y_valid, pred_gbr)))

    # métricas extra sobre el mejor (gbr)
    pred_clip = np.clip(pred_gbr, 0, 20)
    rmse = float(np.sqrt(mean_squared_error(y_valid, pred_clip)))
    mae  = float(mean_absolute_error(y_valid, pred_clip))
    r2   = float(r2_score(y_valid, pred_clip))
    mask = y_valid != 0
    mape = float(np.mean(np.abs((y_valid[mask] - pred_clip[mask]) / y_valid[mask]))) if mask.any() else None

    # reentrenar con train + valid (como en tu notebook)
    X_all = pd.concat([X_train, X_valid], ignore_index=True)
    y_all = pd.concat([y_train, y_valid], ignore_index=True)
    gbr.fit(X_all, y_all)

    # guardar modelo
    model_path = art_dir / "model.joblib"
    joblib.dump(gbr, model_path)

    # guardar métricas
    metrics = {
        "rmse_ridge_valid": rmse_ridge,
        "rmse_gbr_valid": rmse_gbr,
        "rmse_valid_clipped": rmse,
        "mae_valid_clipped": mae,
        "r2_valid_clipped": r2,
        "mape_valid_clipped": mape,
        "features": list(X_train.columns),
        "model": "GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)"
    }
    (art_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Train listo")
    print(f"   Modelo guardado en: {model_path}")
    print(f"   RMSE valid (Ridge): {rmse_ridge:.4f}")
    print(f"   RMSE valid (GBR):   {rmse_gbr:.4f}")


if __name__ == "__main__":
    main()
