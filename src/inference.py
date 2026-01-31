from pathlib import Path
import numpy as np
import pandas as pd
import joblib


def main():
    ROOT = Path(__file__).resolve().parents[1]

    inf_dir  = ROOT / "data" / "inference"
    pred_dir = ROOT / "data" / "predictions"
    art_dir  = ROOT / "artifacts"

    pred_dir.mkdir(parents=True, exist_ok=True)

    x_test_path = inf_dir / "X_test.csv"
    model_path  = art_dir / "model.joblib"

    if not x_test_path.exists():
        raise FileNotFoundError(
            f"No existe {x_test_path}. Primero genera X_test con prep.py (debe guardarse en data/inference/)."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe {model_path}. Primero corre: uv run python -m src.train"
        )

    X_test = pd.read_csv(x_test_path)

    
    if "ID" not in X_test.columns:
        raise ValueError("X_test.csv debe incluir columna ID para poder guardar submission.csv (ID, item_cnt_month).")

    ids = X_test["ID"].copy()
    X_feat = X_test.drop(columns=["ID"])

    model = joblib.load(model_path)
    preds = model.predict(X_feat)
    preds = np.clip(preds, 0, 20)

    submission = pd.DataFrame({"ID": ids, "item_cnt_month": preds})
    out_path = pred_dir / "submission.csv"
    submission.to_csv(out_path, index=False)

    print("✅ Inference listo")
    print(f"   Predicciones guardadas en: {out_path}")


if __name__ == "__main__":
    main()
