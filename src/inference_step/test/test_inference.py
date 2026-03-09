import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

import src.inference as inf


def test_inference_genera_submission(tmp_path, monkeypatch):
    # Inputs fake
    x_test = pd.DataFrame({"a": [1, 2, 3]})
    raw_test = pd.DataFrame({"ID": [0, 1, 2]})

    x_path = tmp_path / "X_test.csv"
    raw_path = tmp_path / "test.csv"
    model_path = tmp_path / "model.joblib"
    out_path = tmp_path / "submission.csv"

    x_test.to_csv(x_path, index=False)
    raw_test.to_csv(raw_path, index=False)

    # Modelo dummy (predice 1.0 siempre)
    m = DummyRegressor(strategy="constant", constant=1.0)
    m.fit(x_test, np.zeros(len(x_test)))
    joblib.dump(m, model_path)

    # Simula CLI args para tu parse_args()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference.py",
            "--x-test-path", str(x_path),
            "--raw-test-path", str(raw_path),
            "--model-path", str(model_path),
            "--output-path", str(out_path),
            "--clip-min", "0",
            "--clip-max", "20",
        ],
    )

    inf.main()

    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert list(df.columns) == ["ID", "item_cnt_month"]
    assert len(df) == 3
