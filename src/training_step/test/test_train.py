import pandas as pd
import numpy as np
import pytest
import src.train as tr


class DummyLogger:
    def info(self, *args, **kwargs):
        pass


def test_evaluar_prediccion_perfecta():
    y_true = pd.Series([0, 1, 2, 3], name=tr.TARGET_COL)
    y_pred = np.array([0, 1, 2, 3], dtype=float)

    res = tr.evaluar(y_true, y_pred, clip=True)

    assert res.rmse == pytest.approx(0.0)
    assert res.mae == pytest.approx(0.0)
    assert res.r2 == pytest.approx(1.0)
    # mape ignora y_true==0, para el resto debe ser 0
    assert res.mape == pytest.approx(0.0)


def test_cargar_datos_prep_lee_archivos(tmp_path):
    # CSVs mínimos con headers correctos
    (tmp_path / "X_train.csv").write_text("a,b\n1,2\n3,4\n")
    (tmp_path / "y_train.csv").write_text(f"{tr.TARGET_COL}\n0.1\n0.2\n")
    (tmp_path / "X_valid.csv").write_text("a,b\n5,6\n")
    (tmp_path / "y_valid.csv").write_text(f"{tr.TARGET_COL}\n0.3\n")

    x_train, y_train, x_valid, y_valid = tr.cargar_datos_prep(DummyLogger(), prep_dir=tmp_path)

    assert x_train.shape == (2, 2)
    assert y_train.shape[0] == 2
    assert x_valid.shape == (1, 2)
    assert y_valid.shape[0] == 1
