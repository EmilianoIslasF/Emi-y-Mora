import pandas as pd
import src.prep as prep


class DummyLogger:
    def info(self, *args, **kwargs):
        pass


def test_crear_date_features_genera_columnas():
    df = pd.DataFrame({"date": ["01.01.2013", "15.02.2013"]})
    out = prep.crear_date_features(df, DummyLogger())

    assert "year" in out.columns
    assert "month" in out.columns
    assert "date_block_num" in out.columns
    assert out["year"].notna().all()
    assert out["month"].notna().all()


def test_agregar_mensual_target_aplica_clipping():
    # Ya trae date_block_num (como lo deja crear_date_features)
    sales = pd.DataFrame(
        {
            "date_block_num": [0, 0, 0, 1],
            "shop_id": [1, 1, 1, 1],
            "item_id": [10, 10, 10, 10],
            "item_cnt_day": [10, 15, 100, -5],  # suma mes0=125 -> clip a 20
        }
    )

    monthly = prep.agregar_mensual_target(sales, DummyLogger())

    # fila correspondiente a date_block_num=0 debe quedar clippeada a CLIP_MAX
    v0 = monthly.loc[monthly["date_block_num"] == 0, prep.COLUMNA_TARGET].iloc[0]
    assert v0 == prep.CLIP_MAX
