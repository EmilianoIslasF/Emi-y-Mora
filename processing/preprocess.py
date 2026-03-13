from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COL = "item_cnt_month"
CLIP_MIN, CLIP_MAX = 0, 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    p.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    p.add_argument("--lags", type=str, default="1,2,3")
    return p.parse_args()


def parse_lags(lags_str: str) -> list[int]:
    parts = [x.strip() for x in lags_str.replace(",", " ").split()]
    return [int(x) for x in parts if x]


def agregar_target_lags_ligeros(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    df = df.sort_values(["shop_id", "item_id", "date_block_num"]).copy()
    grp = df.groupby(["shop_id", "item_id"], sort=False)[TARGET_COL]

    for lag in lags:
        col = f"{TARGET_COL}_lag_{lag}"
        df[col] = grp.shift(lag).fillna(0).astype(np.float32)

    return df


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lags = parse_lags(args.lags)

    sales = pd.read_csv(input_dir / "sales_train.csv")
    items = pd.read_csv(input_dir / "items.csv")
    test = pd.read_csv(input_dir / "test.csv")

    sales["date"] = pd.to_datetime(sales["date"], format="%d.%m.%Y")
    sales["date_block_num"] = sales["date"].dt.to_period("M").factorize()[0].astype(np.int8)

    monthly = sales.groupby(
        ["date_block_num", "shop_id", "item_id"], as_index=False
    ).agg(item_cnt_month=("item_cnt_day", "sum"))

    monthly[TARGET_COL] = monthly[TARGET_COL].clip(CLIP_MIN, CLIP_MAX).astype(np.float32)
    monthly["date_block_num"] = monthly["date_block_num"].astype(np.int8)
    monthly["shop_id"] = monthly["shop_id"].astype(np.int16)
    monthly["item_id"] = monthly["item_id"].astype(np.int32)

    grid = []
    blocks = monthly["date_block_num"].unique()

    for block in blocks:
        cur = monthly[monthly["date_block_num"] == block]
        shops_in_month = cur["shop_id"].unique()
        items_in_month = cur["item_id"].unique()

        block_df = pd.DataFrame(
            [(block, s, i) for s in shops_in_month for i in items_in_month],
            columns=["date_block_num", "shop_id", "item_id"],
        )
        block_df["date_block_num"] = block_df["date_block_num"].astype(np.int8)
        block_df["shop_id"] = block_df["shop_id"].astype(np.int16)
        block_df["item_id"] = block_df["item_id"].astype(np.int32)
        grid.append(block_df)

    matrix = pd.concat(grid, ignore_index=True)
    matrix = matrix.merge(monthly, on=["date_block_num", "shop_id", "item_id"], how="left")
    matrix[TARGET_COL] = matrix[TARGET_COL].fillna(0).astype(np.float32)

    items_small = items[["item_id", "item_category_id"]].copy()
    items_small["item_id"] = items_small["item_id"].astype(np.int32)
    items_small["item_category_id"] = items_small["item_category_id"].astype(np.int16)

    matrix = matrix.merge(items_small, on="item_id", how="left")
    matrix["item_category_id"] = matrix["item_category_id"].fillna(-1).astype(np.int16)

    matrix["month"] = (matrix["date_block_num"] % 12).astype(np.int8)
    matrix["year"] = (matrix["date_block_num"] // 12).astype(np.int8)

    test_matrix = test.copy()
    test_matrix["date_block_num"] = 34
    test_matrix = test_matrix.merge(items_small, on="item_id", how="left")
    test_matrix["shop_id"] = test_matrix["shop_id"].astype(np.int16)
    test_matrix["item_id"] = test_matrix["item_id"].astype(np.int32)
    test_matrix["item_category_id"] = test_matrix["item_category_id"].fillna(-1).astype(np.int16)
    test_matrix["date_block_num"] = test_matrix["date_block_num"].astype(np.int8)
    test_matrix["month"] = (test_matrix["date_block_num"] % 12).astype(np.int8)
    test_matrix["year"] = (test_matrix["date_block_num"] // 12).astype(np.int8)
    test_matrix[TARGET_COL] = np.float32(0)

    common_cols = [
        "date_block_num",
        "shop_id",
        "item_id",
        "item_category_id",
        "month",
        "year",
        TARGET_COL,
    ]

    all_data = pd.concat([matrix[common_cols], test_matrix[common_cols]], ignore_index=True)
    all_data = agregar_target_lags_ligeros(all_data, lags)

    train_data = all_data[all_data["date_block_num"] <= 33].copy()
    test_data = all_data[all_data["date_block_num"] == 34].copy()

    x_train = train_data[train_data["date_block_num"] < 33].drop(columns=[TARGET_COL])
    y_train = train_data[train_data["date_block_num"] < 33][TARGET_COL]

    x_valid = train_data[train_data["date_block_num"] == 33].drop(columns=[TARGET_COL])
    y_valid = train_data[train_data["date_block_num"] == 33][TARGET_COL]

    x_test = test_data.drop(columns=[TARGET_COL])

    x_train.to_csv(output_dir / "X_train.csv", index=False)
    y_train.to_frame(TARGET_COL).to_csv(output_dir / "y_train.csv", index=False)
    x_valid.to_csv(output_dir / "X_valid.csv", index=False)
    y_valid.to_frame(TARGET_COL).to_csv(output_dir / "y_valid.csv", index=False)


if __name__ == "__main__":
    main()
