import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.max_columns", 100)

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent  

RAW = ROOT / "data" / "raw"

sales = pd.read_csv(RAW / "sales_train.csv")
items = pd.read_csv(RAW / "items.csv")
shops = pd.read_csv(RAW / "shops.csv")
item_categories = pd.read_csv(RAW / "item_categories.csv")
test = pd.read_csv(RAW / "test.csv")
sales["date"] = pd.to_datetime(sales["date"], format="%d.%m.%Y")
sales["year"] = sales["date"].dt.year
sales["month"] = sales["date"].dt.month

# indice mensual 34 meses 
sales["date_block_num"] = sales["date"].dt.to_period("M").factorize()[0]
#Convertimos la fecha a fecha, y el respectivo mes comenzando la cuenta en cero como se nos pidió en el ejercicio 
#Ahora vamos a pasar de cantidad de producto diaro y cantidad de producto mensual 
monthly = (
    sales
    .groupby(["date_block_num", "shop_id", "item_id"], as_index=False)
    .agg(item_cnt_month=("item_cnt_day", "sum"))
)

# también haremos un clipping
monthly["item_cnt_month"] = monthly["item_cnt_month"].clip(0, 20)

grid = []
for block in monthly["date_block_num"].unique():
    cur = monthly[monthly["date_block_num"] == block]
    shops_in_month = cur["shop_id"].unique()
    items_in_month = cur["item_id"].unique()
    grid.append(
        pd.DataFrame(
            [(block, s, i) for s in shops_in_month for i in items_in_month],
            columns=["date_block_num", "shop_id", "item_id"]
        )
    )

matrix = pd.concat(grid, ignore_index=True)

matrix = matrix.merge(monthly, on=["date_block_num", "shop_id", "item_id"], how="left")
matrix["item_cnt_month"] = matrix["item_cnt_month"].fillna(0)
#si un producto no aparece en un mes, en realidad vendió 0, pero no hay fila.
#Esto crea filas para poder aprender 0 ventas también, esto para completar el modelo y que sea preciso 
items_small = items[["item_id", "item_category_id"]]
matrix = matrix.merge(items_small, on="item_id", how="left")
#Ahora le agregamos la columna de caterogría del producto en cuestión , es para ayudar al modelo a reconocer que los productos de esta categoría se comportan así

matrix["month"] = (matrix["date_block_num"] % 12).astype(np.int8)
matrix["year"] = (matrix["date_block_num"] // 12).astype(np.int8)
#Nuestro modelo no sabe qué mes del año es ni en qué año va, solo ve un número creciente, por eso le calculamos el residuo para que esos meses 
#se parezcan entre sí. También calcula el año relativo (primer segundo año y así)

def add_lag(df, lags, col):
    tmp = df[["date_block_num", "shop_id", "item_id", col]]
    for lag in lags:
        shifted = tmp.copy()
        shifted["date_block_num"] += lag
        shifted = shifted.rename(columns={col: f"{col}_lag_{lag}"})
        df = df.merge(shifted, on=["date_block_num", "shop_id", "item_id"], how="left")
    return df

matrix = add_lag(matrix, lags=[1,2,3,6,12], col="item_cnt_month")

lag_cols = [c for c in matrix.columns if "lag_" in c]
matrix[lag_cols] = matrix[lag_cols].fillna(0)
#Le hacemos recordar al modelo las ventas anteriores en otra columna(3 es trimestral, 6 es ventas de hace medio año)
matrix.head()
shop_month = (
    matrix.groupby(["date_block_num", "shop_id"], as_index=False)
    .agg(shop_cnt_month=("item_cnt_month", "mean"))
)
matrix = matrix.merge(shop_month, on=["date_block_num", "shop_id"], how="left")
matrix = add_lag(matrix, [1,2,3,6,12], "shop_cnt_month")
matrix.drop(columns=["shop_cnt_month"], inplace=True)
matrix.head()

item_month = (
    matrix.groupby(["date_block_num", "item_id"], as_index=False)
    .agg(item_cnt_month_mean=("item_cnt_month", "mean"))
)
matrix = matrix.merge(item_month, on=["date_block_num", "item_id"], how="left")
matrix = add_lag(matrix, [1,2,3,6,12], "item_cnt_month_mean")
matrix.drop(columns=["item_cnt_month_mean"], inplace=True)

matrix = matrix.fillna(0)
matrix.head()

test["date_block_num"] = 34
test_matrix = test.merge(items_small, on="item_id", how="left")

test_matrix["month"] = (test_matrix["date_block_num"] % 12).astype(np.int8)
test_matrix["year"] = (test_matrix["date_block_num"] // 12).astype(np.int8)

# placeholder del target
test_matrix["item_cnt_month"] = 0
test_matrix.head()

common_cols = ["date_block_num","shop_id","item_id","item_category_id","month","year","item_cnt_month"]
all_data = pd.concat([matrix[common_cols], test_matrix[common_cols]], ignore_index=True)

all_data = add_lag(all_data, [1,2,3,6,12], "item_cnt_month")
lag_cols = [c for c in all_data.columns if "lag_" in c]
all_data[lag_cols] = all_data[lag_cols].fillna(0)

train_data = all_data[all_data["date_block_num"] <= 33].copy()
test_data  = all_data[all_data["date_block_num"] == 34].copy()

#Ahora hacemos el split
X_train = train_data[train_data["date_block_num"] < 33].drop(columns=["item_cnt_month"])
y_train = train_data[train_data["date_block_num"] < 33]["item_cnt_month"]

X_valid = train_data[train_data["date_block_num"] == 33].drop(columns=["item_cnt_month"])
y_valid = train_data[train_data["date_block_num"] == 33]["item_cnt_month"]

X_test  = test_data.drop(columns=["item_cnt_month"])
#Este split preveien que usemos fechas del futuro con fechas del pasado. 

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

prep_dir = ROOT / "data" / "prep"
inf_dir  = ROOT / "data" / "inference"

prep_dir.mkdir(parents=True, exist_ok=True)
inf_dir.mkdir(parents=True, exist_ok=True)

X_train.to_csv(prep_dir / "X_train.csv", index=False)
y_train.to_frame("item_cnt_month").to_csv(prep_dir / "y_train.csv", index=False)

X_valid.to_csv(prep_dir / "X_valid.csv", index=False)
y_valid.to_frame("item_cnt_month").to_csv(prep_dir / "y_valid.csv", index=False)

# Según entregable: X_test para batch inference
X_test.to_csv(inf_dir / "X_test.csv", index=False)

print("Listo: CSV creados en data/prep/ y data/inference/")
