# Emi-y-Mora — ML Pipeline (Tarea 03)

Este repositorio implementa un pipeline **end-to-end** para un problema de predicción de demanda mensual por **tienda–producto**. A partir de datos crudos (`data/raw/`), el proyecto construye un dataset a nivel mensual, genera *features* (categoría, estacionalidad, lags y agregados) y produce particiones temporales de entrenamiento/validación.

Luego entrena un modelo baseline (Ridge) y un modelo principal (Gradient Boosting Regressor), evalúa con métricas estándar y guarda artefactos reproducibles. Finalmente, corre inferencia en batch para generar el archivo `submission.csv` 


---
## Estructura del repositorio 
.
├── artifacts
│   ├── logs
│   │   ├── inference_20260205_114954.log
│   │   ├── prep_20260205_103330.log
│   │   ├── prep_20260206_180603.log
│   │   └── train_20260205_105322.log
│   ├── 1C_Reporte.pdf
│   ├── metrics.json
│   ├── model.joblib
│   └── modelo_final.joblib
├── assets
│   └── pylint_10of10.png
├── data
│   ├── predictions
│   │   └── submission.csv
│   ├── prep
│   │   ├── X_test.csv
│   │   ├── X_valid.csv
│   │   ├── y_train.csv
│   │   └── y_valid.csv
│   └── raw
│       ├── item_categories.csv
│       ├── items.csv
│       ├── sales_train.csv
│       ├── sample_submission.csv
│       ├── shops.csv
│       └── test.csv
├── notebooks
│   ├── Entre_eval_prediccion.ipynb
│   ├── eda01.ipynb
│   └── transform_fitures.ipynb
├── src
│   ├── utils
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── logging_config.py
│   │   └── paths.py
│   ├── __init__.py
│   ├── inference.py
│   ├── prep.py
│   └── train.py
├── 1C_Reporte.pdf
├── README.md
├── pyproject.toml
└── uv.lock



## Instalación y setup 
Requisitos: 
 Linux / WSL recomendado
 Python 3.x
 uv instalado

```bash
git clone <https://github.com/EmilianoIslasF/Emi-y-Mora.git>
cd Emi-y-Mora
uv sync
```
## Como ejecutar el pipeline 
1) Preparación de datos (prep)

```bash
uv run python -m src.prep
```
2) Entrenamiento (train)

```bash
uv run python -m src.train

```
3) Inferencia / predicción (inference)

```bash
uv run python -m src.inference

```

## Scripts: descripción (inputs/outputs)

src/prep.py
Carga datos crudos, valida, limpia y genera features. Escribe datasets procesados en data/prep/.

src/train.py
Entrena el modelo usando los datasets procesados. Guarda el modelo y métricas en artifacts/.

src/inference.py
Carga el modelo entrenado y produce el archivo final de predicciones (submission.csv) en data/predictions/.


## Métrica y resultados

Métrica: RMSE

RMSE (validación): <0.9655>

Kaggle leaderboard / score: <1.00129>

## Dependencias principales

### Runtime
- pandas
- numpy
- scikit-learn
- joblib

### Desarrollo
- uv
- ipykernel
- ruff
- pylint


## Calidad de código (Pylint)

![Pylint 10/10](assets/pylint_10of10.png)

```



