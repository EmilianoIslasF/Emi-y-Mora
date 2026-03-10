# Emi-y-Mora — ML Pipeline (Tarea 04 · MLOps)

Este repositorio implementa un pipeline **end-to-end** para un problema de predicción de demanda mensual por **tienda–producto**. A partir de datos crudos (`data/raw/`), el proyecto construye un dataset a nivel mensual, genera *features* (categoría, estacionalidad, lags y agregados) y produce particiones temporales de entrenamiento/validación e insumos para inferencia.

Luego entrena un modelo baseline (**Ridge**) y un modelo principal (**GradientBoostingRegressor**), evalúa con métricas estándar y guarda artefactos reproducibles. Finalmente, corre inferencia en batch para generar `submission.csv`.

En esta **Tarea 04 (MLOps)** se agrega: estrategia profesional de branching, contenedores Docker por step, ejecución en EC2, ejecución por CLI con argumentos e hiperparámetros, y pruebas unitarias con pytest organizadas por step.

---

## Descripción del proyecto

- **Objetivo:** predecir `item_cnt_month` (ventas mensuales) por `shop_id` × `item_id`.
- **Entrada:** CSVs del dataset en `data/raw/`.
- **Salida final:** `data/predictions/submission.csv`.
- **Clipping del target/predicción:** `[0, 20]`.

---

## Estructura del repositorio

>

## Estructura del repositorio 
.
├── .dockerignore
├── .gitignore
├── .pylintrc
├── 1C_Reporte.pdf
├── README.md
├── artifacts
│   ├── 1C_Reporte.pdf
│   ├── logs
│   │   ├── inference_20260205_114954.log
│   │   ├── prep_20260205_103330.log
│   │   ├── prep_20260206_180603.log
│   │   ├── prep_20260302_192156.log
│   │   ├── prep_20260302_193819.log
│   │   ├── prep_20260302_194009.log
│   │   ├── prep_20260302_194337.log
│   │   ├── prep_20260302_195217.log
│   │   ├── prep_20260303_004525.log
│   │   ├── train_20260205_105322.log
│   │   ├── train_20260302_192349.log
│   │   ├── train_20260302_192419.log
│   │   ├── train_20260302_192444.log
│   │   ├── train_20260302_201232.log
│   │   ├── train_20260302_215404.log
│   │   ├── train_20260302_215907.log
│   │   └── train_20260302_222312.log
│   ├── metrics.json
│   ├── model.joblib
│   └── modelo_final.joblib
├── assets
│   └── pylint_10of10.png
├── conftest.py
├── data
│   ├── inference
│   │   └── X_test.csv
│   ├── predictions
│   │   └── submission.csv
│   ├── prep
│   │   ├── X_test.csv
│   │   ├── X_train.csv
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
├── docker
│   ├── inference
│   │   └── Dockerfile
│   ├── prep
│   │   └── Dockerfile
│   └── train
│       └── Dockerfile
├── notebooks
│   ├── Entre_eval_prediccion.ipynb
│   ├── eda01.ipynb
│   └── transform_fitures.ipynb
├── pyproject.toml
├── repo_tree.txt
├── src
│   ├── __init__.py
│   ├── inference.py
│   ├── inference_step
│   │   └── test
│   │       └── test_inference.py
│   ├── prep.py
│   ├── preprocessing_step
│   │   └── test
│   │       └── test_prep.py
│   ├── train.py
│   ├── training_step
│   │   └── test
│   │       └── test_train.py
│   └── utils
│       ├── __init__.py
│       ├── features.py
│       ├── logging_config.py
│       └── paths.py
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

## Docker por step (imágenes por etapa)

Se dockeriza cada step del pipeline:

prep (preprocessing / feature engineering)

train (training)

inference (batch predictions)

Build de imágenes (en EC2)

Importante: las imágenes se construyen dentro de la instancia EC2.

docker build -t ml-prep:latest  -f docker/prep/Dockerfile .
docker build -t ml-train:latest -f docker/train/Dockerfile .
docker build -t ml-infer:latest -f docker/inference/Dockerfile .
Ejecución del pipeline completo (Docker)

Recomendación: prep y train pueden tardar; se puede usar tmux para evitar perder el proceso si se corta la conexión.

1) Preprocessing (raw → prep + inference)
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  ml-prep:latest \
  --raw-dir data/raw \
  --prep-dir data/prep \
  --inference-dir data/inference \
  --clip-min 0 --clip-max 20 \
  --lags "1,2,3,6,12"

Outputs esperados:

data/prep/X_train.csv, data/prep/y_train.csv, data/prep/X_valid.csv, data/prep/y_valid.csv

data/inference/X_test.csv

2) Training (prep → artifacts)
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  ml-train:latest \
  --prep-dir data/prep \
  --artifacts-dir artifacts \
  --model-name model.joblib \
  --metrics-name metrics.json \
  --alpha 1.0 \
  --n-estimators 100 \
  --learning-rate 0.05 \
  --max-depth 4 \
  --seed 42 \
  --clip-min 0 --clip-max 20

Outputs esperados:

artifacts/model.joblib

artifacts/metrics.json

logs en artifacts/logs/

3) Inference (X_test + model → submission)
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  ml-infer:latest \
  --x-test-path data/inference/X_test.csv \
  --raw-test-path data/raw/test.csv \
  --model-path artifacts/model.joblib \
  --output-path data/predictions/submission.csv \
  --clip-min 0 --clip-max 20

Output esperado:

data/predictions/submission.csv

Ejecución de Contenedores (CLI + argumentos)

Cada step soporta CLI con argparse. Para inspeccionar argumentos:

docker run --rm ml-prep:latest --help
docker run --rm ml-train:latest --help
docker run --rm ml-infer:latest --help

Ejemplo (inference con argumentos):

docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  ml-infer:latest \
  --x-test-path data/inference/X_test.csv \
  --raw-test-path data/raw/test.csv \
  --model-path artifacts/model.joblib \
  --output-path data/predictions/submission.csv \
  --clip-min 0 --clip-max 20

Evidencia requerida (EC2):

Screenshot del docker run ... --help

Screenshot del docker run ... (con logs visibles)

Pruebas Unitarias (pytest)

Las pruebas unitarias viven dentro de cada step:

src/preprocessing_step/test/test_prep.py

src/training_step/test/test_train.py

src/inference_step/test/test_inference.py

Ejecutar tests desde la raíz:

uv run pytest src/ -v

Se incluye conftest.py para asegurar que el paquete src sea importable durante la ejecución de pruebas.




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


## Calidad de código (Pylint)<img width="678" height="253" alt="Captura de pantalla 2026-03-02 130243" src="https://github.com/user-attachments/assets/f94fab97-6fa1-44b2-896a-f8b2cfbd13fb" />

#Screenshots ![Uploading Captura de pantalla 2026-03-02 130243.png…]()

<img width="1110" height="414" alt="Captura de pantalla 2026-03-02 175402" src="https://github.com/user-attachments/assets/f23aee16-2fe8-4023-86c3-8e8667c3aa54" />

<img width="1051" height="289" alt="Captura de pantalla 2026-03-02 175608" src="https://github.com/user-attachments/assets/1f4e1c2c-e822-4871-aae4-c00ffa7e5669" />


![Uploading Captura de pantalla 2026-03-02 130243.png…]()
<img width="858" height="363" alt="Captura de pantalla 2026-03-02 141146" src="https://github.com/user-attachments/assets/af8b4f8f-274b-4dc3-99c0-a0acaee56e08" />

<img width="798" height="442" alt="Captura de pantalla 2026-03-02 184358" src="https://github.com/user-attachments/assets/17e7bed5-a295-40a6-9b6f-f8ce53c3e58a" />


![Pylint 10/10](assets/pylint_10of10.png)<img width="1090" height="381" alt="Captura de pantalla 2026-03-02 191359" src="https://github.com/user-attachments/assets/82eacc48-57f6-4873-bd10-429579bee4a3" />
#Tarea 5

<img width="921" height="250" alt="image" src="https://github.com/user-attachments/assets/83b01ed0-37f2-4f7c-a006-53861a062ec7" />
<img width="921" height="166" alt="image" src="https://github.com/user-attachments/assets/30bfd32f-9552-412e-809d-685208b882bd" />
<img width="921" height="479" alt="image" src="https://github.com/user-attachments/assets/a1a13ed0-b216-4b0c-9a53-3a4d537d2213" />
<img width="921" height="409" alt="image" src="https://github.com/user-attachments/assets/7f21cf42-a0b0-495a-a2a9-8af428c57602" />
<img width="921" height="382" alt="image" src="https://github.com/user-attachments/assets/87438a80-7d0e-47f3-9c3e-8a39a6691d1f" />







---


