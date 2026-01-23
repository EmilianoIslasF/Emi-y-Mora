Emiliano Islas 
y
Raúl Mora

METODOS DE GRAN ESCALA 



Tarea 01. Predicción de Demanda en Retail con Machine Learning


Flujo de trabajo (3 scripts): 


-----Exploración y entendimiento de datos (eda01.ipynb)
Análisis descriptivo de ventas diarias y mensuales, identificación de alta dispersión en ventas y patrones básicos por tienda, producto y categoría.


-----Transformación y preparación de features (transform_features.ipynb)
Agregación de ventas a nivel mensual
Creación de variables rezagadas (lags de 1, 2, 3, 6 y 12 meses), clipping del target para estabilizar el entrenamiento,construcción de datasets de entrenamiento, 
validación y test.

------Entrenamiento, evaluación y predicción (Entre_eval_prediccion.ipynb)



Entrenamiento de modelos (Ridge y Gradient Boosting), evaluación con métricas RMSE, MAE, R² y MAPE, comparación entre modelos, generación de predicciones finales
para Kaggle (submission.csv) y guardado del modelo final (modelo_final.joblib)

El proyecto utiliza uv para la gestión de dependencias. Para reproducir el entorno:
uv sync

