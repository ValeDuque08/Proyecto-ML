Proyectos de Análisis Predictivo
Este repositorio contiene dos proyectos de machine learning desarrollados en Jupyter Notebook, cada uno abordando un problema de negocio distinto: uno de clasificación y otro de regresión.
Contenido
Notebook	Problema	Tipo	Dataset
`P_A_Bank_Marketing.ipynb`	Predecir si un cliente suscribirá un depósito a plazo fijo	Clasificación binaria	Bank Marketing (OpenML ID 222)
`P_A_Diamonds.ipynb`	Predecir el precio de un diamante	Regresión	Diamonds (OpenML ID 42225)
---
1. Bank Marketing — Clasificación
Contexto: campañas de telemercadeo de una institución bancaria portuguesa, en las que se ofrece a los clientes contratar un depósito a plazo fijo. El objetivo es anticipar qué clientes tienen mayor probabilidad de aceptar, para priorizar los esfuerzos del call center.
Dataset: 45.211 registros, 16 variables predictoras (demográficas, financieras y de la campaña) y 1 variable objetivo (`y`: suscribió / no suscribió). Presenta un desbalance de clases considerable (11.7% "yes" vs. 88.3% "no").
Pasos principales del notebook:
Carga del dataset con `ucimlrepo`.
Análisis exploratorio (EDA): estadística descriptiva, valores nulos, duplicados, distribución de variables numéricas y categóricas, correlaciones.
Tratamiento de valores faltantes (imputación por moda para `job`/`education`; nueva categoría explícita para `contact`/`poutcome`).
Exclusión de la variable `duration` por fuga de información (data leakage).
Preprocesamiento con `Pipeline` (`StandardScaler` + `OneHotEncoder`) y regresión logística.
Validación cruzada estratificada (10 folds) y búsqueda de hiperparámetros con `GridSearchCV`.
Evaluación final con precisión, recall, F1-score y matriz de confusión sobre el conjunto de prueba.
Resultado: el modelo optimizado alcanza precision = 0.267, recall = 0.624 y F1 = 0.374 para la clase "yes", identificando al 62% de los clientes que sí suscriben el depósito.
2. Diamonds — Regresión
Contexto: el mercado de diamantes no cuenta con un precio de referencia estandarizado, por lo que se busca un modelo que estime el precio de venta a partir de las características físicas y de calidad de la piedra (motores de fijación de precios, tasaciones y avalúos automatizados).
Dataset: cerca de 54.000 diamantes, con variables numéricas (`carat`, `depth`, `table`, `x`, `y`, `z`) y categóricas ordinales (`cut`, `color`, `clarity`), y `price` como variable objetivo.
Pasos principales del notebook:
Carga del dataset con la librería `openml`.
EDA: estadística descriptiva, valores nulos, duplicados, distribuciones, outliers y correlaciones.
Limpieza de datos: eliminación de 146 filas duplicadas y de 19 registros con dimensiones físicas en 0 (errores de medición).
Preprocesamiento con `Pipeline` (`StandardScaler` + `OrdinalEncoder` respetando el orden de calidad de cada variable).
Entrenamiento con `SGDRegressor`, validación cruzada de 10 folds (`KFold`).
Optimización de hiperparámetros (`penalty`, `alpha`, `eta0`) con `GridSearchCV`.
Resultado: el modelo optimizado (penalización L2, `alpha`=0.0001, `eta0`=0.001) reduce el ECM de 1.500.746,59 a 1.490.294,09, con un RMSE promedio de aproximadamente $1.220 por diamante.
---
Requisitos
Ambos notebooks usan Python con las siguientes librerías principales:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
ucimlrepo      # solo para P_A_Bank_Marketing.ipynb
openml         # solo para P_A_Diamonds.ipynb
```
Cómo ejecutar
Clonar el repositorio.
Instalar las dependencias (`pip install -r requirements.txt` si se agrega uno, o instalarlas manualmente).
Abrir el notebook deseado con Jupyter Notebook / JupyterLab / Google Colab y ejecutar las celdas en orden.
