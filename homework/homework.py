# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


import os
import gzip
import json
import pickle
import zipfile
import logging
from typing import List
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)

# Configuración inicial
DATA_DIR = "files/input/"
OUTPUT_DIR = "files/output/"
MODEL_DIR = "files/models/"
TRAIN_ZIP = "train_data.csv.zip"
TEST_ZIP = "test_data.csv.zip"
TRAIN_CSV = "train_default_of_credit_card_clients.csv"
TEST_CSV = "test_default_of_credit_card_clients.csv"
MODEL_NAME = "model.pkl.gz"
METRICS_FILE = "metrics.json"

CATEGORICAL_VARS = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC_VARS = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger()


def unzip_dataframe(zip_path: str, csv_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(csv_name) as f:
            return pd.read_csv(f)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.dropna()
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    return df


def build_pipeline() -> Pipeline:
    transformer = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_VARS),
        ("num", MinMaxScaler(), NUMERIC_VARS)
    ])
    pipeline = Pipeline(steps=[
        ("transform", transformer),
        ("select", SelectKBest(score_func=f_classif)),
        ("model", LogisticRegression(max_iter=1000, solver="saga", random_state=42))
    ])
    return pipeline


def get_param_grid():
    return {
        "select__k": range(1, 11),
        "model__penalty": ["l1", "l2"],
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100]
    }


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


def compute_metrics(y_true, y_pred, dataset: str):
    return {
        "type": "metrics",
        "dataset": dataset,
        "precision": precision_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }


def compute_confusion(y_true, y_pred, dataset: str):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)}
    }


def save_metrics(metrics: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for entry in metrics:
            json.dump(entry, f)
            f.write("\n")


def main():
    # Paso 1: Cargar y limpiar datos
    train_df = unzip_dataframe(os.path.join(DATA_DIR, TRAIN_ZIP), TRAIN_CSV)
    test_df = unzip_dataframe(os.path.join(DATA_DIR, TEST_ZIP), TEST_CSV)
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    X_train = train_df.drop("default", axis=1)
    y_train = train_df["default"]
    X_test = test_df.drop("default", axis=1)
    y_test = test_df["default"]

    # Paso 2-4: Pipeline y optimización
    pipe = build_pipeline()
    grid = GridSearchCV(pipe, get_param_grid(), cv=10, scoring="balanced_accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    # Paso 5: Guardar modelo
    save_model(grid, os.path.join(MODEL_DIR, MODEL_NAME))

    # Paso 6-7: Métricas
    y_train_pred = grid.predict(X_train)
    y_test_pred = grid.predict(X_test)

    results = [
        compute_metrics(y_train, y_train_pred, "train"),
        compute_metrics(y_test, y_test_pred, "test"),
        compute_confusion(y_train, y_train_pred, "train"),
        compute_confusion(y_test, y_test_pred, "test"),
    ]

    save_metrics(results, os.path.join(OUTPUT_DIR, METRICS_FILE))


main()