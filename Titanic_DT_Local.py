# -*- coding: utf-8 -*-
"""
Titanic Decision Tree - Script de Entrenamiento Independiente
Entrena un modelo de Árbol de Decisión con los mismos datos y columnas
que la Red Neuronal existente, y exporta el modelo como .pkl.

COLUMNAS (orden estricto): ["Fare", "Pclass", "Gender", "Age", "SibSp"]

NOTA: Este modelo NO necesita el scaler (mi_scaler.pkl) porque los
Árboles de Decisión son insensibles a la escala de los datos.
"""

import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import joblib

# ── Nombre del archivo de salida ──────────────────────────────────────────────
DT_MODELO_ARCHIVO = "modelo_dt_titanic.pkl"

# ── 1. CARGA DE DATOS ─────────────────────────────────────────────────────────
print("Cargando datos de entrenamiento...")
try:
    training = pd.read_csv("titanic-train.csv")
except FileNotFoundError:
    print("Error: No se encontró 'titanic-train.csv'. Asegúrate de ejecutar")
    print("       este script desde la raíz del proyecto.")
    exit()

print(f"  → {len(training)} registros cargados.")

# ── 2. PREPROCESAMIENTO (idéntico al de la Red Neuronal) ──────────────────────
training["Gender"] = training["Gender"].apply(
    lambda toLabel: 0 if toLabel == "male" else 1
)
training["Age"] = training["Age"].fillna(training["Age"].mean())

# Orden ESTRICTO de columnas (debe coincidir con lo que recibe app.py)
COLUMNS = ["Fare", "Pclass", "Gender", "Age", "SibSp"]

X = training[COLUMNS].values
y = training["Survived"].values

print(f"  → Features: {COLUMNS}")
print(f"  → Distribución de clases: {dict(zip(*np.unique(y, return_counts=True)))}")

# ── 3. DIVISIÓN TRAIN / VALIDACIÓN ────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nDivisión: {len(X_train)} entrenamiento / {len(X_val)} validación")

# ── 4. ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN ───────────────────────────────────
print("\nEntrenando el Árbol de Decisión...")

dt_model = DecisionTreeClassifier(
    max_depth=5,          # Limita la profundidad para evitar sobreajuste
    min_samples_split=10, # Mínimo de muestras para dividir un nodo
    min_samples_leaf=5,   # Mínimo de muestras en una hoja
    class_weight="balanced",  # Compensa el desbalance de clases
    random_state=42
)

dt_model.fit(X_train, y_train)
print("Entrenamiento finalizado.")

# ── 5. EVALUACIÓN ─────────────────────────────────────────────────────────────
print("\n--- EVALUACIÓN EN CONJUNTO DE VALIDACIÓN ---")
y_pred_val = dt_model.predict(X_val)

print(f"\nPrecisión (Accuracy): {(y_pred_val == y_val).mean() * 100:.2f}%")
print("\nMatriz de Confusión:")
print(confusion_matrix(y_val, y_pred_val))
print("\nMétricas de Clasificación:")
print(classification_report(y_val, y_pred_val, target_names=["Falleció", "Sobrevivió"]))

# ── 6. EVALUACIÓN CON ARCHIVO DE TEST (opcional) ──────────────────────────────
try:
    test = pd.read_csv("titanic-test.csv")
    print("--- EVALUACIÓN EN CONJUNTO DE TEST ---")

    test["Gender"] = test["Gender"].apply(
        lambda toLabel: 0 if toLabel == "male" else 1
    )
    test["Age"] = test["Age"].fillna(test["Age"].mean())

    X_test = test[COLUMNS].values
    y_test = test["Survived"].values

    y_pred_test = dt_model.predict(X_test)
    print(f"Precisión en Test: {(y_pred_test == y_test).mean() * 100:.2f}%")
    print("\nMatriz de Confusión (Test):")
    print(confusion_matrix(y_test, y_pred_test))
    print("\nMétricas de Clasificación (Test):")
    print(classification_report(y_test, y_pred_test, target_names=["Falleció", "Sobrevivió"]))

except FileNotFoundError:
    print("\nNota: No se encontró 'titanic-test.csv'. Se omite evaluación en test.")

# ── 7. GUARDAR EL MODELO ──────────────────────────────────────────────────────
joblib.dump(dt_model, DT_MODELO_ARCHIVO)
print(f"\n✔ Modelo guardado exitosamente como '{DT_MODELO_ARCHIVO}'")
print("  Puedes cargarlo en app.py con: joblib.load('modelo_dt_titanic.pkl')")

# ── 8. PRUEBA RÁPIDA DE PREDICCIÓN INDIVIDUAL ─────────────────────────────────
print("\n--- PRUEBA DE PREDICCIÓN INDIVIDUAL ---")
# Datos: [Fare, Pclass, Gender, Age, SibSp]
pasajero_nuevo = np.array([[83.475, 1, 1, 35, 1]])
prediccion_clase = dt_model.predict(pasajero_nuevo)[0]
prediccion_proba = dt_model.predict_proba(pasajero_nuevo)[0]  # [prob_fallece, prob_sobrevive]

estado = "Sobrevive" if prediccion_clase == 1 else "Fallece"
print(f"Pasajero de prueba → {estado}")
print(f"  Probabilidad de sobrevivir: {prediccion_proba[1] * 100:.2f}%")
print(f"  Probabilidad de fallecer:   {prediccion_proba[0] * 100:.2f}%")
