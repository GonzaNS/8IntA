# -*- coding: utf-8 -*-
"""
Anemia Decision Tree - Script de Entrenamiento Independiente
Lee anemia.csv y divide internamente en 80% train / 20% test.
Exporta: anemia_modelo_dt.pkl

COLUMNAS (orden estricto): ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
OBJETIVO: Result  (0 = No anémico, 1 = Anémico)

NOTA: El Árbol de Decisión NO necesita scaler — es insensible a la escala.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# ── Nombre del archivo de salida ──────────────────────────────────────────────
DT_MODELO_ARCHIVO = "anemia_modelo_dt.pkl"

# ── 1. CARGA DE DATOS ─────────────────────────────────────────────────────────
print("Cargando datos desde 'anemia.csv'...")
try:
    df = pd.read_csv("anemia.csv")
except FileNotFoundError:
    print("Error: No se encontró 'anemia.csv'.")
    print("       Asegúrate de ejecutar este script desde la raíz del proyecto.")
    exit()

print(f"  → {len(df)} registros cargados.")

# ── 2. PREPROCESAMIENTO ───────────────────────────────────────────────────────
if df.isnull().sum().any():
    print("  → Valores nulos detectados. Rellenando con la mediana...")
    df = df.fillna(df.median(numeric_only=True))

COLUMNS = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
TARGET   = "Result"

X = df[COLUMNS].values
y = df[TARGET].values

print(f"\n  Distribución de clases → No anémico (0): {(y==0).sum()} | Anémico (1): {(y==1).sum()}")

# ── 3. DIVISIÓN TRAIN / TEST  (80 / 20) ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nDivisión: {len(X_train)} entrenamiento / {len(X_test)} prueba")

# ── 4. ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN ───────────────────────────────────
print("\nEntrenando el Árbol de Decisión...")
dt_model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)
dt_model.fit(X_train, y_train)
print("Entrenamiento finalizado.")

# ── 5. EVALUACIÓN ─────────────────────────────────────────────────────────────
print("\n--- EVALUACIÓN EN CONJUNTO DE PRUEBA (20%) ---")
y_pred = dt_model.predict(X_test)

print(f"Accuracy en test: {(y_pred == y_test).mean() * 100:.2f}%")
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nMétricas de Clasificación:")
print(classification_report(y_test, y_pred, target_names=["No anémico", "Anémico"]))

# ── 6. IMPORTANCIA DE FEATURES ────────────────────────────────────────────────
print("\nImportancia de Variables:")
importances = dt_model.feature_importances_
for col, imp in sorted(zip(COLUMNS, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {col:<12} {imp:.4f}  {bar}")

# ── 7. GUARDAR EL MODELO ──────────────────────────────────────────────────────
joblib.dump(dt_model, DT_MODELO_ARCHIVO)
print(f"\n✔ Modelo guardado como '{DT_MODELO_ARCHIVO}'")
print("  Cárgalo en app.py con: joblib.load('anemia_modelo_dt.pkl')")

# ── 8. PRUEBA DE PREDICCIÓN INDIVIDUAL ───────────────────────────────────────
print("\n--- PRUEBA DE PREDICCIÓN INDIVIDUAL ---")
# Datos: [Gender, Hemoglobin, MCH, MCHC, MCV]
paciente_ejemplo = np.array([[1, 9.5, 22.0, 29.0, 78.0]])  # probable anémico
pred_clase = dt_model.predict(paciente_ejemplo)[0]
pred_proba = dt_model.predict_proba(paciente_ejemplo)[0]   # [prob_no, prob_anemia]
estado     = "Anémico" if pred_clase == 1 else "No anémico"
print(f"Paciente ejemplo → {estado}")
print(f"  Probabilidad de anemia:      {pred_proba[1]*100:.2f}%")
print(f"  Probabilidad de no anemia:   {pred_proba[0]*100:.2f}%")
