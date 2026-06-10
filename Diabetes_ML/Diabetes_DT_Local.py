# -*- coding: utf-8 -*-
"""
Diabetes Prediction — Random Forest — Script de Entrenamiento Independiente
Lee diabetes_prediction_dataset.csv y divide en 80% train / 20% test.
Exporta: diabetes_modelo_dt.pkl

COLUMNAS (orden estricto):
  ["gender", "age", "hypertension", "heart_disease",
   "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]
OBJETIVO: diabetes  (0 = No diabético, 1 = Diabético)

CODIFICACIÓN DE VARIABLES CATEGÓRICAS:
  gender:          Female=0, Male=1, Other=2
  smoking_history: never=0, No Info=1, current=2, former=3, not current=4, ever=5

NOTA: El Random Forest NO necesita scaler — es insensible a la escala.
Se usa RandomForestClassifier (100 arboles) para obtener probabilidades
distribuidas en lugar de valores extremos 0%/100%.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# ── Nombre del archivo de salida ──────────────────────────────────────────────
DT_MODELO_ARCHIVO = "diabetes_modelo_dt.pkl"

# ── 1. CARGA DE DATOS ─────────────────────────────────────────────────────────
print("Cargando datos desde 'diabetes_prediction_dataset.csv'...")
try:
    df = pd.read_csv("diabetes_prediction_dataset.csv")
except FileNotFoundError:
    print("Error: No se encontro 'diabetes_prediction_dataset.csv'.")
    print("       Asegurate de ejecutar este script desde la carpeta Anemia/.")
    exit()

print(f"  -> {len(df)} registros cargados.")

# ── 2. PREPROCESAMIENTO ───────────────────────────────────────────────────────

# Codificar variables categóricas a numéricas
GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}
SMOKING_MAP = {
    "never":       0,
    "No Info":     1,
    "current":     2,
    "former":      3,
    "not current": 4,
    "ever":        5,
}
df["gender"]          = df["gender"].map(GENDER_MAP)
df["smoking_history"] = df["smoking_history"].map(SMOKING_MAP)

if df.isnull().sum().any():
    print("  -> Valores nulos detectados. Rellenando con la mediana...")
    df = df.fillna(df.median(numeric_only=True))

COLUMNS = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
]
TARGET = "diabetes"

X = df[COLUMNS].values
y = df[TARGET].values

print(f"\n  Distribucion de clases -> No diabetico (0): {(y==0).sum()} | Diabetico (1): {(y==1).sum()}")

# ── 3. DIVISIÓN TRAIN / TEST  (80 / 20) ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nDivision: {len(X_train)} entrenamiento / {len(X_test)} prueba")

# ── 4. ENTRENAMIENTO DEL RANDOM FOREST ───────────────────────────────────────
# RandomForest promedia 100 arboles con distintos subconjuntos de datos,
# produciendo probabilidades suaves en lugar de 0% o 100%.
# class_weight="balanced" compensa el desbalance 91.5% / 8.5%.
print("\nEntrenando el Random Forest (100 arboles)...")
dt_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,           # mas profundidad para capturar patrones del dataset grande
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1               # usa todos los nucleos disponibles
)
dt_model.fit(X_train, y_train)
print("Entrenamiento finalizado.")

# ── 5. EVALUACIÓN ─────────────────────────────────────────────────────────────
print("\n--- EVALUACION EN CONJUNTO DE PRUEBA (20%) ---")
y_pred = dt_model.predict(X_test)

print(f"Accuracy en test: {(y_pred == y_test).mean() * 100:.2f}%")
print("\nMatriz de Confusion:")
print(confusion_matrix(y_test, y_pred))
print("\nMetricas de Clasificacion:")
print(classification_report(y_test, y_pred, target_names=["No diabetico", "Diabetico"]))

# ── 6. IMPORTANCIA DE FEATURES ────────────────────────────────────────────────
print("\nImportancia de Variables:")
importances = dt_model.feature_importances_
for col, imp in sorted(zip(COLUMNS, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {col:<22} {imp:.4f}  {bar}")

# ── 7. GUARDAR EL MODELO ──────────────────────────────────────────────────────
joblib.dump(dt_model, DT_MODELO_ARCHIVO)
print(f"\n OK Random Forest guardado como '{DT_MODELO_ARCHIVO}'")
print("  Cargalo en app.py con: joblib.load('anemia_modelo_dt.pkl')")

# ── 8. PRUEBA DE PREDICCIÓN INDIVIDUAL ───────────────────────────────────────
print("\n--- PRUEBA DE PREDICCION INDIVIDUAL ---")
# [gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]
# Ejemplo: mujer, 60 años, hipertensa, sin cardiopatía, nunca fumó, BMI=30, HbA1c=7.5, glucosa=200
paciente_ejemplo = np.array([[0, 60.0, 1, 0, 0, 30.0, 7.5, 200]])
pred_clase = dt_model.predict(paciente_ejemplo)[0]
pred_proba = dt_model.predict_proba(paciente_ejemplo)[0]  # [prob_no, prob_diabetes]
estado     = "Diabetico" if pred_clase == 1 else "No diabetico"
print(f"Paciente ejemplo -> {estado}")
print(f"  Probabilidad de diabetes:      {pred_proba[1]*100:.2f}%")
print(f"  Probabilidad de no diabetes:   {pred_proba[0]*100:.2f}%")
