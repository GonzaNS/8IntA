# -*- coding: utf-8 -*-
"""
Diabetes Prediction — Red Neuronal — Script de Entrenamiento Independiente
Lee diabetes_prediction_dataset.csv y divide en 80% train / 20% test.
Exporta: diabetes_modelo_rn.h5  y  diabetes_scaler.pkl

COLUMNAS (orden estricto):
  ["gender", "age", "hypertension", "heart_disease",
   "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]
OBJETIVO: diabetes  (0 = No diabético, 1 = Diabético)

CODIFICACIÓN DE VARIABLES CATEGÓRICAS:
  gender:          Female=0, Male=1, Other=2
  smoking_history: never=0, No Info=1, current=2, former=3, not current=4, ever=5
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, regularizers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# ── 1. FIJAR SEMILLAS PARA REPRODUCIBILIDAD ───────────────────────────────────
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ── Nombres de los archivos de salida ─────────────────────────────────────────
MODELO_ARCHIVO = "diabetes_modelo_rn.h5"
SCALER_ARCHIVO = "diabetes_scaler.pkl"

# ── 2. CARGA DE DATOS ─────────────────────────────────────────────────────────
print("Cargando datos desde 'diabetes_prediction_dataset.csv'...")
try:
    df = pd.read_csv("diabetes_prediction_dataset.csv")
except FileNotFoundError:
    print("Error: No se encontro 'diabetes_prediction_dataset.csv'.")
    print("       Asegurate de ejecutar este script desde la carpeta Anemia/.")
    exit()

print(f"  -> {len(df)} registros cargados.")
print(f"  -> Columnas: {list(df.columns)}")

# ── 3. PREPROCESAMIENTO ───────────────────────────────────────────────────────

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

# Verificar valores nulos (puede haber NaN si algún valor no estaba en el mapa)
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

# ── 4. DIVISIÓN TRAIN / TEST  (80 / 20) ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nDivision: {len(X_train)} entrenamiento / {len(X_test)} prueba")

# ── 5. NORMALIZACIÓN ──────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Guardar el scaler — CRITICO: se usa en app.py en tiempo de inferencia
joblib.dump(scaler, SCALER_ARCHIVO)
print(f"  -> Scaler guardado como '{SCALER_ARCHIVO}'")

# ── 6. DEFINICIÓN DEL MODELO ──────────────────────────────────────────────────
# Dropout y L2 para evitar sobreajuste sobre el dataset desbalanceado (91.5% vs 8.5%).
model = keras.Sequential([
    layers.Dense(128, input_dim=len(COLUMNS), activation="relu",
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu",
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1,  activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# ── 7. ENTRENAMIENTO ──────────────────────────────────────────────────────────
# EarlyStopping evita sobreajuste al restaurar los mejores pesos.
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1
)

print("\nEntrenando la Red Neuronal (max 150 epocas con EarlyStopping)...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=150,
    batch_size=64,          # batch mayor por el tamaño del dataset (100k)
    validation_split=0.15,
    callbacks=[early_stop],
    verbose=0
)

# Mostrar resultado de la última época
final_loss = history.history["val_loss"][-1]
final_acc  = history.history["val_accuracy"][-1]
print(f"  -> Validacion — Loss: {final_loss:.4f} | Accuracy: {final_acc*100:.2f}%")

# ── 8. GUARDAR EL MODELO ──────────────────────────────────────────────────────
model.save(MODELO_ARCHIVO)
print(f"\n OK Modelo guardado como '{MODELO_ARCHIVO}'")

# ── 9. EVALUACIÓN EN CONJUNTO DE PRUEBA ──────────────────────────────────────
print("\n--- EVALUACION EN CONJUNTO DE PRUEBA (20%) ---")
y_pred_prob = model.predict(X_test_scaled, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

print(f"Accuracy en test: {(y_pred == y_test).mean() * 100:.2f}%")
print("\nMatriz de Confusion:")
print(confusion_matrix(y_test, y_pred))
print("\nMetricas de Clasificacion:")
print(classification_report(y_test, y_pred, target_names=["No diabetico", "Diabetico"]))

# ── 10. PRUEBA DE PREDICCIÓN INDIVIDUAL ───────────────────────────────────────
print("\n--- PRUEBA DE PREDICCION INDIVIDUAL ---")
# [gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]
# Ejemplo: mujer, 60 años, hipertensa, sin cardiopatía, nunca fumó, BMI=30, HbA1c=7.5, glucosa=200
paciente_ejemplo = np.array([[0, 60.0, 1, 0, 0, 30.0, 7.5, 200]])
paciente_scaled  = scaler.transform(paciente_ejemplo)
prob_diabetes    = float(model.predict(paciente_scaled, verbose=0)[0][0])
estado           = "Diabetico" if prob_diabetes >= 0.5 else "No diabetico"
print(f"Paciente ejemplo -> {estado}  ({prob_diabetes*100:.2f}% probabilidad de diabetes)")
