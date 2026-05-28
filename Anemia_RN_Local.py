# -*- coding: utf-8 -*-
"""
Anemia Neural Network - Script de Entrenamiento Independiente
Lee anemia.csv y divide internamente en 80% train / 20% test.
Exporta: anemia_modelo_rn.h5  y  anemia_scaler.pkl

COLUMNAS (orden estricto): ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
OBJETIVO: Result  (0 = No anémico, 1 = Anémico)
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
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
MODELO_ARCHIVO = "anemia_modelo_rn.h5"
SCALER_ARCHIVO = "anemia_scaler.pkl"

# ── 2. CARGA DE DATOS ─────────────────────────────────────────────────────────
print("Cargando datos desde 'anemia.csv'...")
try:
    df = pd.read_csv("anemia.csv")
except FileNotFoundError:
    print("Error: No se encontró 'anemia.csv'.")
    print("       Asegúrate de ejecutar este script desde la raíz del proyecto.")
    exit()

print(f"  → {len(df)} registros cargados.")
print(f"  → Columnas: {list(df.columns)}")

# ── 3. PREPROCESAMIENTO ───────────────────────────────────────────────────────
# Verificar valores nulos
if df.isnull().sum().any():
    print("  → Valores nulos detectados. Rellenando con la mediana...")
    df = df.fillna(df.median(numeric_only=True))

COLUMNS = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
TARGET   = "Result"

X = df[COLUMNS].values
y = df[TARGET].values

print(f"\n  Distribución de clases → No anémico (0): {(y==0).sum()} | Anémico (1): {(y==1).sum()}")

# ── 4. DIVISIÓN TRAIN / TEST  (80 / 20) ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nDivisión: {len(X_train)} entrenamiento / {len(X_test)} prueba")

# ── 5. NORMALIZACIÓN ──────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Guardar el scaler — CRÍTICO: se usa en app.py en tiempo de inferencia
joblib.dump(scaler, SCALER_ARCHIVO)
print(f"  → Scaler guardado como '{SCALER_ARCHIVO}'")

# ── 6. DEFINICIÓN DEL MODELO ──────────────────────────────────────────────────
model = keras.Sequential([
    layers.Dense(64, input_dim=len(COLUMNS), activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1,  activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# ── 7. ENTRENAMIENTO ──────────────────────────────────────────────────────────
print("\nEntrenando la Red Neuronal (300 épocas)...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=300,
    batch_size=32,
    validation_split=0.15,
    verbose=0
)

# Mostrar resultado de la última época
final_loss   = history.history["val_loss"][-1]
final_acc    = history.history["val_accuracy"][-1]
print(f"  → Validación — Loss: {final_loss:.4f} | Accuracy: {final_acc*100:.2f}%")

# ── 8. GUARDAR EL MODELO ──────────────────────────────────────────────────────
model.save(MODELO_ARCHIVO)
print(f"\n✔ Modelo guardado como '{MODELO_ARCHIVO}'")

# ── 9. EVALUACIÓN EN CONJUNTO DE PRUEBA ──────────────────────────────────────
print("\n--- EVALUACIÓN EN CONJUNTO DE PRUEBA (20%) ---")
y_pred_prob = model.predict(X_test_scaled, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

print(f"Accuracy en test: {(y_pred == y_test).mean() * 100:.2f}%")
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nMétricas de Clasificación:")
print(classification_report(y_test, y_pred, target_names=["No anémico", "Anémico"]))

# ── 10. PRUEBA DE PREDICCIÓN INDIVIDUAL ───────────────────────────────────────
print("\n--- PRUEBA DE PREDICCIÓN INDIVIDUAL ---")
# Datos: [Gender, Hemoglobin, MCH, MCHC, MCV]
paciente_ejemplo = np.array([[1, 9.5, 22.0, 29.0, 78.0]])  # probable anémico
paciente_scaled  = scaler.transform(paciente_ejemplo)
prob_anemia      = float(model.predict(paciente_scaled, verbose=0)[0][0])
estado           = "Anémico" if prob_anemia >= 0.5 else "No anémico"
print(f"Paciente ejemplo → {estado}  ({prob_anemia*100:.2f}% probabilidad de anemia)")
