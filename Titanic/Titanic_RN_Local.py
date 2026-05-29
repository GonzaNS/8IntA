# -*- coding: utf-8 -*-
"""
Titanic Neural Network - Local Version (VS Code)
Flujo optimizado: Entrena y guarda la primera vez. Solo predice las siguientes.
"""

import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import joblib  # Librería para guardar nuestro normalizador (Scaler)

# --- 1. FIJAR SEMILLAS PARA REPRODUCIBILIDAD ---
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Nombres de los archivos a guardar
MODELO_ARCHIVO = "mimodelo_completo.h5"
SCALER_ARCHIVO = "mi_scaler.pkl"

# --- LÓGICA PRINCIPAL: ¿ENTRENAR O CARGAR? ---
if os.path.exists(MODELO_ARCHIVO) and os.path.exists(SCALER_ARCHIVO):
    print(
        f"Archivos encontrados. Cargando modelo '{MODELO_ARCHIVO}' y omitiendo entrenamiento...")
    # Cargar el modelo pre-entrenado
    model = load_model(MODELO_ARCHIVO)
    # Cargar el normalizador
    scaler = joblib.load(SCALER_ARCHIVO)
    print("Modelo y Scaler cargados exitosamente.")

else:
    print("No se encontró el modelo guardado. Iniciando fase de entrenamiento...")

    # --- A. CARGA DE DATOS DE ENTRENAMIENTO ---
    try:
        training = pd.read_csv("titanic-train.csv")
    except FileNotFoundError:
        print("Error: No se encontró 'titanic-train.csv'.")
        exit()

    # --- B. PREPROCESAMIENTO ---
    training['Gender'] = training['Gender'].apply(
        lambda toLabel: 0 if toLabel == 'male' else 1)
    training["Age"] = training["Age"].fillna(training["Age"].mean())

    y_target = training["Survived"].values
    columns = ["Fare", "Pclass", "Gender", "Age", "SibSp"]
    x_input = training[list(columns)].values

    # --- C. NORMALIZACIÓN DE DATOS ---
    scaler = StandardScaler()
    x_input_scaled = scaler.fit_transform(x_input)

    # ¡IMPORTANTE! Guardamos el scaler para usarlo en el futuro
    joblib.dump(scaler, SCALER_ARCHIVO)

    # --- D. DEFINICIÓN DEL MODELO ---
    model = keras.Sequential([
        layers.Dense(32, input_dim=5, activation='relu'),
        layers.Dense(32, activation="relu", name="layer1"),
        layers.Dense(1, activation='sigmoid', name="layer2")
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # --- E. ENTRENAMIENTO ---
    print("Entrenando la red neuronal (400 épocas)...")
    model.fit(x_input_scaled, y_target, epochs=400, verbose=0)

    # Guardamos el modelo completo (Arquitectura + Pesos + Optimizador) en un solo archivo
    model.save(MODELO_ARCHIVO)
    print(
        f"Entrenamiento finalizado. Modelo guardado como '{MODELO_ARCHIVO}'.")


# =====================================================================
# FASE DE PREDICCIÓN (Esta parte siempre se ejecuta, sin importar
# si el modelo se acaba de entrenar o si se cargó del disco duro)
# =====================================================================

print("\n--- INICIANDO FASE DE PREDICCIÓN ---")

# 1. Predicción Individual (Ejemplo de un pasajero nuevo)
# Datos: [Fare, Pclass, Gender, Age, SibSp]
pasajero_nuevo = np.array([[83.475, 1, 1, 35, 1]])
# Usamos el scaler cargado/creado para transformar los datos del nuevo pasajero
pasajero_escalado = scaler.transform(pasajero_nuevo)
prediccion = model.predict(pasajero_escalado, verbose=0)
estado = "Sobrevive" if prediccion.round()[0][0] == 1 else "Fallece"
print(
    f"Predicción para pasajero de prueba: {estado} ({prediccion[0][0]*100:.2f}% de probabilidad)")

# 2. Evaluación Masiva con el archivo de Test
try:
    test = pd.read_csv("titanic-test.csv")
    print("\nEvaluando con 'titanic-test.csv'...")

    test['Gender'] = test['Gender'].apply(
        lambda toLabel: 0 if toLabel == 'male' else 1)
    test["Age"] = test["Age"].fillna(test["Age"].mean())

    columns = ["Fare", "Pclass", "Gender", "Age", "SibSp"]
    x_test = test[list(columns)].values
    y_test = test["Survived"].values

    # Escalamos los datos de prueba usando nuestro scaler guardado
    x_test_scaled = scaler.transform(x_test)

    # Hacemos las predicciones
    y_simuladot = model.predict(x_test_scaled, verbose=0).round().flatten()

    # Resultados
    conf_mat_test = confusion_matrix(y_true=y_test, y_pred=y_simuladot)
    print('\nMatriz de Confusión:')
    print(conf_mat_test)
    print('\nMétricas de Clasificación:')
    print(classification_report(y_test, y_simuladot))

    # Gráfico
    labels = ['Falleció', 'Sobrevivió']
    fig, ax = plt.subplots()
    cax = ax.matshow(conf_mat_test, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel('Predicho por el Modelo')
    plt.ylabel('Dato Real')
    plt.title('Matriz de Confusión - Test')
    plt.show()

except FileNotFoundError:
    print("\nNota: No se encontró 'titanic-test.csv'. Fin de la ejecución.")
