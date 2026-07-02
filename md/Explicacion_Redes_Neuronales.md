# Guía Práctica: Entendiendo el Código de la Red Neuronal

Este documento explica paso a paso cómo funcionan los scripts de entrenamiento de la **Red Neuronal** (`Diabetes_RN_Local.py` y `Titanic_RN_Local.py`). Está escrito para que cualquier persona, sin importar su experiencia, pueda entender qué hace cada línea importante.

> [!NOTE]
> **¿Qué es una Red Neuronal?**
> Imagina el cerebro humano: está formado por millones de neuronas conectadas entre sí que se "activan" o "silencian" dependiendo de la información que reciben. Una Red Neuronal Artificial imita este proceso de forma matemática. Le damos muchos ejemplos del pasado (datos de pacientes, pasajeros, etc.) y el modelo aprende a reconocer patrones para predecir casos nuevos que nunca ha visto.

---

## Bloque 1: Importar las Herramientas

```python
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
```

**Explicación práctica:**
Antes de construir cualquier cosa, necesitas sacar las herramientas. Aquí le decimos a Python qué necesitamos:

| Librería | Para qué sirve |
|---|---|
| `numpy` / `pandas` | Manejar los datos del CSV como una tabla |
| `tensorflow` / `keras` | Construir y entrenar la Red Neuronal |
| `keras.layers` | Las "capas" que forman la arquitectura de la red |
| `keras.regularizers` | Penalización L2 para evitar sobreajuste |
| `EarlyStopping` | Detener el entrenamiento automáticamente cuando deja de mejorar |
| `train_test_split` | Dividir los datos en parte de estudio y parte de examen |
| `StandardScaler` | Normalizar los datos (explicado más adelante) |
| `confusion_matrix` | Ver en detalle dónde acertó y falló el modelo |
| `joblib` | Guardar el normalizador en un archivo para usarlo en la web |

---

## Bloque 2: Fijar Semillas para Reproducibilidad

```python
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
```

**Explicación práctica:**
Las Redes Neuronales se inicializan con números **aleatorios**. Esto significa que si entrenas el mismo modelo dos veces, podrías obtener resultados ligeramente distintos cada vez.

Fijar la "semilla" (`seed`) es como decirle a la computadora: *"usa siempre la misma secuencia de números aleatorios"*. Así el modelo produce los mismos resultados cada vez que lo entrenas, lo cual es fundamental para poder comparar y depurar experimentos. El número `42` es simplemente una convención popular (viene de la novela *"The Hitchhiker's Guide to the Galaxy"*).

---

## Bloque 3: Lógica de Carga Inteligente (solo en Titanic_RN_Local.py)

```python
if os.path.exists(MODELO_ARCHIVO) and os.path.exists(SCALER_ARCHIVO):
    model = load_model(MODELO_ARCHIVO)
    scaler = joblib.load(SCALER_ARCHIVO)
else:
    # ... entrenar desde cero
```

**Explicación práctica:**
Este bloque es una optimización de tiempo. Entrenar una red neuronal puede tomar minutos. Si ya tienes el modelo guardado del entrenamiento anterior, ¿para qué volver a esperar?

El script verifica si los archivos `.h5` y `.pkl` ya existen en el disco:
- **Si existen** → Los carga directamente y salta al paso de predicción.
- **Si no existen** → Entra al bloque `else` y entrena la red desde cero.

Es como un estudiante que ya tiene sus apuntes guardados: si los tiene, los reutiliza; si no, los escribe de nuevo.

---

## Bloque 4: Cargar y Preparar los Datos

```python
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Rellenar valores faltantes con la mediana
df = df.fillna(df.median(numeric_only=True))

COLUMNS = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
TARGET   = "Result"

X = df[COLUMNS].values  # Datos de entrada (las "pistas")
y = df[TARGET].values   # Respuestas correctas (0 o 1)
```

**Explicación práctica:**
1. **Leemos el CSV** con `pandas`: lo convierte en una tabla fácil de manipular.
2. **Rellenamos huecos**: Si a algún paciente le falta un dato en el archivo, lo reemplazamos por el valor más representativo (la mediana) para que el modelo no se rompa.
3. **Separamos en X e y**: Dividimos la tabla en dos partes:
   - **`X` (las pistas):** Los valores que el modelo recibirá para hacer su análisis (hemoglobina, género, etc.).
   - **`y` (la respuesta correcta):** La etiqueta real del paciente (0 = sano, 1 = enfermo). Con esto el modelo sabe si acertó durante el entrenamiento.

---

## Bloque 5: Dividir para Estudiar y para el Examen

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

**Explicación práctica:**
Cortamos nuestros datos en dos grupos antes de que el modelo los vea:

| Grupo | Tamaño | Uso |
|---|---|---|
| **Train (Entrenamiento)** | 80% | El modelo aprende con estos datos |
| **Test (Prueba)** | 20% | El "examen final" con datos que el modelo nunca vio |

El parámetro `stratify=y` es un detalle importante: asegura que ambos grupos tengan la misma proporción de enfermos y sanos. Sin esto, podría ocurrir que el 100% de los enfermos quede en el set de entrenamiento y el modelo nunca los vea durante la evaluación.

---

## Bloque 6: Normalización de los Datos (el Scaler)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Guardar el scaler — se necesita en app.py
joblib.dump(scaler, "diabetes_scaler.pkl")
```

**Explicación práctica:**
Este bloque responde a una pregunta clave: ¿por qué el Árbol de Decisión no necesita esto pero la Red Neuronal sí?

Imagina que tienes dos columnas:
- **Hemoglobina:** valores entre 5 y 20
- **MCV:** valores entre 60 y 120

La Red Neuronal funciona con multiplicaciones matemáticas. Si un número es 100 veces más grande que otro, el modelo le dará automáticamente más "importancia" solo por su tamaño, aunque no sea más relevante. El `StandardScaler` ajusta todos los valores para que estén en la misma escala (alrededor de 0, con desviación estándar de 1).

**Diferencia crítica entre `fit_transform` y `transform`:**
- `fit_transform(X_train)`: *Aprende* la escala de los datos de entrenamiento y los transforma.
- `transform(X_test)`: Aplica la **misma escala ya aprendida** a los datos de prueba. ¡Nunca debe aprender del test!

El scaler se guarda en `.pkl` porque en `app.py` necesitamos aplicar exactamente esta misma transformación a los datos del usuario en tiempo real.

---

## Bloque 7: Definir la Arquitectura de la Red Neuronal

```python
# Modelo de Diabetes ML (con Dropout y regularización L2)
model = keras.Sequential([
    layers.Dense(64, input_dim=5, activation="relu",
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu",
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1,  activation="sigmoid")
])
```

**Explicación práctica:**
Aquí es donde "construimos" la red. `Sequential` significa que las capas van una detrás de otra, como eslabones de una cadena. Cada `Dense` es una **capa de neuronas**:

```
ENTRADA (5 datos)
      │
  ┌───▼───┐
  │  64   │  ← Capa 1: 64 neuronas + regularización L2
  └───┬───┘
  ┌───▼───┐
  │Dropout│  ← Apaga el 30% de neuronas aleatoriamente (evita memorizar)
  └───┬───┘
  ┌───▼───┐
  │  32   │  ← Capa 2: 32 neuronas + regularización L2
  └───┬───┘
  ┌───▼───┐
  │Dropout│  ← Apaga el 20% de neuronas aleatoriamente
  └───┬───┘
  ┌───▼───┐
  │  16   │  ← Capa 3: 16 neuronas (consolidan la información)
  └───┬───┘
  ┌───▼───┐
  │   1   │  ← Capa de salida: 1 neurona con valor entre 0.0 y 1.0
  └───────┘
```

**¿Qué son las `activation` (funciones de activación)?**

- **`relu`** *(Rectified Linear Unit)*: Si el valor es negativo, lo pone en 0; si es positivo, lo deja igual. Permite a la red aprender relaciones complejas y no lineales. Es la función más usada en capas intermedias.
- **`sigmoid`**: Toma cualquier número y lo "aplana" entre 0 y 1. Perfecta para la capa final cuando queremos interpretar el resultado como una **probabilidad** (ej: 0.87 = 87% de probabilidad de diabetes).

**¿Qué es el `Dropout`?**
Durante cada iteración de entrenamiento, el Dropout "apaga" aleatoriamente un porcentaje de neuronas (30% en la primera capa, 20% en la segunda). Esto obliga a la red a no depender de ningún camino único de cálculo y a aprender patrones más generales. Sin Dropout, el dataset de diabetes (que está desbalanceado 91.5%/8.5%) podría llevar al modelo a predecir siempre la clase mayoritaria.

**¿Qué es la regularización L2 (`kernel_regularizer`)?**
Penaliza los pesos muy grandes dentro de la red. Los pesos grandes tienden a producir predicciones extremas (saturación de la función sigmoid). L2 empuja a la red a mantener pesos más moderados, contribuyendo a probabilidades menos extremas.

---

## Bloque 8: Compilar el Modelo

```python
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
```

**Explicación práctica:**
Compilar es como configurar el motor de aprendizaje antes de que arranque.

- **`loss="binary_crossentropy"`** (función de pérdida): Es la "nota" que el modelo se pone a sí mismo. Le dice qué tan equivocado estuvo en sus predicciones. Se usa `binary_crossentropy` cuando el resultado es binario (sí/no, 0/1). El objetivo del entrenamiento es minimizar este número hasta que sea lo más cercano a 0 posible.

- **`optimizer="adam"`**: Es el "método de estudio" del modelo. Adam (*Adaptive Moment Estimation*) es uno de los optimizadores más populares y eficientes. Ajusta automáticamente qué tan rápido aprende el modelo y en qué dirección deben moverse los pesos de las neuronas.

- **`metrics=["accuracy"]`**: Solo para monitoreo visual. Le pedimos que también nos muestre el porcentaje de aciertos durante el entrenamiento.

---

## Bloque 9: Entrenar el Modelo

```python
# EarlyStopping: detiene el entrenamiento si val_loss no mejora en 20 épocas
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stop],
    verbose=0
)
```

**Explicación práctica:**
¡Este es el momento donde el modelo aprende! `model.fit()` es el corazón del proceso.

| Parámetro | Valor | Qué significa |
|---|---|---|
| `epochs=150` | Máx 150 vueltas | Límite superior; EarlyStopping lo detiene antes si ya no mejora |
| `batch_size=32` | 32 ejemplos | No procesa todos los datos a la vez; los divide en grupos de 32 para ir ajustando los pesos gradualmente |
| `validation_split=0.15` | 15% | Reserva otro 15% del entrenamiento para monitorear si el modelo está aprendiendo bien o memorizando |
| `verbose=0` | Silencioso | No imprime nada por pantalla en cada época (sería demasiado texto) |

**¿Qué hace `EarlyStopping`?**
Monitorea la pérdida de validación (`val_loss`) después de cada época. Si pasan 20 épocas seguidas sin mejorar, detiene el entrenamiento y **restaura los pesos de la mejor época** (`restore_best_weights=True`). Esto evita el sobreajuste (*overfitting*) que ocurría antes con 300 épocas y que producía salidas extremas de 0% o 100%.

**¿Qué pasa en cada época?**
1. El modelo recibe 32 ejemplos (un *batch*).
2. Hace sus predicciones.
3. Compara con las respuestas correctas usando la función de pérdida.
4. El optimizador ajusta todos los pesos internos un poquito en la dirección correcta.
5. Repite hasta procesar todos los datos → eso es 1 época. Luego comienza de nuevo.

---

## Bloque 10: Guardar el Modelo

```python
model.save("diabetes_modelo_rn.h5")
```

**Explicación práctica:**
El modelo ya aprendió. Ahora guardamos su "cerebro entrenado" en un archivo `.h5`.

El formato **HDF5** (`.h5`) es un formato estándar para guardar modelos de Keras. Contiene todo lo que el modelo necesita para funcionar:
- La arquitectura (cuántas capas y neuronas tiene)
- Los pesos aprendidos (millones de números ajustados durante el entrenamiento)
- El estado del optimizador

Gracias a esto, `app.py` puede cargar este archivo al iniciar y el modelo estará listo para predecir al instante, sin necesidad de volver a entrenar.

---

## Bloque 11: Evaluar con los Datos de Prueba

```python
y_pred_prob = model.predict(X_test_scaled, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Explicación práctica:**
Le pasamos el 20% de datos que el modelo **nunca ha visto** y pedimos sus predicciones.

- `model.predict()` devuelve probabilidades (ej: `[0.92, 0.07, 0.74...]`).
- `>= 0.5` convierte esas probabilidades en decisiones: si supera el 50%, clasifica como positivo (1).

**¿Qué es la Matriz de Confusión?**
Es una tabla de 2x2 que resume los errores y aciertos del modelo:

```
                  Predicho: 0    Predicho: 1
Real: 0 (Sano)  [ Correcto ✓  | Falso Positivo ✗ ]
Real: 1 (Enf.)  [ Falso Neg. ✗ |  Correcto ✓      ]
```

- **Falso Positivo (FP):** El modelo dijo que estaba enfermo pero estaba sano.
- **Falso Negativo (FN):** El modelo dijo que estaba sano pero estaba enfermo. *(El más peligroso en contexto médico)*

---

## Bloque 12: Predicción Individual de Ejemplo

```python
paciente_ejemplo = np.array([[1, 9.5, 22.0, 29.0, 78.0]])
paciente_scaled  = scaler.transform(paciente_ejemplo)
prob_diabetes    = float(model.predict(paciente_scaled, verbose=0)[0][0])
estado           = "Diabético" if prob_diabetes >= 0.5 else "No diabético"
```

**Explicación práctica:**
Aquí probamos el modelo con un paciente inventado para verificar que todo funciona.

El orden de los datos **es crítico** y debe ser exactamente el mismo que se usó durante el entrenamiento: `[Gender, Hemoglobin, MCH, MCHC, MCV]`. Si cambias el orden, el modelo recibirá datos incorrectos y producirá predicciones sin sentido.

Este es exactamente el mismo proceso que ocurre cuando un usuario completa el formulario web en la página de Diabetes ML.

---

## Bloque 13: Temperature Scaling en app.py (calibración de probabilidades)

```python
# En app.py, rama de predicción de diabetes ML con Red Neuronal:
prob_raw = float(prediccion[0][0])

TEMPERATURE = 3.5
logit = np.log(prob_raw / (1 - prob_raw + 1e-8))
prob_calibrada = 1 / (1 + np.exp(-logit / TEMPERATURE))
probabilidad = prob_calibrada * 100
```

**Explicación práctica:**
El dataset de diabetes tiene un desbalance de clases (91.5%/8.5%), lo que hace que el modelo tienda a ser conservador con la clase positiva. El **Temperature Scaling** es una técnica de calibración post-entrenamiento que suaviza las predicciones extremas *sin reentrenar el modelo*.

**¿Cómo funciona?**
1. Extrae el **logit** (valor antes de aplicar sigmoid): `log(p / (1-p))`.
2. Divide el logit por la temperatura `T = 3.5`. Un valor T > 1 comprime el logit hacia 0, produciendo probabilidades más intermedias.
3. Re-aplica sigmoid para volver al rango [0, 1].

```
Sin calibrar:   prob = 0.00002  →  mostrado como  0.002%
Con T=3.5:      prob = 0.0452   →  mostrado como  4.52%
```

El diagnóstico final ("Anémico" / "No anémico") no cambia: el umbral sigue siendo 50%. Solo los porcentajes se muestran de forma más legible.

> [!NOTE]
> El Temperature Scaling es solo aplicado en la predicción en vivo (`app.py`). El modelo `.h5` guardado no se modifica.

---

## Resumen del Flujo Completo

```
CSV con datos reales
        │
        ▼
Preprocesamiento (limpiar nulos, codificar texto)
        │
        ▼
División 80/20 (Train / Test)
        │
        ▼
Normalización con StandardScaler
        │       └──► Guardar scaler.pkl  ─────────────────► app.py
        ▼
Definir arquitectura
 (Dense + Dropout + L2 regularización)
        │
        ▼
Compilar (loss, optimizer, metrics)
        │
        ▼
Entrenar: model.fit() — máx 150 épocas + EarlyStopping
        │       └──► Guardar modelo.h5  ──────────────────► app.py
        ▼
Evaluar con el 20% reservado
        │
        ▼
Matriz de Confusión + Reporte de Clasificación
        │
        ▼
app.py: predicción en vivo
 └── Escalar datos del usuario con scaler.pkl
 └── Predecir con modelo.h5
 └── Aplicar Temperature Scaling (T=3.5)
 └── Mostrar probabilidad calibrada en pantalla
```

> [!IMPORTANT]
> El archivo `.h5` (modelo) y el `.pkl` (scaler) siempre deben ir en pareja. Si reentrenan el modelo, también se genera un nuevo scaler. Usar un scaler viejo con un modelo nuevo (o viceversa) producirá predicciones incorrectas porque la escala de los datos no coincidirá.

