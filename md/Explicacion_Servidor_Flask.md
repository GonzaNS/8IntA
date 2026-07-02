# Guía Práctica: Entendiendo el Servidor Flask (`app.py`)

Este documento explica paso a paso cómo funciona `app.py`, el archivo central del proyecto. Es el **servidor web** que conecta el formulario HTML con los modelos de Machine Learning. Está escrito para que cualquier persona pueda entenderlo sin experiencia previa en desarrollo web.

> [!NOTE]
> **¿Qué es un servidor web?**
> Cuando abres un navegador y escribes `http://localhost:5000`, tu navegador le está "preguntando" algo a un programa que está corriendo en tu computadora. Ese programa es el servidor. Flask es el framework que nos permite crear ese servidor con muy pocas líneas de Python.

---

## La analogía del restaurante

Para entender cómo funciona Flask, piensa en un restaurante:

| Restaurante | Flask / app.py |
|---|---|
| El menú con los platos disponibles | Las **rutas** (`@app.route`) |
| El mesero que toma tu pedido | Flask recibiendo la petición del navegador |
| La cocina que prepara el plato | La función de Python que procesa los datos |
| El plato que llega a tu mesa | La página HTML que el servidor devuelve |
| Los insumos de cocina | Los modelos `.h5` y `.pkl` cargados en memoria |

---

## Bloque 1: Importar las Herramientas

```python
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import joblib
import numpy as np

from diabetes_expert import diagnosticar, SINTOMAS, REGLAS_PROLOG
```

**Explicación práctica:**

| Importación | Para qué sirve |
|---|---|
| `Flask` | La clase principal que crea la aplicación web |
| `render_template` | Le dice a Flask que devuelva un archivo HTML con datos incluidos |
| `request` | Permite leer los datos que el usuario envió desde el formulario |
| `load_model` | Carga un modelo de Red Neuronal desde un archivo `.h5` |
| `joblib` | Carga el scaler y los modelos de Árbol de Decisión desde `.pkl` |
| `numpy` | Para organizar los datos del formulario en un array numérico |
| `diagnosticar`, etc. | Funciones del Sistema Experto de Diabetes (archivo separado) |

---

## Bloque 2: Crear la Aplicación

```python
app = Flask(__name__)
```

**Explicación práctica:**
Esta única línea crea toda la aplicación web. `Flask(__name__)` le dice a Flask en qué carpeta buscar los archivos HTML (la carpeta `templates/`) y los archivos estáticos (CSS, imágenes).

`__name__` es una variable especial de Python que contiene el nombre del archivo actual. Flask la usa internamente para ubicar los recursos del proyecto.

---

## Bloque 3: Cargar los Modelos al Iniciar 🔑

```python
# ▶ Red Neuronal Titanic
nn_model = load_model("mimodelo_completo.h5")
scaler   = joblib.load("mi_scaler.pkl")

# ▶ Árbol de Decisión Titanic (carga condicional)
if os.path.exists("modelo_dt_titanic.pkl"):
    dt_model = joblib.load("modelo_dt_titanic.pkl")
else:
    dt_model = None

# ▶ Red Neuronal Diabetes ML (carga condicional)
if os.path.exists("Diabetes_ML/diabetes_modelo_rn.h5") and os.path.exists("Diabetes_ML/diabetes_scaler.pkl"):
    anemia_nn_model = load_model("Diabetes_ML/diabetes_modelo_rn.h5")
    anemia_scaler   = joblib.load("Diabetes_ML/diabetes_scaler.pkl")
else:
    anemia_nn_model = None
    anemia_scaler   = None

# ▶ Random Forest Diabetes ML (carga condicional)
if os.path.exists("Diabetes_ML/diabetes_modelo_dt.pkl"):
    anemia_dt_model = joblib.load("Diabetes_ML/diabetes_modelo_dt.pkl")
else:
    anemia_dt_model = None
```

**Explicación práctica:**
Este bloque se ejecuta **una sola vez**, en el momento en que arrancas el servidor (`python app.py`). Los modelos quedan guardados en la memoria RAM de la computadora.

**¿Por qué cargamos los modelos aquí y no dentro de cada función?**
Si cargáramos el modelo dentro de la función de predicción, cada vez que un usuario hiciera clic en "Predecir" habría que abrir el archivo del disco duro y cargar el modelo de nuevo. Eso podría tardar varios segundos. Al cargarlo una vez al inicio, la predicción es **instantánea** porque el modelo ya está en memoria.

**Carga condicional con `os.path.exists()`:**
Los modelos generados por los scripts de entrenamiento son opcionales (el usuario debe ejecutarlos primero). Si el archivo `.pkl` no existe, la variable se establece en `None`. Más adelante, cada ruta verifica si la variable es `None` antes de usarla y muestra un mensaje de error amigable si el modelo falta.

```
Arrancar servidor
      │
      ▼
¿Existe mimodelo_completo.h5?  →  Sí: cargar  →  nn_model = <modelo>
                               →  No: ERROR (este es obligatorio)

¿Existe modelo_dt_titanic.pkl? →  Sí: cargar  →  dt_model = <modelo>
                               →  No: ignorar →  dt_model = None
```

---

## Bloque 4: Las Rutas — El Menú del Restaurante

Una **ruta** (`route`) es la asociación entre una URL y una función de Python. Cuando el navegador visita una URL, Flask ejecuta la función correspondiente y devuelve lo que esa función retorna.

```python
@app.route("/")
def home():
    return render_template("index.html")
```

El `@app.route("/")` es un **decorador**: una instrucción especial que le dice a Flask *"cuando alguien visite la URL `/`, ejecuta la función `home()`"*.

### Tabla completa de rutas del proyecto

| URL | Método HTTP | Función | Qué hace |
|---|---|---|---|
| `/` | GET | `home()` | Muestra el formulario del Titanic vacío |
| `/predecir` | POST | `predecir()` | Recibe datos Titanic y retorna predicción |
| `/diabetes_ml` | GET | `anemia_form()` | Muestra el formulario de Diabetes ML vacío |
| `/diabetes_ml/predecir` | POST | `anemia_predecir()` | Recibe datos Diabetes ML y retorna predicción |
| `/diabetes` | GET | `diabetes_form()` | Muestra el formulario de Diabetes vacío |
| `/diabetes/diagnosticar` | POST | `diabetes_diagnosticar()` | Recibe síntomas y retorna diagnóstico |

**¿Qué son los métodos HTTP GET y POST?**
- **GET:** El navegador *pide* una página. Se usa para mostrar formularios vacíos. La URL cambia pero no se envían datos sensibles.
- **POST:** El navegador *envía* datos al servidor (los campos del formulario). Se usa para procesar información.

---

## Bloque 5: Ruta de Predicción — Titanic

```python
@app.route("/predecir", methods=["POST"])
def predecir():
    # 1. Leer los datos del formulario
    fare   = float(request.form["fare"])
    pclass = int(request.form["pclass"])
    gender = int(request.form["gender"])
    age    = float(request.form["age"])
    sibsp  = int(request.form["sibsp"])

    # 2. Saber qué modelo eligió el usuario
    modelo_elegido = request.form.get("modelo", "red_neuronal")

    # 3. Organizar los datos en el orden correcto
    datos_entrada = np.array([[fare, pclass, gender, age, sibsp]])

    # 4. Procesar según el modelo elegido
    if modelo_elegido == "red_neuronal":
        datos_escalados = scaler.transform(datos_entrada)
        prediccion      = nn_model.predict(datos_escalados)
        probabilidad    = float(prediccion[0][0]) * 100
        estado          = "Sobrevive" if probabilidad >= 50 else "Fallece"

    elif modelo_elegido == "arbol_decision":
        if dt_model is None:
            return render_template("index.html",
                resultado="Error: ejecuta Titanic_DT_Local.py primero",
                probabilidad=0.0,
                modelo_usado=modelo_elegido)
        proba        = dt_model.predict_proba(datos_entrada)[0]
        probabilidad = float(proba[1]) * 100
        estado       = "Sobrevive" if probabilidad >= 50 else "Fallece"

    # 5. Devolver la página con el resultado
    return render_template("index.html",
        resultado    = estado,
        probabilidad = probabilidad,
        modelo_usado = modelo_elegido)
```

**Explicación práctica paso a paso:**

**Paso 1 — Leer los datos del formulario:**
`request.form["fare"]` es como abrir el sobre que llegó del navegador y leer el valor del campo con `name="fare"` del HTML. Todo llega como texto (`string`), por eso se convierte explícitamente a `float` o `int`.

**Paso 2 — Saber qué modelo eligió el usuario:**
El formulario tiene dos radio buttons con `name="modelo"`. El valor del que esté seleccionado llega aquí. Si el usuario no seleccionó nada (caso raro), se usa `"red_neuronal"` como valor por defecto.

**Paso 3 — Organizar los datos:**
`np.array([[...]])` crea una matriz de una sola fila. Los modelos esperan recibir los datos en este formato. El doble corchete `[[...]]` es importante: el modelo espera una tabla, aunque sea de una sola fila.

**Paso 4 — El despacho condicional:**
Dependiendo de qué modelo eligió el usuario, se toma uno de dos caminos:
- **Red Neuronal:** Primero normaliza con el `scaler`, luego predice. La salida es una probabilidad entre 0 y 1.
- **Árbol de Decisión:** Predice directamente sin normalizar. Usa `predict_proba()` para obtener la probabilidad en lugar de solo la clase.

**Paso 5 — `render_template()`:**
Esta función hace dos cosas a la vez:
1. Abre el archivo `templates/index.html`.
2. Reemplaza las variables como `{{ resultado }}` y `{{ probabilidad }}` en el HTML con los valores reales calculados.
El resultado es una página HTML completa que el navegador puede mostrar.

---

## Bloque 6: Ruta de Predicción — Diabetes ML

La estructura es idéntica a la del Titanic, pero con las columnas y modelos de Diabetes ML:

```python
@app.route("/diabetes_ml/predecir", methods=["POST"])
def anemia_predecir():
    gender     = int(request.form["gender"])
    hemoglobin = float(request.form["hemoglobin"])
    mch        = float(request.form["mch"])
    mchc       = float(request.form["mchc"])
    mcv        = float(request.form["mcv"])

    modelo_elegido = request.form.get("modelo", "red_neuronal")
    datos_entrada  = np.array([[gender, hemoglobin, mch, mchc, mcv]])

    if modelo_elegido == "red_neuronal":
        datos_escalados = anemia_scaler.transform(datos_entrada)
        prediccion      = anemia_nn_model.predict(datos_escalados)
        probabilidad    = float(prediccion[0][0]) * 100
        estado          = "Anémico" if probabilidad >= 50 else "No anémico"

    elif modelo_elegido == "arbol_decision":
        proba        = anemia_dt_model.predict_proba(datos_entrada)[0]
        probabilidad = float(proba[1]) * 100
        estado       = "Anémico" if probabilidad >= 50 else "No anémico"

    return render_template("anemia.html",
        resultado    = estado,
        probabilidad = probabilidad,
        modelo_usado = modelo_elegido)
```

**Diferencias importantes respecto al Titanic:**
- Se usan las variables `anemia_nn_model`, `anemia_scaler` y `anemia_dt_model` (específicas de anemia).
- Las etiquetas de resultado son `"Anémico"` / `"No anémico"` en lugar de `"Sobrevive"` / `"Fallece"`.
- El orden de columnas es `[Gender, Hemoglobin, MCH, MCHC, MCV]` en lugar de `[Fare, Pclass, Gender, Age, SibSp]`.

---

## Bloque 7: Ruta del Sistema Experto — Diabetes

```python
@app.route("/diabetes")
def diabetes_form():
    return render_template("diabetes.html",
        sintomas_marcados = [],
        resultado         = None,
        reglas_prolog     = REGLAS_PROLOG)

@app.route("/diabetes/diagnosticar", methods=["POST"])
def diabetes_diagnosticar():
    sintomas_presentes = request.form.getlist("sintomas")
    resultado          = diagnosticar(sintomas_presentes)

    return render_template("diabetes.html",
        sintomas_marcados = sintomas_presentes,
        resultado         = resultado,
        reglas_prolog     = REGLAS_PROLOG)
```

**Explicación práctica:**
El Sistema Experto es más simple de manejar en el servidor porque no hay modelos que cargar ni datos numéricos que normalizar.

`request.form.getlist("sintomas")` es especial: cuando hay múltiples checkboxes con el mismo `name="sintomas"`, esta función devuelve una lista con todos los que fueron marcados. Si ninguno fue marcado, devuelve una lista vacía `[]`.

La función `diagnosticar()` (definida en `diabetes_expert.py`) recibe esa lista de síntomas y ejecuta el motor de inferencia, devolviendo un diccionario con el diagnóstico, nivel de riesgo y recomendaciones.

---

## Bloque 8: Arrancar el Servidor

```python
if __name__ == "__main__":
    app.run(debug=True)
```

**Explicación práctica:**
Esta línea solo se ejecuta cuando corres el archivo directamente (`python app.py`), no cuando otro archivo lo importa.

`debug=True` activa el modo de desarrollo, que tiene dos ventajas:
1. Si cambias el código, el servidor se reinicia automáticamente sin que tengas que detenerlo y volver a iniciarlo.
2. Si ocurre un error, muestra el mensaje completo en el navegador en lugar de una página en blanco.

> [!WARNING]
> `debug=True` nunca debe usarse en un servidor real de producción (un sitio publicado en internet). Solo es para desarrollo local, ya que expone información interna del código en caso de error.

---

## Flujo Completo de una Petición (de principio a fin)

```
Usuario llena el formulario y hace clic en "Predecir"
        │
        │  HTTP POST a /anemia/predecir
        ▼
Flask recibe la petición y ejecuta anemia_predecir()
        │
        ├── request.form["hemoglobin"] → "9.5"  (string)
        ├── float("9.5")               → 9.5    (número)
        │
        ├── request.form.get("modelo") → "red_neuronal"
        │
        ├── np.array([[1, 9.5, 22.0, 29.0, 78.0]])
        │
        ├── [Si RN]  anemia_scaler.transform(datos)
        │             anemia_nn_model.predict(datos_escalados)
        │             → 0.923  →  92.3%  →  "Anémico"
        │
        └── [Si DT]  anemia_dt_model.predict_proba(datos)
                      → [0.08, 0.92]  →  92.0%  →  "Anémico"
        │
        ▼
render_template("anemia.html",
    resultado    = "Anémico",
    probabilidad = 92.3,
    modelo_usado = "red_neuronal")
        │
        ▼
Jinja2 inserta los valores en el HTML
        │
        ▼
El navegador muestra la página con el resultado y la barra de probabilidad
```

---

## Resumen de Variables Globales de Modelos

| Variable | Contenido | Se usa en |
|---|---|---|
| `nn_model` | Red Neuronal Titanic (`.h5`) | `/predecir` (rama RN) |
| `scaler` | Normalizador Titanic (`.pkl`) | `/predecir` (rama RN) |
| `dt_model` | Árbol de Decisión Titanic (`.pkl`) | `/predecir` (rama DT) |
| `anemia_nn_model` | Red Neuronal Anemia (`.h5`) | `/anemia/predecir` (rama RN) |
| `anemia_scaler` | Normalizador Anemia (`.pkl`) | `/anemia/predecir` (rama RN) |
| `anemia_dt_model` | Árbol de Decisión Anemia (`.pkl`) | `/anemia/predecir` (rama DT) |

> [!IMPORTANT]
> Todas estas variables son **globales**: se cargan una vez al iniciar el servidor y están disponibles para todas las funciones de rutas. Esto es intencional para maximizar la velocidad de respuesta. Si una variable es `None` (porque el archivo `.pkl` o `.h5` no existe), la ruta correspondiente mostrará un mensaje de error en lugar de caerse.
