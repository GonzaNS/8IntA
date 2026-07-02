# Arquitectura del Proyecto — Plataforma de Análisis Predictivo Clínico

Este documento explica cómo está construido el proyecto: qué tecnologías se usan, por qué se eligieron, y cómo funcionan juntas. Está escrito para que cualquier persona pueda entenderlo, tenga o no experiencia en programación.

---

## ¿Qué es este proyecto?

Es una **aplicación web de diagnóstico asistido por Machine Learning** que permite al usuario ingresar datos de un paciente y recibir una predicción médica. Actualmente cuenta con tres módulos:

| Módulo | Tipo de modelo | ¿Qué predice? |
|---|---|---|
| **Titanic** | Red Neuronal + Árbol de Decisión | Probabilidad de supervivencia |
| **Diabetes ML** | Red Neuronal + Random Forest | Si el paciente tiene diabetes |
| **Diabetes (Experto)** | Sistema Experto con reglas | Nivel de riesgo de Diabetes Tipo 2 |

---

## Vista general de la arquitectura

```
┌──────────────────────────────────────────────────────────────┐
│                     NAVEGADOR DEL USUARIO                     │
│          (escribe datos en un formulario HTML/CSS)            │
└─────────────────────────┬────────────────────────────────────┘
                          │  HTTP POST (envía datos del form)
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                    SERVIDOR  —  Flask (Python)                 │
│                         app.py                                │
│                                                               │
│  Recibe los datos → elige el modelo → calcula → responde     │
└───────────┬────────────────────────┬─────────────────────────┘
            │                        │
            ▼                        ▼
  ┌─────────────────┐      ┌──────────────────────┐
  │  Modelos  .h5   │      │  Modelos  .pkl        │
  │  (Keras/TF)     │      │  (scikit-learn/joblib)│
  │  Red Neuronal   │      │  Árbol de Decisión    │
  │                 │      │  Sistema Experto      │
  └─────────────────┘      └──────────────────────┘
```

---

## Lenguaje de Programación: Python 🐍

**¿Qué es?** Python es un lenguaje de programación muy popular, especialmente en el mundo de la ciencia de datos e inteligencia artificial. Su sintaxis es cercana al inglés, lo que lo hace fácil de leer y aprender.

**¿Por qué se usa aquí?** Porque las mejores herramientas para Machine Learning (Keras, scikit-learn) y para crear servidores web rápidos (Flask) están escritas en Python y funcionan perfectamente juntas en el mismo proyecto.

---

## Parte 1 — El Servidor Web: Flask

**¿Qué es Flask?**
Flask es un **microframework web** para Python. Un "framework" es como una plantilla prefabricada que te ahorra escribir cientos de líneas de código repetitivo. "Micro" significa que es ligero y minimalista: te da lo justo y necesario sin complicaciones.

**¿Cómo funciona en este proyecto?**
`app.py` es el corazón del servidor. Cuando el usuario abre el navegador y escribe la dirección, Flask:
1. Escucha la petición del navegador.
2. Decide qué página o función ejecutar (esto se llama **ruta** o *route*).
3. Devuelve una página HTML con los resultados.

**Las rutas del proyecto:**

| URL | Qué hace |
|---|---|
| `/` | Muestra la página del Titanic |
| `/predecir` | Recibe los datos del formulario Titanic y devuelve la predicción |
| `/diabetes_ml` | Muestra la página de predicción de Diabetes (ML) |
| `/diabetes_ml/predecir` | Recibe los datos del formulario de diabetes ML y devuelve la predicción |
| `/diabetes` | Muestra la página del Sistema Experto de Diabetes |
| `/diabetes/diagnosticar` | Recibe los síntomas marcados y devuelve el diagnóstico |

---

## Parte 2 — El Frontend: HTML + CSS

**¿Qué es el Frontend?**
Es todo lo que el usuario **ve e interactúa** en el navegador: los formularios, los botones, los colores, las animaciones.

**HTML (HyperText Markup Language)**
Es el esqueleto de la página. Define la estructura: dónde va el título, dónde va el formulario, dónde aparece el resultado. Cada página del proyecto tiene su propio archivo `.html`:
- `templates/index.html` → Titanic
- `templates/diabetes_ml.html` → Diabetes (Predicción ML)
- `templates/diabetes.html` → Diabetes (Sistema Experto)

**CSS (Cascading Style Sheets)**
Es el maquillaje de la página. Controla el diseño visual: paleta de colores oscuros, tipografía elegante, animaciones de los íconos, el efecto de brillo en el fondo, etc. En este proyecto el CSS está escrito directamente dentro de cada archivo HTML (en la etiqueta `<style>`), lo que simplifica la estructura de archivos.

**Jinja2 (Motor de plantillas)**
Flask usa Jinja2 para insertar datos dinámicos en el HTML. Por ejemplo, cuando el modelo calcula "Sobrevive - 87.3%", Flask le pasa esa información al HTML y Jinja2 la muestra en la página usando la sintaxis `{{ variable }}`. Sin Jinja2, el HTML es estático y nunca cambiaría.

---

## Parte 3 — Los Modelos de Machine Learning

### 3.1 Red Neuronal con Keras / TensorFlow

**¿Qué es TensorFlow/Keras?**
- **TensorFlow** es la librería de Google para Machine Learning, una de las más potentes del mundo.
- **Keras** es la interfaz simplificada que se usa *encima* de TensorFlow. Es como el volante de un auto de Fórmula 1: hace que el motor complicado sea fácil de controlar.

**¿Qué hace la Red Neuronal?**
Imita (de forma muy simplificada) cómo funciona el cerebro humano. Aprende a reconocer patrones en los datos durante el entrenamiento y los aplica para hacer predicciones sobre casos nuevos.

**Archivos generados:**

| Archivo | Qué contiene |
|---|---|
| `Titanic/mimodelo_completo.h5` | El cerebro entrenado de la Red Neuronal del Titanic |
| `Diabetes_ML/diabetes_modelo_rn.h5` | El cerebro entrenado de la Red Neuronal de Diabetes ML |
| `Titanic/mi_scaler.pkl` | El normalizador de datos del Titanic (para RN) |
| `Diabetes_ML/diabetes_scaler.pkl` | El normalizador de datos de Diabetes ML (para RN) |

> **¿Por qué un scaler?** Las Redes Neuronales son sensibles a la escala de los números. Si una columna tiene valores entre 0 y 1, y otra entre 0 y 500, el modelo se confunde. El `StandardScaler` de scikit-learn transforma todos los datos para que estén en la misma escala antes de entrar a la red.

### 3.2 Árbol de Decisión con scikit-learn

**¿Qué es scikit-learn?**
Es la librería de Machine Learning más usada en Python para modelos "clásicos" (no deep learning). Es simple, rápida y muy bien documentada.

**¿Qué hace el Árbol de Decisión?**
Crea una serie de preguntas de Sí/No basadas en los datos, como un diagrama de flujo médico. No necesita scaler porque no le importa la escala de los números, solo compara valores entre sí.

**Archivos generados:**

| Archivo | Qué contiene |
|---|---|
| `Titanic/modelo_dt_titanic.pkl` | El Random Forest entrenado del Titanic |
| `Diabetes_ML/diabetes_modelo_dt.pkl` | El Random Forest entrenado de Diabetes ML |

### 3.3 Sistema Experto (Diabetes)

**¿Qué es un Sistema Experto?**
A diferencia del Machine Learning, un Sistema Experto **no aprende de datos**: usa reglas lógicas escritas manualmente por un experto humano. Es como codificar el conocimiento de un médico en forma de condiciones `Si... Entonces...`.

El archivo `diabetes_expert.py` contiene un motor de inferencia propio que evalúa los síntomas marcados por el usuario y los compara contra un conjunto de reglas predefinidas (escritas en estilo Prolog, un lenguaje de lógica).

---

## Parte 4 — Persistencia de Modelos: joblib

**¿Qué es joblib?**
Es una librería de Python que sirve para guardar y cargar objetos de Python en archivos (`.pkl`). Su uso es fundamental aquí: una vez que un modelo es entrenado (lo que puede tardar varios minutos), se guarda en un archivo `.pkl` para no tener que reentrenarlo cada vez que se inicia el servidor.

**El flujo completo:**

```
Entrenar → Guardar (.h5 o .pkl) → Cargar en app.py al iniciar → Predecir al instante
```

---

## Flujo Completo de una Predicción (ejemplo: Diabetes ML)

```
1. El usuario abre /diabetes_ml en su navegador
       │
       ▼
2. Flask ejecuta anemia_form() → devuelve diabetes_ml.html vacío
       │
       ▼
3. El usuario llena el formulario y hace clic en "Analizar Paciente"
       │  (HTTP POST a /diabetes_ml/predecir)
       ▼
4. Flask ejecuta anemia_predecir()
   ├── Codifica variables categóricas (gender, smoking_history)
   ├── Lee los datos del formulario (age, BMI, HbA1c, glucosa, etc.)
   ├── Lee qué modelo eligió el usuario (radio button)
   │
   ├── Si eligió "Red Neuronal":
   │     └── Normaliza datos con diabetes_scaler → predict() → Temperature Scaling → probabilidad
   │
   └── Si eligió "Random Forest":
         └── predict_proba() directamente (promedio de 100 árboles) → probabilidad
       │
       ▼
5. Flask devuelve diabetes_ml.html con el resultado ya incluido
       │
       ▼
6. El navegador muestra el diagnóstico con su animación y barra de probabilidad
```

---

## Resumen de Tecnologías

| Tecnología | Categoría | ¿Para qué se usa? |
|---|---|---|
| **Python 3** | Lenguaje | Todo el backend |
| **Flask** | Framework web | Servidor, rutas, lógica |
| **Jinja2** | Motor de plantillas | Insertar datos en HTML |
| **Keras / TensorFlow** | Deep Learning | Redes Neuronales |
| **scikit-learn** | Machine Learning | Árbol de Decisión, StandardScaler |
| **joblib** | Utilidad | Guardar y cargar modelos `.pkl` |
| **NumPy** | Matemáticas | Arrays y operaciones numéricas |
| **Pandas** | Datos | Leer y manipular archivos CSV |
| **HTML5** | Frontend | Estructura de las páginas |
| **CSS3** | Frontend | Diseño visual y animaciones |

---

> [!NOTE]
> Los scripts de entrenamiento (`Titanic/Titanic_RN_Local.py`, `Titanic/Titanic_DT_Local.py`, `Diabetes_ML/Diabetes_RN_Local.py`, `Diabetes_ML/Diabetes_DT_Local.py`) se ejecutan **una sola vez** de forma local para generar los archivos `.h5` y `.pkl`. El servidor Flask **nunca** entrena modelos por sí solo; solo los carga y usa.
