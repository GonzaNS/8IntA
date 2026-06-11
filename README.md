# 🧠 Plataforma de Análisis Predictivo Clínico

Aplicación web de diagnóstico asistido por **Machine Learning** y **Sistema Experto**, construida con Python + Flask. Permite ingresar datos de un paciente y obtener predicciones médicas mediante tres módulos independientes.

---

## 📋 Módulos disponibles

| Módulo | Modelos | ¿Qué predice? |
|---|---|---|
| **Titanic** | Red Neuronal + Árbol de Decisión | Probabilidad de supervivencia |
| **Diabetes ML** | Red Neuronal + Random Forest | Predicción de diabetes usando Machine Learning |
| **Diabetes** | Sistema Experto con reglas lógicas | Nivel de riesgo de Diabetes Tipo 2 |

---

## 🗂️ Estructura del proyecto

```
8IntA/
├── app.py                     # Servidor Flask — punto de entrada principal
├── requeriments.txt           # Dependencias del proyecto
│
├── Titanic/                   # Módulo Titanic
│   ├── Titanic_RN_Local.py    # Entrena la Red Neuronal
│   ├── Titanic_DT_Local.py    # Entrena el Árbol de Decisión
│   ├── titanic-train.csv      # Dataset de entrenamiento
│   ├── titanic-test.csv       # Dataset de prueba
│   ├── mimodelo_completo.h5   # Modelo RN generado (git-ignorado)
│   ├── mi_scaler.pkl          # Normalizador RN generado (git-ignorado)
│   └── modelo_dt_titanic.pkl  # Modelo DT generado (git-ignorado)
│
├── Diabetes_ML/               # Módulo Diabetes ML
│   ├── Diabetes_RN_Local.py   # Entrena la Red Neuronal
│   ├── Diabetes_DT_Local.py   # Entrena el Random Forest
│   ├── diabetes_prediction_dataset.csv # Dataset de entrenamiento
│   ├── diabetes_modelo_rn.h5  # Modelo RN generado (git-ignorado)
│   ├── diabetes_scaler.pkl    # Normalizador RN generado (git-ignorado)
│   └── diabetes_modelo_dt.pkl # Modelo RF generado (git-ignorado)
│
├── Diabetes/                  # Módulo Diabetes
│   └── diabetes_expert.py     # Motor de inferencia del Sistema Experto
│
└── templates/                 # Páginas HTML (Jinja2)
    ├── index.html             # Página Titanic
    ├── diabetes_ml.html       # Página Diabetes ML
    └── diabetes.html          # Página Diabetes
```

> **Nota:** Los archivos `.h5` y `.pkl` están en `.gitignore`. Deben generarse localmente ejecutando los scripts de entrenamiento antes de iniciar el servidor.

---

## ⚙️ Instalación y puesta en marcha

### 1. Clonar el repositorio

```bash
git clone https://github.com/GonzaNS/8IntA.git
cd 8IntA
```

### 2. Crear y activar el entorno virtual

```bash
# Crear el entorno virtual
python -m venv venv

# Activar en Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activar en macOS / Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requeriments.txt
```

> La instalación puede tardar 5–15 minutos por el tamaño de TensorFlow.

### 4. Generar los modelos (una sola vez)

Ejecuta los scripts de entrenamiento desde la **raíz del proyecto**:

```bash
# Módulo Titanic
python Titanic/Titanic_RN_Local.py
python Titanic/Titanic_DT_Local.py

# Módulo Diabetes ML
python Diabetes_ML/Diabetes_RN_Local.py
python Diabetes_ML/Diabetes_DT_Local.py
```

> El Sistema Experto de Diabetes **no requiere entrenamiento** — sus reglas están en `Diabetes/diabetes_expert.py`.

### 5. Iniciar el servidor

```bash
python app.py
```

Abre el navegador en: **http://localhost:5000**

---

## 🌐 Rutas de la aplicación

| URL | Módulo |
|---|---|
| `http://localhost:5000/` | Predicción Titanic |
| `http://localhost:5000/diabetes_ml` | Detección de Diabetes con ML |
| `http://localhost:5000/diabetes` | Diagnóstico de Diabetes con Sistema Experto |

---

## 🛠️ Stack tecnológico

| Tecnología | Uso |
|---|---|
| **Python 3** | Lenguaje principal |
| **Flask** | Servidor web y rutas |
| **Keras / TensorFlow** | Redes Neuronales |
| **scikit-learn** | Random Forest, Árbol de Decisión, StandardScaler |
| **joblib** | Serialización de modelos `.pkl` |
| **NumPy / Pandas** | Manipulación de datos |
| **HTML5 + CSS3** | Frontend con Jinja2 |

---

## ❗ Solución de problemas

**`No module named 'flask'`** — El entorno virtual no está activo:
```bash
.\venv\Scripts\Activate.ps1
pip install -r requeriments.txt
```

**`AVISO: modelo no encontrado`** — Falta ejecutar el script de entrenamiento correspondiente y reiniciar el servidor.

**Error al activar entorno en PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
