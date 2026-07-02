# Guía de Instalación y Ejecución del Proyecto

Esta guía explica, paso a paso y desde cero, cómo instalar todas las dependencias y poner en marcha la aplicación web de análisis predictivo clínico. Está escrita para que cualquier persona pueda seguirla sin experiencia previa.

---

## Requisitos previos

Antes de empezar, necesitas tener instalado en tu computadora:

| Requisito | Versión mínima | ¿Cómo verificar? |
|---|---|---|
| **Python** | 3.10 o superior | Abre una terminal y escribe `python --version` |
| **pip** | Incluido con Python | Escribe `pip --version` |
| **Git** (opcional) | Cualquiera | Para clonar el repositorio |

> [!IMPORTANT]
> Este proyecto fue desarrollado y probado en **Windows**. Si usas macOS o Linux, los comandos son casi idénticos pero usa `python3` en lugar de `python` donde corresponda.

---

## Paso 1 — Obtener el proyecto

### Opción A: Clonar con Git
Abre una terminal (PowerShell o CMD) en la carpeta donde quieres guardar el proyecto y ejecuta:

```bash
git clone https://github.com/GonzaNS/8IntA.git
cd 8IntA
```

### Opción B: Descarga manual
1. Descarga el ZIP del repositorio desde GitHub.
2. Extráelo en la carpeta de tu elección.
3. Abre una terminal **dentro de esa carpeta**.

> [!TIP]
> En Windows, puedes abrir una terminal directamente en una carpeta haciendo clic derecho dentro de ella y seleccionando "Abrir en Terminal" o "Abrir PowerShell aquí".

---

## Paso 2 — Crear un Entorno Virtual (recomendado)

Un entorno virtual es un espacio aislado donde se instalan las librerías del proyecto **sin afectar** otras instalaciones de Python en tu computadora. Es como una caja de herramientas exclusiva para este proyecto.

```bash
# Crear el entorno virtual
python -m venv venv

# Activarlo en Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activarlo en Windows (CMD)
venv\Scripts\activate.bat

# Activarlo en macOS / Linux
source venv/bin/activate
```

Cuando el entorno está activo, verás `(venv)` al inicio de la línea de la terminal:
```
(venv) PS C:\...\8IntA>
```

Para desactivarlo cuando termines de trabajar:
```bash
deactivate
```

---

## Paso 3 — Instalar las Dependencias

Con el entorno virtual activado, instala todas las librerías del proyecto con un solo comando:

```bash
pip install -r requeriments.txt
```

Este comando lee el archivo `requeriments.txt` e instala las versiones exactas con las que el proyecto fue desarrollado y probado. El proceso puede tardar **entre 5 y 15 minutos** dependiendo de tu conexión a internet, ya que TensorFlow es una librería grande.

### Dependencias principales del proyecto

| Librería | Versión | Para qué se usa |
|---|---|---|
| `Flask` | 3.1.3 | Servidor web |
| `tensorflow` | 2.21.0 | Red Neuronal (motor) |
| `keras` | 3.14.1 | Red Neuronal (interfaz simplificada) |
| `scikit-learn` | 1.8.0 | Árbol de Decisión y StandardScaler |
| `joblib` | 1.5.3 | Guardar y cargar modelos `.pkl` |
| `numpy` | 2.4.4 | Operaciones numéricas con arrays |
| `pandas` | 3.0.3 | Leer y manipular archivos CSV |
| `matplotlib` | 3.10.9 | Generar gráficos de la Matriz de Confusión |
| `Jinja2` | 3.1.6 | Motor de plantillas HTML (incluido con Flask) |
| `h5py` | 3.14.0 | Leer/escribir archivos `.h5` (modelos Keras) |

---

## Paso 4 — Entrenar los Modelos

Los archivos `.h5` (Redes Neuronales) y `.pkl` (Árboles de Decisión y Scalers) **no están incluidos** en el repositorio. Debes generarlos ejecutando los scripts de entrenamiento. Esto solo necesitas hacerlo **una vez**.

> [!WARNING]
> Ejecuta cada script **desde la raíz del proyecto** (la carpeta `8IntA/`), no desde dentro de ninguna subcarpeta. De lo contrario, Python no encontrará los archivos CSV.

### 4.1 — Modelos del Titanic

```bash
# Red Neuronal (genera: mimodelo_completo.h5 + mi_scaler.pkl)
python Titanic_RN_Local.py

# Árbol de Decisión (genera: modelo_dt_titanic.pkl)
python Titanic_DT_Local.py
```

**Tiempo estimado:**
- `Titanic_RN_Local.py`: 1–3 minutos (400 épocas de entrenamiento)
- `Titanic_DT_Local.py`: segundos

### 4.2 — Modelos de Diabetes ML

> [!WARNING]
> Los scripts de Diabetes ML deben ejecutarse **desde dentro de la carpeta `Diabetes_ML/`**, ya que el CSV está en esa carpeta.

```bash
# Ir a la carpeta
cd Diabetes_ML

# Red Neuronal (genera: diabetes_modelo_rn.h5 + diabetes_scaler.pkl)
python Diabetes_RN_Local.py

# Random Forest (genera: diabetes_modelo_dt.pkl)
python Diabetes_DT_Local.py

# Volver a la raíz del proyecto
cd ..
```

**Tiempo estimado:**
- `Diabetes_RN_Local.py`: 3–8 minutos (150 épocas, dataset de 100,000 registros)
- `Diabetes_DT_Local.py`: 10–30 segundos (100 árboles, paralelizado)

> [!NOTE]
> El Sistema Experto de Diabetes **no requiere entrenamiento**. Sus reglas están escritas directamente en `diabetes_expert.py` y están listas para usarse.

### Verificación: archivos esperados después del entrenamiento

Al terminar los 4 scripts, deberías ver estos archivos en sus respectivas carpetas:

```
8IntA/
├── Titanic/
│   ├── mimodelo_completo.h5       ✅ Red Neuronal Titanic
│   ├── mi_scaler.pkl              ✅ Normalizador Titanic (RN)
│   └── modelo_dt_titanic.pkl      ✅ Random Forest Titanic
├── Diabetes_ML/
│   ├── diabetes_modelo_rn.h5      ✅ Red Neuronal Diabetes ML
│   ├── diabetes_scaler.pkl        ✅ Normalizador Diabetes ML (RN)
│   └── diabetes_modelo_dt.pkl     ✅ Random Forest Diabetes ML
└── ... (resto de archivos)
```

---

## Paso 5 — Iniciar el Servidor Flask

Con todos los modelos generados, arranca la aplicación web:

```bash
python app.py
```

Si todo está correcto, verás en la terminal una salida similar a esta:

```
Cargando Red Neuronal y scaler...
Red Neuronal lista.
Árbol de Decisión Titanic cargado.
Red Neuronal de Diabetes (ML) cargada.
Random Forest de Diabetes (ML) cargado.
Servidor listo para recibir peticiones.
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

---

## Paso 6 — Usar la Aplicación

Abre tu navegador web (Chrome, Firefox, Edge, etc.) y visita:

```
http://localhost:5000
```

Desde ahí puedes navegar a los tres módulos:

| URL | Módulo |
|---|---|
| `http://localhost:5000/` | Predicción de supervivencia Titanic |
| `http://localhost:5000/diabetes_ml` | Predicción de Diabetes (ML) |
| `http://localhost:5000/diabetes` | Diagnóstico de Diabetes (Sistema Experto) |

Para detener el servidor, vuelve a la terminal y presiona `Ctrl + C`.

---

## Solución de Problemas Frecuentes

### ❌ "No module named 'flask'" o similar
El entorno virtual no está activo o las dependencias no se instalaron correctamente.
```bash
# Activa el entorno virtual
.\venv\Scripts\Activate.ps1
# Vuelve a instalar dependencias
pip install -r requeriments.txt
```

### ❌ "No se encontró 'diabetes_prediction_dataset.csv'" al ejecutar scripts de entrenamiento
Estás ejecutando el script desde la carpeta equivocada.
```bash
# Asegúrate de estar dentro de la carpeta Diabetes_ML
cd C:\ruta\hacia\8IntA\Diabetes_ML
python Diabetes_RN_Local.py
```

### ❌ "AVISO: 'modelo_dt_titanic.pkl' no encontrado" en el servidor
El servidor inicia igual, pero si intentas usar el Árbol de Decisión en la web aparecerá un error. Solución: ejecuta el script correspondiente.
```bash
python Titanic_DT_Local.py
```

### ❌ Error al activar el entorno virtual en PowerShell
Windows a veces bloquea la ejecución de scripts. Ejecuta este comando una vez:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Luego vuelve a intentar activar el entorno virtual.

### ❌ La página web muestra "Error: ejecuta el script primero"
Significa que el archivo `.h5` o `.pkl` de ese módulo no existe todavía. Ejecuta el script de entrenamiento correspondiente y reinicia el servidor.

---

## Orden recomendado para empezar desde cero

```
1. python -m venv venv
2. .\venv\Scripts\Activate.ps1
3. pip install -r requeriments.txt
4. python Titanic\Titanic_RN_Local.py
5. python Titanic\Titanic_DT_Local.py
6. cd Diabetes_ML
7. python Diabetes_RN_Local.py
8. python Diabetes_DT_Local.py
9. cd ..
10. python app.py
11. Abrir http://localhost:5000 en el navegador
```
