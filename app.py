from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Sistema Experto — Diabetes Mellitus Tipo 2
from Diabetes.diabetes_expert import diagnosticar, SINTOMAS, REGLAS_PROLOG

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────
# 1. Cargamos los modelos y el scaler en la memoria al encender el servidor
# ──────────────────────────────────────────────────────────────────

# ▶ Red Neuronal (modelo original — intacto)
print("Cargando Red Neuronal y scaler...")
nn_model = load_model("Titanic/mimodelo_completo.h5")
scaler = joblib.load("Titanic/mi_scaler.pkl")
print("Red Neuronal lista.")

# ▶ Árbol de Decisión Titanic (se carga si el .pkl ya fue generado)
import os as _os
DT_MODEL_PATH = "Titanic/modelo_dt_titanic.pkl"
if _os.path.exists(DT_MODEL_PATH):
    dt_model = joblib.load(DT_MODEL_PATH)
    print("Árbol de Decisión Titanic cargado.")
else:
    dt_model = None
    print(
        "AVISO: 'Titanic/modelo_dt_titanic.pkl' no encontrado. "
        "Ejecuta 'Titanic/Titanic_DT_Local.py' para generarlo antes de usar el Árbol de Decisión."
    )

# ══════════════════════════════════════════════════════════════
#  MODELOS DE DIABETES (Red Neuronal + Random Forest)
# ══════════════════════════════════════════════════════════════

# ▶ Red Neuronal — Diabetes
DIABETES_ML_RN_PATH     = "Diabetes_ML/diabetes_modelo_rn.h5"
DIABETES_ML_SCALER_PATH = "Diabetes_ML/diabetes_scaler.pkl"
if _os.path.exists(DIABETES_ML_RN_PATH) and _os.path.exists(DIABETES_ML_SCALER_PATH):
    anemia_nn_model = load_model(DIABETES_ML_RN_PATH)
    anemia_scaler   = joblib.load(DIABETES_ML_SCALER_PATH)
    print("Red Neuronal de Diabetes (ML) cargada.")
else:
    anemia_nn_model = None
    anemia_scaler   = None
    print(
        "AVISO: Modelos de diabetes ML no encontrados. "
        "Ejecuta 'Diabetes_ML/Diabetes_RN_Local.py' para generarlos."
    )

# ▶ Random Forest — Diabetes
DIABETES_ML_DT_PATH = "Diabetes_ML/diabetes_modelo_dt.pkl"
if _os.path.exists(DIABETES_ML_DT_PATH):
    anemia_dt_model = joblib.load(DIABETES_ML_DT_PATH)
    print("Random Forest de Diabetes (ML) cargado.")
else:
    anemia_dt_model = None
    print(
        "AVISO: 'Diabetes_ML/diabetes_modelo_dt.pkl' no encontrado. "
        "Ejecuta 'Diabetes_ML/Diabetes_DT_Local.py' para generarlo."
    )

print("Servidor listo para recibir peticiones.")

# 2. Ruta principal: Solo muestra la página web vacía


@app.route("/")
def home():
    return render_template("index.html")

# 3. Ruta de predicción: Recibe los datos del formulario, calcula y devuelve la página


@app.route("/predecir", methods=["POST"])
def predecir():
    if request.method == "POST":
        # Extraer los datos enviados desde el formulario HTML
        fare = float(request.form["fare"])
        pclass = int(request.form["pclass"])
        gender = int(request.form["gender"])
        age = float(request.form["age"])
        sibsp = int(request.form["sibsp"])

        # Modelo elegido por el usuario (por defecto: Red Neuronal)
        modelo_elegido = request.form.get("modelo", "red_neuronal")

        # ORDEN ESTRICTO: Debe ser el mismo con el que se entrenó cada modelo
        # columns = ["Fare", "Pclass", "Gender", "Age", "SibSp"]
        datos_entrada = np.array([[fare, pclass, gender, age, sibsp]])

        # ── Rama 1: Red Neuronal ──────────────────────────────────────────────
        if modelo_elegido == "red_neuronal":
            datos_escalados = scaler.transform(datos_entrada)
            prediccion = nn_model.predict(datos_escalados)
            probabilidad = float(prediccion[0][0]) * 100  # ya en porcentaje
            estado = "Sobrevive" if probabilidad >= 50 else "Fallece"

        # ── Rama 2: Árbol de Decisión ────────────────────────────────────────
        elif modelo_elegido == "arbol_decision":
            if dt_model is None:
                # El .pkl aún no ha sido generado
                return render_template(
                    "index.html",
                    resultado="Error: entrena el Árbol de Decisión primero (ejecuta Titanic/Titanic_DT_Local.py)",
                    probabilidad=0.0,
                    modelo_usado=modelo_elegido,
                )
            # El Árbol de Decisión NO necesita scaler
            proba = dt_model.predict_proba(datos_entrada)[0]  # [prob_fallece, prob_sobrevive]
            probabilidad = float(proba[1]) * 100  # % de sobrevivir
            estado = "Sobrevive" if probabilidad >= 50 else "Fallece"

        else:
            estado = "Modelo desconocido"
            probabilidad = 0.0

        # Volvemos a cargar index.html con las variables de respuesta
        return render_template(
            "index.html",
            resultado=estado,
            probabilidad=probabilidad,
            modelo_usado=modelo_elegido,
        )


# ══════════════════════════════════════════════════════════════
#  ANÁLISIS PREDICTIVO — DIABETES ML (Red Neuronal + Random Forest)
# ══════════════════════════════════════════════════════════════

# 6. Ruta principal: muestra el formulario vacío
@app.route("/diabetes_ml")
def anemia_form():
    return render_template("diabetes_ml.html")


# 7. Ruta de predicción de diabetes ML
@app.route("/diabetes_ml/predecir", methods=["POST"])
def anemia_predecir():
    if request.method == "POST":
        # ── Mapas de codificación (deben ser idénticos a los usados en entrenamiento) ──
        GENDER_MAP  = {"Female": 0, "Male": 1, "Other": 2}
        SMOKING_MAP = {
            "never": 0, "No Info": 1, "current": 2,
            "former": 3, "not current": 4, "ever": 5,
        }

        # Extraer datos del formulario y codificar categóricos
        gender              = GENDER_MAP.get(request.form["gender"], 0)
        age                 = float(request.form["age"])
        hypertension        = int(request.form["hypertension"])
        heart_disease       = int(request.form["heart_disease"])
        smoking_history     = SMOKING_MAP.get(request.form["smoking_history"], 1)
        bmi                 = float(request.form["bmi"])
        hba1c_level         = float(request.form["hba1c_level"])
        blood_glucose_level = float(request.form["blood_glucose_level"])

        # Modelo elegido por el usuario
        modelo_elegido = request.form.get("modelo", "red_neuronal")

        # ORDEN ESTRICTO: debe coincidir con COLUMNS en los scripts de entrenamiento
        datos_entrada = np.array([[
            gender, age, hypertension, heart_disease,
            smoking_history, bmi, hba1c_level, blood_glucose_level
        ]])

        # ── Rama 1: Red Neuronal ─────────────────────────────────────────
        if modelo_elegido == "red_neuronal":
            if anemia_nn_model is None or anemia_scaler is None:
                return render_template(
                    "diabetes_ml.html",
                    resultado="Error: ejecuta Diabetes_ML/Diabetes_RN_Local.py para generar los modelos.",
                    probabilidad=0.0,
                    modelo_usado=modelo_elegido,
                )
            datos_escalados = anemia_scaler.transform(datos_entrada)
            prediccion  = anemia_nn_model.predict(datos_escalados)
            prob_raw = float(prediccion[0][0])
            # Temperature scaling: suaviza predicciones extremas (0% o 100%)
            # sin reentrenar el modelo. T>1 produce valores más intermedios.
            TEMPERATURE = 3.5
            logit = np.log(prob_raw / (1 - prob_raw + 1e-8))
            prob_calibrada = 1 / (1 + np.exp(-logit / TEMPERATURE))
            probabilidad = prob_calibrada * 100
            estado = "Diabético" if probabilidad >= 50 else "No diabético"

        # ── Rama 2: Random Forest ─────────────────────────────────────
        elif modelo_elegido == "arbol_decision":
            if anemia_dt_model is None:
                return render_template(
                    "diabetes_ml.html",
                    resultado="Error: ejecuta Diabetes_ML/Diabetes_DT_Local.py para generar el modelo.",
                    probabilidad=0.0,
                    modelo_usado=modelo_elegido,
                )
            proba = anemia_dt_model.predict_proba(datos_entrada)[0]  # [prob_no, prob_diabetes]
            probabilidad = float(proba[1]) * 100
            estado = "Diabético" if probabilidad >= 50 else "No diabético"

        else:
            estado = "Modelo desconocido"
            probabilidad = 0.0

        return render_template(
            "diabetes_ml.html",
            resultado=estado,
            probabilidad=probabilidad,
            modelo_usado=modelo_elegido,
        )


# ══════════════════════════════════════════════════════════════
#  SISTEMA EXPERTO — DIABETES MELLITUS TIPO 2
# ══════════════════════════════════════════════════════════════

# 4. Ruta principal del sistema experto: muestra el formulario vacío
@app.route("/diabetes")
def diabetes_form():
    return render_template(
        "diabetes.html",
        sintomas_marcados=[],
        resultado=None,
        reglas_prolog=REGLAS_PROLOG,
    )


# 5. Ruta de diagnóstico: recibe los síntomas y ejecuta el motor de inferencia
@app.route("/diabetes/diagnosticar", methods=["POST"])
def diabetes_diagnosticar():
    if request.method == "POST":
        # Los checkboxes con name="sintomas" envían una lista de valores
        # Solo los checkboxes MARCADOS aparecen en el POST.
        # request.form.getlist() devuelve [] si ninguno fue marcado.
        sintomas_presentes = request.form.getlist("sintomas")

        # Ejecutar el motor de inferencia del sistema experto
        resultado = diagnosticar(sintomas_presentes)

        # Re-renderizar la página con el resultado y los síntomas marcados
        return render_template(
            "diabetes.html",
            sintomas_marcados=sintomas_presentes,
            resultado=resultado,
            reglas_prolog=REGLAS_PROLOG,
        )


if __name__ == "__main__":
    # Iniciar el servidor local en el puerto 5000
    app.run(debug=True)
