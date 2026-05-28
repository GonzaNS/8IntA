from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Sistema Experto — Diabetes Mellitus Tipo 2
from diabetes_expert import diagnosticar, SINTOMAS, REGLAS_PROLOG

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────
# 1. Cargamos los modelos y el scaler en la memoria al encender el servidor
# ──────────────────────────────────────────────────────────────────

# ▶ Red Neuronal (modelo original — intacto)
print("Cargando Red Neuronal y scaler...")
nn_model = load_model("mimodelo_completo.h5")
scaler = joblib.load("mi_scaler.pkl")
print("Red Neuronal lista.")

# ▶ Árbol de Decisión (se carga si el .pkl ya fue generado)
import os as _os
DT_MODEL_PATH = "modelo_dt_titanic.pkl"
if _os.path.exists(DT_MODEL_PATH):
    dt_model = joblib.load(DT_MODEL_PATH)
    print("Árbol de Decisión cargado.")
else:
    dt_model = None
    print(
        "AVISO: 'modelo_dt_titanic.pkl' no encontrado. "
        "Ejecuta 'Titanic_DT_Local.py' para generarlo antes de usar el Árbol de Decisión."
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
                    resultado="Error: entrena el Árbol de Decisión primero (ejecuta Titanic_DT_Local.py)",
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
