from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Sistema Experto — Diabetes Mellitus Tipo 2
from diabetes_expert import diagnosticar, SINTOMAS, REGLAS_PROLOG

app = Flask(__name__)

# 1. Cargamos el modelo y el scaler en la memoria al encender el servidor
print("Cargando modelo y scaler...")
model = load_model("mimodelo_completo.h5")
scaler = joblib.load("mi_scaler.pkl")
print("Listo para recibir peticiones.")

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

        # ORDEN ESTRICTO: Debe ser el mismo con el que entrenaste el modelo
        # columns = ["Fare", "Pclass", "Gender", "Age", "SibSp"]
        datos_entrada = np.array([[fare, pclass, gender, age, sibsp]])

        # Transformar los datos con el Scaler
        datos_escalados = scaler.transform(datos_entrada)

        # Realizar la predicción
        prediccion = model.predict(datos_escalados)
        probabilidad = float(prediccion[0][0])  # Obtener el número decimal

        # Lógica de respuesta
        estado = "Sobrevive" if probabilidad >= 0.5 else "Fallece"

        # Volvemos a cargar index.html, pero ahora pasándole las variables de respuesta
        return render_template("index.html", resultado=estado, probabilidad=probabilidad*100)


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
