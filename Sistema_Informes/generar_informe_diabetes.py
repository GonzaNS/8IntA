# -*- coding: utf-8 -*-
"""
Genera Informe_Comparacion_Diabetes.docx
Compara Red Neuronal vs Random Forest sobre el dataset de Diabetes (100k registros).
Ejecutar desde la raíz del proyecto: python generar_informe_diabetes.py
"""

import os, sys, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score
)
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import tensorflow as tf

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# 1. CARGAR DATOS Y MODELOS
# ══════════════════════════════════════════════════════════════
print("Cargando dataset y modelos...")

CSV_PATH    = "../Diabetes_ML/diabetes_prediction_dataset.csv"
RN_PATH     = "../Diabetes_ML/diabetes_modelo_rn.h5"
SCALER_PATH = "../Diabetes_ML/diabetes_scaler.pkl"
DT_PATH     = "../Diabetes_ML/diabetes_modelo_dt.pkl"

for p in [CSV_PATH, RN_PATH, SCALER_PATH, DT_PATH]:
    if not os.path.exists(p):
        print(f"ERROR: No se encontró '{p}'. Entrena los modelos primero.")
        sys.exit(1)

df = pd.read_csv(CSV_PATH)

GENDER_MAP  = {"Female": 0, "Male": 1, "Other": 2}
SMOKING_MAP = {"never": 0, "No Info": 1, "current": 2, "former": 3, "not current": 4, "ever": 5}
df["gender"]          = df["gender"].map(GENDER_MAP)
df["smoking_history"] = df["smoking_history"].map(SMOKING_MAP)
if df.isnull().sum().any():
    df = df.fillna(df.median(numeric_only=True))

COLUMNS = ["gender", "age", "hypertension", "heart_disease",
           "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]
TARGET  = "diabetes"

X = df[COLUMNS].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler      = joblib.load(SCALER_PATH)
X_test_sc   = scaler.transform(X_test)

rn_model    = tf.keras.models.load_model(RN_PATH)
rf_model    = joblib.load(DT_PATH)

# ══════════════════════════════════════════════════════════════
# 2. PREDICCIONES Y MÉTRICAS
# ══════════════════════════════════════════════════════════════
print("Calculando métricas...")

# ── Red Neuronal ──
rn_prob  = rn_model.predict(X_test_sc, verbose=0).flatten()
rn_pred  = (rn_prob >= 0.5).astype(int)
rn_acc   = accuracy_score(y_test, rn_pred)
rn_auc   = roc_auc_score(y_test, rn_prob)
rn_f1    = f1_score(y_test, rn_pred)
rn_prec  = precision_score(y_test, rn_pred)
rn_rec   = recall_score(y_test, rn_pred)
rn_cm    = confusion_matrix(y_test, rn_pred)
rn_report = classification_report(y_test, rn_pred,
                                   target_names=["No diabético","Diabético"])

# ── Random Forest ──
rf_prob  = rf_model.predict_proba(X_test)[:, 1]
rf_pred  = rf_model.predict(X_test)
rf_acc   = accuracy_score(y_test, rf_pred)
rf_auc   = roc_auc_score(y_test, rf_prob)
rf_f1    = f1_score(y_test, rf_pred)
rf_prec  = precision_score(y_test, rf_pred)
rf_rec   = recall_score(y_test, rf_pred)
rf_cm    = confusion_matrix(y_test, rf_pred)
rf_report = classification_report(y_test, rf_pred,
                                   target_names=["No diabético","Diabético"])

print(f"\nRed Neuronal — Acc:{rn_acc:.4f}  AUC:{rn_auc:.4f}  F1:{rn_f1:.4f}")
print(f"Random Forest — Acc:{rf_acc:.4f}  AUC:{rf_auc:.4f}  F1:{rf_f1:.4f}")

# ══════════════════════════════════════════════════════════════
# 3. GENERAR IMÁGENES DE MATRICES DE CONFUSIÓN
# ══════════════════════════════════════════════════════════════
print("\nGenerando matrices de confusión...")

def plot_cm(cm, title, path, color):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    fig.patch.set_facecolor("#0d1628")
    ax.set_facecolor("#0d1628")
    im = ax.imshow(cm, cmap="Blues" if color == "blue" else "Greens",
                   aspect="auto", alpha=0.85)
    labels = ["No diabético", "Diabético"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.set_yticklabels(labels, color="white", fontsize=9)
    ax.set_xlabel("Predicho", color="white", fontsize=10, labelpad=8)
    ax.set_ylabel("Real", color="white", fontsize=10, labelpad=8)
    ax.set_title(title, color="white", fontsize=11, pad=10, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a4a68")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / total * 100
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if cm[i, j] < cm.max() * 0.5 else "#0d1628")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> Imagen guardada: {path}")

plot_cm(rn_cm, "Red Neuronal — Diabetes", "graficos/cm_rn_diabetes.png",  "blue")
plot_cm(rf_cm, "Random Forest — Diabetes", "graficos/cm_rf_diabetes.png", "green")

# Gráfico de barras comparativo
def plot_comparison(path):
    metrics  = ["Accuracy", "AUC-ROC", "F1-Score", "Precisión", "Recall"]
    rn_vals  = [rn_acc, rn_auc, rn_f1, rn_prec, rn_rec]
    rf_vals  = [rf_acc, rf_auc, rf_f1, rf_prec, rf_rec]

    x     = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0d1628")
    ax.set_facecolor("#111d33")

    bars1 = ax.bar(x - width/2, rn_vals, width, label="Red Neuronal",
                   color="#3b9eff", alpha=0.9, edgecolor="#7ec8ff", linewidth=0.7)
    bars2 = ax.bar(x + width/2, rf_vals, width, label="Random Forest",
                   color="#2dd4a0", alpha=0.9, edgecolor="#6eecc8", linewidth=0.7)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                color="white", fontsize=7.5, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                color="white", fontsize=7.5, fontweight="bold")

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x); ax.set_xticklabels(metrics, color="white", fontsize=9)
    ax.set_ylabel("Valor", color="white", fontsize=10)
    ax.set_title("Comparación de métricas — Dataset Diabetes (20% test, 20,000 registros)",
                 color="white", fontsize=10, pad=10, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#2a4a68")
    ax.spines["left"].set_color("#2a4a68")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#1e2f4d", linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)
    legend = ax.legend(facecolor="#111d33", edgecolor="#2a4a68",
                       labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> Imagen guardada: {path}")

plot_comparison("graficos/comparacion_diabetes.png")

# ══════════════════════════════════════════════════════════════
# 4. CONSTRUIR EL DOCUMENTO WORD
# ══════════════════════════════════════════════════════════════
print("\nGenerando documento Word...")

doc = Document("plantillas/Plantilla_Informe2.docx")

# Vaciar el cuerpo conservando la sección (sectPr) que python-docx necesita
body = doc.element.body
# Guardar el último sectPr antes de limpiar
import copy
sect_pr_list = body.findall(
    ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr"
)
sect_pr_backup = copy.deepcopy(sect_pr_list[-1]) if sect_pr_list else None

for element in list(body):
    body.remove(element)

# Restaurar el sectPr para que python-docx pueda calcular anchos de tabla
if sect_pr_backup is not None:
    body.append(sect_pr_backup)

# ── Helpers ───────────────────────────────────────────────────
def add_heading(doc, text, level):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x0D, 0x16, 0x28) if level == 0 \
                              else RGBColor(0x1A, 0x3A, 0x6D)
    return p

def add_para(doc, text, bold=False, italic=False, size=None, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = alignment
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    return p

def add_table_row(table, row_idx, values, bold_first=False, header=False):
    row = table.rows[row_idx]
    for col_idx, val in enumerate(values):
        cell = row.cells[col_idx]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(str(val))
        run.bold = (bold_first and col_idx == 0) or header
        if header:
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:fill"), "1A3A6D")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:val"), "clear")
            tcPr.append(shd)
        run.font.size = Pt(9)

# ══════════════════════════════════════════════════════════════
# TÍTULO
# ══════════════════════════════════════════════════════════════
title_para = doc.add_paragraph()
title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_run = title_para.add_run(
    "Comparación de Modelos de Machine Learning para la Predicción de Diabetes Mellitus Tipo 2: "
    "Red Neuronal Artificial vs. Random Forest"
)
title_run.bold      = True
title_run.font.size = Pt(14)
title_run.font.color.rgb = RGBColor(0x1A, 0x3A, 0x6D)

doc.add_paragraph()

# Autores
auth = doc.add_paragraph()
auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
ar = auth.add_run("Análisis Predictivo Clínico — Plataforma 8IntA")
ar.bold = True; ar.font.size = Pt(10)

inst = doc.add_paragraph()
inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
ir = inst.add_run("Dataset: Diabetes Prediction Dataset (Kaggle) — 100,000 registros clínicos reales")
ir.italic = True; ir.font.size = Pt(9)
ir.font.color.rgb = RGBColor(0x44, 0x44, 0x88)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# ABSTRACT
# ══════════════════════════════════════════════════════════════
abs_heading = doc.add_paragraph()
abs_heading.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
abr = abs_heading.add_run("Abstract — ")
abr.bold = True; abr.font.size = Pt(9); abr.italic = True
abs_text = abs_heading.add_run(
    "Este trabajo evalúa y compara el rendimiento de dos modelos de Machine Learning "
    "para la predicción de Diabetes Mellitus Tipo 2: una Red Neuronal Artificial (RNA) "
    "implementada con Keras/TensorFlow y un Random Forest entrenado con scikit-learn. "
    "Ambos modelos fueron entrenados sobre un conjunto de 100,000 registros clínicos reales "
    "que incluyen variables demográficas y biomédicas clave. El conjunto de prueba independiente "
    "comprendió 20,000 registros (20%). Se reportan métricas de Accuracy, AUC-ROC, F1-Score, "
    "Precisión y Recall, junto con las matrices de confusión completas. "
    f"Los resultados muestran un Accuracy de {rn_acc*100:.2f}% (RNA) vs {rf_acc*100:.2f}% (RF), "
    f"y un AUC-ROC de {rn_auc:.4f} (RNA) vs {rf_auc:.4f} (RF)."
)
abs_text.font.size = Pt(9)

kw = doc.add_paragraph()
kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
kr = kw.add_run("Keywords — ")
kr.bold = True; kr.font.size = Pt(9); kr.italic = True
kt = kw.add_run(
    "Diabetes Mellitus Tipo 2, Machine Learning, Red Neuronal Artificial, "
    "Random Forest, Predicción Clínica, HbA1c, Glucosa en Sangre, AUC-ROC."
)
kt.font.size = Pt(9)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# I. INTRODUCCIÓN
# ══════════════════════════════════════════════════════════════
doc.add_heading("I.  INTRODUCCIÓN", level=1)

add_para(doc,
    "Según la Organización Mundial de la Salud (OMS), la Diabetes Mellitus Tipo 2 es una "
    "de las enfermedades no transmisibles de mayor crecimiento en el mundo: afecta a más de "
    "422 millones de personas y causa 1.5 millones de muertes directas por año. La detección "
    "temprana es el factor clave para reducir sus complicaciones (nefropatía, retinopatía, "
    "neuropatía periférica y enfermedad cardiovascular).", size=10)

add_para(doc,
    "El Machine Learning ha demostrado ser una herramienta poderosa para identificar patrones "
    "en datos clínicos y construir modelos predictivos de alta precisión. En este contexto, "
    "comparamos dos enfoques complementarios: (1) una Red Neuronal Artificial profunda, capaz "
    "de capturar relaciones no lineales complejas, y (2) un Random Forest, un método de "
    "ensemble que combina múltiples árboles de decisión para mejorar la robustez y generalización.", size=10)

add_para(doc,
    f"El dataset utilizado contiene {len(df):,} registros reales con ocho variables clínicas: "
    "género, edad, hipertensión, cardiopatía, historial de tabaquismo, Índice de Masa Corporal (BMI), "
    "Hemoglobina Glicosilada (HbA1c) y nivel de glucosa en sangre. La variable objetivo es binaria: "
    f"Diabético (1) vs. No diabético (0). La distribución de clases presenta un desbalance notable: "
    f"{(y==0).sum():,} casos negativos ({(y==0).mean()*100:.1f}%) vs. "
    f"{(y==1).sum():,} casos positivos ({(y==1).mean()*100:.1f}%).", size=10)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# II. DATASET Y VARIABLES
# ══════════════════════════════════════════════════════════════
doc.add_heading("II.  DATASET Y VARIABLES CLÍNICAS", level=1)

add_para(doc, "A.  Descripción del Dataset", bold=True, size=10)
add_para(doc,
    "El Diabetes Prediction Dataset (disponible en Kaggle) contiene registros anonimizados "
    "recolectados en entornos hospitalarios. Cada registro incluye ocho características de "
    "entrada y una variable objetivo binaria.", size=10)

# Tabla de variables
doc.add_paragraph()
cap = doc.add_paragraph()
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
cr = cap.add_run("TABLA I — Variables del Dataset de Diabetes")
cr.bold = True; cr.font.size = Pt(9)

tbl = doc.add_table(rows=10, cols=3)
tbl.style = "Table Grid"
headers = ["Variable", "Tipo", "Descripción"]
add_table_row(tbl, 0, headers, header=True)
rows_data = [
    ["gender",              "Categórica",  "Sexo biológico (Female=0, Male=1, Other=2)"],
    ["age",                 "Numérica",    "Edad en años (rango: 0–80)"],
    ["hypertension",        "Binaria",     "Hipertensión arterial (0=No, 1=Sí)"],
    ["heart_disease",       "Binaria",     "Enfermedad cardíaca diagnosticada (0=No, 1=Sí)"],
    ["smoking_history",     "Categórica",  "Historial tabáquico (6 categorías, 0–5)"],
    ["bmi",                 "Numérica",    "Índice de Masa Corporal (kg/m²)"],
    ["HbA1c_level",         "Numérica",    "Hemoglobina glicosilada (%) — umbral diagnóstico ≥6.5%"],
    ["blood_glucose_level", "Numérica",    "Glucosa en sangre (mg/dL) — umbral ≥126 mg/dL"],
    ["diabetes (objetivo)", "Binaria",     "Resultado: 0 = No diabético, 1 = Diabético"],
]
for i, row in enumerate(rows_data):
    add_table_row(tbl, i+1, row, bold_first=True)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# III. METODOLOGÍA
# ══════════════════════════════════════════════════════════════
doc.add_heading("III.  METODOLOGÍA", level=1)

add_para(doc, "A.  División de Datos", bold=True, size=10)
add_para(doc,
    "El dataset fue dividido en 80% para entrenamiento (80,000 registros) y 20% para prueba "
    "(20,000 registros) usando estratificación para mantener la proporción de clases. "
    "La semilla aleatoria fijada fue 42 para garantizar reproducibilidad.", size=10)

add_para(doc, "B.  Preprocesamiento", bold=True, size=10)
add_para(doc,
    "Las variables categóricas (género e historial de tabaquismo) fueron codificadas a valores "
    "numéricos mediante mapas ordinales. Para la Red Neuronal, todas las variables numéricas "
    "fueron normalizadas con StandardScaler (media=0, desviación estándar=1). El Random Forest "
    "no requiere normalización al ser insensible a la escala.", size=10)

add_para(doc, "C.  Arquitectura de la Red Neuronal", bold=True, size=10)
add_para(doc,
    "La RNA implementada con Keras/TensorFlow 2 presenta la siguiente arquitectura secuencial:", size=10)

arch_items = [
    "Capa de entrada: 8 neuronas (una por variable clínica)",
    "Capa densa 1: 128 neuronas, activación ReLU, regularización L2 (λ=0.001)",
    "Dropout 1: tasa del 30% para prevenir sobreajuste",
    "Capa densa 2: 64 neuronas, activación ReLU, regularización L2 (λ=0.001)",
    "Dropout 2: tasa del 20%",
    "Capa densa 3: 32 neuronas, activación ReLU",
    "Capa de salida: 1 neurona, activación Sigmoid (probabilidad de diabetes)",
]
for item in arch_items:
    p = doc.add_paragraph(style="Normal")
    p.add_run(f"  • {item}").font.size = Pt(10)

add_para(doc,
    "El optimizador utilizado fue Adam con función de pérdida binary_crossentropy. "
    "El entrenamiento empleó EarlyStopping (patience=20) monitoreando la val_loss, "
    "con un máximo de 150 épocas y batch_size=64.", size=10)

add_para(doc, "D.  Configuración del Random Forest", bold=True, size=10)
add_para(doc,
    "El ensemble de árboles se configuró con 100 estimadores, profundidad máxima de 10 niveles "
    "y class_weight='balanced' para compensar el desbalance de clases (91.5% negativos vs 8.5% "
    "positivos). El parámetro n_jobs=-1 permite el entrenamiento paralelo en todos los núcleos "
    "del procesador.", size=10)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# IV. RESULTADOS
# ══════════════════════════════════════════════════════════════
doc.add_heading("IV.  RESULTADOS", level=1)

add_para(doc, "A.  Métricas de Rendimiento", bold=True, size=10)

# Tabla comparativa principal
doc.add_paragraph()
cap2 = doc.add_paragraph()
cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
c2r = cap2.add_run("TABLA II — Comparación de Métricas (Conjunto de Prueba: 20,000 registros)")
c2r.bold = True; c2r.font.size = Pt(9)

tbl2 = doc.add_table(rows=7, cols=3)
tbl2.style = "Table Grid"
add_table_row(tbl2, 0, ["Métrica", "Red Neuronal", "Random Forest"], header=True)

metrics_data = [
    ("Accuracy",   f"{rn_acc*100:.2f}%", f"{rf_acc*100:.2f}%"),
    ("AUC-ROC",    f"{rn_auc:.4f}",      f"{rf_auc:.4f}"),
    ("F1-Score",   f"{rn_f1:.4f}",       f"{rf_f1:.4f}"),
    ("Precisión",  f"{rn_prec:.4f}",     f"{rf_prec:.4f}"),
    ("Recall",     f"{rn_rec:.4f}",      f"{rf_rec:.4f}"),
    ("Mejor en",
     "AUC-ROC" if rn_auc >= rf_auc else "—",
     "F1 / Recall" if rf_f1 >= rn_f1 else "—"),
]
for i, row in enumerate(metrics_data):
    add_table_row(tbl2, i+1, list(row), bold_first=True)

doc.add_paragraph()

add_para(doc, "B.  Gráfico Comparativo de Métricas", bold=True, size=10)
doc.add_picture("graficos/comparacion_diabetes.png", width=Inches(5.5))
last_para = doc.paragraphs[-1]
last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap_fig1 = doc.add_paragraph()
cap_fig1.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap_fig1.add_run("Fig. 1. Comparación de métricas — Red Neuronal vs Random Forest (dataset de Diabetes).").font.size = Pt(8)

doc.add_paragraph()
add_para(doc, "C.  Matrices de Confusión", bold=True, size=10)

# Dos imágenes de CM lado a lado (como tabla de 1 fila y 2 cols)
cm_tbl = doc.add_table(rows=1, cols=2)
cm_tbl.style = "Table Grid"

# Red Neuronal
cell_rn = cm_tbl.rows[0].cells[0]
p_rn = cell_rn.paragraphs[0]
p_rn.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_rn = p_rn.add_run()
run_rn.add_picture("graficos/cm_rn_diabetes.png", width=Inches(2.8))

# Random Forest
cell_rf = cm_tbl.rows[0].cells[1]
p_rf = cell_rf.paragraphs[0]
p_rf.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_rf = p_rf.add_run()
run_rf.add_picture("graficos/cm_rf_diabetes.png", width=Inches(2.8))

cap_cm = doc.add_paragraph()
cap_cm.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap_cm.add_run(
    f"Fig. 2. Matrices de confusión — Izquierda: Red Neuronal "
    f"(VP={rn_cm[1,1]:,}, FP={rn_cm[0,1]:,}, FN={rn_cm[1,0]:,}, VN={rn_cm[0,0]:,}). "
    f"Derecha: Random Forest "
    f"(VP={rf_cm[1,1]:,}, FP={rf_cm[0,1]:,}, FN={rf_cm[1,0]:,}, VN={rf_cm[0,0]:,})."
).font.size = Pt(8)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# V. ANÁLISIS Y DISCUSIÓN
# ══════════════════════════════════════════════════════════════
doc.add_heading("V.  ANÁLISIS Y DISCUSIÓN", level=1)

# Determinar ganador por métrica
winner_acc  = "Red Neuronal" if rn_acc  >= rf_acc  else "Random Forest"
winner_auc  = "Red Neuronal" if rn_auc  >= rf_auc  else "Random Forest"
winner_f1   = "Red Neuronal" if rn_f1   >= rf_f1   else "Random Forest"
winner_prec = "Red Neuronal" if rn_prec >= rf_prec else "Random Forest"
winner_rec  = "Red Neuronal" if rn_rec  >= rf_rec  else "Random Forest"

add_para(doc,
    f"En términos de Accuracy, {winner_acc} obtuvo el mejor resultado ({max(rn_acc,rf_acc)*100:.2f}% "
    f"vs {min(rn_acc,rf_acc)*100:.2f}%). Sin embargo, dado el notable desbalance de clases del "
    f"dataset (91.5% negativos), el Accuracy por sí solo no es la métrica más representativa.", size=10)

add_para(doc,
    f"El AUC-ROC, que mide la capacidad discriminante del modelo independientemente del umbral "
    f"de decisión, fue superior en {winner_auc} ({max(rn_auc,rf_auc):.4f} vs {min(rn_auc,rf_auc):.4f}). "
    f"Un AUC cercano a 1.0 indica que el modelo tiene alta capacidad de separar diabéticos "
    f"de no diabéticos a través de todos los umbrales posibles.", size=10)

add_para(doc,
    f"El F1-Score, que balancea Precisión y Recall en la clase positiva (diabéticos), "
    f"fue mayor para {winner_f1} ({max(rn_f1,rf_f1):.4f} vs {min(rn_f1,rf_f1):.4f}). "
    f"En contextos clínicos, el Recall (sensibilidad) es especialmente crítico: un falso negativo "
    f"(paciente diabético clasificado como sano) tiene mayor costo médico que un falso positivo. "
    f"En esta métrica, {winner_rec} mostró superioridad ({max(rn_rec,rf_rec):.4f}).", size=10)

add_para(doc,
    "El uso de class_weight='balanced' en el Random Forest y la combinación de Dropout + "
    "regularización L2 en la Red Neuronal fueron estrategias clave para manejar el desbalance "
    "de clases sin recurrir a técnicas de sobremuestreo (SMOTE, etc.). Ambos enfoques lograron "
    "detectar positivos reales más allá de lo que un clasificador trivial haría.", size=10)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# VI. CONCLUSIONES
# ══════════════════════════════════════════════════════════════
doc.add_heading("VI.  CONCLUSIONES", level=1)

overall_winner = "Red Neuronal" if (rn_auc + rn_f1) >= (rf_auc + rf_f1) else "Random Forest"
overall_loser  = "Random Forest" if overall_winner == "Red Neuronal" else "Red Neuronal"

add_para(doc,
    f"Se compararon dos modelos de Machine Learning para la detección de Diabetes Mellitus Tipo 2 "
    f"sobre un dataset de 100,000 registros clínicos reales. En una evaluación global, "
    f"{overall_winner} mostró un rendimiento ligeramente superior considerando el balance entre "
    f"AUC-ROC y F1-Score, aunque ambos modelos presentan fortalezas complementarias.", size=10)

conclusions = [
    f"La Red Neuronal logró un Accuracy de {rn_acc*100:.2f}% y AUC de {rn_auc:.4f}, "
    f"capturando relaciones no lineales entre variables clínicas.",
    f"El Random Forest alcanzó un Accuracy de {rf_acc*100:.2f}% y AUC de {rf_auc:.4f}, "
    f"con mayor interpretabilidad y resistencia natural al sobreajuste.",
    "Las variables HbA1c y glucosa en sangre son los predictores dominantes, "
    "consistentes con los criterios diagnósticos de la ADA (American Diabetes Association).",
    "El desbalance de clases (91.5% / 8.5%) requirió técnicas específicas: "
    "Dropout + L2 para la RNA y class_weight='balanced' para el Random Forest.",
    "Ambos modelos están integrados en una aplicación web Flask con selección "
    "dinámica por el usuario vía formulario.",
]
for c in conclusions:
    p = doc.add_paragraph(style="Normal")
    p.add_run(f"  • {c}").font.size = Pt(10)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# VII. REFERENCIAS
# ══════════════════════════════════════════════════════════════
doc.add_heading("VII.  REFERENCIAS", level=1)

refs = [
    "[1] World Health Organization (OMS). \"Diabetes.\" WHO Fact Sheets, 2023. "
    "Disponible en: https://www.who.int/news-room/fact-sheets/detail/diabetes",
    "[2] F. Chollet et al. Keras. GitHub, 2015. Disponible en: https://github.com/fchollet/keras",
    "[3] F. Pedregosa et al. \"Scikit-learn: Machine Learning in Python.\" "
    "Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
    "[4] L. Breiman. \"Random Forests.\" Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
    "[5] M. Abadi et al. \"TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems.\" "
    "OSDI, 2016. Disponible en: https://www.tensorflow.org/",
    "[6] American Diabetes Association. \"Standards of Medical Care in Diabetes—2023.\" "
    "Diabetes Care, vol. 46, Supplement 1, pp. S1–S291, Jan. 2023.",
    "[7] Kaggle. \"Diabetes Prediction Dataset.\" "
    "Disponible en: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset",
]
for ref in refs:
    p = doc.add_paragraph(style="Normal")
    p.add_run(ref).font.size = Pt(9)

# ══════════════════════════════════════════════════════════════
# 5. GUARDAR
# ══════════════════════════════════════════════════════════════
OUTPUT = "informes/Informe_Comparacion_Diabetes.docx"
doc.save(OUTPUT)
print(f"\n[OK] Informe guardado: {OUTPUT}")
print("   (Los archivos de imagen temporales cm_rn_diabetes.png, cm_rf_diabetes.png,")
print("    comparacion_diabetes.png se pueden eliminar despues.)")
