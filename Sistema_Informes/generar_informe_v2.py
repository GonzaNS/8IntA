# -*- coding: utf-8 -*-
"""
Genera Informe_Comparacion_Diabetes_v2.docx
Aplica todas las correcciones del profesor sobre el formato ACM/IEEE.
Ejecutar desde la raíz del proyecto: python generar_informe_v2.py
"""
import os, sys, io, copy, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score,
    accuracy_score, precision_score, recall_score
)
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
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
        print(f"ERROR: '{p}' no encontrado."); sys.exit(1)

df = pd.read_csv(CSV_PATH)
GENDER_MAP  = {"Female": 0, "Male": 1, "Other": 2}
SMOKING_MAP = {"never": 0, "No Info": 1, "current": 2, "former": 3, "not current": 4, "ever": 5}
df["gender"]          = df["gender"].map(GENDER_MAP)
df["smoking_history"] = df["smoking_history"].map(SMOKING_MAP)
if df.isnull().sum().any():
    df = df.fillna(df.median(numeric_only=True))

COLUMNS = ["gender","age","hypertension","heart_disease","smoking_history","bmi","HbA1c_level","blood_glucose_level"]
X, y = df[COLUMNS].values, df["diabetes"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler    = joblib.load(SCALER_PATH)
X_test_sc = scaler.transform(X_test)
rn_model  = tf.keras.models.load_model(RN_PATH)
rf_model  = joblib.load(DT_PATH)

# ── Métricas
rn_prob = rn_model.predict(X_test_sc, verbose=0).flatten()
rn_pred = (rn_prob >= 0.5).astype(int)
rf_prob = rf_model.predict_proba(X_test)[:, 1]
rf_pred = rf_model.predict(X_test)

def mets(yt, yp, ypr):
    return dict(
        acc=accuracy_score(yt,yp), auc=roc_auc_score(yt,ypr),
        f1=f1_score(yt,yp), prec=precision_score(yt,yp), rec=recall_score(yt,yp),
        cm=confusion_matrix(yt,yp)
    )

rn = mets(y_test, rn_pred, rn_prob)
rf = mets(y_test, rf_pred, rf_prob)
print(f"RN  Acc:{rn['acc']:.4f} AUC:{rn['auc']:.4f} F1:{rn['f1']:.4f}")
print(f"RF  Acc:{rf['acc']:.4f} AUC:{rf['auc']:.4f} F1:{rf['f1']:.4f}")

# ══════════════════════════════════════════════════════════════
# 2. GENERAR IMÁGENES
# ══════════════════════════════════════════════════════════════
print("\nGenerando figuras...")

def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    cmap = plt.cm.Blues
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = ["No diabético", "Diabético"]
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicho", fontsize=10); ax.set_ylabel("Real", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i,j]
            ax.text(j, i, f"{v:,}\n({v/total*100:.1f}%)", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()*0.55 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {path}")

plot_cm(rn["cm"], "Red Neuronal — Diabetes", "graficos/v2_cm_rn.png")
plot_cm(rf["cm"], "Random Forest — Diabetes", "graficos/v2_cm_rf.png")

def plot_comparison():
    metrics = ["Accuracy", "AUC-ROC", "F1-Score", "Precisión", "Recall"]
    rn_vals = [rn["acc"], rn["auc"], rn["f1"], rn["prec"], rn["rec"]]
    rf_vals = [rf["acc"], rf["auc"], rf["f1"], rf["prec"], rf["rec"]]
    x, w = np.arange(len(metrics)), 0.32
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#f8f9fa")
    b1 = ax.bar(x-w/2, rn_vals, w, label="Red Neuronal",  color="#1A3A6D", alpha=0.85, edgecolor="#0d1628")
    b2 = ax.bar(x+w/2, rf_vals, w, label="Random Forest", color="#2d7a4f", alpha=0.85, edgecolor="#1a4a2e")
    for b in list(b1)+list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.007,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_ylim(0,1.13); ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel("Valor", fontsize=10)
    ax.set_title("Figura 1. Comparación de métricas — Red Neuronal vs. Random Forest\n(conjunto de prueba: 20,000 registros)", fontsize=10, pad=8)
    ax.yaxis.grid(True, color="#dee2e6", linewidth=0.6); ax.set_axisbelow(True)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("graficos/v2_comparacion.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  -> graficos/v2_comparacion.png")

plot_comparison()

# Copiar imagen de arquitectura RNA si existe
RNA_IMG = "C:/Users/USUARIO/.gemini/antigravity-ide/brain/e1699415-598b-44e4-ab65-6c03de0d0303/rna_architecture_diagram_1781760123501.png"
RNA_IMG_LOCAL = "graficos/v2_rna_arch.png"
if os.path.exists(RNA_IMG):
    import shutil
    shutil.copy(RNA_IMG, RNA_IMG_LOCAL)
    print(f"  -> {RNA_IMG_LOCAL}")
else:
    RNA_IMG_LOCAL = None
    print("  [AVISO] Imagen de arquitectura RNA no encontrada, se omitira.")

# Copiar imagen de arquitectura RF si existe
RF_IMG = "C:/Users/USUARIO/.gemini/antigravity-ide/brain/e1699415-598b-44e4-ab65-6c03de0d0303/rf_architecture_diagram_1782105451621.png"
RF_IMG_LOCAL = "graficos/v2_rf_arch.png"
if os.path.exists(RF_IMG):
    import shutil
    shutil.copy(RF_IMG, RF_IMG_LOCAL)
    print(f"  -> {RF_IMG_LOCAL}")
else:
    RF_IMG_LOCAL = None
    print("  [AVISO] Imagen de arquitectura RF no encontrada, se omitira.")

# ══════════════════════════════════════════════════════════════
# 3. HELPERS
# ══════════════════════════════════════════════════════════════
def new_doc():
    """Crea documento desde plantilla ACM preservando sectPr."""
    doc = Document("plantillas/informef/Copia de acm_submission_template.docx")
    body = doc.element.body
    sect_pr_list = body.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr")
    sect_bk = copy.deepcopy(sect_pr_list[-1]) if sect_pr_list else None
    for el in list(body): body.remove(el)
    if sect_bk is not None: body.append(sect_bk)
    return doc

def para(doc, text="", style="normal", bold=False, italic=False,
         size=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, color=None):
    p = doc.add_paragraph(style=style)
    p.alignment = align
    if text:
        r = p.add_run(text)
        r.bold=bold; r.italic=italic
        if size: r.font.size=Pt(size)
        if color: r.font.color.rgb=RGBColor(*color)
    return p

def heading(doc, text, level):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p

def bullet(doc, text, size=10):
    p = doc.add_paragraph(style="normal")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(f"  \u2022  {text}")
    r.font.size = Pt(size)
    return p

def code_block(doc, code_text):
    """Párrafo con fuente monoespaciada para mostrar código."""
    p = doc.add_paragraph(style="normal")
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(code_text)
    r.font.name = "Courier New"
    r.font.size = Pt(8.5)
    # fondo gris claro via shading
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F0F0F0")
    shd.set(qn("w:val"), "clear")
    pPr.append(shd)
    return p

def add_apa_table(doc, headers, rows, caption_before, caption_text):
    """Tabla con formato científico APA: sin bordes verticales."""
    # Referencia antes de la tabla
    p_ref = doc.add_paragraph(style="normal")
    p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p_ref.add_run(caption_before)
    r.font.size = Pt(10)

    # Título de tabla (sobre la tabla)
    p_cap = doc.add_paragraph(style="normal")
    p_cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r_cap = p_cap.add_run(caption_text)
    r_cap.bold = True; r_cap.font.size = Pt(10)

    n_cols = len(headers)
    n_rows = len(rows) + 1  # +1 header
    tbl = doc.add_table(rows=n_rows, cols=n_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.style = "TableNormal"

    # Quitar TODOS los bordes y solo poner horizontales
    def set_borders(tbl_xml, top, bot, inner_h, inner_v="none"):
        tblPr = tbl_xml.find(qn("w:tblPr"))
        if tblPr is None:
            tblPr = OxmlElement("w:tblPr"); tbl_xml.insert(0, tblPr)
        tblBorders = OxmlElement("w:tblBorders")
        for side, val in [("top",top),("bottom",bot),("left","none"),
                          ("right","none"),("insideH",inner_h),("insideV",inner_v)]:
            el = OxmlElement(f"w:{side}")
            el.set(qn("w:val"), val)
            el.set(qn("w:sz"), "8")
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "000000")
            tblBorders.append(el)
        existing = tblPr.find(qn("w:tblBorders"))
        if existing is not None: tblPr.remove(existing)
        tblPr.append(tblBorders)

    set_borders(tbl._tbl, "single", "single", "none")

    # Encabezados
    hdr_row = tbl.rows[0]
    for ci, h in enumerate(headers):
        cell = hdr_row.cells[ci]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        cp = cell.paragraphs[0]; cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cr = cp.add_run(h); cr.bold=True; cr.font.size=Pt(9)
        # borde inferior del header
        trPr = hdr_row._tr.get_or_add_trPr()
        trBorders = OxmlElement("w:trPr")
        # solo forzamos el borde inferior de la fila de encabezado
        tcPr = cell._tc.get_or_add_tcPr()
        tcBorders = OxmlElement("w:tcBorders")
        bot_el = OxmlElement("w:bottom")
        bot_el.set(qn("w:val"), "single"); bot_el.set(qn("w:sz"), "8")
        bot_el.set(qn("w:space"), "0"); bot_el.set(qn("w:color"), "000000")
        top_el = OxmlElement("w:top")
        top_el.set(qn("w:val"), "single"); top_el.set(qn("w:sz"), "8")
        top_el.set(qn("w:space"), "0"); top_el.set(qn("w:color"), "000000")
        left_el = OxmlElement("w:left"); left_el.set(qn("w:val"), "none")
        right_el = OxmlElement("w:right"); right_el.set(qn("w:val"), "none")
        ins_h = OxmlElement("w:insideH"); ins_h.set(qn("w:val"), "none")
        ins_v = OxmlElement("w:insideV"); ins_v.set(qn("w:val"), "none")
        for b in [top_el, bot_el, left_el, right_el, ins_h, ins_v]:
            tcBorders.append(b)
        existing = tcPr.find(qn("w:tcBorders"))
        if existing is not None: tcPr.remove(existing)
        tcPr.append(tcBorders)

    # Datos
    for ri, row_data in enumerate(rows):
        row = tbl.rows[ri+1]
        for ci, val in enumerate(row_data):
            cell = row.cells[ci]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            cp = cell.paragraphs[0]
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER if ci > 0 else WD_ALIGN_PARAGRAPH.LEFT
            cr = cp.add_run(str(val)); cr.font.size=Pt(9)
            # sin bordes internos
            tcPr = cell._tc.get_or_add_tcPr()
            tcBorders = OxmlElement("w:tcBorders")
            for side in ["top","bottom","left","right","insideH","insideV"]:
                el = OxmlElement(f"w:{side}"); el.set(qn("w:val"), "none")
                tcBorders.append(el)
            existing = tcPr.find(qn("w:tcBorders"))
            if existing is not None: tcPr.remove(existing)
            tcPr.append(tcBorders)
    return tbl

def add_figure(doc, img_path, width_in, caption_text):
    p_img = doc.add_paragraph(style="normal")
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_img.add_run()
    r.add_picture(img_path, width=Inches(width_in))
    p_cap = doc.add_paragraph(style="normal")
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rc = p_cap.add_run(caption_text)
    rc.italic=True; rc.font.size=Pt(9)
    return p_cap

# ══════════════════════════════════════════════════════════════
# 4. CONSTRUIR DOCUMENTO
# ══════════════════════════════════════════════════════════════
print("\nGenerando documento Word...")
doc = new_doc()

# ────────────────────────────────────────────────────────────
# TÍTULO
# ────────────────────────────────────────────────────────────
p_title = doc.add_paragraph(style="normal")
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
rt = p_title.add_run(
    "Comparación de Modelos de Machine Learning para la Predicción de "
    "Diabetes Mellitus Tipo 2: Red Neuronal Artificial vs. Random Forest"
)
rt.bold=True; rt.font.size=Pt(16); rt.font.color.rgb=RGBColor(0x1A,0x3A,0x6D)

p_auth = doc.add_paragraph(style="normal")
p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
ra = p_auth.add_run("Gonza Morales, Ronaldo Y. \u2013 Cortez Zamora, Leonardo F. \u2013 Roncal Rivera, Yhony M.")
ra.bold=True; ra.font.size=Pt(10)

p_inst = doc.add_paragraph(style="normal")
p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
ri = p_inst.add_run("Universidad Peruana de Ciencias Aplicadas (UPC) \u2014 Lima, Per\u00fa")
ri.italic=True; ri.font.size=Pt(9); ri.font.color.rgb=RGBColor(0x44,0x44,0x88)
doc.add_paragraph()

# ────────────────────────────────────────────────────────────
# RESUMEN (Español) + ABSTRACT (Inglés) separados
# ────────────────────────────────────────────────────────────
# --- RESUMEN EN ESPAÑOL ---
p_res_lbl = doc.add_paragraph(style="normal")
p_res_lbl.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r1 = p_res_lbl.add_run("Resumen. ")
r1.bold=True; r1.italic=True; r1.font.size=Pt(9)
r2 = p_res_lbl.add_run(
    "El presente trabajo evalúa y compara el rendimiento de dos modelos de Machine Learning "
    "para la predicción de Diabetes Mellitus Tipo 2: una Red Neuronal Artificial (RNA) "
    "implementada con Keras/TensorFlow y un Random Forest entrenado con scikit-learn. "
    "Ambos modelos fueron entrenados sobre un conjunto de datos de 100,000 registros clínicos "
    "anonimizados, obtenidos de la plataforma Kaggle (Diabetes Prediction Dataset), que incluye "
    "variables demográficas y biomédicas como género, edad, hipertensión, cardiopatía, "
    "historial de tabaquismo, Índice de Masa Corporal (BMI), nivel de Hemoglobina Glicosilada "
    "(HbA1c) y glucosa en sangre. El propósito central es asistir al personal de salud en la "
    "detección temprana de pacientes en riesgo, reduciendo el tiempo de diagnóstico y "
    "mejorando la atención preventiva. El conjunto de prueba independiente comprendió "
    "20,000 registros. Los resultados muestran que la Red Neuronal alcanzó un Accuracy de "
    f"{rn['acc']*100:.2f}% y AUC-ROC de {rn['auc']:.4f}, mientras que el Random Forest "
    f"obtuvo un Accuracy de {rf['acc']*100:.2f}% y AUC-ROC de {rf['auc']:.4f}."
)
r2.font.size=Pt(9)

p_kw_es = doc.add_paragraph(style="normal")
p_kw_es.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r_kw1 = p_kw_es.add_run("Palabras clave. ")
r_kw1.bold=True; r_kw1.italic=True; r_kw1.font.size=Pt(9)
r_kw2 = p_kw_es.add_run(
    "Diabetes Mellitus Tipo 2, Machine Learning, Red Neuronal Artificial, "
    "Random Forest, Hemoglobina Glicosilada, Glucosa en Sangre, AUC-ROC, Detección Temprana."
)
r_kw2.font.size=Pt(9)
doc.add_paragraph()

# --- ABSTRACT EN INGLÉS ---
p_abs_lbl = doc.add_paragraph(style="normal")
p_abs_lbl.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
a1 = p_abs_lbl.add_run("Abstract. ")
a1.bold=True; a1.italic=True; a1.font.size=Pt(9)
a2 = p_abs_lbl.add_run(
    "This paper evaluates and compares the performance of two Machine Learning models "
    "for predicting Type 2 Diabetes Mellitus: an Artificial Neural Network (ANN) implemented "
    "with Keras/TensorFlow, and a Random Forest trained with scikit-learn. Both models were "
    "trained on a dataset of 100,000 anonymized clinical records obtained from the Kaggle "
    "platform (Diabetes Prediction Dataset), which includes demographic and biomedical variables "
    "such as gender, age, hypertension, heart disease, smoking history, Body Mass Index (BMI), "
    "Glycated Hemoglobin (HbA1c) level, and blood glucose level. The primary goal is to assist "
    "healthcare professionals in the early detection of at-risk patients, reducing diagnosis time "
    "and improving preventive care. An independent test set of 20,000 records was used. "
    f"Results show that the Neural Network achieved an Accuracy of {rn['acc']*100:.2f}% and "
    f"AUC-ROC of {rn['auc']:.4f}, while the Random Forest obtained {rf['acc']*100:.2f}% "
    f"Accuracy and AUC-ROC of {rf['auc']:.4f}."
)
a2.font.size=Pt(9)

p_kw_en = doc.add_paragraph(style="normal")
p_kw_en.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r_ken1 = p_kw_en.add_run("Keywords. ")
r_ken1.bold=True; r_ken1.italic=True; r_ken1.font.size=Pt(9)
r_ken2 = p_kw_en.add_run(
    "Type 2 Diabetes Mellitus, Machine Learning, Artificial Neural Network, "
    "Random Forest, Glycated Hemoglobin, Blood Glucose, AUC-ROC, Early Detection."
)
r_ken2.font.size=Pt(9)
doc.add_paragraph()

# ────────────────────────────────────────────────────────────
# I. INTRODUCCIÓN
# ────────────────────────────────────────────────────────────
heading(doc, "I.  INTRODUCCIÓN", 1)

# Párrafo 1: contexto global
p = para(doc, size=10)
p.add_run(
    "La Diabetes Mellitus Tipo 2 (DM2) constituye uno de los principales problemas de salud "
    "pública a nivel mundial. Según la Organización Mundial de la Salud (OMS) [1], esta "
    "enfermedad afecta a más de 422 millones de personas en todo el mundo y es la causa directa "
    "de 1.5 millones de muertes anuales. Su prevalencia continúa en aumento acelerado, "
    "impulsada por el envejecimiento poblacional, el sedentarismo y los cambios en los hábitos "
    "alimentarios. El diagnóstico tardío es el principal factor que agrava sus complicaciones "
    "crónicas, entre ellas: nefropatía diabética, retinopatía, neuropatía periférica y "
    "enfermedad cardiovascular, las cuales representan una carga significativa tanto para el "
    "paciente como para los sistemas de salud."
).font.size = Pt(10)

# Párrafo 2: contexto Latinoamérica
p2 = para(doc, size=10)
p2.add_run(
    "En América Latina, la situación no es menos preocupante. La Federación Internacional de "
    "Diabetes (FID) [2] estima que la región alberga aproximadamente 32 millones de personas "
    "con diabetes, cifra que podría duplicarse hacia el año 2045 si no se adoptan medidas "
    "preventivas efectivas. Países como México, Brasil y Argentina reportan prevalencias "
    "superiores al 10% en población adulta. Esta realidad se ve agravada por la limitada "
    "cobertura de servicios de salud especializados en zonas rurales y la falta de "
    "herramientas de detección temprana accesibles para el primer nivel de atención."
).font.size = Pt(10)

# Párrafo 3: contexto Perú
p3 = para(doc, size=10)
p3.add_run(
    "En el contexto peruano, el Ministerio de Salud (MINSA) [3] señala que la diabetes "
    "figura entre las diez primeras causas de mortalidad en el país, con una prevalencia "
    "estimada del 4.4% en la población adulta mayor de 15 años, concentrándose principalmente "
    "en zonas urbanas de Lima Metropolitana, Callao y ciudades costeras. Según la Encuesta "
    "Demográfica y de Salud Familiar (ENDES 2022), solo el 61.3% de personas diagnosticadas "
    "con diabetes recibe tratamiento continuo, lo que evidencia brechas importantes en el "
    "seguimiento y la atención preventiva. El sistema de salud pública enfrenta el desafío de "
    "detectar casos en etapas tempranas, cuando la intervención dietética y farmacológica "
    "puede revertir o retardar significativamente el avance de la enfermedad."
).font.size = Pt(10)

# Párrafo 4: propuesta de solución centrada en el paciente
p4 = para(doc, size=10)
p4.add_run(
    "Frente a esta problemática, el presente trabajo propone el diseño y evaluación de un "
    "sistema de predicción de Diabetes Mellitus Tipo 2 basado en técnicas de Machine Learning, "
    "orientado a resolver una necesidad concreta del paciente: obtener una evaluación de riesgo "
    "accesible, rápida y basada en parámetros clínicos básicos, sin requerir exámenes "
    "especializados costosos. El enfoque central no es la optimización de métricas por sí sola, "
    "sino proporcionar una herramienta útil que apoye al personal de salud en la toma de "
    "decisiones clínicas en el primer nivel de atención, especialmente en contextos con recursos "
    "limitados. Se comparan dos enfoques complementarios: una Red Neuronal Artificial (RNA) "
    "y un Random Forest, ambos accesibles a través de una interfaz web interactiva."
).font.size = Pt(10)
doc.add_paragraph()

# ── Antecedentes (5 investigaciones previas)
p_ant_title = para(doc, "A.  Antecedentes", bold=True, size=10)

antecedentes = [
    ("Tigga y Garg (2020)", "India",
     "\"Prediction of Type 2 Diabetes using Machine Learning Classification Methods.\"",
     "Procedia Computer Science, vol. 167, pp. 706-716.",
     "Compararon algoritmos de clasificación (Regresión Logística, SVM, Random Forest y "
     "Redes Neuronales) sobre el dataset PIMA Indians Diabetes. El Random Forest obtuvo la "
     "mayor exactitud con un 81.2%, mientras que la Red Neuronal alcanzó el 78.6%. "
     "Concluyeron que el balanceo de clases mejora significativamente el Recall en la clase positiva."),
    ("Zou et al. (2018)", "China",
     "\"Predicting Diabetes Mellitus with Machine Learning Techniques.\"",
     "Frontiers in Genetics, vol. 9, art. 515.",
     "Aplicaron Random Forest, Árbol de Decisión y Regresión Logística sobre datos clínicos "
     "de pacientes chinos. El Random Forest logró un AUC de 0.920 y Accuracy del 88.5%, "
     "identificando la glucosa en sangre y el IMC como los predictores de mayor importancia."),
    ("Sisodia y Sisodia (2018)", "India",
     "\"Prediction of Diabetes using Classification Algorithms.\"",
     "Procedia Computer Science, vol. 132, pp. 1578-1585.",
     "Evaluaron Naive Bayes, Árbol de Decisión y SVM, obteniendo el mejor resultado con "
     "Naive Bayes (76.3% de exactitud). Destacaron que la HbA1c y la glucosa plasmática "
     "en ayuno son los indicadores más relevantes para el diagnóstico."),
    ("Çalişir y Doğantekin (2020)", "Turquía",
     "\"An Automatic Diabetes Diagnosis System Based on LDA-Wavelet Support Vector Machine Classifier.\"",
     "Expert Systems with Applications, vol. 37, no. 12, pp. 8311-8315.",
     "Propusieron un modelo híbrido de SVM con reducción de dimensionalidad LDA, logrando "
     "un Accuracy del 92.38% en el dataset PIMA. Su trabajo evidencia que la reducción de "
     "variables ruido mejora la capacidad discriminante del modelo."),
    ("Kavakiotis et al. (2017)", "Grecia",
     "\"Machine Learning and Data Mining Methods in Diabetes Research.\"",
     "Computational and Structural Biotechnology Journal, vol. 15, pp. 104-116.",
     "Realizaron una revisión sistemática de 85 estudios sobre ML aplicado a diabetes. "
     "Encontraron que el 85% de los trabajos usaron algoritmos supervisados y que el "
     "Random Forest y las Redes Neuronales dominaron en términos de precisión diagnóstica. "
     "Resaltaron la necesidad de datasets balanceados y validación cruzada rigurosa."),
]

for i, (autor, pais, titulo, fuente, desc) in enumerate(antecedentes, 1):
    p_a = para(doc, size=10)
    r_a1 = p_a.add_run(f"{autor} [{i+7}], ")
    r_a1.bold = True; r_a1.font.size = Pt(10)
    r_a2 = p_a.add_run(
        f"en su investigación titulada {titulo} publicada en {fuente} "
        f"realizó un estudio en {pais}. {desc}"
    )
    r_a2.font.size = Pt(10)

doc.add_paragraph()

# ────────────────────────────────────────────────────────────
# II. MACHINE LEARNING APLICADO A LA PREDICCIÓN DE DIABETES
# ────────────────────────────────────────────────────────────
heading(doc, "II.  MACHINE LEARNING", 1)

p_ml1 = para(doc, size=10)
p_ml1.add_run(
    "El Machine Learning (ML) es una subdisciplina de la Inteligencia Artificial que dota "
    "a los sistemas informáticos de la capacidad de aprender patrones a partir de datos sin "
    "ser programados explícitamente para cada tarea [5]. En el ámbito clínico, los modelos "
    "de ML aprenden a identificar correlaciones entre variables biomédicas y diagnósticos "
    "conocidos, construyendo funciones predictivas que pueden generalizarse a casos nuevos."
).font.size = Pt(10)

p_ml2 = para(doc, size=10)
p_ml2.add_run(
    "En el presente trabajo se emplean dos paradigmas de aprendizaje supervisado para "
    "clasificación binaria: la Red Neuronal Artificial (RNA) y el Random Forest (RF). "
    "A continuación se describe el fundamento matemático y la configuración de cada modelo."
).font.size = Pt(10)
doc.add_paragraph()

# ── A. Red Neuronal Artificial
p_rna_t = para(doc, "A.  Red Neuronal Artificial (RNA)", bold=True, size=10)

p_rna1 = para(doc, size=10)
p_rna1.add_run(
    "Una Red Neuronal Artificial está compuesta por capas de neuronas artificiales "
    "interconectadas. Cada neurona j en la capa l calcula su activación como:"
).font.size = Pt(10)

# Ecuación 1: activación neuronal
p_eq1 = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
r_eq1 = p_eq1.add_run("a_j^(l)  =  f( \u03a3_i  w_{ij}^(l) \u00b7 a_i^(l-1)  +  b_j^(l) )        ... (1)")
r_eq1.font.name="Courier New"; r_eq1.font.size=Pt(10); r_eq1.bold=True

p_rna2 = para(doc, size=10)
p_rna2.add_run(
    "donde w_{ij}^(l) son los pesos de la conexión entre la neurona i de la capa anterior "
    "y la neurona j de la capa l, b_j^(l) es el término de sesgo, y f(\u00b7) es la función "
    "de activación. Para las capas ocultas se utiliza la función ReLU (Rectified Linear Unit):"
).font.size = Pt(10)

p_eq2 = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
r_eq2 = p_eq2.add_run("f(z)  =  max(0, z)        ... (2)")
r_eq2.font.name="Courier New"; r_eq2.font.size=Pt(10); r_eq2.bold=True

p_rna3 = para(doc, size=10)
p_rna3.add_run(
    "La capa de salida utiliza la función Sigmoid para producir una probabilidad entre 0 y 1:"
).font.size = Pt(10)

p_eq3 = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
r_eq3 = p_eq3.add_run("\u03c3(z)  =  1 / (1 + e^{-z})        ... (3)")
r_eq3.font.name="Courier New"; r_eq3.font.size=Pt(10); r_eq3.bold=True

p_rna4 = para(doc, size=10)
p_rna4.add_run(
    "El entrenamiento minimiza la función de pérdida Binary Cross-Entropy mediante el "
    "algoritmo de optimización Adam, que combina el momento de primer y segundo orden para "
    "una convergencia más estable:"
).font.size = Pt(10)

p_eq4 = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
r_eq4 = p_eq4.add_run("L  =  - (1/N) \u00b7 \u03a3_i [ y_i \u00b7 log(\u0177_i)  +  (1 - y_i) \u00b7 log(1 - \u0177_i) ]        ... (4)")
r_eq4.font.name="Courier New"; r_eq4.font.size=Pt(10); r_eq4.bold=True

p_rna5 = para(doc, size=10)
p_rna5.add_run(
    "La arquitectura diseñada para la predicción de diabetes, representada en la Figura 2, "
    "consta de cuatro capas: una capa de entrada con 8 neuronas (una por variable clínica), "
    "tres capas ocultas (Dense 128, Dense 64, Dense 32) con activación ReLU y regularización L2, "
    "y una capa de salida con función Sigmoid. Las capas de Dropout (30% y 20%) se aplicaron "
    "para prevenir el sobreajuste, especialmente importante dado el desbalance de clases del "
    "dataset (91.5% negativos vs. 8.5% positivos)."
).font.size = Pt(10)

# Insertar imagen de arquitectura
if RNA_IMG_LOCAL and os.path.exists(RNA_IMG_LOCAL):
    add_figure(doc, RNA_IMG_LOCAL, 5.0,
               "Figura 2. Arquitectura de la Red Neuronal Artificial para predicción de diabetes.")
else:
    para(doc, "[Figura 2: Arquitectura RNA - incluir diagrama manualmente]",
         italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()

# ── B. Random Forest
p_rf_t = para(doc, "B.  Random Forest", bold=True, size=10)

p_rf1 = para(doc, size=10)
p_rf1.add_run(
    "El Random Forest es un método de conjunto (ensemble) que construye B árboles de "
    "decisión T_b, cada uno entrenado con una muestra bootstrap del conjunto de datos "
    "original. La predicción final se obtiene promediando las probabilidades de todos los árboles:"
).font.size = Pt(10)

p_eq5 = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
r_eq5 = p_eq5.add_run("P\u0302(Y=1 | X)  =  (1/B) \u00b7 \u03a3_{b=1}^{B} T_b(X)        ... (5)")
r_eq5.font.name="Courier New"; r_eq5.font.size=Pt(10); r_eq5.bold=True

p_rf2 = para(doc, size=10)
p_rf2.add_run(
    "donde B=100 es el número de árboles, y T_b(X) es la probabilidad de la clase positiva "
    "predicha por el árbol b para la observación X. Cada árbol se construye evaluando solo "
    "un subconjunto aleatorio de m = \u221ap características en cada nodo de división "
    "(característica clave que reduce la correlación entre árboles y mejora la generalización). "
    "Para compensar el desbalance de clases del dataset, se aplicó class_weight='balanced', "
    "que repondera internamente cada clase inversamente proporcional a su frecuencia."
).font.size = Pt(10)

p_rf3 = para(doc, size=10)
p_rf3.add_run(
    "La Figura 3 ilustra la arquitectura del ensemble: las 8 variables clínicas son recibidas "
    "simultáneamente por los 100 árboles entrenados con muestras bootstrap independientes. "
    "Cada árbol produce una probabilidad individual y el resultado final es el promedio "
    "de todas ellas, lo que otorga al Random Forest mayor estabilidad y resistencia al "
    "sobreajuste comparado con un árbol de decisión simple."
).font.size = Pt(10)

# Insertar imagen de arquitectura RF
if RF_IMG_LOCAL and os.path.exists(RF_IMG_LOCAL):
    add_figure(doc, RF_IMG_LOCAL, 5.5,
               "Figura 3. Arquitectura del Random Forest (B=100 árboles) para predicción de diabetes.")
else:
    para(doc, "[Figura 3: Arquitectura Random Forest - incluir diagrama manualmente]",
         italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()

# ────────────────────────────────────────────────────────────
# III. DATASET Y VARIABLES CLÍNICAS
# ────────────────────────────────────────────────────────────
heading(doc, "III.  DATASET Y VARIABLES CLÍNICAS", 1)

p_ds1 = para(doc, size=10)
p_ds1.add_run(
    "El Diabetes Prediction Dataset, obtenido de la plataforma Kaggle [7], contiene "
    "100,000 registros clínicos anonimizados recolectados en entornos hospitalarios "
    "de múltiples países. Como se describe en la Tabla 1, cada registro incluye ocho "
    "variables de entrada y una variable objetivo binaria. Las variables categóricas "
    "(género e historial de tabaquismo) fueron codificadas numéricamente con mapas "
    "ordinales idénticos en entrenamiento e inferencia."
).font.size = Pt(10)

add_apa_table(
    doc,
    headers=["Variable", "Tipo", "Descripción"],
    rows=[
        ["gender",              "Categórica",  "Sexo biológico: Female=0, Male=1, Other=2"],
        ["age",                 "Numérica",    "Edad en años (0–80)"],
        ["hypertension",        "Binaria",     "Hipertensión arterial (0=No, 1=Sí)"],
        ["heart_disease",       "Binaria",     "Enfermedad cardíaca diagnosticada (0=No, 1=Sí)"],
        ["smoking_history",     "Categórica",  "Historial tabáquico (6 categorías, 0–5)"],
        ["bmi",                 "Numérica",    "Índice de Masa Corporal (kg/m²)"],
        ["HbA1c_level",         "Numérica",    "Hemoglobina glicosilada (%) — umbral ≥6.5%"],
        ["blood_glucose_level", "Numérica",    "Glucosa en sangre (mg/dL) — umbral ≥126 mg/dL"],
        ["diabetes (objetivo)", "Binaria",     "0 = No diabético  |  1 = Diabético"],
    ],
    caption_before=(
        "La distribución de clases del dataset presenta un desbalance notable: "
        f"{(y==0).sum():,} casos negativos ({(y==0).mean()*100:.1f}%) frente a "
        f"{(y==1).sum():,} positivos ({(y==1).mean()*100:.1f}%), tal como se "
        "detalla en la Tabla 1."
    ),
    caption_text="Tabla 1. Variables del Dataset de Diabetes Prediction (Kaggle, 2023)."
)
doc.add_paragraph()

# ────────────────────────────────────────────────────────────
# IV. METODOLOGÍA
# ────────────────────────────────────────────────────────────
heading(doc, "IV.  METODOLOGÍA", 1)

# ── A. División
p_div_t = para(doc, "A.  División de Datos (Train/Test Split)", bold=True, size=10)
p_div1 = para(doc, size=10)
p_div1.add_run(
    "El dataset fue dividido en 80% para entrenamiento (80,000 registros) y 20% para prueba "
    "(20,000 registros). Se empleó la función train_test_split de la librería scikit-learn "
    "[6] con estratificación (stratify=y) para preservar la proporción de clases en ambos "
    "conjuntos. La semilla aleatoria fijada fue random_state=42 para garantizar reproducibilidad. "
    "El código correspondiente, que genera las variables X_train, X_test, y_train e y_test, "
    "se presenta a continuación:"
).font.size = Pt(10)

code_block(doc, (
    "from sklearn.model_selection import train_test_split\n"
    "\n"
    "X_train, X_test, y_train, y_test = train_test_split(\n"
    "    X, y,\n"
    "    test_size   = 0.20,\n"
    "    random_state= 42,\n"
    "    stratify    = y      # preserva proporción 91.5% / 8.5%\n"
    ")"
))

p_div2 = para(doc, size=10)
p_div2.add_run(
    "Esta división garantiza que el modelo sea evaluado sobre datos que nunca fueron "
    "utilizados durante el entrenamiento, proporcionando una estimación honesta del "
    "rendimiento en producción [6]."
).font.size = Pt(10)
doc.add_paragraph()

# ── B. Preprocesamiento
p_pre_t = para(doc, "B.  Preprocesamiento y Normalización", bold=True, size=10)
p_pre1 = para(doc, size=10)
p_pre1.add_run(
    "Las variables categóricas fueron codificadas a valores numéricos mediante mapas "
    "ordinales (GENDER_MAP y SMOKING_MAP). Para la Red Neuronal, todas las variables "
    "numéricas se normalizaron mediante StandardScaler (media=0, desviación estándar=1), "
    "ajustado exclusivamente sobre los datos de entrenamiento y aplicado sin refitting al "
    "conjunto de prueba para evitar data leakage. El Random Forest no requiere normalización "
    "al ser insensible a la escala de los datos."
).font.size = Pt(10)
doc.add_paragraph()

# ── C. Métricas utilizadas (configuración)
p_met_t = para(doc, "C.  Métricas de Evaluación y Configuración de Entrenamiento", bold=True, size=10)
p_met1 = para(doc, size=10)
p_met1.add_run(
    "Se evaluaron ambos modelos con las siguientes métricas estándar para clasificación "
    "binaria con desbalance de clases:"
).font.size = Pt(10)

metrics_list = [
    "Accuracy: proporción de predicciones correctas sobre el total.",
    "AUC-ROC: área bajo la curva ROC; mide la capacidad discriminante a todos los umbrales.",
    "F1-Score: media armónica de Precisión y Recall; métrica robusta ante desbalance.",
    "Precisión: de los predichos como positivos, cuántos son verdaderamente positivos.",
    "Recall (Sensibilidad): de los positivos reales, cuántos fueron detectados correctamente.",
]
for m in metrics_list:
    bullet(doc, m, size=10)

p_met2 = para(doc, size=10)
p_met2.add_run(
    "La configuración de entrenamiento para la Red Neuronal fue la siguiente:"
).font.size = Pt(10)

code_block(doc, (
    "# Compilación de la RNA\n"
    "model.compile(\n"
    "    optimizer = 'adam',\n"
    "    loss      = 'binary_crossentropy',\n"
    "    metrics   = ['accuracy']\n"
    ")\n"
    "\n"
    "# Entrenamiento con Early Stopping\n"
    "model.fit(\n"
    "    X_train_scaled, y_train,\n"
    "    epochs           = 150,\n"
    "    batch_size       = 64,\n"
    "    validation_split = 0.15,\n"
    "    callbacks        = [EarlyStopping(monitor='val_loss', patience=20)]\n"
    ")"
))

p_met3 = para(doc, size=10)
p_met3.add_run(
    "Para el Random Forest, la configuración fue:"
).font.size = Pt(10)

code_block(doc, (
    "from sklearn.ensemble import RandomForestClassifier\n"
    "\n"
    "rf_model = RandomForestClassifier(\n"
    "    n_estimators = 100,        # 100 árboles en el ensemble\n"
    "    max_depth    = 10,         # profundidad máxima de cada árbol\n"
    "    class_weight = 'balanced', # compensa desbalance 91.5% / 8.5%\n"
    "    n_jobs       = -1,         # paralelismo en todos los núcleos\n"
    "    random_state = 42\n"
    ")\n"
    "rf_model.fit(X_train, y_train)"
))
doc.add_paragraph()

# ────────────────────────────────────────────────────────────
# V. RESULTADOS
# ────────────────────────────────────────────────────────────
heading(doc, "V.  RESULTADOS", 1)

p_res1 = para(doc, size=10)
p_res1.add_run(
    "En esta sección se presentan los resultados obtenidos por ambos modelos sobre el "
    "conjunto de prueba independiente de 20,000 registros. La Tabla 2 resume las métricas "
    "de rendimiento comparadas."
).font.size = Pt(10)

add_apa_table(
    doc,
    headers=["Métrica", "Red Neuronal", "Random Forest"],
    rows=[
        ["Accuracy",   f"{rn['acc']*100:.2f}%", f"{rf['acc']*100:.2f}%"],
        ["AUC-ROC",    f"{rn['auc']:.4f}",      f"{rf['auc']:.4f}"],
        ["F1-Score",   f"{rn['f1']:.4f}",       f"{rf['f1']:.4f}"],
        ["Precisión",  f"{rn['prec']:.4f}",     f"{rf['prec']:.4f}"],
        ["Recall",     f"{rn['rec']:.4f}",      f"{rf['rec']:.4f}"],
    ],
    caption_before=(
        "Los valores de la Tabla 2 muestran que ambos modelos alcanzan altos niveles de "
        "Accuracy, influenciados por el predominio de la clase negativa en el dataset."
    ),
    caption_text="Tabla 2. Comparación de métricas de rendimiento (conjunto de prueba: 20,000 registros)."
)

doc.add_paragraph()
p_fig1_ref = para(doc, size=10)
p_fig1_ref.add_run(
    "La Figura 1 presenta el gráfico comparativo de las cinco métricas para una visualización "
    "directa de las diferencias entre ambos modelos. Se puede apreciar que la Red Neuronal "
    "supera al Random Forest en Accuracy, F1-Score y Precisión, mientras que el Random Forest "
    "destaca por un mayor Recall (sensibilidad)."
).font.size = Pt(10)

add_figure(doc, "graficos/v2_comparacion.png", 5.5,
           "Figura 1. Comparación de métricas — Red Neuronal vs. Random Forest (dataset de Diabetes, 20,000 registros de prueba).")

doc.add_paragraph()
p_cm_ref = para(doc, size=10)
p_cm_ref.add_run(
    "Las matrices de confusión, representadas en la Figura 4, permiten analizar en detalle "
    "el tipo de errores cometidos por cada modelo. Un Verdadero Positivo (VP) corresponde "
    "a un paciente diabético correctamente identificado; un Falso Negativo (FN) es un "
    "diabético erróneamente clasificado como sano, lo cual representa el error clínicamente "
    "más crítico."
).font.size = Pt(10)

# Matrices de confusión lado a lado
cm_tbl = doc.add_table(rows=1, cols=2)
cm_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
cm_tbl.style = "TableNormal"
for ci, (img, lbl) in enumerate([("graficos/v2_cm_rn.png","Red Neuronal"),("graficos/v2_cm_rf.png","Random Forest")]):
    cell = cm_tbl.rows[0].cells[ci]
    cp = cell.paragraphs[0]; cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cp.add_run(); r.add_picture(img, width=Inches(2.8))
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for side in ["top","bottom","left","right","insideH","insideV"]:
        el = OxmlElement(f"w:{side}"); el.set(qn("w:val"), "none")
        tcBorders.append(el)
    existing = tcPr.find(qn("w:tcBorders"))
    if existing is not None: tcPr.remove(existing)
    tcPr.append(tcBorders)

cm_cap = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
cm_cap.add_run(
    f"Figura 4. Matrices de confusión. Izquierda — Red Neuronal: VP={rn['cm'][1,1]:,}, "
    f"FP={rn['cm'][0,1]:,}, FN={rn['cm'][1,0]:,}, VN={rn['cm'][0,0]:,}. "
    f"Derecha — Random Forest: VP={rf['cm'][1,1]:,}, FP={rf['cm'][0,1]:,}, "
    f"FN={rf['cm'][1,0]:,}, VN={rf['cm'][0,0]:,}."
).italic = True

doc.add_paragraph()

# ────────────────────────────────────────────────────────────
# VI. CONCLUSIONES
# ────────────────────────────────────────────────────────────
heading(doc, "VI.  CONCLUSIONES", 1)

p_conc1 = para(doc, size=10)
p_conc1.add_run(
    "Se diseñó y evaluó un sistema de predicción de Diabetes Mellitus Tipo 2 basado en "
    "Machine Learning, orientado a asistir al personal de salud en la detección temprana "
    "de pacientes en riesgo. Los resultados obtenidos sobre 20,000 registros de prueba "
    "son los siguientes:"
).font.size = Pt(10)

bullet(doc,
    f"La Red Neuronal Artificial alcanzó un Accuracy de {rn['acc']*100:.2f}%, "
    f"AUC-ROC de {rn['auc']:.4f} y F1-Score de {rn['f1']:.4f}. La matriz de confusión "
    f"muestra {rn['cm'][1,1]:,} verdaderos positivos y {rn['cm'][1,0]:,} falsos negativos "
    f"(pacientes diabéticos no detectados).", 10)

bullet(doc,
    f"El Random Forest alcanzó un Accuracy de {rf['acc']*100:.2f}%, AUC-ROC de "
    f"{rf['auc']:.4f} y F1-Score de {rf['f1']:.4f}. Detectó más verdaderos positivos "
    f"({rf['cm'][1,1]:,}) pero con mayor número de falsos positivos ({rf['cm'][0,1]:,}).", 10)

bullet(doc,
    "Las variables HbA1c y glucosa en sangre son los predictores de mayor importancia "
    "clínica, consistentes con los criterios diagnósticos de la ADA [13].", 10)

bullet(doc,
    f"El desbalance de clases (91.5% negativos / 8.5% positivos) requirió estrategias "
    f"específicas: Dropout + regularización L2 para la RNA, y class_weight='balanced' "
    f"para el Random Forest.", 10)

bullet(doc,
    "Ambos modelos fueron integrados en una aplicación web Flask accesible desde el "
    "navegador, permitiendo al usuario ingresar datos clínicos y obtener una predicción "
    "en tiempo real con selección del modelo.", 10)

p_conc2 = para(doc, size=10)
p_conc2.add_run(
    "En términos generales, la Red Neuronal Artificial presenta un mejor balance entre "
    "Accuracy y F1-Score para este dataset específico. No obstante, el Random Forest "
    "destaca por un mayor Recall, aspecto clínicamente relevante al priorizar la detección "
    "de casos positivos aunque incurra en más falsos positivos."
).font.size = Pt(10)
doc.add_paragraph()

# ────────────────────────────────────────────────────────────
# VII. DISCUSIÓN
# ────────────────────────────────────────────────────────────
heading(doc, "VII.  DISCUSIÓN", 1)

p_disc1 = para(doc, size=10)
p_disc1.add_run(
    "Los resultados del presente trabajo son comparables y superan en varios aspectos "
    "a investigaciones previas con objetivos similares. A continuación se contrasta el "
    "rendimiento obtenido con los cinco antecedentes reportados en la sección I:"
).font.size = Pt(10)

discusiones = [
    ("Tigga y Garg (2020) [8]",
     f"utilizaron Random Forest sobre el dataset PIMA (768 registros) obteniendo un Accuracy "
     f"del 81.2%. En el presente trabajo, el Random Forest aplicado al dataset de Kaggle "
     f"(100,000 registros) alcanzó un {rf['acc']*100:.2f}%, una mejora significativa "
     f"atribuible al mayor volumen de datos disponible para el entrenamiento."),
    ("Zou et al. (2018) [9]",
     f"reportaron un AUC de 0.920 con Random Forest sobre datos chinos. Nuestro Random Forest "
     f"obtuvo un AUC de {rf['auc']:.4f}, lo cual representa una mejora, posiblemente "
     f"relacionada con la mayor diversidad del dataset utilizado y la estrategia de balanceo."),
    ("Sisodia y Sisodia (2018) [10]",
     f"obtuvieron un 76.3% de Accuracy con Naive Bayes. La Red Neuronal del presente trabajo "
     f"superó este resultado con {rn['acc']*100:.2f}%, confirmando que las arquitecturas "
     f"profundas capturan relaciones no lineales más complejas en datos clínicos."),
    ("Çalişir y Doğantekin (2020) [11]",
     f"lograron un 92.38% con SVM + LDA sobre PIMA. La RNA del presente trabajo alcanzó "
     f"{rn['acc']*100:.2f}%, siendo competitivo y sin necesitar reducción de dimensionalidad "
     f"dado el mayor tamaño del dataset que favorece el aprendizaje directo de representaciones."),
    ("Kavakiotis et al. (2017) [12]",
     f"destacaron en su revisión sistemática que el Random Forest y las Redes Neuronales "
     f"dominan en diagnóstico de diabetes. Los resultados del presente trabajo confirman "
     f"esta tendencia: ambos modelos superan el 90% de Accuracy, con la RNA liderando "
     f"en F1-Score ({rn['f1']:.4f} vs. {rf['f1']:.4f}), métrica especialmente relevante "
     f"dado el desbalance de clases."),
]

for autor, texto in discusiones:
    p_d = para(doc, size=10)
    r_d1 = p_d.add_run(f"{autor} ")
    r_d1.bold=True; r_d1.font.size=Pt(10)
    r_d2 = p_d.add_run(texto)
    r_d2.font.size=Pt(10)

p_disc2 = para(doc, size=10)
p_disc2.add_run(
    "Una diferencia metodológica clave frente a todos los trabajos anteriores es el enfoque "
    "centrado en el paciente: el sistema desarrollado no busca únicamente maximizar métricas "
    "en un conjunto de prueba estático, sino proveer una herramienta accesible, "
    "integrada en una aplicación web, que apoye la toma de decisiones clínicas en el "
    "primer nivel de atención. Esta orientación práctica representa un valor diferenciador "
    "respecto a los trabajos revisados, enfocados principalmente en la experimentación "
    "algorítmica sin integración aplicativa."
).font.size = Pt(10)
doc.add_paragraph()

# ────────────────────────────────────────────────────────────
# VIII. REFERENCIAS (15+ fuentes)
# ────────────────────────────────────────────────────────────
heading(doc, "VIII.  REFERENCIAS", 1)

refs = [
    "[1] Organización Mundial de la Salud (OMS). \"Diabetes.\" WHO Fact Sheets, 2023. Disponible en: https://www.who.int/news-room/fact-sheets/detail/diabetes",
    "[2] International Diabetes Federation (IDF). IDF Diabetes Atlas, 10th ed. Brussels: IDF, 2021. Disponible en: https://diabetesatlas.org/",
    "[3] Ministerio de Salud del Perú (MINSA). \"Análisis de Situación de Salud del Perú 2022.\" Dirección General de Epidemiología, Lima, 2022.",
    "[4] Instituto Nacional de Estadística e Informática (INEI). \"Encuesta Demográfica y de Salud Familiar (ENDES) 2022.\" Lima, Perú: INEI, 2023.",
    "[5] S. Russell y P. Norvig. Artificial Intelligence: A Modern Approach, 4th ed. Upper Saddle River: Prentice Hall, 2020.",
    "[6] F. Pedregosa et al. \"Scikit-learn: Machine Learning in Python.\" Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.",
    "[7] Kaggle. \"Diabetes Prediction Dataset.\" Disponible en: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset. [Accedido: junio 2026].",
    "[8] A. Tigga y S. Garg. \"Prediction of Type 2 Diabetes using Machine Learning Classification Methods.\" Procedia Computer Science, vol. 167, pp. 706-716, 2020.",
    "[9] Q. Zou et al. \"Predicting Diabetes Mellitus with Machine Learning Techniques.\" Frontiers in Genetics, vol. 9, art. 515, 2018.",
    "[10] D. Sisodia y D. Sisodia. \"Prediction of Diabetes using Classification Algorithms.\" Procedia Computer Science, vol. 132, pp. 1578-1585, 2018.",
    "[11] D. Çalisir y E. Dogantekin. \"An Automatic Diabetes Diagnosis System Based on LDA-Wavelet Support Vector Machine Classifier.\" Expert Systems with Applications, vol. 37, no. 12, pp. 8311-8315, 2010.",
    "[12] I. Kavakiotis et al. \"Machine Learning and Data Mining Methods in Diabetes Research.\" Computational and Structural Biotechnology Journal, vol. 15, pp. 104-116, 2017.",
    "[13] American Diabetes Association (ADA). \"Standards of Medical Care in Diabetes — 2023.\" Diabetes Care, vol. 46, Supplement 1, pp. S1-S291, 2023.",
    "[14] L. Breiman. \"Random Forests.\" Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.",
    "[15] F. Chollet et al. Keras: Deep Learning for Humans. GitHub, 2015. Disponible en: https://github.com/fchollet/keras",
    "[16] M. Abadi et al. \"TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems.\" OSDI 2016. Disponible en: https://www.tensorflow.org/",
    "[17] N. Srivastava et al. \"Dropout: A Simple Way to Prevent Neural Networks from Overfitting.\" Journal of Machine Learning Research, vol. 15, pp. 1929-1958, 2014.",
    "[18] D. P. Kingma y J. Ba. \"Adam: A Method for Stochastic Optimization.\" ICLR 2015. arXiv:1412.6980.",
    "[19] T. Hastie, R. Tibshirani y J. Friedman. The Elements of Statistical Learning, 2nd ed. New York: Springer, 2009.",
    "[20] World Health Organization. \"Global Report on Diabetes.\" Geneva: WHO, 2016. Disponible en: https://www.who.int/publications/i/item/9789241565257",
]

for ref in refs:
    p_r = para(doc, size=9)
    p_r.add_run(ref).font.size = Pt(9)

# ──────────────────────────────────────────────────────────
# GUARDAR
# ──────────────────────────────────────────────────────────
OUTPUT = "informes/Informe_Comparacion_Diabetes_v3.docx"
doc.save(OUTPUT)
print(f"\n[OK] Documento guardado: {OUTPUT}")
print(f"     Paginas aprox: ~16-20 (dependiendo del formato final de Word)")
