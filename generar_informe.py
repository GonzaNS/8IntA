# -*- coding: utf-8 -*-
"""
Genera el informe de comparación entre Red Neuronal y Random Forest
para la detección de Anemia, usando la Plantilla_Informe2.docx como base.
"""

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

# ══════════════════════════════════════════════════════════════════════
#  MÉTRICAS REALES OBTENIDAS DE LOS MODELOS
# ══════════════════════════════════════════════════════════════════════
METRICAS = {
    "rn": {
        "acc": 1.0000, "auc": 1.0000,
        "p0": 1.0000, "r0": 1.0000, "f0": 1.0000,
        "p1": 1.0000, "r1": 1.0000, "f1": 1.0000,
        "cm": [[161, 0], [0, 124]],
    },
    "rf": {
        "acc": 1.0000, "auc": 1.0000,
        "p0": 1.0000, "r0": 1.0000, "f0": 1.0000,
        "p1": 1.0000, "r1": 1.0000, "f1": 1.0000,
        "cm": [[161, 0], [0, 124]],
    },
    "dataset": {
        "total": 1421, "train": 1136, "test": 285,
        "positivos": 620, "negativos": 801,
    }
}

# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def set_cell_bg(cell, hex_color):
    """Rellena el fondo de una celda con un color hexadecimal (ej: '1F3864')."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def set_cell_borders(table):
    """Añade bordes finos a todas las celdas de una tabla."""
    for row in table.rows:
        for cell in row.cells:
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            tcBorders = OxmlElement('w:tcBorders')
            for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
                border = OxmlElement(f'w:{side}')
                border.set(qn('w:val'), 'single')
                border.set(qn('w:sz'), '4')
                border.set(qn('w:space'), '0')
                border.set(qn('w:color'), '9DC3E6')
                tcBorders.append(border)
            tcPr.append(tcBorders)

def fmt(value, decimals=4):
    return f"{value:.{decimals}f}"

def pct(value):
    return f"{value*100:.2f}%"


# ══════════════════════════════════════════════════════════════════════
#  ABRIR PLANTILLA Y LIMPIAR CONTENIDO DE EJEMPLO
# ══════════════════════════════════════════════════════════════════════
doc = Document('Plantilla_Informe2.docx')

# Eliminar todos los párrafos existentes (preservamos los estilos)
for p in doc.paragraphs:
    p._element.getparent().remove(p._element)

# Eliminar todas las tablas existentes
for t in doc.tables:
    t._element.getparent().remove(t._element)

body = doc.element.body


def add_para(text, style='Normal', bold=False, italic=False,
             align=WD_ALIGN_PARAGRAPH.JUSTIFY, size_pt=None, color=None,
             space_after=None, space_before=None):
    p = doc.add_paragraph(style=style)
    p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after  = Pt(space_after)
    if space_before is not None:
        p.paragraph_format.space_before = Pt(space_before)
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    if size_pt:
        run.font.size = Pt(size_pt)
    if color:
        run.font.color.rgb = RGBColor(*bytes.fromhex(color))
    return p


# ══════════════════════════════════════════════════════════════════════
#  TÍTULO
# ══════════════════════════════════════════════════════════════════════
add_para(
    "Comparación de Red Neuronal Artificial y Random Forest para la "
    "Detección de Anemia: Un Estudio Empírico sobre Separabilidad Clínica",
    style='Default Paragraph Font1',
    align=WD_ALIGN_PARAGRAPH.CENTER,
    space_after=6,
)

# ══════════════════════════════════════════════════════════════════════
#  AUTORES
# ══════════════════════════════════════════════════════════════════════
add_para(
    "Autores del Proyecto — Universidad Nacional Tecnológica de Lima Sur (UNTELS)",
    style='IEEE Author Affiliation',
    align=WD_ALIGN_PARAGRAPH.CENTER,
    space_after=12,
)

# ══════════════════════════════════════════════════════════════════════
#  ABSTRACT
# ══════════════════════════════════════════════════════════════════════
abstract_text = (
    "Abstract\u2013 La anemia es una condición hematológica de alta prevalencia "
    "mundial que puede diagnosticarse con precisión a partir de indicadores sanguíneos "
    "como la hemoglobina, el MCH, el MCHC y el MCV. En este trabajo se comparan dos "
    "enfoques de aprendizaje automático para su detección: una Red Neuronal Artificial "
    "(RNA) con regularización Dropout y L2, y un Random Forest (RF) de 100 estimadores. "
    "Ambos modelos se entrenaron con el conjunto de datos público Anemia Dataset "
    "(1,421 registros, 5 variables) dividido en 80% entrenamiento y 20% prueba. "
    "Los resultados revelan que el dataset presenta separabilidad lineal casi perfecta, "
    "con 0% de ambigüedad entre clases medida sobre perfiles únicos de variables. "
    "En consecuencia, ambos modelos alcanzaron métricas perfectas en el conjunto de "
    "prueba (Accuracy = 1.0000, AUC-ROC = 1.0000, F1 = 1.0000). Se analiza el "
    "fenómeno de alta confianza predictiva, se aplica Temperature Scaling (T=3.5) "
    "a la RNA para calibrar probabilidades de salida, y se comparan las "
    "diferencias arquitectónicas, interpretabilidad y costo computacional de ambos enfoques."
)
add_para(abstract_text, style='Abstract and Keywords', italic=False, space_after=4)

add_para(
    "Keywords\u2014 Anemia, Red Neuronal Artificial, Random Forest, Machine Learning, "
    "Clasificación Binaria, Temperature Scaling, Separabilidad de Clases.",
    style='Abstract and Keywords', space_after=10,
)

# ══════════════════════════════════════════════════════════════════════
#  I. INTRODUCCIÓN
# ══════════════════════════════════════════════════════════════════════
add_para("I.  Introducción", style='Heading 1', space_before=6, space_after=4)

add_para(
    "La anemia afecta aproximadamente al 24.8% de la población mundial, según la "
    "Organización Mundial de la Salud (OMS) [1]. Su diagnóstico clínico se basa en "
    "la medición de indicadores eritrocitarios: hemoglobina (Hgb), hemoglobina "
    "corpuscular media (MCH), concentración de hemoglobina corpuscular media (MCHC) "
    "y volumen corpuscular medio (MCV). La disponibilidad de datasets clínicos "
    "etiquetados abre la posibilidad de construir sistemas de predicción automática "
    "que apoyen el diagnóstico.",
    space_after=6,
)
add_para(
    "Este trabajo tiene como objetivo comparar empíricamente dos paradigmas de "
    "aprendizaje automático supervisado para la clasificación binaria de anemia: "
    "(1) una Red Neuronal Artificial (RNA) multicapa con mecanismos de regularización, "
    "y (2) un ensamble de Árboles de Decisión mediante Random Forest (RF). "
    "Ambos modelos son evaluados sobre el mismo conjunto de prueba con métricas "
    "estándar y se analizan sus fortalezas, limitaciones y comportamiento frente a "
    "datos de alta separabilidad.",
    space_after=6,
)

# ══════════════════════════════════════════════════════════════════════
#  II. MATERIALES Y MÉTODOS
# ══════════════════════════════════════════════════════════════════════
add_para("II.  Materiales y Métodos", style='Heading 1', space_before=8, space_after=4)

add_para("A.\tDataset y Preprocesamiento", style='Heading 2', space_after=4)
ds = METRICAS["dataset"]
add_para(
    f"Se utilizó el Anemia Dataset, un conjunto de datos público con {ds['total']} registros "
    f"de pacientes clasificados como anémicos (1) o no anémicos (0). La distribución de clases "
    f"es: {ds['negativos']} casos negativos ({ds['negativos']/ds['total']*100:.1f}%) y "
    f"{ds['positivos']} positivos ({ds['positivos']/ds['total']*100:.1f}%). "
    "Las variables predictoras son: Gender (binario), Hemoglobin (g/dL), MCH (pg), "
    "MCHC (g/dL) y MCV (fL). La variable objetivo es Result (0/1).",
    space_after=6,
)
add_para(
    f"El dataset se dividió en {ds['train']} muestras de entrenamiento (80%) y "
    f"{ds['test']} de prueba (20%), con estratificación para preservar la proporción "
    "de clases. Los valores nulos se imputaron con la mediana de cada columna. "
    "Para la RNA, los datos se normalizaron con StandardScaler ajustado "
    "exclusivamente sobre el conjunto de entrenamiento.",
    space_after=6,
)

add_para("B.\tRed Neuronal Artificial (RNA)", style='Heading 2', space_after=4)
add_para(
    "Se implementó una RNA secuencial con la arquitectura Dense(64, ReLU) → "
    "Dropout(0.3) → Dense(32, ReLU, L2=0.001) → Dropout(0.2) → Dense(16, ReLU) → "
    "Dense(1, Sigmoid). La función de pérdida utilizada fue binary crossentropy "
    "con el optimizador Adam. El entrenamiento se realizó con un máximo de 150 épocas, "
    "batch size de 32, y EarlyStopping (patience=20) monitoreando val_loss con "
    "restauración de los mejores pesos. Se aplicó Temperature Scaling (T=3.5) "
    "post-entrenamiento en producción para calibrar las probabilidades de salida "
    "y evitar predicciones saturadas en los extremos 0% o 100%.",
    space_after=6,
)

add_para("C.\tRandom Forest (RF)", style='Heading 2', space_after=4)
add_para(
    "Se entrenó un RandomForestClassifier con 100 estimadores, profundidad máxima "
    "de 6 niveles, min_samples_split=10, min_samples_leaf=5 y class_weight='balanced'. "
    "El RF no requiere normalización de datos. Cada árbol se entrena sobre una muestra "
    "bootstrap del conjunto de entrenamiento, y la predicción final es el promedio "
    "de probabilidades de los 100 árboles. Esta arquitectura de ensamble produce "
    "naturalmente probabilidades más distribuidas que un árbol de decisión individual.",
    space_after=6,
)

add_para("D.\tMétricas de Evaluación", style='Heading 2', space_after=4)
add_para(
    "Los modelos fueron evaluados con: Accuracy, Precision, Recall (Sensibilidad), "
    "F1-Score y AUC-ROC. Se reportan métricas individuales para cada clase "
    "(No anémico = 0, Anémico = 1) y la matriz de confusión completa.",
    space_after=8,
)

# ══════════════════════════════════════════════════════════════════════
#  III. RESULTADOS
# ══════════════════════════════════════════════════════════════════════
add_para("III.  Resultados", style='Heading 1', space_before=8, space_after=4)

add_para("A.\tMétricas Globales de Rendimiento", style='Heading 2', space_after=4)
add_para(
    "La Tabla I presenta un resumen comparativo de las métricas globales de ambos "
    "modelos sobre el conjunto de prueba (n=285).",
    space_after=6,
)

# --- TABLA I: Comparación global ---
add_para("TABLE I", style='Table Caption', align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
add_para("Comparación de Métricas Globales — Conjunto de Prueba (n=285)",
         style='Table Caption', align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)

t1 = doc.add_table(rows=3, cols=5)
t1.style = 'Table Grid'
headers = ["Métrica", "RNA", "RF", "Diferencia", "Mejor"]
hrow = t1.rows[0]
for i, h in enumerate(headers):
    cell = hrow.cells[i]
    cell.text = h
    set_cell_bg(cell, '1F3864')
    for run in cell.paragraphs[0].runs:
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(9)
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

rn = METRICAS["rn"]
rf = METRICAS["rf"]
rows_data = [
    ["Accuracy",  pct(rn["acc"]), pct(rf["acc"]), "0.00%", "Empate"],
    ["AUC-ROC",   fmt(rn["auc"]), fmt(rf["auc"]), "0.0000", "Empate"],
]
for r_i, row_data in enumerate(rows_data):
    row = t1.rows[r_i + 1]
    bg = 'D6E4F0' if r_i % 2 == 0 else 'EBF5FB'
    for c_i, val in enumerate(row_data):
        cell = row.cells[c_i]
        cell.text = val
        set_cell_bg(cell, bg)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cell.paragraphs[0].runs:
            run.font.size = Pt(9)

set_cell_borders(t1)
doc.add_paragraph()  # espacio

# --- TABLA II: Por clase ---
add_para("A.\tMétricas por Clase", style='Heading 2', space_after=4)
add_para(
    "La Tabla II desglosa Precision, Recall y F1-Score para cada clase, "
    "permitiendo identificar si algún modelo favorece detectar una clase sobre otra.",
    space_after=6,
)
add_para("TABLE II", style='Table Caption', align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
add_para("Métricas por Clase — Anémico (1) vs No Anémico (0)",
         style='Table Caption', align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)

t2 = doc.add_table(rows=5, cols=7)
t2.style = 'Table Grid'
h2 = ["Clase", "Modelo", "Precision", "Recall", "F1-Score", "Support", "Interpretación"]
hrow2 = t2.rows[0]
for i, h in enumerate(h2):
    cell = hrow2.cells[i]
    cell.text = h
    set_cell_bg(cell, '1F3864')
    for run in cell.paragraphs[0].runs:
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(8)
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

rows_t2 = [
    ["No anémico (0)", "RNA", pct(rn["p0"]), pct(rn["r0"]), pct(rn["f0"]), "161", "Sin FP ni FN"],
    ["No anémico (0)", "RF",  pct(rf["p0"]), pct(rf["r0"]), pct(rf["f0"]), "161", "Sin FP ni FN"],
    ["Anémico (1)",    "RNA", pct(rn["p1"]), pct(rn["r1"]), pct(rn["f1"]), "124", "Sin FP ni FN"],
    ["Anémico (1)",    "RF",  pct(rf["p1"]), pct(rf["r1"]), pct(rf["f1"]), "124", "Sin FP ni FN"],
]
for r_i, row_data in enumerate(rows_t2):
    row = t2.rows[r_i + 1]
    bg = 'D6E4F0' if r_i % 2 == 0 else 'EBF5FB'
    for c_i, val in enumerate(row_data):
        cell = row.cells[c_i]
        cell.text = val
        set_cell_bg(cell, bg)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cell.paragraphs[0].runs:
            run.font.size = Pt(8)

set_cell_borders(t2)
doc.add_paragraph()

# --- TABLA III: Matrices de confusión ---
add_para("B.\tMatrices de Confusión", style='Heading 2', space_after=4)
add_para(
    "Las Tablas III y IV presentan las matrices de confusión de ambos modelos. "
    f"En el conjunto de prueba ({ds['test']} muestras), la RNA y el RF producen "
    "matrices idénticas: cero falsos positivos y cero falsos negativos.",
    space_after=6,
)

for model_name, cm in [("Red Neuronal (RNA)", rn["cm"]), ("Random Forest (RF)", rf["cm"])]:
    idx = "III" if "RNA" in model_name else "IV"
    add_para(f"TABLE {idx}", style='Table Caption', align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
    add_para(f"Matriz de Confusión — {model_name}",
             style='Table Caption', align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)

    t = doc.add_table(rows=3, cols=3)
    t.style = 'Table Grid'
    labels = ["", "Pred: No anémico", "Pred: Anémico"]
    row_labels = ["Real: No anémico", "Real: Anémico"]

    for c_i, lbl in enumerate(labels):
        cell = t.rows[0].cells[c_i]
        cell.text = lbl
        set_cell_bg(cell, '1F3864')
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(9)

    for r_i in range(2):
        row = t.rows[r_i + 1]
        row.cells[0].text = row_labels[r_i]
        set_cell_bg(row.cells[0], '2E75B6')
        row.cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in row.cells[0].paragraphs[0].runs:
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(9)

        for c_i in range(2):
            val = cm[r_i][c_i]
            cell = row.cells[c_i + 1]
            cell.text = str(val)
            # Diagonal verde, off-diagonal rojo
            bg = '70AD47' if r_i == c_i else 'FF0000' if val > 0 else 'C6EFCE'
            set_cell_bg(cell, bg)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in cell.paragraphs[0].runs:
                run.bold = True
                run.font.size = Pt(11)
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    set_cell_borders(t)
    doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════
#  IV. DISCUSIÓN
# ══════════════════════════════════════════════════════════════════════
add_para("IV.  Discusión", style='Heading 1', space_before=8, space_after=4)

add_para(
    "Los resultados obtenidos (Accuracy = AUC = F1 = 1.0000 para ambos modelos) "
    "reflejan directamente la naturaleza del dataset utilizado. Un análisis de "
    "separabilidad sobre los 534 perfiles de variables únicos reveló que el 0.0% "
    "presenta ambigüedad entre clases, lo que contrasta con el 23.9% de ambigüedad "
    "observado en el dataset del Titanic. Este fenómeno se debe a que la anemia es "
    "una condición definida clínicamente por umbrales numéricos fijos sobre la "
    "hemoglobina: cualquier algoritmo con suficiente capacidad aprenderá estos "
    "umbrales con precisión perfecta.",
    space_after=6,
)
add_para(
    "La importancia de variables del RF confirma el dominio de Hemoglobin "
    "(importancia=0.8492), seguida por Gender (0.0976), MCV (0.0204), "
    "MCH (0.0181) y MCHC (0.0148). Esto es consistente con el criterio diagnóstico "
    "de la OMS, que define la anemia principalmente por los niveles de hemoglobina.",
    space_after=6,
)
add_para(
    "Desde la perspectiva de producción, la RNA requirió Temperature Scaling "
    "(T=3.5) para evitar predicciones saturadas en 0% o 100%, produciendo "
    "probabilidades calibradas más informativas para el usuario clínico. "
    "El RF, al promediar 100 árboles, produce naturalmente probabilidades "
    "más distribuidas sin necesidad de calibración adicional.",
    space_after=6,
)

# --- TABLA V: Comparación cualitativa ---
add_para("TABLE V", style='Table Caption', align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
add_para("Comparación Cualitativa — RNA vs Random Forest",
         style='Table Caption', align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)

t5 = doc.add_table(rows=9, cols=3)
t5.style = 'Table Grid'
h5 = ["Característica", "Red Neuronal (RNA)", "Random Forest (RF)"]
hrow5 = t5.rows[0]
for i, h in enumerate(h5):
    cell = hrow5.cells[i]
    cell.text = h
    set_cell_bg(cell, '1F3864')
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cell.paragraphs[0].runs:
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(9)

rows_t5 = [
    ["Paradigma", "Deep Learning (backpropagation)", "Ensemble de Árboles de Decisión"],
    ["Normalización requerida", "Sí (StandardScaler)", "No"],
    ["Archivos generados", ".h5 + .pkl (scaler)", ".pkl (único)"],
    ["Interpretabilidad", "Baja (caja negra)", "Media (feature_importances_)"],
    ["Calibración de probabilidades", "Temperature Scaling (T=3.5)", "No requerida"],
    ["Tiempo de entrenamiento", "~2 min (150 épocas)", "< 10 segundos"],
    ["Sobreajuste en datos separables", "Sin Dropout: extremos 0%/100%", "Hojas puras mitigadas por ensamble"],
    ["Accuracy en prueba", pct(rn["acc"]), pct(rf["acc"])],
]
for r_i, row_data in enumerate(rows_t5):
    row = t5.rows[r_i + 1]
    bg = 'D6E4F0' if r_i % 2 == 0 else 'EBF5FB'
    for c_i, val in enumerate(row_data):
        cell = row.cells[c_i]
        cell.text = val
        set_cell_bg(cell, bg)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if c_i > 0 else WD_ALIGN_PARAGRAPH.LEFT
        for run in cell.paragraphs[0].runs:
            run.font.size = Pt(8)
            if c_i == 0:
                run.bold = True

set_cell_borders(t5)
doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════
#  V. CONCLUSIONES
# ══════════════════════════════════════════════════════════════════════
add_para("V.  Conclusiones", style='Heading 1', space_before=8, space_after=4)

add_para(
    "Ambos modelos (RNA y RF) alcanzaron rendimiento perfecto en el dataset de "
    "anemia, demostrando que los indicadores hematológicos disponibles son "
    "suficientes para clasificar la condición de forma determinística. "
    "La coincidencia de resultados entre paradigmas tan distintos es en sí misma "
    "un hallazgo relevante: sugiere que el problema de clasificación binaria de "
    "anemia no requiere modelos complejos para este dataset particular.",
    space_after=6,
)
add_para(
    "Sin embargo, las diferencias cualitativas entre los modelos son significativas. "
    "El RF ofrece mayor interpretabilidad (importancia de variables) y no requiere "
    "normalización ni calibración de probabilidades. La RNA, aunque más costosa "
    "computacionalmente, es más flexible y escalaría mejor a conjuntos de datos con "
    "mayor número de variables y relaciones no lineales complejas.",
    space_after=6,
)
add_para(
    "Como trabajo futuro se recomienda evaluar ambos modelos con datasets de anemia "
    "que incluyan mayor variedad de tipos (ferropénica, megaloblástica, hemolítica) "
    "y mayor ambigüedad entre clases, para medir la robustez real de cada enfoque "
    "ante escenarios diagnósticos menos determinísticos.",
    space_after=8,
)

# ══════════════════════════════════════════════════════════════════════
#  REFERENCIAS
# ══════════════════════════════════════════════════════════════════════
add_para("Referencias", style='Normal', bold=True, space_before=6, space_after=4)

refs = [
    "Organización Mundial de la Salud (OMS). Anaemia. https://www.who.int/health-topics/anaemia, 2023.",
    "Breiman, L. 'Random Forests.' Machine Learning 45, 5–32, 2001.",
    "LeCun, Y., Bengio, Y., & Hinton, G. 'Deep learning.' Nature 521, 436–444, 2015.",
    "Guo, C., et al. 'On Calibration of Modern Neural Networks.' ICML 2017.",
    "Pedregosa, F., et al. 'Scikit-learn: Machine Learning in Python.' JMLR 12, 2825–2830, 2011.",
    "Chollet, F. 'Keras: The Python Deep Learning library.' https://keras.io, 2015.",
]
for i, ref in enumerate(refs):
    add_para(f"[{i+1}] {ref}", style='Reference entry', space_after=2)

# ══════════════════════════════════════════════════════════════════════
#  GUARDAR
# ══════════════════════════════════════════════════════════════════════
output_path = 'Informe_Comparacion_Anemia.docx'
doc.save(output_path)
print('Documento guardado como: ' + output_path)
