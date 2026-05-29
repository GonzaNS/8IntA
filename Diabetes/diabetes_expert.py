# =============================================================
#  Sistema Experto — Diagnóstico de Diabetes Mellitus Tipo 2
#  Motor de Inferencia basado en Lógica Proposicional
#
#  Las reglas se evalúan en orden jerárquico con if/elif/else.
#  No se usan puntajes ni heurísticas: cada regla es una
#  expresión booleana explícita (AND / OR) sobre los hechos.
# =============================================================

# ---------------------------------------------------------------
# Lista canónica de síntomas reconocidos por el sistema
# ---------------------------------------------------------------
SINTOMAS = [
    "poliuria",               # Orinar con frecuencia y en grandes cantidades
    "polidipsia",             # Sed extrema y constante
    "polifagia",              # Hambre excesiva aunque se haya comido
    "perdida_peso",           # Pérdida de peso inexplicada
    "fatiga",                 # Cansancio persistente o falta de energía
    "vision_borrosa",         # Dificultad para enfocar o visión nublada
    "neuropatia",             # Hormigueo / entumecimiento en manos o pies
    "curacion_lenta",         # Heridas o cortes que tardan en sanar
    "infecciones_recurrentes",# Infecciones frecuentes de orina, piel, encías
    "acantosis_nigricans",    # Manchas oscuras en pliegues de la piel
]


# ---------------------------------------------------------------
# Representación formal de las reglas en sintaxis Prolog
# ---------------------------------------------------------------
REGLAS_PROLOG = [
    {
        "num": 1,
        "descripcion": "Riesgo Crítico",
        "regla": "riesgo_critico(X) :-\n    poliuria(X),\n    polidipsia(X),\n    polifagia(X),\n    acantosis_nigricans(X).",
    },
    {
        "num": 2,
        "descripcion": "Riesgo Alto",
        "regla": "riesgo_alto(X) :-\n    (poliuria(X) ; polidipsia(X)),\n    perdida_peso(X),\n    (neuropatia(X) ; curacion_lenta(X)).",
    },
    {
        "num": 3,
        "descripcion": "Riesgo Moderado-Alto",
        "regla": "riesgo_moderado_alto(X) :-\n    fatiga(X),\n    vision_borrosa(X),\n    (infecciones_recurrentes(X) ; curacion_lenta(X)).",
    },
    {
        "num": 4,
        "descripcion": "Alerta Temprana",
        "regla": "alerta_temprana(X) :-\n    (poliuria(X) ; polidipsia(X) ; polifagia(X)).",
    },
    {
        "num": 5,
        "descripcion": "Riesgo Moderado",
        "regla": "riesgo_moderado(X) :-\n    (acantosis_nigricans(X) ; neuropatia(X) ; fatiga(X)).",
    },
    {
        "num": 0,
        "descripcion": "Baja Probabilidad (Por Defecto)",
        "regla": "baja_probabilidad(X) :-\n    \\+riesgo_critico(X),\n    \\+riesgo_alto(X),\n    \\+riesgo_moderado_alto(X),\n    \\+alerta_temprana(X),\n    \\+riesgo_moderado(X).",
    },
]


# ---------------------------------------------------------------
# Motor de Inferencia
# ---------------------------------------------------------------
def diagnosticar(sintomas_presentes: list[str]) -> dict:
    """
    Recibe la lista de síntomas confirmados por el usuario y aplica
    5 reglas de lógica proposicional en orden jerárquico descendente.

    Parámetros
    ----------
    sintomas_presentes : list[str]
        Nombres de síntomas (en snake_case, según SINTOMAS) que
        el usuario marcó como presentes.

    Retorna
    -------
    dict con claves:
        - riesgo        : str   Etiqueta del nivel de riesgo
        - diagnostico   : str   Descripción del diagnóstico
        - regla         : int   Número de regla que se activó (0 = ninguna)
        - sintomas_clave: list  Síntomas que dispararon la regla activa
        - total         : int   Total de síntomas marcados
        - porcentaje    : float Porcentaje sobre el total posible (10)
        - recomendacion : str   Recomendación clínica
    """

    # ── Extraer hechos booleanos del conjunto de síntomas ──────────────────
    hechos = set(sintomas_presentes)

    poliuria               = "poliuria"               in hechos
    polidipsia             = "polidipsia"             in hechos
    polifagia              = "polifagia"              in hechos
    perdida_peso           = "perdida_peso"           in hechos
    fatiga                 = "fatiga"                 in hechos
    vision_borrosa         = "vision_borrosa"         in hechos
    neuropatia             = "neuropatia"             in hechos
    curacion_lenta         = "curacion_lenta"         in hechos
    infecciones            = "infecciones_recurrentes" in hechos
    acantosis_nigricans    = "acantosis_nigricans"    in hechos

    total      = len(hechos)
    porcentaje = round((total / len(SINTOMAS)) * 100, 1)

    # ══════════════════════════════════════════════════════════════════════
    #  REGLAS — Lógica Proposicional (evaluación jerárquica)
    # ══════════════════════════════════════════════════════════════════════

    # REGLA 1 — Riesgo Crítico
    # SI poliuria AND polidipsia AND polifagia AND acantosis_nigricans
    if poliuria and polidipsia and polifagia and acantosis_nigricans:
        return {
            "riesgo":         "Crítico",
            "diagnostico":    "Alta probabilidad de Diabetes Mellitus Tipo 2 en estado avanzado.",
            "regla":          1,
            "prolog":         "riesgo_critico(X) :-\n    poliuria(X),\n    polidipsia(X),\n    polifagia(X),\n    acantosis_nigricans(X).",
            "sintomas_clave": ["poliuria", "polidipsia", "polifagia", "acantosis_nigricans"],
            "total":          total,
            "porcentaje":     porcentaje,
            "recomendacion":  (
                "Acuda a urgencias o a un endocrinólogo de forma inmediata. "
                "Se requieren análisis de glucemia en ayunas, HbA1c y PTOG urgentes."
            ),
        }

    # REGLA 2 — Riesgo Alto
    # SI (poliuria OR polidipsia) AND perdida_peso AND (neuropatia OR curacion_lenta)
    elif (poliuria or polidipsia) and perdida_peso and (neuropatia or curacion_lenta):
        sintomas_clave = (
            (["poliuria"] if poliuria else []) +
            (["polidipsia"] if polidipsia else []) +
            ["perdida_peso"] +
            (["neuropatia"] if neuropatia else []) +
            (["curacion_lenta"] if curacion_lenta else [])
        )
        return {
            "riesgo":         "Alto",
            "diagnostico":    "Perfil clínico de alto riesgo para Diabetes Mellitus Tipo 2.",
            "regla":          2,
            "prolog":         "riesgo_alto(X) :-\n    (poliuria(X) ; polidipsia(X)),\n    perdida_peso(X),\n    (neuropatia(X) ; curacion_lenta(X)).",
            "sintomas_clave": sintomas_clave,
            "total":          total,
            "porcentaje":     porcentaje,
            "recomendacion":  (
                "Consulte a su médico con carácter de urgencia. "
                "Se recomienda glucemia en ayunas y hemoglobina glucosilada (HbA1c)."
            ),
        }

    # REGLA 3 — Riesgo Moderado-Alto
    # SI fatiga AND vision_borrosa AND (infecciones OR curacion_lenta)
    elif fatiga and vision_borrosa and (infecciones or curacion_lenta):
        sintomas_clave = (
            ["fatiga", "vision_borrosa"] +
            (["infecciones_recurrentes"] if infecciones else []) +
            (["curacion_lenta"] if curacion_lenta else [])
        )
        return {
            "riesgo":         "Moderado-Alto",
            "diagnostico":    "Combinación de síntomas con riesgo moderado-alto de Diabetes Tipo 2.",
            "regla":          3,
            "prolog":         "riesgo_moderado_alto(X) :-\n    fatiga(X),\n    vision_borrosa(X),\n    (infecciones_recurrentes(X) ; curacion_lenta(X)).",
            "sintomas_clave": sintomas_clave,
            "total":          total,
            "porcentaje":     porcentaje,
            "recomendacion":  (
                "Programe una cita médica pronto. Realice análisis de glucosa "
                "y evalúe cambios en dieta y actividad física."
            ),
        }

    # REGLA 4 — Alerta Temprana
    # SI poliuria OR polidipsia OR polifagia
    elif poliuria or polidipsia or polifagia:
        sintomas_clave = (
            (["poliuria"]    if poliuria    else []) +
            (["polidipsia"]  if polidipsia  else []) +
            (["polifagia"]   if polifagia   else [])
        )
        return {
            "riesgo":         "Alerta Temprana",
            "diagnostico":    "Presencia de síntomas cardinales. Se recomienda vigilancia activa.",
            "regla":          4,
            "prolog":         "alerta_temprana(X) :-\n    (poliuria(X) ; polidipsia(X) ; polifagia(X)).",
            "sintomas_clave": sintomas_clave,
            "total":          total,
            "porcentaje":     porcentaje,
            "recomendacion":  (
                "Consulte a su médico de cabecera para un chequeo preventivo. "
                "Monitoree la evolución de los síntomas en las próximas semanas."
            ),
        }

    # REGLA 5 — Riesgo Moderado
    # SI acantosis_nigricans OR neuropatia OR fatiga
    elif acantosis_nigricans or neuropatia or fatiga:
        sintomas_clave = (
            (["acantosis_nigricans"] if acantosis_nigricans else []) +
            (["neuropatia"]          if neuropatia          else []) +
            (["fatiga"]              if fatiga              else [])
        )
        return {
            "riesgo":         "Moderado",
            "diagnostico":    "Síntomas de alerta moderada compatibles con factores de riesgo.",
            "regla":          5,
            "prolog":         "riesgo_moderado(X) :-\n    (acantosis_nigricans(X) ; neuropatia(X) ; fatiga(X)).",
            "sintomas_clave": sintomas_clave,
            "total":          total,
            "porcentaje":     porcentaje,
            "recomendacion":  (
                "Adopte hábitos saludables (dieta balanceada y ejercicio regular). "
                "Realice un control de glucemia en su próxima visita médica."
            ),
        }

    # REGLA POR DEFECTO — Baja probabilidad
    else:
        return {
            "riesgo":         "Baja probabilidad",
            "diagnostico":    "No se detectaron síntomas suficientes para indicar riesgo de Diabetes Tipo 2.",
            "regla":          0,
            "prolog":         "baja_probabilidad(X) :-\n    \\+riesgo_critico(X),\n    \\+riesgo_alto(X),\n    \\+riesgo_moderado_alto(X),\n    \\+alerta_temprana(X),\n    \\+riesgo_moderado(X).",
            "sintomas_clave": [],
            "total":          total,
            "porcentaje":     porcentaje,
            "recomendacion":  (
                "Continue con sus chequeos médicos anuales de rutina "
                "y mantenga un estilo de vida saludable."
            ),
        }
