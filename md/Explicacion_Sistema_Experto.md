# Guía Práctica: Entendiendo el Sistema Experto (`diabetes_expert.py`)

Este documento explica paso a paso cómo funciona el Sistema Experto de diagnóstico de Diabetes Mellitus Tipo 2. A diferencia de los modelos de Machine Learning (Red Neuronal y Árbol de Decisión), este sistema **no aprende de datos**: razona usando reglas lógicas escritas por un experto médico. Está escrito para que cualquier persona pueda entenderlo.

---

## ¿Qué es un Sistema Experto y en qué se diferencia del ML?

| Característica | Machine Learning (RN / DT) | Sistema Experto |
|---|---|---|
| ¿Cómo adquiere conocimiento? | Aprendiendo de miles de ejemplos pasados | Reglas escritas manualmente por un experto |
| ¿Necesita datos de entrenamiento? | Sí (CSV con cientos de filas) | No |
| ¿Puede explicar su decisión? | Difícil (especialmente la RN) | Sí: indica exactamente qué regla se activó |
| ¿Qué pasa si cambian los criterios? | Hay que reentrenar el modelo | Solo se modifica la regla en el código |
| Archivos generados | `.h5` y `.pkl` | Ninguno (todo está en el código Python) |

**La analogía perfecta:** El Machine Learning es como un médico que aprendió a diagnosticar leyendo miles de historias clínicas. El Sistema Experto es como un médico que sigue un **protocolo escrito**: *"Si el paciente tiene poliuria, polidipsia y pérdida de peso, el riesgo es Alto"*.

---

## ¿Qué es Prolog y por qué se menciona?

Prolog es un lenguaje de programación lógica usado históricamente para construir Sistemas Expertos. Sus reglas se escriben de esta forma:

```prolog
riesgo_critico(X) :-
    poliuria(X),
    polidipsia(X),
    polifagia(X),
    acantosis_nigricans(X).
```

Esto se lee: *"X tiene riesgo crítico **si** tiene poliuria **Y** polidipsia **Y** polifagia **Y** acantosis nigricans"*.

En este proyecto **no usamos Prolog directamente** (usamos Python), pero las reglas fueron diseñadas con esa lógica formal y se almacenan en `REGLAS_PROLOG` para mostrarlas visualmente en la página web. Sirven como documentación del razonamiento del sistema.

---

## Bloque 1: La Lista de Síntomas Reconocidos

```python
SINTOMAS = [
    "poliuria",               # Orinar con frecuencia y en grandes cantidades
    "polidipsia",             # Sed extrema y constante
    "polifagia",              # Hambre excesiva aunque se haya comido
    "perdida_peso",           # Pérdida de peso inexplicada
    "fatiga",                 # Cansancio persistente
    "vision_borrosa",         # Dificultad para enfocar
    "neuropatia",             # Hormigueo en manos o pies
    "curacion_lenta",         # Heridas que tardan en sanar
    "infecciones_recurrentes",# Infecciones frecuentes
    "acantosis_nigricans",    # Manchas oscuras en pliegues de la piel
]
```

**Explicación práctica:**
Esta lista define el **vocabulario** del sistema: los únicos síntomas que el sistema entiende y puede evaluar. Cada nombre está en formato `snake_case` (minúsculas separadas por guiones bajos) porque son los mismos valores que llegan del formulario HTML cuando el usuario marca los checkboxes.

Los **3 síntomas cardinales** de la diabetes son los primeros tres (las "3 Polis"):
- **Poliuria** (orinar mucho)
- **Polidipsia** (beber mucho)
- **Polifagia** (comer mucho)

Su presencia simultánea es una señal de alarma muy fuerte, por eso aparecen en la regla de mayor prioridad.

---

## Bloque 2: Las Reglas Formales (REGLAS_PROLOG)

```python
REGLAS_PROLOG = [
    {
        "num": 1,
        "descripcion": "Riesgo Crítico",
        "regla": "riesgo_critico(X) :-\n    poliuria(X),\n    polidipsia(X),\n    polifagia(X),\n    acantosis_nigricans(X).",
    },
    # ... más reglas ...
]
```

**Explicación práctica:**
Esta es una lista de diccionarios de Python que almacena la representación **formal y legible** de cada regla. No se usa para calcular nada: su único propósito es mostrarse en la página web de diabetes como documentación del razonamiento del sistema.

Cada diccionario tiene:
- `num`: El número de la regla (para identificarla en la interfaz)
- `descripcion`: El nombre del nivel de riesgo
- `regla`: El código en sintaxis Prolog que describe la condición lógica

---

## Bloque 3: La Función Diagnóstica — El Motor de Inferencia ⭐

```python
def diagnosticar(sintomas_presentes: list[str]) -> dict:
```

Esta es la función principal y el corazón del archivo. Recibe una lista de síntomas marcados y devuelve un diccionario con el diagnóstico completo.

### Paso 3.1 — Convertir la lista en un conjunto de hechos

```python
hechos = set(sintomas_presentes)

poliuria            = "poliuria"            in hechos
polidipsia          = "polidipsia"          in hechos
polifagia           = "polifagia"           in hechos
perdida_peso        = "perdida_peso"        in hechos
fatiga              = "fatiga"              in hechos
vision_borrosa      = "vision_borrosa"      in hechos
neuropatia          = "neuropatia"          in hechos
curacion_lenta      = "curacion_lenta"      in hechos
infecciones         = "infecciones_recurrentes" in hechos
acantosis_nigricans = "acantosis_nigricans" in hechos
```

**Explicación práctica:**
Primero convertimos la lista en un `set` (conjunto). Los conjuntos en Python permiten verificar si un elemento está dentro con la expresión `"valor" in conjunto`, que es instantánea sin importar el tamaño.

Luego creamos **variables booleanas** (`True` o `False`) para cada síntoma. Esto hace que las reglas que vienen después sean fáciles de leer:

```python
# En lugar de escribir esto en cada regla:
if "poliuria" in sintomas_presentes and "polidipsia" in sintomas_presentes:

# Solo escribimos esto (mucho más claro):
if poliuria and polidipsia:
```

---

## Bloque 4: Las 5 Reglas — Lógica Proposicional Jerárquica

El sistema evalúa las reglas **en orden, de mayor a menor gravedad**, usando una cadena `if / elif / else`. En cuanto una regla se cumple, retorna el resultado y no evalúa las siguientes.

```
¿Se cumple Regla 1? → Sí → RETORNA "Crítico"   (fin)
         ↓ No
¿Se cumple Regla 2? → Sí → RETORNA "Alto"      (fin)
         ↓ No
¿Se cumple Regla 3? → Sí → RETORNA "Mod-Alto"  (fin)
         ↓ No
¿Se cumple Regla 4? → Sí → RETORNA "Alerta"    (fin)
         ↓ No
¿Se cumple Regla 5? → Sí → RETORNA "Moderado"  (fin)
         ↓ No
                            RETORNA "Baja prob." (defecto)
```

### Regla 1 — Riesgo Crítico

```python
# CONDICIÓN: poliuria AND polidipsia AND polifagia AND acantosis_nigricans
if poliuria and polidipsia and polifagia and acantosis_nigricans:
    return {
        "riesgo":        "Crítico",
        "diagnostico":   "Alta probabilidad de Diabetes Mellitus Tipo 2 en estado avanzado.",
        "regla":         1,
        "sintomas_clave": ["poliuria", "polidipsia", "polifagia", "acantosis_nigricans"],
        "recomendacion": "Acuda a urgencias o a un endocrinólogo de forma inmediata...",
        ...
    }
```

**Explicación práctica:** Se necesitan los **4 síntomas específicos al mismo tiempo** (operador `AND`). La presencia simultánea de las 3 Polis más la acantosis (manchas oscuras en la piel) es una señal clínica de diabetes avanzada que requiere atención inmediata.

---

### Regla 2 — Riesgo Alto

```python
# CONDICIÓN: (poliuria OR polidipsia) AND perdida_peso AND (neuropatia OR curacion_lenta)
elif (poliuria or polidipsia) and perdida_peso and (neuropatia or curacion_lenta):
    sintomas_clave = (
        (["poliuria"]    if poliuria    else []) +
        (["polidipsia"]  if polidipsia  else []) +
        ["perdida_peso"] +
        (["neuropatia"]  if neuropatia  else []) +
        (["curacion_lenta"] if curacion_lenta else [])
    )
```

**Explicación práctica:** Aquí aparecen los operadores `OR`: el sistema acepta variantes. No importa si tiene poliuria o polidipsia (con uno basta), siempre que también haya pérdida de peso más algún síntoma de daño nervioso o de circulación. Esta combinación sugiere diabetes con posibles complicaciones tempranas.

La construcción de `sintomas_clave` es dinámica: agrega a la lista solo los síntomas que realmente estaban presentes (`if poliuria else []`). Esta lista se muestra en la interfaz web para que el usuario vea exactamente qué síntomas dispararon la regla.

---

### Regla 3 — Riesgo Moderado-Alto

```python
# CONDICIÓN: fatiga AND vision_borrosa AND (infecciones OR curacion_lenta)
elif fatiga and vision_borrosa and (infecciones or curacion_lenta):
```

**Explicación práctica:** Esta combinación representa síntomas secundarios de la diabetes que solos no serían tan específicos, pero juntos forman un perfil de riesgo claro: el cansancio crónico y la visión borrosa son consecuencias del nivel alto de glucosa en sangre, y las infecciones frecuentes o la mala cicatrización son señales de un sistema inmune comprometido.

---

### Regla 4 — Alerta Temprana

```python
# CONDICIÓN: poliuria OR polidipsia OR polifagia
elif poliuria or polidipsia or polifagia:
```

**Explicación práctica:** Con un solo síntoma cardinal (cualquiera de las 3 Polis), el sistema ya emite una alerta. Es la regla más "amplia" entre las de alto riesgo: basta con uno de los tres síntomas clásicos para activarla. Esto garantiza que ningún caso con síntomas cardinales pase desapercibido, incluso si solo está en etapas iniciales.

---

### Regla 5 — Riesgo Moderado

```python
# CONDICIÓN: acantosis_nigricans OR neuropatia OR fatiga
elif acantosis_nigricans or neuropatia or fatiga:
```

**Explicación práctica:** Síntomas que individualmente pueden tener muchas causas, pero que en el contexto de un chequeo preventivo merecen atención. La acantosis nigricans en particular es un marcador visual de resistencia a la insulina (un precursor de la diabetes).

---

### Regla por defecto — Baja probabilidad

```python
else:
    return {
        "riesgo":    "Baja probabilidad",
        "diagnostico": "No se detectaron síntomas suficientes...",
        "regla":     0,
        ...
    }
```

**Explicación práctica:** Si ninguna de las 5 reglas anteriores se cumplió, el sistema concluye que no hay indicadores suficientes de riesgo. Esto corresponde a la regla Prolog `baja_probabilidad(X)` que usa la negación (`\+`) para verificar que ninguna de las otras reglas aplica.

---

## Bloque 5: El Resultado — El Diccionario de Respuesta

Cada rama retorna un diccionario con la misma estructura de claves:

```python
{
    "riesgo":         "Alto",           # Etiqueta del nivel de riesgo
    "diagnostico":    "Perfil clínico...", # Descripción para el usuario
    "regla":          2,               # Número de regla que se activó
    "prolog":         "riesgo_alto...", # Código Prolog de esa regla
    "sintomas_clave": ["poliuria", "perdida_peso", "neuropatia"],
    "total":          5,               # Cuántos síntomas marcó el usuario
    "porcentaje":     50.0,            # % del total posible (5 de 10 = 50%)
    "recomendacion":  "Consulte a su médico..."
}
```

**Explicación práctica:**
Este diccionario es lo que Flask recibe desde `diagnosticar()` y luego pasa a `diabetes.html` para mostrarlo. Jinja2 en el HTML accede a cada clave con `{{ resultado.riesgo }}`, `{{ resultado.diagnostico }}`, etc.

El campo `porcentaje` se calcula así:
```python
total      = len(hechos)                        # Ej: 5 síntomas marcados
porcentaje = round((total / len(SINTOMAS)) * 100, 1)  # (5/10)*100 = 50.0%
```
No indica probabilidad de tener diabetes (eso lo determina la regla), sino qué fracción del total de síntomas conocidos presenta el paciente.

---

## Flujo Completo del Sistema Experto

```
Usuario marca checkboxes en el formulario web
        │
        │  HTTP POST a /diabetes/diagnosticar
        ▼
Flask llama a: diagnosticar(["poliuria", "perdida_peso", "neuropatia"])
        │
        ▼
Convierte lista → set de hechos booleanos
  poliuria=True, perdida_peso=True, neuropatia=True, ...resto=False
        │
        ▼
Evalúa Regla 1: poliuria AND polidipsia AND polifagia AND acantosis
  → False (falta polidipsia, polifagia, acantosis) → siguiente
        │
        ▼
Evalúa Regla 2: (poliuria OR polidipsia) AND perdida_peso AND (neuropatia OR curacion_lenta)
  → (True OR False) AND True AND (True OR False)
  → True AND True AND True
  → ¡SE ACTIVA!
        │
        ▼
Retorna diccionario con riesgo="Alto", regla=2, sintomas_clave=[...], ...
        │
        ▼
Flask pasa el diccionario a diabetes.html
        │
        ▼
La página muestra el nivel de riesgo, los síntomas detectados y la recomendación
```

---

## Resumen de las 6 Reglas

| # | Nombre | Condición lógica | Operador dominante |
|---|---|---|---|
| 1 | Crítico | poliuria **Y** polidipsia **Y** polifagia **Y** acantosis | AND total |
| 2 | Alto | (poliuria **O** polidipsia) **Y** perdida_peso **Y** (neuropatia **O** curacion_lenta) | AND con OR interno |
| 3 | Moderado-Alto | fatiga **Y** vision_borrosa **Y** (infecciones **O** curacion_lenta) | AND con OR interno |
| 4 | Alerta Temprana | poliuria **O** polidipsia **O** polifagia | OR (cualquiera basta) |
| 5 | Moderado | acantosis **O** neuropatia **O** fatiga | OR (cualquiera basta) |
| 0 | Baja probabilidad | Ninguna de las anteriores | Caso por defecto |

> [!TIP]
> Para agregar una nueva regla al sistema, solo tienes que añadir un nuevo bloque `elif` dentro de la función `diagnosticar()` en el lugar jerárquico correcto (según su gravedad), y agregar su representación formal al array `REGLAS_PROLOG`. No se requiere reentrenamiento ni archivos adicionales.
