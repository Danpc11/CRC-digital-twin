# crc-digital-twin

Gemelo digital mecanicista de cáncer colorrectal: modela los cuatro
subtipos moleculares consensuados (CMS1-4) como atractores de una red
tipo Hopfield continua, calibrable contra datos reales, con validación
contra desenlaces de supervivencia y un módulo de pronóstico
longitudinal para seguimiento post-quirúrgico — todo diseñado para
operar sobre paneles medibles por **qPCR/RT-qPCR únicamente** (sin
ddPCR, sin NGS, sin secuenciación de exosomas).

## Instalación

```bash
pip install -r requirements.txt
```

## Estructura

```
src/attractor_model.py       modelo dinámico, dinámica, clasificador (con patrones demo)
src/calibration.py           calibración contra datos reales etiquetados
src/survival_validation.py   validación contra desenlaces de supervivencia (Kaplan-Meier, log-rank)
src/prognosis.py             módulo de trayectoria/pronóstico longitudinal post-quirúrgico
src/synthetic_data.py        generador de datos sintéticos para probar el pipeline sin datos reales
run_pipeline.py              CLI: calibración + validación en un solo comando
tests/                       suite de regresión (pytest), 20 tests
figures/                     salidas gráficas
data/                        datos (sintéticos incluidos; reales NO incluidos, ver abajo)
```

## Correr con datos sintéticos (prueba de que el pipeline funciona)

```bash
python3 src/synthetic_data.py
python3 run_pipeline.py --input data/synthetic_cohort.tsv --output results/
python3 -m pytest tests/ -v
```

## Bases de datos abiertas para calibración con datos REALES

| Fuente | Qué aporta | Acceso |
|---|---|---|
| **GSE39582** (GEO, Marisa et al. 2013) | Expresión + supervivencia libre de recidiva + MSI + mutaciones KRAS/BRAF/TP53, 566 pacientes | Descarga libre, series matrix desde GEO |
| **TCGA-COAD/READ** (GDC Data Portal) | RNA-seq + mutaciones (MAF) + clínico, ~620 pacientes | GDC API / gdc-client, o cBioPortal |
| **Synapse syn2623706** (consorcio CMS, Guinney et al. 2015) | Etiquetas CMS oficiales ya asignadas para TCGA, GSE39582, GSE14333, GSE17536 | Cuenta gratuita en Synapse |
| **GSE17536 / GSE17537** (Smith cohort) | Cohorte externa independiente para validación, ~230 pacientes | Descarga libre desde GEO |

**Combinación recomendada**: etiquetas CMS de Synapse aplicadas a GSE39582 — expresión + subtipo CMS validado + supervivencia real, todo en una sola cohorte.

## Esquema de datos esperado (TSV, no CSV)

```
sample_id   cms_label            GENE1   GENE2   ...  relapse_free_months  relapse_event
TCGA-01     CMS1_MSI_immune      4.21    -1.02   ...  38.2                 0
TCGA-02     CMS2_canonical_WNT   -0.88   3.55    ...  12.1                 1
```

- `cms_label`: uno de `CMS1_MSI_immune`, `CMS2_canonical_WNT`, `CMS3_metabolic`, `CMS4_mesenchymal`, o `none` para muestras no clasificadas limpiamente (~13-15% en el consorcio original — no se deben forzar a una etiqueta)
- Columnas de genes: cualquier subconjunto numérico, no tiene que coincidir con el panel placeholder de 8 genes de `attractor_model.py` — `calibration.py` infiere las columnas de genes automáticamente
- `relapse_free_months` / `relapse_event`: opcionales, solo necesarias para `survival_validation.py`

Este proyecto **no descarga datos automáticamente** — el usuario construye este TSV a partir de las fuentes de la tabla de arriba. La red del entorno de desarrollo usado para este esqueleto no tiene acceso a GEO/GDC/Synapse.

## Correr con datos reales

```bash
python3 run_pipeline.py --input data/gse39582_cms_labeled.tsv --output results/
```

Esto:
1. Calibra `CMS_PATTERNS` como centroides z-score reales por subtipo (reemplaza los placeholders hechos a mano)
2. Corre validación de supervivencia (Kaplan-Meier + log-rank test multivariado) por subtipo predicho
3. Guarda un reporte en lenguaje llano que **no sobreclama**: p < 0.05 es evidencia de utilidad pronóstica en esa cohorte, no validación clínica — eso requiere replicación en cohorte externa (ej. GSE17536/GSE17537) antes de cualquier uso más allá de investigación

## Panel PCR-compatible (restricción de diseño)

Todo el panel está restringido a técnicas de PCR en tiempo real estándar:

| Capa | Técnica | Qué mide |
|---|---|---|
| Expresión (subtipo CMS) | RT-qPCR relativa | pronóstico basal por subtipo |
| Mutación conductora (BRAF/KRAS) | qPCR alelo-específico (ARMS-PCR) o HRM | marcador pronóstico independiente |
| Estatus MSI | HRM de marcadores mononucleotídicos (BAT-25/BAT-26) | favorable en estadio II |
| Seguimiento longitudinal (MRD) | qPCR alelo-específico dirigido a la mutación ya identificada en el tumor primario del paciente | vigilancia post-quirúrgica sin NGS |

## Módulo de pronóstico longitudinal (`prognosis.py`)

Convierte una serie temporal de mediciones post-quirúrgicas en una señal de riesgo continua:
- Vector de estado cerca de cero a lo largo del tiempo → sin enfermedad residual detectable → buen pronóstico
- Vector que se aleja del origen hacia un atractor → alerta de recurrencia

**Importante**: el score de riesgo (`hazard_from_trajectory`) es actualmente **ordinal, no una probabilidad calibrada** — la calibración real requeriría datos de seguimiento longitudinal + tiempo a recurrencia (tipo DYNAMIC/GALAXY), que no son de acceso público fácil. Esto queda documentado como limitación explícita, no oculto.

## Limitaciones explícitas (léase antes de usar para nada serio)

1. Sin datos reales de calibración, todo el pronóstico es hipotético — arquitectura validada, biología no.
2. El módulo de trayectoria longitudinal no está calibrado contra tiempo a recurrencia real.
3. Ningún componente de este proyecto predice respuesta a tratamiento todavía (ver conversación de diseño — eso requiere un término de perturbación farmacodinámica que no existe en el modelo actual).
4. p < 0.05 en una sola cohorte no es validación clínica.

## Estado del arte y panorama comercial (para referencia)

- **Oncotype DX Colon Recurrence Score**: panel RT-qPCR de 12 genes, ya clínicamente validado — el precedente directo de este diseño
- **Cellworks Biosimulation Platform**: "gemelo digital" comercial real (NGS + Therapy Response Index), no compatible con restricción de qPCR
- **Signatera / Haystack MRD**: monitoreo de ctDNA vía NGS personalizado — este proyecto busca una aproximación de menor costo vía qPCR alelo-específico
