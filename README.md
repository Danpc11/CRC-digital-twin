# crc-digital-twin

Gemelo digital mecanicista de cáncer colorrectal: modela los cuatro subtipos moleculares
consensuados (CMS1-4) como atractores de una red tipo Hopfield continua, calibrable contra
datos reales, con validación contra desenlaces de supervivencia y un módulo de pronóstico
longitudinal para seguimiento post-quirúrgico — diseñado para operar sobre paneles medibles
por **qPCR/RT-qPCR únicamente** (sin ddPCR, sin NGS, sin secuenciación de exosomas).

Para el estado actual del proyecto (qué evidencia hay, qué falta), ver `PROJECT_STATUS.md`.
Para el historial de cambios, ver `CHANGELOG.md`.

## Instalación

```bash
pip install -r requirements.txt
```

## Estructura

```
run_pipeline.py                      CLI: calibración + validación (cohorte de entrenamiento)

src/
  attractor_model.py                 modelo de atractores (dinámica ODE tipo Hopfield)
  calibration.py                     calibración contra datos reales etiquetados
  survival_validation.py             Kaplan-Meier / log-rank
  prognosis.py                       trayectoria/pronóstico longitudinal
  prognosis_demo.py                  demo end-to-end con patrones calibrados reales
  external_validation.py             aplica patrones YA calibrados a cohorte externa (sin recalibrar)
  pooled_cox_validation.py           Cox estratificado combinando múltiples cohortes externas
  concordance_analysis.py            matriz de concordancia modelo vs. etiqueta oficial
  feature_selection.py               selección data-driven de genes (AUC/ANOVA/Random Forest)
  synthetic_data.py                  generador de datos de prueba
  plot_trajectories.py               figura de trayectorias del modelo
  plot_survival_curves.py            figuras de curvas Kaplan-Meier
  download_synapse_data.py           descarga datos del consorcio CMS (Synapse)
  parse_geo_series_matrix.py         parser de series_matrix de GEO (por celda, no por posición)
  format_to_schema.py                construye dataset TCGA
  build_gse39582_dataset.py          construye dataset GSE39582 (entrenamiento)
  build_gse17536_dataset.py          construye dataset GSE17536 (validación externa)
  build_gse17537_dataset.py          construye dataset GSE17537 (validación externa)
  build_external_cohort_generic.py   construye cualquier otra cohorte CRCSC con etiqueta CMS oficial

tests/            suite de regresión (pytest)
data/             datos (no versionados, ver .gitignore)
figures/          salidas gráficas
```

## Quickstart (datos sintéticos, sin credenciales)

```bash
python3 src/synthetic_data.py
python3 run_pipeline.py --input data/synthetic_cohort.tsv --output results/
python3 -m pytest tests/ -v
```

## Panel actual (10 genes, todos compatibles con RT-qPCR)

`MLH1`, `GNLY`, `USP18` (eje CMS1) · `MYC`, `AXIN2` (eje CMS2) · `FABP1`, `CPS1`, `SI` (eje
CMS3) · `VIM`, `TGFB1` (eje CMS4). Ver `PROJECT_STATUS.md` para la justificación de cada gen.

## Cómo descargar los datos reales

### Synapse (etiquetas CMS del consorcio)

1. Cuenta gratuita en https://www.synapse.org, genera un Personal Access Token
   (Settings → Personal Access Tokens)
2. `pip install synapseclient`
3. `export SYNAPSE_AUTH_TOKEN="tu_token"`
4. `python3 src/download_synapse_data.py` — descarga `syn4961785` (dataset combinado CRCSC),
   `syn4978511` (etiquetas CMS de TCGA) y `syn2023932` (RNA-seq TCGA)

IDs de Synapse verificados:
- `syn4961785` — dataset combinado CRCSC (incluye `cms_labels_public_all.txt`)
- `syn4978511` — etiquetas CMS específicas de TCGA-COADREAD
- `syn2023932` — RNA-seq procesado de TCGA-COADREAD

### GEO (expresión + supervivencia)

```bash
curl -L -o data/raw_geo/GSE39582_series_matrix.txt.gz \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE39nnn/GSE39582/matrix/GSE39582_series_matrix.txt.gz"
```

El patrón de URL de GEO agrupa por los últimos 3 dígitos del número de serie reemplazados
por `nnn` (para IDs de 3 dígitos o menos, el bucket es literalmente `GPLnnn`/`GSEnnn`, sin
prefijo numérico — verificar si falla).

### Anotación de plataforma (mapeo probe → gen)

```bash
curl -L -o data/raw_geo/GPL570.txt \
  "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?targ=self&acc=GPL570&form=text&view=full"
```

**No usar** `GPL570_family.soft.gz` — ese archivo incluye todas las series históricas que
usaron la plataforma (decenas de GB para GPL570). El comando de arriba trae solo el registro
propio de la plataforma (~80 MB).

## Flujo completo con datos reales

```bash
# 1. Construir dataset de entrenamiento (GSE39582)
python3 src/build_gse39582_dataset.py
python3 run_pipeline.py --input data/gse39582_cms_labeled.tsv --output results_gse39582/
python3 src/concordance_analysis.py --input results_gse39582/scored_cohort.tsv

# 2. Construir y validar una cohorte externa (patrones YA calibrados, sin recalibrar)
python3 src/build_gse17536_dataset.py
python3 src/external_validation.py \
  --patterns results_gse39582/calibrated_patterns.tsv \
  --input data/gse17536_cms_labeled.tsv \
  --output results_external_gse17536/

# 3. Combinar múltiples cohortes externas en un solo análisis (más poder estadístico)
python3 src/pooled_cox_validation.py \
  --cohort GSE17536 results_external_gse17536/scored_external_cohort.tsv \
  --cohort GSE17537 results_external_gse17537/scored_external_cohort.tsv \
  --output results_pooled_cox/
```

### Agregar una cohorte CRCSC nueva

Flujo de dos pasos — diagnosticar antes de construir, nunca asumir nombres de columna:

```bash
curl -L -o data/raw_geo/GSE13294_series_matrix.txt.gz \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13294/matrix/GSE13294_series_matrix.txt.gz"

python3 src/build_external_cohort_generic.py --gse GSE13294 --diagnose
# revisar las columnas que imprime, luego:
python3 src/build_external_cohort_generic.py --gse GSE13294 \
  --dataset gse13294 \
  --duration-col "characteristics__XXX" \
  --event-col "characteristics__YYY" \
  --event-map "valorA=1,valorB=0"   # solo si el evento no es ya 0/1
```

## Esquema de datos esperado (TSV, no CSV)

```
sample_id   cms_label            GENE1   GENE2   ...  relapse_free_months  relapse_event
TCGA-01     CMS1_MSI_immune      4.21    -1.02   ...  38.2                 0
TCGA-02     CMS2_canonical_WNT   -0.88   3.55    ...  12.1                 1
```

- `cms_label`: uno de `CMS1_MSI_immune`, `CMS2_canonical_WNT`, `CMS3_metabolic`,
  `CMS4_mesenchymal`, o `none` para muestras no clasificadas por el consorcio
- Columnas de genes: cualquier subconjunto numérico — `calibration.py` infiere las columnas
  de genes automáticamente
- `relapse_free_months` / `relapse_event`: opcionales, solo necesarias para
  `survival_validation.py` / `external_validation.py`

## Panel PCR-compatible (restricción de diseño)

| Capa | Técnica | Qué mide |
|---|---|---|
| Expresión (subtipo CMS) | RT-qPCR relativa | pronóstico basal por subtipo |
| Mutación conductora (BRAF/KRAS) | qPCR alelo-específico (ARMS-PCR) o HRM | marcador pronóstico independiente |
| Estatus MSI | HRM de marcadores mononucleotídicos (BAT-25/BAT-26) | favorable en estadio II |
| Seguimiento longitudinal (MRD) | qPCR alelo-específico dirigido a la mutación ya identificada en el tumor primario del paciente | vigilancia post-quirúrgica sin NGS |

## Módulo de pronóstico longitudinal (`prognosis.py`)

Convierte una serie temporal de mediciones post-quirúrgicas en una señal de riesgo ordinal:
vector de estado cerca de cero = sin enfermedad residual; vector que se aleja hacia un
atractor = alerta de recurrencia. Demo end-to-end con patrones calibrados reales:

```bash
python3 src/prognosis_demo.py --patterns results_gse39582/calibrated_patterns.tsv
```
