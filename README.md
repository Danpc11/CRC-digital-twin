# CRC-digital-twin

[![Tests](https://github.com/Danpc11/CRC-digital-twin/actions/workflows/tests.yml/badge.svg)](https://github.com/Danpc11/CRC-digital-twin/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
[![Docker](https://img.shields.io/badge/Docker-pipelinesinmegen%2Fcoloq-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/pipelinesinmegen/coloq)
![Tests count](https://img.shields.io/badge/tests-176%20passing-brightgreen)

Gemelo digital de cáncer colorrectal: modela los cuatro subtipos moleculares
consensuados de cáncer colorrectal (*Consensus Molecular Subtypes*, CMS1–CMS4) como atractores de una red tipo Hopfield continua, calibrable contra
datos reales, con validación contra desenlaces de supervivencia y un módulo de pronóstico
longitudinal para seguimiento post-quirúrgico — diseñado para operar sobre paneles medibles
únicamente mediante **qPCR/RT-qPCR** (reacción en cadena de la polimerasa cuantitativa o con transcripción inversa), sin PCR digital por gotículas (ddPCR), secuenciación de nueva generación (NGS) ni secuenciación de exosomas.

El clasificador clínico principal sigue siendo la correlación con centroides CMS congelados.
El repositorio incluye además una recuperación dinámica **Modern Hopfield experimental** con
energía explícita, criterios de convergencia/estabilidad y comparación longitudinal V1/V2;
se reporta por separado y no sustituye silenciosamente la clasificación validada.

Para el estado actual del proyecto (qué evidencia hay, qué falta), ver `PROJECT_STATUS.md`.
Para el historial de cambios, ver `CHANGELOG.md`. Para el fundamento matemático del modelo
(la dinámica, por qué regla de proyección y no Hebb, cómo se calibra), ver `MODEL.md`.

## Instalación

Tres opciones equivalentes — todas con las mismas versiones fijadas, verificadas con la suite
completa de regresión (176 pruebas). Se recomienda Python 3.12; la aplicación admite Python
3.11 o versiones posteriores.

### pip

```bash
pip install -r requirements.txt
```

### conda

```bash
conda env create -f environment.yml
conda activate coloq
python3 cli.py test          # verificar que el entorno quedó sano
```

Se recomienda `conda config --set channel_priority strict` antes de crearlo. El archivo usa
dos canales: `conda-forge` para casi todo, y `bioconda` para `synapseclient`, que solo existe
ahí.

### Con Docker (recomendado para reproducibilidad)

Imagen publicada en Docker Hub — no hace falta construir nada localmente:

```bash
docker pull pipelinesinmegen/coloq:latest

docker run --rm pipelinesinmegen/coloq:latest test                          # suite de regresión
docker run --rm -v "$(pwd)/results:/app/results_demo" pipelinesinmegen/coloq:latest demo
docker run --rm -p 8501:8501 pipelinesinmegen/coloq:latest app              # interfaz web
```

O construir la imagen localmente desde el `Dockerfile` en `docker/` — **el contexto de build
es la raíz del repositorio**, correr siempre desde ahí con `-f`:

```bash
docker build -f docker/Dockerfile -t coloq:latest .

docker run --rm coloq:latest test                          # suite de regresión
docker run --rm -v "$(pwd)/results:/app/results_demo" coloq:latest demo
docker run --rm -p 8501:8501 coloq:latest app              # interfaz web
```

O con `docker compose` (construye localmente, no usa la imagen de Docker Hub):

```bash
docker compose -f docker/docker-compose.yml run --rm coloq test
docker compose -f docker/docker-compose.yml run --rm coloq demo
docker compose -f docker/docker-compose.yml up app         # http://localhost:8501
```

La imagen corre la suite de tests durante el build: si algo falla, la imagen no se construye.

## Uso

Todo el pipeline se opera desde un solo punto de entrada:

```bash
python3 cli.py --help
python3 cli.py demo                  # pipeline completo sobre datos sintéticos
python3 cli.py app                   # interfaz web (Streamlit)
python3 cli.py test                  # suite de regresión
```

Subcomandos disponibles: `demo`, `calibrate`, `classify`, `validate-external`, `pooled-cox`,
`cox-diagnostics`, `dynamics-diagnostics`, `modern-hopfield`, `prognosis`,
`simulate-treatment`, `app`, `test`. Cada uno delega en el script correspondiente
de `src/` — el CLI solo orquesta, no duplica lógica.

Para ejecutar el diagnóstico dinámico completo y guardar todas las tablas reproducibles:

```bash
python3 cli.py dynamics-diagnostics \
  --patterns results_gse39582/calibrated_patterns.tsv \
  --beta 10 --n-samples 300 --find-interval --full \
  --output results_dynamics/
```

`--full` refina individualmente los estados no clasificados, deduplica equilibrios
estables por distancia, mide cuencas locales con criterios de correlación, distancia y
residuo, ejecuta sensibilidad a ruido/umbral y compara las trayectorias forzadas para
β=2 y β=10. Es diagnóstico de investigación, no selección clínica de β.

Para comparar en los cuatro CMS exactamente las mismas fuerzas sin estabilizador (V1) y con
corrección basal/estabilizador (V2):

```bash
python3 cli.py modern-hopfield \
  --patterns results_gse39582_final/calibrated_patterns.tsv \
  --beta 3.0 --compare-stabilized-sweep \
  --forcing-candidates 0.7 1.5 3 5 8 12 20 \
  --output results_modern_hopfield/
```

La salida incluye el residuo del campo en el origen, dirección del desplazamiento basal,
detalle V1/V2 por CMS y los primeros candidatos que alcanzan el criterio de recuperación.

### Interfaz web

`python3 cli.py app` inicia una aplicación con cinco pestañas: clasificación de muestras contra
patrones calibrados, pronóstico longitudinal post-quirúrgico (con alerta, fuerza de evidencia
del atractor y tratamientos aplicables), simulación contrafactual de tratamiento y una
pestaña de método con el panel, la evidencia acumulada y las limitaciones.

**Pronóstico e Intervención usan Modern Hopfield V2 como único motor dinámico** (β=3.0,
fuerza máxima=5.0 por defecto) — reemplaza por completo la dinámica de proyección anterior,
no es una opción alternativa. Verificado con datos reales de GSE39582 con el criterio más
estricto disponible (`relax_after_forcing_withdrawal`: éxito medido *después* de retirar el
forzamiento): los 4 subtipos CMS se alcanzan de forma robusta, incluido CMS2 (antes
inalcanzable — llegaba a correlación negativa con el objetivo incluso con fuerza=20 bajo la
dinámica anterior). Ver `MODEL.md` sección 10 para la tabla completa de umbrales V1 vs V2.
La dinámica de proyección histórica (`projection_legacy`) sigue disponible por script/CLI
únicamente para reproducibilidad científica de resultados previos, no en la interfaz.

En **Muestras** puede activarse la recuperación Modern Hopfield experimental. Una etiqueta
dinámica solo se acepta si converge, el equilibrio es estable, el residuo es pequeño y la
correlación supera el umbral; las discordancias se muestran sin reemplazar el CMS principal.
Esta es una función distinta del motor dinámico de Pronóstico/Intervención: aquí sigue siendo
experimental y opcional, mientras que el motor de trayectorias ya no lo es.

### Ejecutable de un solo archivo (sin instalar Python)

Para alguien que solo quiere abrir la app sin instalar nada — construye un ejecutable
independiente con la interfaz web ya empacada:

```bash
bash build_executable.sh      # construye dist/ColoQ (~170 MB, primera vez tarda varios minutos)
./dist/ColoQ                  # doble clic en el explorador de archivos hace lo mismo
```

Abre el navegador automáticamente. No requiere Python, `pip`, ni conexión a internet más
allá de la primera construcción. El build es específico de la plataforma donde se corre
(un ejecutable hecho en Linux no corre en Windows/Mac) — hay que construirlo una vez en cada
sistema operativo que se quiera soportar. El `.spec` usado
(`ColoQ.spec`) está documentado con los flags exactos que hacen falta para que Streamlit
empaque correctamente (no es un `pyinstaller app.py` directo — a Streamlit le faltan sus
assets estáticos sin flags adicionales).

## Estructura

```
cli.py                               punto de entrada único (todos los subcomandos)
app.py                               interfaz web (Streamlit)
launcher.py                          arranque para el ejecutable de un solo archivo
ColoQ.spec                           spec de PyInstaller (flags ya resueltos, ver comentarios)
build_executable.sh                  construye dist/ColoQ (ejecutable independiente)
run_pipeline.py                      calibración + validación (invocado por `cli.py calibrate`)
requirements.txt                     dependencias con versiones fijadas (pip)
environment.yml                      entorno conda equivalente (mismas versiones)
.dockerignore                        (en la raíz: Docker lo lee del contexto de build)

docker/
  Dockerfile                         imagen reproducible
  docker-compose.yml                 orquestación (contexto de build = raíz)

src/
  attractor_model.py                 modelo de atractores (ecuación diferencial ordinaria tipo Hopfield)
  calibration.py                     calibración contra datos reales etiquetados
  survival_validation.py             Kaplan-Meier / log-rank
  prognosis.py                       trayectoria/pronóstico longitudinal
  prognosis_demo.py                  demo end-to-end con patrones calibrados reales
  external_validation.py             aplica patrones YA calibrados a cohorte externa (sin recalibrar)
  pooled_cox_validation.py           Cox estratificado combinando múltiples cohortes externas
  cox_diagnostics.py                 diagnósticos formales del Cox (Schoenfeld, influyentes, heterogeneidad)
  dynamics_diagnostics.py            equilibrios/estabilidad reales de la dinámica no lineal
  modern_hopfield.py                  energía moderna, clasificación dinámica y barrido V1/V2
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
  build_external_cohort_generic.py   construye cualquier otra cohorte del Consorcio de Subtipificación del Cáncer Colorrectal (CRCSC) con etiqueta CMS oficial
  clinical_covariates.py             normaliza covariables clínicas, incluido el estadio
  batch_add_cohorts.py               descarga y prepara un lote de cohortes externas
  diagnose_id_mapping.py             diagnostica correspondencias entre identificadores de muestras
  download_geo_gse39582.py           descarga los datos de GSE39582 desde GEO
  error_analysis.py                  analiza errores y umbrales de clasificación
  pooled_cox_validation.py           ejecuta análisis de Cox estratificado entre cohortes
  cox_diagnostics.py                 verifica supuestos del Cox (mismo modelo estratificado)
  power_analysis.py                  estima el poder estadístico del análisis de supervivencia
  treatment_perturbation.py          define las perturbaciones de tratamiento
  treatment_simulation_demo.py       simula trayectorias con y sin tratamiento

tests/            suite de regresión (pytest)
data/             datos (no versionados, ver .gitignore)
figures/          salidas gráficas
```

## Inicio rápido (datos sintéticos, sin credenciales)

```bash
python3 cli.py demo
```

Equivalente corriendo los scripts uno por uno:

```bash
python3 src/synthetic_data.py
python3 run_pipeline.py --input data/synthetic_cohort.tsv --output results/
python3 -m pytest tests/ -v
```

## Panel actual (10 genes, todos compatibles con RT-qPCR)

`MLH1`, `GNLY`, `USP18` (eje CMS1) · `MYC`, `AXIN2` (eje CMS2) · `FABP1`, `CPS1`, `SI` (eje
CMS3) · `VIM`, `TGFB1` (eje CMS4). Ver `PROJECT_STATUS.md` para la justificación de cada gen.
Este es el panel del pipeline calibrado. El modelo conceptual predeterminado de
`src/attractor_model.py` conserva un panel histórico de ocho genes para sus demostraciones y
pruebas unitarias; no se utiliza para calibrar ni validar cohortes reales.

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

Para ajustar el análisis por estadio clínico, indique primero `--stage-col` al construir cada
cohorte y después use `--adjust-stage` en `pooled-cox`. Consulte la ayuda de ambos comandos
para conocer los nombres de columna y las opciones disponibles.

Antes de reportar los HR de `pooled-cox` como definitivos, corra los diagnósticos formales
sobre el mismo modelo estratificado (supuesto de riesgos proporcionales, observaciones
influyentes, heterogeneidad entre cohortes):

```bash
python3 src/cox_diagnostics.py \
  --input results_external_gse17536/scored_external_cohort.tsv \
  --input results_external_gse17537/scored_external_cohort.tsv \
  --adjust-stage --output results_cox_diagnostics/
```

`--adjust-stage` agrega `stage_harmonized` como covariable adicional (requiere que las
cohortes de entrada ya traigan columna `stage`). Por defecto el modelo se ajusta
estratificado por cohorte (`strata=["cohort"]`), igual que `pooled-cox` — usar
`--no-stratify` solo para comparación/depuración explícita, nunca para el resultado
reportado.

### Agregar una cohorte nueva del CRCSC

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

## Panel compatible con PCR (restricción de diseño)

| Capa | Técnica | Qué mide |
|---|---|---|
| Expresión (subtipo CMS) | RT-qPCR relativa | pronóstico basal por subtipo |
| Mutación conductora (BRAF/KRAS) | qPCR alelo-específico (sistema de mutación refractaria a amplificación, ARMS-PCR) o análisis de alta resolución de curvas de fusión (HRM) | marcador pronóstico independiente |
| Estatus MSI (inestabilidad de microsatélites) | HRM de marcadores mononucleotídicos (BAT-25/BAT-26) | favorable en estadio II |
| Seguimiento longitudinal (enfermedad residual mínima, MRD) | qPCR alelo-específico dirigido a la mutación ya identificada en el tumor primario del paciente | vigilancia post-quirúrgica sin NGS |

## Módulo de pronóstico longitudinal (`prognosis.py`)

Convierte una serie temporal de mediciones post-quirúrgicas en una señal de riesgo ordinal:
vector de estado cerca de cero = sin enfermedad residual; vector que se aleja hacia un
atractor = alerta de recurrencia. Demo end-to-end con patrones calibrados reales:

```bash
python3 src/prognosis_demo.py --patterns results_gse39582/calibrated_patterns.tsv
```

## Licencia

MIT — ver [`LICENSE`](LICENSE).
