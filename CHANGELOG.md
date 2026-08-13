# Historial de cambios

**Formato:** más reciente primero.
No sigue un versionado semántico estricto (es un proyecto de investigación)
— cada entrada es un hito de desarrollo.

## Modern Hopfield V2 reemplaza por completo el motor de Pronóstico/Intervención

- Verificado con datos reales de GSE39582 (`--compare-stabilized-sweep`, criterio de éxito
  post-retirada de forzamiento): V2 iguala o mejora a V1 en los 4 CMS. CMS2 pasa de umbral
  3.0 a **1.5**; antes con V1 llegaba a correlación **negativa** con el objetivo incluso con
  fuerza=20 (terminaba pareciéndose a CMS1, el patrón dominante, en vez del objetivo). CMS3
  de 3.0 a 0.7. CMS4 de 8.0 a 5.0 (el más costoso de los 4, único con correlación positiva
  con CMS1).
- `app.py`: se quitó el selector de motor dinámico (`projection_legacy` vs `modern_hopfield`)
  de la barra lateral — Modern Hopfield V2 es ahora el único motor para las pestañas
  Pronóstico e Intervención, con β=3.0 y fuerza máxima=5.0 como valores por defecto
  (verificados, no arbitrarios). `projection_legacy` sigue existiendo en
  `prognosis_demo.py`/`treatment_simulation_demo.py` solo para reproducibilidad científica
  vía script/CLI, ya no como elección en la interfaz.
- Se agregaron 3 pruebas de regresión sin cobertura previa (`modern_hopfield_baseline`,
  `relax_after_forcing_withdrawal`, `summarize_forcing_thresholds`) y se corrigió un import
  roto en un test. Suite actual: **158 pruebas aprobadas**.

## Modern Hopfield: estabilización basal, barrido V1/V2 y clasificación experimental

- Se agregó la energía Modern Hopfield y su flujo de gradiente continuo, con Jacobiano
  simétrico, verificación de energía, equilibrios, estabilidad y cuencas.
- La fase de reposo ahora resta el campo basal uniforme
  $b=X(1/M)$ y añade $-kx$; así $F(0)=0$ incluso cuando los centroides reales no suman
  cero sin ponderar. La energía modificada es $E+b^Tx+(k/2)||x||^2$.
- La transición hacia recaída apaga gradualmente la corrección basal y el estabilizador en
  vez de retirarlos cuando la fuerza todavía vale cero.
- Nuevo `--compare-stabilized-sweep`: compara V1 y V2 con idénticos β, tiempos, objetivos
  y candidatos de fuerza para los cuatro CMS; guarda detalle y umbrales en TSV.
- La app conserva la correlación CMS como clasificación principal y permite ejecutar una
  recuperación Modern Hopfield experimental con criterios explícitos de convergencia,
  estabilidad, residuo y correlación. Las discordancias no se fuerzan.
- El análisis de cuencas ahora refina y verifica estabilidad/residuo; el barrido de β exige
  separación mínima real entre los equilibrios.
- Suite actual: **135 pruebas aprobadas**.

## Ronda de coherencia: panel y números de evidencia alineados en todo el código

- **`attractor_model.py` seguía con el panel placeholder viejo de 8 genes** (`GZMB`,
  `KRAS_sig`) pese a que el panel congelado es de 10 — actualizado a los 10 genes
  actuales, con `MLH1` con signo **negativo** en CMS1 (baja expresión por silenciamiento
  epigenético, consistente con el descubrimiento documentado abajo y con
  `synthetic_data.py`). El self-check del módulo sigue convergiendo a los 4 atractores
  correctos (r ≥ 0.986).
- **`prognosis_demo.py::EVIDENCE_STRENGTH` tenía los números del Cox de 3 cohortes**
  (n=326, CMS4 HR=1.88 p=0.03, CMS1 p=0.09 "tendencia") — actualizado al modelo
  ajustado por estadio de las 4 cohortes (n=388): CMS4 HR=2.06 (p=0.018, robusto al
  ajuste), CMS1 HR=2.09 (p=0.016 — significativo por primera vez, marcado
  explícitamente como resultado provisional pendiente de confirmación), CMS3 p=0.11
  con la salvedad de poder (45%).
- **`app.py`** (pestaña Método): la tabla de cohortes no incluía GSE33113 y el pie
  citaba el Cox viejo (n=326, p=0.045) — actualizado a las 4 cohortes externas
  (n=415/388, p<0.001). El medidor de evidencia por atractor se actualiza solo
  vía `EVIDENCE_STRENGTH`.
- **Tests**: `test_state_dimension_matches_gene_panel` esperaba N=8 — ahora N=10 con
  nota apuntando a los archivos que hay que tocar si el panel cambia. El test de
  drivers era tautológico (`or True` lo hacía pasar siempre) — reescrito para
  verificar forma, magnitud y alineación real con los patrones.
- **Referencias "41 tests"** en `requirements.txt` y `environment.yml` — la suite ya
  tiene 42 (el badge del README estaba correcto).
- **`feature_selection.py::CURRENT_PANEL`** incluía `GZMB` y omitía `GNLY`/`USP18` —
  corregido; el reporte de candidatos marcaba mal qué genes son "nuevos".
- **`error_analysis.py`**: paleta de colores CMS distinta del resto del proyecto
  (CMS4 en rosa en vez de bermellón) — unificada a la asignación estándar de
  `app.py`/`plot_survival_curves.py`; texto interpretativo que mencionaba `GZMB`
  actualizado al eje inmune actual (`GNLY`/`USP18`).
- **`download_synapse_data.py`**: el docstring apuntaba a `scripts/` — la carpeta
  es `src/`.

## Cuarta cohorte externa (GSE33113) y modelo ajustado por estadio robusto

- **Bug de parser encontrado y corregido**: `!Sample_title` (que en GSE33113 trae el ID
  interno `col001` usado por las etiquetas CMS del consorcio, distinto del GSM de GEO) venía
  ANTES de `!Sample_geo_accession` en el archivo, orden contrario al asumido por el parser de
  una sola pasada. Se descartaba en silencio. Corregido con parser de dos pasadas: primero
  localiza `sample_ids` sin importar su posición, luego procesa el resto de los campos.
- **Puente de ID agregado a `build_external_cohort_generic.py`**: cuando el cruce directo por
  GSM da cero coincidencias, prueba automáticamente `Sample_title`/`Sample_description` como
  puente antes de fallar (con mensaje claro si tampoco funciona, listando los campos
  disponibles para inspección manual).
- **`STAGE_MAP` extendido** con el formato `"AJCC stage X CRC"` (visto en el campo
  `disease status` de GSE33113, cohorte diseñada como estadio II homogéneo).
- `pooled_cox_validation.py` extendido con **modelo crudo restringido a la misma muestra del
  ajustado** (separa "atenuación por ajuste real" de "pérdida de poder por menos eventos") y
  diagnóstico automático por covariable.
- **Resultado con las 4 cohortes (n=415, 100 eventos)**: modelo ajustado por estadio
  significativo globalmente (p<0.001, n=388). CMS4 HR=2.06 (p=0.018), robusto al ajuste
  (apenas se mueve entre crudo restringido y ajustado) — a diferencia de la ronda anterior con
  3 cohortes, donde el efecto de CMS4 sí se atenuaba y perdía significancia al ajustar. CMS1
  también resultó significativo (HR=2.09, p=0.016) — resultado nuevo, no replicado antes de
  esta cohorte, tratar con cautela.
- `power_analysis.py`: el poder de CMS4 subió de 58% a 76% con la cuarta cohorte; CMS1 pasó
  de estar subpotenciado a 75%. CMS3 sigue en 45%.

## Simulación de tratamiento y reporte de evidencia por atractor

- `treatment_perturbation.py`: tres mecanismos de tratamiento (inmunoterapia anti-PD1,
  anti-EGFR, quimioterapia citotóxica), condicionados por biología del paciente, evidencia
  clínica citada por mecanismo:
  - Inmunoterapia: KEYNOTE-177 (Andre et al. 2020 NEJM; actualización 2024 Ann Oncol),
    HR=0.60-0.73 en MSI-H/dMMR. Sin beneficio basal fuera de CMS1 (tumores MSS
    "inmunológicamente fríos" — sin piso artificial de eficacia).
  - Anti-EGFR: Karapetis et al. 2008 NEJM; Douillard et al. 2013 NEJM; Di Nicolantonio
    et al. 2008 JCO — requiere RAS/BRAF wild-type. Sin estatus real disponible, usa
    proxy débil por cercanía a CMS3 (penalizado por incertidumbre), nunca sustituye la
    prueba de mutación real.
  - Quimioterapia citotóxica: eficacia reducida (no nula) en pacientes cerca de CMS4,
    consistente con la evidencia empírica propia del proyecto.
- **Bug de diseño encontrado y corregido durante la construcción**: la primera versión
  empujaba genes específicos del mecanismo hacia arriba (ej. reforzar GNLY/USP18 en
  inmunoterapia), lo cual hundía al paciente más en su propio atractor en vez de
  simular limpieza tumoral — el hazard subía con tratamiento "efectivo" en vez de
  bajar. Corregido: un tratamiento efectivo ahora se representa como una fuerza de
  amortiguamiento que jala el vector de estado hacia el origen, proporcional a la
  eficacia condicionada, no como un empuje sobre genes del propio atractor.
- `treatment_simulation_demo.py`: simulación contrafactual (misma trayectoria de
  recaída, con y sin tratamiento), demuestra divergencia real cuando el mecanismo
  aplica al subtipo del paciente.
- `prognosis_demo.py` extendido: cada alerta de recurrencia ahora reporta (a) hacia
  qué atractor se dirige el paciente, (b) fuerza de evidencia externa de ESE atractor
  específico (`EVIDENCE_STRENGTH`, derivado del Cox combinado de 3 cohortes — CMS4
  fuerte, CMS1 tendencia no significativa, CMS3 sin evidencia), y (c) qué tratamientos
  tienen mecanismo aplicable a ese estado, con aviso explícito cuando `anti_egfr` solo
  se apoya en el proxy débil de RNA.
- Bug encontrado y corregido en `synthetic_data.py`: seguía con el panel viejo de 8
  genes (`GZMB`, `KRAS_sig`) en vez de los 10 genes actuales — causaba fallos en 3
  tests nuevos. Corregido, incluyendo `MLH1` con signo negativo en el centroide CMS1
  (consistente con la dirección real descubierta: baja expresión, no alta).

## Validación combinada de 3 cohortes externas

- `pooled_cox_validation.py`: Cox de riesgos proporcionales estratificado por cohorte,
  combina múltiples cohortes externas en un solo análisis con más poder que log-rank
  tests separados por cohorte.
- Combinando GSE17536+GSE17537+GSE14333 (n=326): test global p=0.045 (primera vez que
  el modelo completo de 4 grupos cruza significancia en validación externa). CMS4:
  HR=1.88, p=0.03. CMS3 sin señal (p=0.70).
- `build_external_cohort_generic.py`: versión generalizada para agregar cualquier cohorte
  CRCSC adicional ya etiquetada por el consorcio, sin asumir plataforma ni nombres de
  columna de supervivencia (dos pasos: `--diagnose` primero, construcción después).

## Corrección de bugs en el parser de GEO (`parse_geo_series_matrix.py`)

- **Bug de desalineación posicional**: el parser original asumía que todas las muestras
  de una serie tienen sus `!Sample_characteristics_ch1` en el mismo orden. En GSE17537
  esto causó que la columna `overall_event` mezclara grados de diferenciación tumoral
  con death/no death. Corregido: parseo por celda (par `atributo: valor` extraído
  directamente de cada celda), no por posición de línea.
  Verificado que GSE39582/GSE17536 NO fueron afectados en las columnas de supervivencia
  usadas (0 diferencias tras re-parsear).
- **Bug de formato empacado**: GSE14333 empaqueta todos sus atributos en una sola celda
  separados por `;` (`"Location: Right; DukesStage: A; DFS_Time: 3.64; DFS_Cens: 1"`),
  a diferencia del formato de una línea por atributo de otras series. El parser ahora
  maneja ambos formatos (split por `;` antes de split por `:`).

## GSE17537 (validación externa fresca)

- Construida con `build_gse17537_dataset.py`. El consorcio CMS **no** etiquetó esta
  subserie (sí etiquetó GSE17536) — se maneja con `cms_label="none"` explícito, sin
  línea base de comparación oficial para esta cohorte específicamente.
- Resultado individual: p=0.71 (n=55, subpotenciada). No usada para ajustar el panel en
  ningún momento — es la única cohorte externa genuinamente no tocada.

## GSE14333

- Construida con `build_external_cohort_generic.py`. Columnas de supervivencia
  (`DFS_Time`/`DFS_Cens`) requirieron verificación de dirección: `DFS_Cens=1` resultó
  significar **censurado** (no evento), no evento como sugeriría la convención moderna
  por defecto — confirmado empíricamente cruzando contra estadio Dukes (proporción de
  `DFS_Cens=1` baja monótonamente con estadio más avanzado, consistente con "censurado",
  no con "evento").
- Resultado individual: p=0.59 (modelo), p=0.13 (etiqueta oficial) — ninguno separa en
  esta cohorte sola, pero aporta poder real al análisis combinado.

## Ablación de `USP18`

- Prueba de remover `USP18` del panel (solo `GNLY` para CMS1): kappa interno bajó
  0.679→0.664, p de supervivencia empeoró 10x (0.0004→0.041). **USP18 restaurado.**

## Selección data-driven de genes (`feature_selection.py`)

- AUC uno-contra-el-resto + ANOVA F-test + Random Forest sobre el transcriptoma completo
  de GSE39582 (23,520 genes), no solo el panel actual.
- Reveló: `GZMB` tenía AUC mediocre (rank #2,837 de 23,520) para CMS1; `MLH1` tenía
  dirección invertida (AUC=0.229, es decir, **baja** expresión en CMS1 — consistente con
  silenciamiento epigenético de MLH1 como causa de MSI esporádica, no con el placeholder
  original que asumía alta expresión).
- `GNLY` (AUC=0.886) y `USP18` (AUC=0.888) identificados como reemplazo de `GZMB`.
  Resultado: kappa interno 0.639→0.679; validación externa (GSE17536) 0.168→0.090.
- `CKLF` (agregado antes por literatura, ver abajo) resultó con AUC=0.633 (rank #3,882) —
  explica por qué no ayudó: su validación original era para pronóstico de recaída dentro
  de CMS1 ya diagnosticado, no para la tarea de clasificación de subtipo.

## `CKLF` agregado y revertido

- Agregado al eje CMS1 por respaldo de literatura (marcador de riesgo de recaída
  específico de CMS1). Validación externa empeoró (GSE17536: p 0.168→0.365).
  **Revertido.** Lección: un gen validado para una tarea (pronóstico dentro de subtipo)
  no garantiza utilidad en otra (clasificación de subtipo).

## Descubrimiento: TCGA no tiene RFS curado

- `TCGACRC_clinical-merged.tsv` tiene `dfsMo`/`dfsStat` (RFS) vacío (0/603 no-nulos) pero
  `osMo`/`osStat` (OS) completo (603/603). `format_to_schema.py` ajustado para usar OS
  con nombres de columna explícitos (`overall_survival_months`/`death_event`, no
  `relapse_free_months`/`relapse_event`) para no mezclar endpoints bajo el mismo nombre.
- Resultado: ni el modelo ni la etiqueta oficial separan supervivencia en TCGA-OS
  (p=0.33 y p=0.35) — evidencia de que el endpoint (OS en una cohorte con seguimiento
  inmaduro), no el panel, es la limitación en ese caso.

## `CPS1` + `SI` agregados (eje CMS3)

- Por literatura (Yuan et al. 2025, *Nat Commun* — diferenciación enterocito-like en
  organoides KRAS-mutantes). Kappa interno 0.533→0.639. Reemplazó el placeholder
  `KRAS_sig` (nunca fue un gen real, solo un marcador de posición del esqueleto inicial).

## Calibración inicial contra datos reales

- `calibration.py`: centroides empíricos por subtipo a partir de TCGA-COAD/READ y
  GSE39582, reemplazando los patrones placeholder hechos a mano del esqueleto original.
- `external_validation.py`: aplica patrones ya calibrados a una cohorte nueva sin
  recalibrar (metodología correcta de validación externa).
- `concordance_analysis.py`: matriz de confusión modelo vs. etiqueta oficial del
  consorcio, Cohen's kappa.

## Esqueleto inicial

- `attractor_model.py`: red tipo Hopfield continua, 4 atractores (CMS1-4), matriz de
  acoplamiento por regla de proyección (Personnaz-Guyon-Dreyfus, no Hebb clásica —
  evita el límite de capacidad de ~0.14N patrones en espacios de baja dimensión).
- Panel placeholder de 8 genes con patrones hechos a mano (no calibrados).
- `prognosis.py`: módulo de trayectoria/pronóstico longitudinal (distancia al origen
  como proxy de riesgo).
