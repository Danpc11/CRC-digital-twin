# Changelog

Formato: más reciente primero. No sigue versionado semántico estricto (proyecto de
investigación, no paquete distribuido) — cada entrada es un hito de desarrollo.

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
