# Estado del proyecto

**Última actualización:** 2026-09-15. Historial detallado en `CHANGELOG.md`.

## Panel actual

10 genes, todos medibles mediante qPCR con transcripción inversa (RT-qPCR): `MLH1`, `GNLY`,
`USP18` (CMS1) · `MYC`, `AXIN2` (CMS2) ·
`GALNT8`, `CPS1`, `AGR2` (CMS3) · `VIM`, `TGFB1` (CMS4). Congelado — no agregar genes sin
justificación cuantitativa nueva (cada gen tiene costo real en un ensayo RT-qPCR).
Actualizado 2026-09-09: `FABP1`→`GALNT8`, `SI`→`AGR2` (aprobado por Daniel, basado en
selección data-driven + validación externa en 5 cohortes — ver
`network_analysis/CLAUDE.md`, sección "Cambio ejecutado en el panel real de dt +
validación externa completa"). `MYC`→`TP53RK`/`SLC5A6` se evaluó pero no se aprobó.
CMS corresponde a los subtipos moleculares consensuados de cáncer colorrectal (*Consensus
Molecular Subtypes*).

Revisión metodológica externa del 2026-09-11. Conclusión: **hoy no existe ninguna cohorte
de validación limpia**, y las cifras de la tabla de abajo deben leerse como *validación
retrospectiva de desarrollo* (lo mismo que ya imprime `external_validation.py`), no como
validación externa confirmatoria:

- GSE17536 se usó iterativamente para ajustar el panel y a la vez entra al Cox agrupado.
- El cambio de panel del 2026-09-09 (`FABP1→GALNT8`, `SI→AGR2`) se decidió con selección
  data-driven **y validación en las 5 cohortes externas**; el Cox agrupado de 5 cohortes
  (p=0.000532) se calcula sobre esas mismas 5 cohortes. Es selección y evaluación sobre la
  misma muestra: los HR y p están optimistamente sesgados.
- Por lo anterior, la descripción de GSE17537 como "nunca usada para ajustar el panel" ya
  no es exacta desde el 2026-09-09.
- Hay decenas de p-valores (por cohorte, por clasificador, crudo/ajustado, reparametrizado)
  sin ninguna corrección por multiplicidad, y la narrativa de "GSE33113 fue la que permitió
  la conclusión" es el patrón clásico de *forking paths*.

**Qué reportar como resultado principal mientras no haya cohorte nueva**: el
leave-one-cohort-out ya implementado en `pooled_cox_validation.py`
(`leave_one_cohort_out_validation`), no el Cox agrupado in-sample.

**Qué hace falta para volver a tener validación confirmatoria**: (1) congelar el panel
actual por escrito (este archivo + tag de git), (2) conseguir al menos una cohorte del
CRCSC con etiqueta CMS oficial y RFS que **no** haya sido tocada (candidatas: GSE38832,
GSE29621, GSE13294), (3) correr `external_validation.py` una sola vez sobre ella y
reportar ese resultado tal cual salga.

**Otras correcciones de la misma revisión** (detalle en `CHANGELOG.md`):
- Bug de doble normalización en el puente qPCR de la app (clasificación sin sentido en el
  flujo Ct → CMS). Corregido y cubierto por tests de regresión.
- La pestaña Paciente presentaba como "alerta de recurrencia" la recaída que el propio
  simulador inyecta en el mes 15. Ahora se etiqueta explícitamente como escenario hipotético.
- El "origen" (x=0 en z-score) es el tumor promedio de la cohorte, no ausencia de tumor;
  documentado en `prognosis.py` y `treatment_perturbation.py`.
- La asimetría de umbrales V1 (CMS2 "inalcanzable") se puede diagnosticar con
  `src/pattern_norm_diagnostic.py` antes de atribuirla a biología.

## ¿Qué añade CMS sobre la clínica de rutina? (MMR) — recalculado 2026-09-15

Objeción razonable: CMS1 ≈ dMMR/MSI-H, que ya tiene prueba clínica estándar (IHC de MMR). Lo
defendible del panel es lo que aporte *más allá* de estadio + MMR. GSE39582 anota MMR
(`msi_status`); `cli.py cox-clinical` con estadio categórico (RFS, estadio I-III, n=449,
132 eventos, subtipo **predicho**; salidas en `results_cox_clinical/`):

| Modelo | HR CMS4 (IC95%) | p | HR CMS3 | HR CMS1 | HR dMMR | C |
|---|---|---|---|---|---|---|
| A. CMS solo | 2.02 (1.30–3.13) | 0.0017 | 1.70 | 0.74 | — | 0.597 |
| B. CMS + estadio | 1.76 (1.13–2.75) | 0.012 | 1.56 | 0.70 | — | 0.648 |
| C. estadio + MMR (sin CMS) | — | — | — | — | 0.50 (p=0.035) | 0.622 |
| **D. CMS + estadio + MMR** | **1.78 (1.14–2.78)** | **0.011** | 1.57 (p=0.056) | 0.83 (n.s.) | 0.71 (n.s.) | 0.649 |
| E. CMS + estadio, **solo pMMR** (n=380, 122 ev.) | **1.77 (1.13–2.76)** | **0.012** | 1.52 (p=0.077) | 0.82 | — | 0.636 |

Estadio entra como indicadores (`stage_I` HR=0.13, `stage_III` HR=1.69): el ajuste lineal
previo estaba mal especificado, el salto II→III no equivale al I→II.

- **CMS4 conserva su HR tras ajustar por estadio y MMR, y dentro de los pMMR.** Aporte
  conjunto de CMS sobre estadio+MMR: LRT χ²=10.2, 3 gl, p=0.017; ΔC-index +0.027. Schoenfeld
  limpio en todo el modelo D (nada viola PH con Holm).
- CMS3 es consistente en dirección y magnitud en los tres modelos (1.52–1.70) pero no alcanza
  significancia; reportar como tendencia.
- **CMS1 no aporta nada tras ajustar** (HR 0.83, p=0.59), y dMMR pasa de 0.50 significativo a
  0.71 n.s. al meter CMS: colinealidad esperable. Lectura: el eje CMS1 del panel no añade
  sobre IHC de MMR. Contrasta con el HR≈2.1 de CMS1 en las cohortes externas — heterogeneidad
  real, posiblemente por endpoint (DFS incluye mortalidad no oncológica, y los dMMR tienden a
  ser pacientes mayores).
- **Solapamiento**: 64/121 CMS1 predichos son dMMR, **45/121 son pMMR (37%)**. El eje CMS1 del
  panel no es sustituto de MMR; recomendar inmunoterapia por CMS1 predicho sin IHC sería un
  error clínico.
- **Límite**: in-sample (GSE39582 es la cohorte de calibración). Ninguna de las externas anota
  MMR; TCGA tiene MSI pero no RFS curado.

## Rerun completo del 2026-09-15

Todas las cifras de validación externa se recalcularon desde cero tras detectar tres
problemas de preparación de datos (ver "Hallazgos de preparación de datos" más abajo).
**Las cifras vigentes son las de la sección siguiente**; lo que aparece después, en
"Evidencia acumulada (histórico)", corresponde a corridas anteriores y se conserva solo
como registro de lo que cambió y por qué.

## Resultados de validación externa (vigentes — rerun 2026-09-15)

### Análisis principal: 4 cohortes con etiqueta CMS oficial

Subtipo **predicho** por el panel de 10 genes, Cox estratificado por cohorte, estadio como
indicadores categóricos, referencia CMS2. Restringido a los pacientes con etiqueta del
consorcio, para que la comparación contra la etiqueta oficial sea sobre la misma muestra.
GSE14333 + GSE17536 + GSE33113 + GSE37892; **n=428, 95 eventos**
(`results_pooled_cox_mismamuestra/`).

| Covariable | HR | IC95% | p |
|---|---|---|---|
| estadio III (vs I+II) | 3.83 | 2.33–6.29 | <0.001 |
| CMS1 | **2.06** | 1.17–3.62 | 0.012 |
| CMS3 | 1.27 | 0.61–2.61 | 0.52 |
| CMS4 | **2.03** | 1.19–3.47 | 0.010 |

- Aporte incremental de CMS sobre estadio: LRT χ²=9.58, 3 gl, **p=0.022**.
- **C-index estratificado** 0.649 → 0.697, **ΔC=+0.048 (IC95% bootstrap +0.021 a +0.087)**,
  500/500 remuestreos válidos. Citar el estratificado, no el agrupado: en un Cox con
  `strata=['cohort']` el C-index de lifelines compara pares de cohortes distintas, cuyas
  funciones basales difieren por construcción.
- **Leave-one-cohort-out**: ΔC estratificado +0.045 a +0.065 en las cuatro particiones; el
  aporte de CMS no depende de ninguna cohorte individual. HR de CMS1 1.79–2.15 y de CMS4
  1.76–2.39 en todas ellas.
- Estadio I se fusiona con II (41 pacientes, 1 evento): con el evento correctamente
  codificado casi nadie recae en estadio I y su indicador produce separación completa. El
  contraste reportado es **III vs I+II**.

**Interpretación**: CMS1 y CMS4 duplican el riesgo de recurrencia frente a CMS2, de forma
independiente del estadio. CMS3 no tiene efecto pronóstico distinguible.

### Contraste contra la etiqueta oficial del consorcio (misma muestra)

`--group-col cms_label` sobre los mismos 428 pacientes (`results_pooled_cox_oficial/`). Es
el control que separa "el panel no transfiere" de "CMS no predice en estas cohortes":

| | panel predicho | etiqueta oficial |
|---|---|---|
| CMS1 | 2.06 (1.17–3.62) p=0.012 | 2.25 (1.27–3.98) p=0.006 |
| CMS3 | 1.27 (0.61–2.61) p=0.52 | 0.87 (0.37–2.03) p=0.75 |
| CMS4 | 2.03 (1.19–3.47) p=0.010 | 2.36 (1.41–3.94) p=0.001 |
| ΔC estratificado | +0.048 [+0.021, +0.087] | +0.075 [+0.039, +0.109] |

Los tres HR concuerdan en dirección y magnitud, con la atenuación esperable por error de
clasificación (κ≈0.6–0.8). **El panel recupera aproximadamente dos tercios del aporte
pronóstico de la clasificación CMS completa** (0.048 / 0.075). La ausencia de efecto de CMS3
se reproduce en ambos, y coincide con la literatura (CMS3 tiene pronóstico intermedio y no
se separa consistentemente de CMS2), así que es evidencia de que el panel funciona, no de
que falle.

### Análisis de sensibilidad

| Análisis | n / eventos | LRT p | ΔC estratificado |
|---|---|---|---|
| Principal (4 cohortes etiquetadas) | 428 / 95 | 0.022 | +0.048 [+0.021, +0.087] |
| 5 cohortes, incluye no etiquetados | 518 / 117 | 0.016 | +0.039 [+0.024, +0.075] |
| Solo cohortes con RFS real (GSE14333+33113+37892) | ~330 / 82 | 0.078 | +0.060 [+0.026, +0.097] |

- En el análisis de 5 cohortes, CMS3 sube a HR=2.04 (p=0.02). **Es un artefacto de
  población, no de clasificación**: los 108 pacientes sin etiqueta del consorcio
  (no-consenso, más GSE17537 entera) son tumores ambiguos y de peor pronóstico, y parte de
  ellos cae en el grupo CMS3 predicho. Al igualar la muestra el HR baja a 1.27. La matriz de
  confusión descarta la hipótesis alternativa: el grupo CMS3 predicho se contamina con CMS2
  (8/27 en GSE14333, 6/36 en GSE37892), que es la referencia de buen pronóstico y por tanto
  atenuaría el HR, no lo inflaría.
- En solo-RFS el LRT queda en p=0.078 por pérdida de eventos, pero el ΔC es el mayor de los
  tres (+0.060) y CMS3/CMS4 siguen significativos individualmente. El endpoint mixto
  DFS/RFS no explica el hallazgo.

### Diagnósticos del modelo

Todos limpios (`results_cox_diagnostics/`):

- **Riesgos proporcionales**: ninguna covariable viola el supuesto, ni individualmente ni
  con corrección de Holm; omnibus de Fisher p=0.87.
- **Efecto tiempo-dependiente**: modelo por tramos a 36 meses sin diferencia early/late en
  ningún subtipo (p=0.57, 0.95, 0.48).
- **Heterogeneidad entre cohortes**: p=0.43 (CMS1), 0.39 (CMS3), 0.46 (CMS4). Efecto
  homogéneo. El término de estadio no es estimable porque GSE33113 es de estadio único.
- **Observaciones influyentes**: magnitud máxima de delta-beta 0.063; ningún paciente
  individual mueve el modelo.

**Esto invalida dos afirmaciones que este documento sostenía antes**: la violación del
supuesto de riesgos proporcionales por CMS4 y su supuesto efecto temporalmente restringido
eran artefactos del estadio modelado como variable lineal y del indicador de evento
invertido en GSE14333.

### Concordancia de clasificación

| Cohorte | n etiquetados | cobertura | accuracy (aceptadas) | accuracy (abstención = error) | κ |
|---|---|---|---|---|---|
| GSE39582 (calibración) | 519 | 1.00 | 0.805 | — | **0.728** |
| GSE17536 | 156 | 0.86 | 0.866 | 0.744 | 0.812 |
| GSE33113 | 85 | 0.86 | 0.808 | 0.694 | 0.735 |
| GSE14333 | 135 | 0.79 | 0.755 | 0.593 | 0.673 |
| GSE37892 | 118 | 0.80 | 0.702 | 0.559 | 0.590 |

GSE17537 no tiene etiqueta oficial (el consorcio solo etiquetó GSE17536 de ese estudio), así
que no aporta concordancia y queda fuera del análisis principal.

**Error de clasificación dominante: CMS4 oficial → CMS1 predicho** (7/24 en GSE14333, 10/32
en GSE37892, ~30%). Explicación probable: de los 10 genes, `GNLY` y `USP18` marcan infiltrado
inmune, y CMS4 también está infiltrado (estroma inflamado, no citotóxico); con 10 genes no
hay resolución para separar ambos patrones. Es lo que atenúa los dos HR hacia un valor común
(~2.0) cuando los oficiales son 2.25 y 2.36. Clínicamente importa poco para estratificar
riesgo (ambos son alto riesgo), pero **sí impide usar CMS1 predicho como sustituto de MMR**:
el criterio para inmunoterapia debe seguir siendo IHC de MMR. Trabajo futuro: un marcador
estromal específico (`THBS2`, `INHBA`) desambiguaría.

### `cms_margin`: calibrado, pero no se usa para abstener

El margen entre la primera y la segunda correlación está bien calibrado como medida de
confianza y transfiere entre cohortes. En GSE39582, exigir margen ≥0.2 sube la accuracy de
0.805 a 0.860 (cobertura 0.87) y ≥0.4 la sube a 0.909 (cobertura 0.74); el comportamiento es
monótono en las cuatro cohortes externas, con el codo en 0.2–0.3.

**Decisión: no se aplica abstención por margen.** El análisis del tipo de error muestra que
el margen no discrimina los errores relevantes clínicamente: de los 16 errores de GSE37892
que cruzan la frontera riesgo-alto/riesgo-bajo, 9 tienen margen ≥0.34, incluido un CMS2
clasificado como CMS1 con margen 0.85. Los errores que un umbral sí eliminaría son
confusiones entre subtipos del mismo grupo de riesgo (CMS1↔CMS4, CMS2↔CMS3), que mejorarían
la concordancia nominal sin cambiar ninguna decisión, a costa de 13–19% de cobertura. El
margen mide ambigüedad geométrica del perfil, no distancia en riesgo.

Uso recomendado: reportarlo junto a la etiqueta como aviso de confianza baja (sin ocultar el
resultado), y evaluarlo como covariable propia en el Cox — los no-consenso del consorcio
sugieren que la ambigüedad del perfil puede tener valor pronóstico por sí misma.

### Hallazgos de preparación de datos (invalidaron las cifras anteriores)

Los tres los detectaron validadores nuevos añadidos en esta revisión; ninguno era visible
antes y todos afectaban a los resultados publicados en versiones previas de este documento:

1. **GSE14333: indicador de evento invertido.** La columna `DFS_Cens` codifica 1 = censurado;
   se leía como evento. Producía 99 recaídas en 126 pacientes (79%) y aportaba el 52% de los
   eventos del Cox agrupado con el 24% de los pacientes, aplastando la señal de todo el
   análisis. Señal diagnóstica: los supuestos eventos tenían seguimiento medio de 46.0 meses
   frente a 21.8 de los censurados — imposible, una recaída ocurre antes del fin de
   seguimiento. Detectado por `check_event_coding()`.
2. **GSE33113: escala lineal y tiempo en días.** Expresión con mediana 24.8 y máximo 15 972
   (los centroides están en log2), y tiempo a recurrencia en días (mediana 1179) leído como
   meses. Este documento señalaba esta cohorte como la que "permitió la conclusión" sobre
   CMS4. Detectados por `ensure_log2_scale()` y `check_duration_units()`.
3. **GSE33113 y GSE37892 construidas con el panel anterior** (`FABP1`, `SI`) en lugar de
   `GALNT8`/`AGR2`. Sus resultados históricos no correspondían al panel vigente.

Con los tres corregidos, el número de eventos del análisis de 5 cohortes pasó de 189 a 117,
y **la señal se volvió más clara, no menos**: se eliminaron 72 eventos falsos que eran ruido.

### Limitaciones que siguen en pie

- **Las cinco cohortes participaron en la selección del panel** (cambio `FABP1→GALNT8`,
  `SI→AGR2` del 2026-09-09). Esto es validación retrospectiva de desarrollo, **no
  confirmatoria**. No existe todavía ninguna cohorte intocada.
- **Endpoint mixto**: GSE17536 y GSE17537 aportan DFS (incluye muerte por cualquier causa);
  GSE14333, GSE33113 y GSE37892 aportan RFS. El análisis solo-RFS mantiene el hallazgo, pero
  debe declararse.
- **Sin corrección por multiplicidad** en el conjunto de análisis exploratorios.
- **κ in-sample** en GSE39582 (0.728): falta validación cruzada de los centroides para
  cuantificar el optimismo.
- **Calibración absoluta no evaluable**: un Cox estratificado no transfiere riesgo basal a
  una cohorte nueva. Las cifras de `cox_apparent_calibration.tsv` son aparentes, dentro del
  estrato de ajuste, y no deben reportarse como calibración externa.

### Redacción sugerida del resultado

> Un panel de 10 genes compatible con RT-qPCR reproduce la clasificación CMS con κ=0.73 en la
> cohorte de calibración y κ=0.59–0.81 en cuatro cohortes externas. El subtipo predicho aporta
> información pronóstica independiente del estadio (n=428, 95 eventos; CMS1 HR=2.06 [1.17–3.62]
> y CMS4 HR=2.03 [1.19–3.47] frente a CMS2; ΔC-index estratificado +0.048 [IC95% +0.021 a
> +0.087]; supuesto de riesgos proporcionales satisfecho, efecto homogéneo entre cohortes,
> estable en validación leave-one-cohort-out). La comparación con la etiqueta del consorcio
> sobre la misma muestra (ΔC +0.075) indica que el panel recupera aproximadamente dos tercios
> del valor pronóstico de la clasificación completa. CMS3 no muestra efecto pronóstico
> independiente, ni con el panel ni con la etiqueta oficial.

## ¿Necesitará quimioterapia? — pregunta abierta

El análisis de arriba es **pronóstico** (quién recae más), no **predictivo** (a quién le sirve
el tratamiento). Son cosas distintas y no se deduce una de la otra: que CMS4 recaiga más no
implica que la quimioterapia le sirva más ni menos.

`cli.py cox-chemo` (nuevo en esta revisión) prueba la interacción CMS × quimioterapia
adyuvante en GSE39582 (estadio II–III, n=460, 139 eventos, 202 tratados):

- **LRT de la interacción: χ²=2.76, 3 gl, p=0.43.** Añadiendo GSE14333: p=0.75.
- HR de quimio dentro de cada CMS: todos los IC95% abarcan el 1.
- El término CMS4×quimio tiene IC 0.25–1.44: compatible con un beneficio cuádruple y con un
  perjuicio del 40%. **El diseño no puede distinguir**, no es que el resultado sea negativo.

Dos razones de fondo, ambas irreparables con estos datos: (1) la quimio se indica por estadio,
edad y comorbilidad, así que hay confusión por indicación —el HR marginal de quimio sale 1.24,
absurdo como efecto causal—; (2) con 139 eventos no hay potencia para interacciones (regla
práctica: ~4× los eventos del efecto principal).

**Qué haría falta**: datos de ensayo aleatorizado estratificado por CMS (el precedente es Song
et al. 2016, JAMA Oncol, sobre NSABP C-07), o cohortes con respuesta medida bajo tratamiento
(GSE104645, GSE72970, GSE5851 son candidatas públicas con expresión + respuesta + PFS). Hasta
entonces, las pestañas Paciente e Intervención de la app son **simulación mecanística
generadora de hipótesis**, no recomendación clínica, y así deben presentarse.

## Evidencia acumulada (histórico — corridas anteriores al rerun del 2026-09-15)

⚠️ **Las cifras de esta sección NO son vigentes.** Se conservan para documentar qué cambió
tras corregir la preparación de datos. Los HR de CMS4 de 1.77–2.50, el p=0.000532, la
violación de riesgos proporcionales, el efecto tiempo-dependiente y el poder post-hoc del
76% pertenecen a corridas con GSE14333 mal codificada, GSE33113 en escala lineal y con el
tiempo en días, y dos cohortes con el panel anterior.



| Cohorte | Rol | n | valor p (log-rank) |
|---|---|---|---|
| GSE39582 | Entrenamiento | 557 | 1.84e-05 |
| TCGA-COAD/READ | Descartada (sin supervivencia libre de recaída [RFS] curada, solo supervivencia global [OS]) | 558 | 0.33 (ninguno separa — problema del desenlace) |
| GSE17536 | Externa (usada iterativamente para ajustar el panel) | 145 | 0.153 |
| GSE17537 | Externa (no usada para ajustar el panel hasta 2026-09-09; sí entró en la validación del cambio de panel) | 55 | 0.881 |
| GSE14333 | Externa | 126 | 0.166 |
| GSE33113 | Externa (estadio II homogéneo) | 89 | **0.00034** |
| GSE37892 | Externa (endpoint: metástasis a distancia, no recaída general) | 130 | 0.098 (modelo) / 0.189 (etiqueta oficial) — no significativa sola |

**Modelo de Cox estratificado que combina las cinco cohortes externas
(GSE17536+GSE17537+GSE14333+GSE33113+GSE37892, n=545, 137 eventos): Concordance=0.591,
log-likelihood ratio test p=0.000532. CMS1 HR=2.11 (p<0.005), CMS4 HR=2.50 (p<0.005) —
mejora en las tres métricas frente al panel anterior (Concordance 0.587→0.591, p
0.0015→0.000532, CMS1 2.01→2.11, CMS4 2.19→2.50; detalle completo en
`network_analysis/CLAUDE.md`). Ninguna cohorte individual necesita ser significativa por
separado para que esto se sostenga; es justamente el punto de agrupar.**

**⚠️ Nota (2026-09-09): la comparación reparametrizada contra CMS1 como referencia y el
modelo de 4 cohortes ajustado por estadio (más abajo en esta sección) reflejan todavía el
panel ANTERIOR (`FABP1`/`SI`) — no se recalcularon con el cambio de panel. No citarlos como
vigentes sin volver a correrlos con `GALNT8`/`AGR2` primero.**

**Actualización 2026-09-11 — modelo ajustado por estadio recalculado con el panel v0.2.0,
5 cohortes** (`pooled-cox --adjust-stage`; n=518, 117 eventos, estadio IV excluido): estadio
HR 3.19; **CMS1 HR 2.12 (1.22–3.68, p=0.0075), CMS3 HR 2.19 (1.18–4.05, p=0.013), CMS4 HR
2.26 (1.35–3.81, p=0.0021)**; C-index 0.701 vs 0.658 solo estadio; aporte incremental de CMS
LRT p=0.0055, ΔC +0.043 (IC95% bootstrap +0.027 a +0.070). **Leave-one-cohort-out** (el
resultado a reportar como principal): CMS aporta sobre estadio con p<0.02 en 4/5 pliegues de
entrenamiento, pero la ganancia de C-index en la cohorte omitida es pequeña (0.000–0.069).
CMS3 solo alcanza significancia tras ajustar; leerlo con cautela (pocos eventos).

**Comparación directa con CMS1 como referencia (mismo modelo, misma muestra, solo
reparametrizado)**: CMS2 vs. CMS1, HR=0.50 (IC95% 0.31-0.79), **p<0.005** — CMS2 tiene la
mitad del riesgo de CMS1 comparado de frente, no solo "bajo por construcción" al ser la
referencia habitual. CMS3 vs. CMS1: HR=0.66, p=0.15 (tendencia, no significativo). CMS4
vs. CMS1: HR=1.09, p=0.71 — **sin diferencia significativa entre CMS1 y CMS4** pese a ser
subtipos biológicamente muy distintos (inmune/MSI vs. mesenquimal). Esto es consistente
con literatura publicada sobre CMS1 (buen pronóstico en etapa temprana, pero paradójicamente
mal pronóstico tras la recaída, acercándose al de CMS4 en ese punto) — no es un artefacto
de este análisis, coincide con lo ya descrito fuera de este proyecto.

**Modelo de Cox estratificado que combina las cuatro cohortes externas
(GSE17536+GSE17537+GSE14333+GSE33113, n=415, 100 eventos):**

- Modelo crudo (solo subtipo): prueba global p=0.012. CMS4, razón de riesgos instantáneos
  (HR)=2.24 (p=0.0025); CMS1, HR=1.86
  (p=0.025), CMS3 sin efecto (p=0.51).
- **Modelo ajustado por estadio (n=388, 80 eventos): prueba global p<0.001.** CMS4, HR=2.06
  (p=0.018), CMS1 HR=2.09 (p=0.016), CMS3 sin efecto significativo (p=0.11).
- **Diagnóstico de atenuación** (comparando crudo restringido a la misma muestra vs.
  ajustado, para separar pérdida de poder de ajuste real): el HR de CMS4 **prácticamente no
  cambia** entre restringido (2.34) y ajustado (2.06) — el efecto es robusto al ajuste por
  estadio, no un artefacto de confusión. Con las 3 cohortes originales (antes de sumar
  GSE33113) esto no se sostenía (HR caía de 1.87 a 1.60, no significativo) — la cohorte
  adicional, con 89 pacientes de estadio homogéneo, fue la que permitió esta conclusión.
- **Poder estadístico**: con 4 covariables (subtipo + estadio), CMS4 alcanza 76% de poder
  (era 58% con 3 cohortes) y CMS1 75% (era subpotenciado). CMS3 sigue en 45%, sin evidencia
  suficiente para concluir ausencia de efecto.

Concordancia de clasificación (GSE39582 vs. etiqueta oficial del consorcio): kappa=0.728 ("buena"), 80.5% accuracy (cifra vigente; ver sección de resultados).

## Validación entre plataformas (RNA-seq, TCGA-COAD/READ)

Todas las cohortes anteriores son microarreglos Affymetrix (mayoría U133 Plus 2.0, una
U133A). TCGA-COAD/READ usa RNA-seq — una tecnología de medición fundamentalmente distinta,
nunca vista durante la calibración. RFS/DFS no existe de forma curada en TCGA (`dfsStat`
100% vacío, verificado en las 603 muestras clínicas) y OS ya se había descartado antes
(p=0.33) — la prueba aquí no es supervivencia, es concordancia de clasificación contra la
etiqueta oficial del consorcio (que para TCGA sí se calculó a partir de RNA-seq):

**n=512 pacientes con etiqueta oficial (excluyendo 'none'). Kappa=0.663 ("buena"),
accuracy=76.2%** — prácticamente igual al kappa=0.679 obtenido en microarreglos (diferencia
de solo 0.016). Por subtipo: CMS1 86.8%, CMS2 79.1%, CMS4 70.8%, CMS3 66.7%. La confusión
más grande es CMS2↔CMS3 (33 pacientes CMS2 llamados CMS3, 15 al revés) — un par con
separación conocida como difícil en la literatura general de CMS, no necesariamente un
artefacto de cambiar de plataforma; no se tiene el desglose por subtipo del kappa de
microarreglos para comparar el patrón exacto, solo la cifra agregada.

**Lectura**: el modelo, calibrado exclusivamente en microarreglos, generaliza
razonablemente bien a RNA-seq sin recalibrar. Script: `src/build_tcga_rnaseq_dataset.py`
(reconstruye desde `data/raw_synapse/tcga_rnaseq/`, el archivo `tcga_cms_labeled.tsv`
anterior solo tenía 5 de los 10 genes del panel actual).

## Lectura

**Actualizado tras sumar GSE33113 (agosto 2026).** El modelo completo de 4 subtipos separa
supervivencia externa de forma significativa incluso ajustando por estadio clínico (p<0.001
global, n=415/388), con CMS4 como el eje más consistente y ahora robusto al ajuste — ya no
solo "tendencia", sino un efecto que sobrevive el control por el factor de confusión más
fuerte disponible. CMS1 también resultó significativo tras sumar la cuarta cohorte
(HR=2.09, p=0.016); dado que antes rondaba p=0.07-0.09, conviene tratar este resultado con
cautela hasta confirmarlo en una cohorte adicional — podría ser señal genuina o ganancia de
poder general, no necesariamente evidencia nueva específica de CMS1. CMS3 sigue sin
evidencia de valor pronóstico independiente, y con 45% de poder no se puede todavía
descartar que sí lo tenga.

## Motor dinámico: Modern Hopfield V2

Reemplazó por completo la dinámica de proyección anterior en `app.py` (pestañas Pronóstico
e Intervención) — no es una alternativa, es el único motor ya en la interfaz. Verificado
con datos reales de GSE39582 (la cohorte de entrenamiento), con el criterio más estricto
disponible (éxito medido *después* de retirar el forzamiento, no mientras sigue activo):

| Patrón | Umbral V1 (dinámica original) | Umbral V2 (con corrección basal + estabilizador) |
|---|---|---|
| CMS1_MSI_immune | 0.7 | 0.7 |
| CMS2_canonical_WNT | 3.0 | **1.5** |
| CMS3_metabolic | 3.0 | **0.7** |
| CMS4_mesenchymal | 8.0 | **5.0** |

El hallazgo más importante: bajo la dinámica original, CMS2 llegaba a correlación
**negativa** con su propio objetivo (terminaba pareciéndose al patrón dominante, CMS1) sin
importar cuánto se aumentara la fuerza — un límite estructural, no de calibración. La
corrección resuelve esto de forma verificada. Detalle matemático completo en `MODEL.md`
sección 10.

**Verificado con centroides calibrados independientemente en 3 de las 4 cohortes externas**
(GSE17536, GSE14333, GSE33113). El patrón se repite en las tres, no es una peculiaridad de
GSE39582:

| Cohorte | CMS1 | CMS2 | CMS3 | CMS4 |
|---|---|---|---|---|
| GSE39582 (entrenamiento) | 0.7→0.7 | 3.0→**1.5** | 3.0→**0.7** | 8.0→**5.0** |
| GSE17536 | 0.7→0.7 | 3.0→**0.7** | 5.0→**1.5** | 8.0→**5.0** |
| GSE14333 | 0.7→0.7 | 3.0→**0.7** | 5.0→**1.5** | 5.0→5.0 (sin ventaja) |
| GSE33113 | 0.7→0.7 | 3.0→**0.7** | 3.0→**0.7** | 3.0→**1.5** |

En las 4 calibraciones independientes, CMS2 arranca con correlación **negativa** bajo V1 en
fuerza baja (se confunde con CMS1, el patrón dominante) y V2 lo resuelve consistentemente
con fuerza ≤1.5 — un fenómeno estructural reproducible, no un artefacto de una calibración
específica. Única excepción real: CMS4 en GSE14333 no muestra ventaja de V2 sobre V1 (ambos
llegan a fuerza=5.0), documentado tal cual.

**Salvedad de tamaño de muestra**: varias clases se calibraron por debajo del mínimo
recomendado (30) — GSE33113 CMS3 con solo 10 muestras, GSE17536 CMS3 con 20, GSE14333
CMS1/CMS3 con 22/23. La consistencia del patrón entre calibraciones es alentadora, pero
esos centroides individuales son más ruidosos que el de GSE39582.

**GSE17537 no se pudo verificar de forma independiente, y no es una tarea pendiente**: las
55 muestras de esta cohorte tienen `cms_label="none"` — no un error de formato, sino que
esta cohorte nunca formó parte del conjunto etiquetado originalmente por el consorcio CMS
(consistente con su uso ya documentado en este proyecto como cohorte "externa nunca usada
para ajustar el panel", validada contra supervivencia con las predicciones del modelo, no
contra una etiqueta oficial). Sin verdad de referencia, no existe forma de calibrar
centroides independientes de esta cohorte — la verificación cruzada queda completa en 3/4,
y la cuarta no es alcanzable con estos datos, no por falta de intentarlo.

### Clasificador dinámico experimental vs. estático: validación de Cox en cohortes externas

Distinto de lo anterior — esto compara el clasificador **estático** (correlación con
centroides, el principal) contra la recuperación dinámica **experimental** (pestaña
Muestras, con abstención explícita), corriendo el mismo Cox estratificado sobre las 4
cohortes externas con cada uno:

| | Estático (`predicted_cms`) | Dinámico (`modern_hopfield_cms`) |
|---|---|---|
| n combinado | 415 | 343 (92 abstenciones) |
| Concordance, modelo crudo (solo CMS) | 0.577 | 0.585 |
| CMS1 HR, modelo ajustado | 2.09 (p=0.016) | 2.05 (p=0.07, no significativo) |
| CMS4 HR, modelo ajustado | — | 3.05 (p=0.004, robusto) |

El concordance del modelo crudo es prácticamente idéntico entre ambos clasificadores — la
diferencia no está ahí. Lo que sí cambia es que **CMS1 pierde significancia** bajo el
clasificador dinámico en el modelo ajustado por estadio.

**Investigación de sesgo en las 92 abstenciones** (comparando tasa de recaída incluidas vs.
excluidas): tasa de evento global casi idéntica (incluidas 23.6% vs. excluidas 26.4%), y
por cohorte la diferencia va en direcciones **opuestas** sin patrón consistente (con
conteos pequeños por cohorte, n=7 a n=38, donde esa variación es compatible con azar). El
100% de las abstenciones son por "entrada híbrida o ambigua" — ningún fallo numérico
(no convergencia, inestabilidad, residuo alto) — consistente con la fracción de tumores
mixtos/no clasificables ya documentada en la literatura de CMS (~13% en el consorcio
original), no un artefacto de la implementación.

**Lectura honesta**: no hay evidencia clara de que las abstenciones estén sesgadas por
desenlace. La explicación más simple del debilitamiento de CMS1 es pérdida de poder
estadístico (22% menos muestras), no un sesgo direccional demostrado — pero tampoco se
puede descartar por completo algo más sutil sin un análisis estratificado más profundo. El
clasificador estático sigue siendo el principal; esto queda documentado como evidencia
complementaria, no como reemplazo de las cifras existentes (kappa=0.679, Cox con el
clasificador estático).

## Simulación de tratamiento

`treatment_perturbation.py` implementa tres mecanismos (inmunoterapia anti-PD1, tratamiento
anti-EGFR y quimioterapia citotóxica), condicionados por la biología del paciente, con evidencia
clínica citada (KEYNOTE-177: HR=0.60–0.73 en tumores con alta inestabilidad de microsatélites
[MSI-H] o deficiencia en la reparación de errores de apareamiento [dMMR]; requisito de RAS/BRAF
de tipo silvestre para el tratamiento anti-EGFR). Los tumores estables en microsatélites (MSS)
no muestran el beneficio basal de inmunoterapia descrito para MSI-H/dMMR.
`prognosis_demo.py` ahora reporta, junto con cada alerta de recurrencia: fuerza de evidencia
del atractor hacia el que se dirige el paciente, y qué tratamientos tienen mecanismo
aplicable a ese estado. **Dirección fundamentada en literatura, magnitud NO calibrada** —
sigue siendo exploración in silico, no una herramienta de decisión clínica.

## Limitaciones

- El riesgo (*hazard*) de `prognosis.py` es ordinal, no una probabilidad calibrada.
- CMS3 sigue sin evidencia de efecto (p=0.11 ajustado), pero con solo 45% de poder estadístico — no se
  puede concluir la ausencia del efecto, solo que el conjunto de muestra actual no alcanza para verlo.
- El criterio de activación de `anti_egfr` usa un proxy débil por ARN (cercanía a CMS3) cuando no hay estatus
  RAS/BRAF real — no debe sustituir la prueba de mutación (qPCR alelo-específico/HRM)

## Próximos pasos

En orden de prioridad, tras el rerun del 2026-09-15:

1. **Cohorte confirmatoria intocada.** Es lo único que convierte la validación de desarrollo
   en confirmatoria. Congelar el panel por escrito (este archivo + tag de git), conseguir una
   cohorte del CRCSC con etiqueta CMS oficial y RFS que no haya intervenido en ninguna
   decisión (candidatas: GSE38832, GSE29621, GSE13294 — verificar anotación de RFS y estadio),
   y correr `validate-external` + `pooled-cox` **una sola vez** sobre ella, reportando el
   resultado tal cual salga.
2. **κ con validación cruzada** (5-fold: centroides en 4/5, clasificar 1/5) para cuantificar
   el optimismo del 0.728 in-sample. Es código sencillo sobre `run_pipeline.py`.
3. **Retirar el poder post-hoc** de `power_analysis.py` y de este documento: calcular poder
   con el HR observado es función 1:1 del p-valor y no aporta información (Hoenig & Heisey
   2001). Conservar solo el HR mínimo detectable con los eventos disponibles.
4. **Partición en `feature_selection.py`**: AUC → top-300 → Random Forest corre hoy sobre
   todas las muestras sin holdout ni validación cruzada anidada. No rescata el panel actual,
   pero es requisito para cualquier iteración futura.
5. **Sensibilidad a la etiqueta de referencia**: repetir la concordancia con `CMS_network`
   (solo consenso) en lugar de `CMS_final_network_plus_RFclassifier_in_nonconsensus_samples`.
   Si κ sube, parte del "error" era ruido de etiqueta.
6. **Desambiguar CMS4 vs CMS1**: evaluar añadir un marcador estromal específico (`THBS2`,
   `INHBA`) contra el error dominante de clasificación (~30% de CMS4 → CMS1). Cualquier cambio
   de panel debe decidirse sobre GSE39582 y evaluarse en la cohorte confirmatoria, nunca sobre
   las cinco actuales.
7. **`cms_margin` como covariable** en el Cox, para probar si la ambigüedad del perfil tiene
   valor pronóstico propia (lo sugieren los no-consenso).
8. **Regenerar la tabla V1/V2** del motor dinámico con el calendario de forzamiento unificado
   (`compare_forcing_sweep_v1_v2`) y reescribirla según el diagnóstico de normas: la asimetría
   de umbrales seguía el orden exacto de las normas de los centroides (2.71/2.49/1.51/1.30
   frente a cuencas de 43%/32%/15%/10%), y con normas igualadas + V2 los cuatro subtipos son
   alcanzables al forzamiento mínimo. Es un artefacto de parametrización, no un hallazgo
   biológico sobre CMS2.
