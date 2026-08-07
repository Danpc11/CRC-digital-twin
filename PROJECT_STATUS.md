# Estado del proyecto — ColoQ / crc-digital-twin

**Última actualización:** agosto 2026
**Estado:** panel congelado, evidencia interna sólida, evidencia externa parcial (eje CMS4)

---

## 1. Qué es esto

Gemelo digital mecanicista de cáncer colorrectal: modela los cuatro subtipos moleculares
consensuados (CMS1–4) como atractores de una red tipo Hopfield continua, calibrable contra
datos reales, con validación contra desenlaces de supervivencia y un módulo de pronóstico
longitudinal para seguimiento post-quirúrgico — diseñado desde el inicio para operar sobre
paneles medibles por **qPCR/RT-qPCR únicamente** (sin ddPCR, sin NGS, sin secuenciación de
exosomas), pensando en accesibilidad para infraestructura de salud pública mexicana.

La arquitectura tiene tres capas:

1. **Mecanística** (`attractor_model.py`) — dinámica tipo Hopfield con 4 atractores (uno
   por subtipo CMS), matriz de acoplamiento construida por regla de proyección
   (Personnaz-Guyon-Dreyfus), no Hebb clásica.
2. **Calibración** (`calibration.py`) — reemplaza los patrones placeholder por centroides
   reales calculados de datos de pacientes.
3. **Pronóstico** (`prognosis.py`) — convierte una serie temporal de mediciones
   post-quirúrgicas en una señal de riesgo ordinal (distancia al origen = ausencia de
   enfermedad residual).

---

## 2. Panel final (10 genes, todos compatibles con RT-qPCR)

| Eje CMS | Genes | Base biológica |
|---|---|---|
| CMS1 (MSI-inmune) | `MLH1`, `GNLY`, `USP18` | MLH1: estatus MMR (dirección real: **baja** expresión en CMS1, por silenciamiento epigenético — no alta, como asumía el panel placeholder original). GNLY/USP18: citotoxicidad NK/T e interferón tipo I |
| CMS2 (canónico/WNT) | `MYC`, `AXIN2` | Activación WNT/MYC clásica |
| CMS3 (metabólico) | `FABP1`, `CPS1`, `SI` | Diferenciación enterocito-like (Yuan et al. 2025, *Nat Commun*) |
| CMS4 (mesenquimal) | `VIM`, `TGFB1` | EMT/activación estromal |

**Precedente clínico de referencia:** el Colon Recurrence Score (Oncotype DX Colon) usa 12
genes por RT-qPCR (7 de cáncer + 5 de referencia). Este panel, con housekeeping, queda en un
orden de magnitud comparable.

---

## 3. Cómo llegamos a este panel (resumen del proceso, no solo el resultado)

| Cambio | Método | Resultado |
|---|---|---|
| Panel placeholder inicial (8 genes, patrones hechos a mano) | Intuición biológica | Solo esqueleto matemático, sin calibrar |
| Calibración contra datos reales (TCGA + GSE39582) | Centroides empíricos | `KRAS_sig` (placeholder sin gen real) descartado |
| `CPS1` + `SI` agregados (eje CMS3) | Literatura (Yuan et al. 2025) | Kappa interno 0.533 → 0.639 |
| `CKLF` agregado (eje CMS1) | Literatura (marcador pronóstico, mal aplicado a tarea de clasificación) | Validación externa **empeoró** (p 0.168 → 0.365) — **revertido** |
| `feature_selection.py` construido | AUC uno-contra-resto + ANOVA + Random Forest sobre transcriptoma completo (23,520 genes) | Reveló que `GZMB` tenía AUC mediocre (rank #2,837) y que `MLH1` tenía dirección invertida |
| `GNLY` + `USP18` reemplazan `GZMB` | Data-driven (AUC top-10 de 23,520) | Kappa interno 0.639 → 0.679; validación externa 0.168 → 0.090 |
| Prueba de remover `USP18` | Ablación controlada | p de supervivencia empeoró 10× (0.0004 → 0.041) → **USP18 se mantiene** |

**Lección metodológica explícita:** un gen con buen respaldo de literatura (`CKLF`) no
garantiza mejora — su validación original era para una tarea distinta (pronóstico dentro de
CMS1 ya diagnosticado, no clasificación de subtipo). La selección data-driven, verificada
empíricamente contra cohortes reales, superó a la selección por hipótesis biológica sola.

---

## 4. Evidencia empírica por cohorte

| Cohorte | Rol | n (con supervivencia) | Endpoint | p-valor (modelo) | p-valor (etiqueta oficial) |
|---|---|---|---|---|---|
| **GSE39582** | Entrenamiento/calibración | 557 | RFS | **0.00039** | 0.00017 |
| **TCGA-COAD/READ** | Descartado como validación de supervivencia | 558 | OS (no RFS — TCGA no tiene RFS curado) | 0.33 | 0.35 (ninguno separa — problema del endpoint, no del panel) |
| **GSE17536** | Validación externa (usada para ajustar el panel) | 145 | RFS | 0.090 | 0.019 |
| **GSE17537** | Validación externa fresca (nunca usada para decidir nada del panel) | 55 | RFS | 0.71 | *(sin etiqueta oficial — el consorcio CMS no etiquetó esta subserie)* |

**Concordancia de clasificación (GSE39582 vs. etiqueta oficial del consorcio):**
Cohen's kappa = 0.679 ("buena"), accuracy = 77.5%.
Por subtipo: CMS1 90.1%, CMS2 80.2%, CMS3 65.2%, CMS4 70.1%.

### Análisis combinado (Cox estratificado por cohorte, GSE17536 + GSE17537, n=200)

Modelo de riesgos proporcionales de Cox, estratificado por cohorte (cada cohorte con su
propia línea base de riesgo, efecto de subtipo estimado conjuntamente). Referencia: CMS2.

| Subtipo | Hazard Ratio | p |
|---|---|---|
| CMS1 | 1.74 (peor) | 0.13 |
| CMS3 | 0.89 (similar) | 0.80 |
| **CMS4** | **1.99 (peor)** | **0.05** |

Test de razón de verosimilitud global: **p = 0.103** (no significativo para el modelo
completo de 4 grupos). Concordance = 0.58.

---

## 5. Interpretación

- **El eje CMS4 es la señal más consistente de todo el proyecto.** Peor pronóstico
  confirmado visualmente en curvas Kaplan-Meier, en cada cohorte, con cada versión del
  panel, y ahora al límite de significancia (p=0.05) en el análisis combinado. Es
  particularmente relevante porque el módulo de pronóstico longitudinal (`prognosis.py`)
  existe justamente para detectar movimiento hacia el atractor de peor pronóstico —
  este es el eje con más respaldo para esa función específica.
- **CMS3 no aporta señal de supervivencia distinguible de CMS2** en la validación externa
  combinada (HR≈0.89, p=0.80), a pesar de tener concordancia de clasificación razonable
  (65-75%).
- **CMS1 muestra dirección contraintuitiva** (HR=1.74, peor pronóstico) en el análisis
  combinado. No se puede descartar que sea ruido, pero hay precedente en literatura de
  un comportamiento bifásico real de CMS1 (buen pronóstico temprano, peor pronóstico
  específicamente tras recaída) que podría explicar el patrón según la composición de
  cada cohorte.
- **El modelo de 4 grupos completo no tiene, todavía, evidencia externa suficiente** para
  uso más allá de investigación. Esto no invalida la arquitectura — es el resultado
  esperable de un primer ciclo de desarrollo de un panel biomarcador nuevo.

---

## 6. Limitaciones explícitas

1. **Sin datos reales de calibración, todo pronóstico es hipotético** — esto ya no aplica
   (el panel está calibrado contra 566 pacientes reales), pero las magnitudes de riesgo no
   están calibradas a probabilidad clínica, solo son ordinales.
2. **El módulo de pronóstico longitudinal (`prognosis.py`) no está calibrado contra tiempo
   a recurrencia real** — usa distancia al origen como proxy de riesgo, sin datos
   longitudinales de seguimiento (tipo DYNAMIC/GALAXY) para calibrar la relación
   riesgo-tiempo.
3. **Ningún componente predice respuesta a tratamiento.** Eso requeriría un término de
   perturbación farmacodinámica que no existe en el modelo actual — línea de investigación
   separada, no iniciada.
4. **p<0.05 en una sola cohorte no es validación clínica.** Incluso el resultado combinado
   (Cox estratificado) requeriría replicación prospectiva independiente antes de cualquier
   uso más allá de investigación.
5. **GSE17536 fue usado iterativamente para decidir qué genes mantener** (CKLF revertido,
   GNLY/USP18 mantenidos) — esto compromete parcialmente su validez como cohorte
   "externa" en sentido estadístico estricto. GSE17537 es la única cohorte genuinamente
   no tocada, pero está subpotenciada (n=55).
6. **GSE17537 no tiene etiqueta CMS oficial del consorcio** — no se pudo hacer la
   comparación de línea base "modelo vs. etiqueta oficial" en esa cohorte específicamente.
7. **TCGA-COAD/READ no sirve para validar RFS** — no tiene ese endpoint curado (0/603
   pacientes). Solo tiene OS, que mostró exactamente cero separación tanto con el modelo
   como con la etiqueta oficial — conclusión: el endpoint, no el panel, es el problema en
   ese caso específico.
8. **Direcciones de gen contraintuitivas deben verificarse, no asumirse:** `MLH1` tiene
   dirección real invertida (baja expresión = CMS1, no alta) respecto al placeholder
   original — la calibración contra datos reales lo corrigió automáticamente, pero es un
   recordatorio de que la intuición biológica sin verificación empírica puede estar
   equivocada incluso en genes bien caracterizados.

---

## 7. Estado del arte y panorama comercial (contexto)

- **Oncotype DX Colon Recurrence Score** — panel RT-qPCR de 12 genes, ya validado
  clínicamente. Precedente metodológico directo de este proyecto.
- **Cellworks Biosimulation Platform** — "gemelo digital" comercial real (NGS +
  simulación de respuesta a fármacos), no compatible con la restricción de qPCR de este
  proyecto.
- **Signatera / Haystack MRD** — monitoreo de ADN tumoral circulante vía NGS
  personalizado. Este proyecto busca una aproximación de menor costo vía qPCR
  alelo-específico para el módulo de seguimiento longitudinal.
- **Relevancia para México:** el argumento central no es "mejor que las herramientas
  comerciales existentes" — es accesibilidad. La infraestructura de qPCR/RT-qPCR ya
  existe ampliamente en laboratorios de salud pública mexicanos; NGS personalizado o
  biosimulación comercial no son accesibles en la práctica para la mayoría de
  instituciones públicas latinoamericanas.

---

## 8. Estructura del repositorio

```
crc-digital-twin/
├── README.md
├── requirements.txt
├── run_pipeline.py                    CLI: calibración + validación (cohorte de entrenamiento)
│
├── src/
│   ├── attractor_model.py             modelo de atractores (dinámica ODE tipo Hopfield)
│   ├── calibration.py                 calibración contra datos reales etiquetados
│   ├── survival_validation.py         Kaplan-Meier / log-rank
│   ├── prognosis.py                   trayectoria/pronóstico longitudinal
│   ├── prognosis_demo.py              demo end-to-end con patrones calibrados reales
│   ├── external_validation.py         aplica patrones YA calibrados a cohorte externa (sin recalibrar)
│   ├── pooled_cox_validation.py       Cox estratificado combinando múltiples cohortes externas
│   ├── concordance_analysis.py        matriz de concordancia modelo vs. etiqueta oficial
│   ├── feature_selection.py           selección data-driven de genes (AUC/ANOVA/Random Forest)
│   ├── synthetic_data.py              generador de datos de prueba
│   ├── plot_trajectories.py           figura de trayectorias del modelo
│   ├── plot_survival_curves.py        figuras de curvas Kaplan-Meier
│   ├── download_synapse_data.py       descarga datos del consorcio CMS (Synapse)
│   ├── download_geo_gse39582.py       descarga GSE39582 (no probado, ver parse_geo_series_matrix.py)
│   ├── parse_geo_series_matrix.py     parser robusto de series_matrix de GEO (por celda, no por posición)
│   ├── format_to_schema.py            construye dataset TCGA en el esquema esperado
│   ├── build_gse39582_dataset.py      construye dataset GSE39582
│   ├── build_gse17536_dataset.py      construye dataset GSE17536 (validación externa)
│   └── build_gse17537_dataset.py      construye dataset GSE17537 (validación externa fresca)
│
├── tests/                             suite de regresión (pytest)
├── data/                              datos (no versionados, ver .gitignore)
└── figures/                           salidas gráficas
```

---

## 9. Próximos pasos sugeridos

1. **Conectar el hallazgo de CMS4 al módulo de pronóstico** — marcar en el reporte de
   alerta de `prognosis_demo.py` qué tan fuerte es la evidencia detrás del atractor hacia
   el que se mueve un paciente (CMS4 con evidencia fuerte, los otros tres con evidencia
   débil/no concluyente).
2. **Ampliar validación externa con más cohortes CRCSC** (`petacc3`, `kfsyscc`, `gse2109`,
   `gse14333`, `gse13294`, `gse13067`, `gse35896`, `gse23878`, `gse37892` — todas ya tienen
   etiqueta CMS oficial en `cms_labels_public_all.txt` de Synapse) antes de intentar
   ajustar el panel otra vez, para no repetir el problema de sobreajuste iterativo sobre
   cohortes pequeñas.
3. **No seguir agregando genes al panel** sin una razón cuantitativa clara — cada gen
   adicional tiene costo real en un ensayo RT-qPCR (multiplexado, optimización, puntos de
   fallo).
4. Diseño del módulo de perturbación farmacodinámica (predicción de respuesta a
   tratamiento) — línea de investigación separada, no iniciada, requiere datos de
   ensayos con brazo de tratamiento vs. observación.
