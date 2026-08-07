# Estado del proyecto

**Última actualización:** agosto 2026. Historial detallado en `CHANGELOG.md`.

## Panel actual

10 genes, todos RT-qPCR: `MLH1`, `GNLY`, `USP18` (CMS1) · `MYC`, `AXIN2` (CMS2) ·
`FABP1`, `CPS1`, `SI` (CMS3) · `VIM`, `TGFB1` (CMS4). Congelado — no agregar genes sin
justificación cuantitativa nueva (cada gen tiene costo real en un ensayo RT-qPCR).

## Evidencia acumulada

| Cohorte | Rol | n | p-valor (log-rank) |
|---|---|---|---|
| GSE39582 | Entrenamiento | 557 | 0.00039 |
| TCGA-COAD/READ | Descartada (sin RFS curado, solo OS) | 558 | 0.33 (ninguno separa — problema de endpoint) |
| GSE17536 | Externa | 145 | 0.090 |
| GSE17537 | Externa (nunca usada para ajustar el panel) | 55 | 0.71 |
| GSE14333 | Externa | 126 | 0.59 |

**Cox estratificado combinando GSE17536+GSE17537+GSE14333 (n=326):**
test global p=0.045 (significativo). CMS4 es el eje con evidencia más fuerte y consistente
(HR=1.88, p=0.03, peor pronóstico en las tres cohortes). CMS3 no aporta señal de
supervivencia distinguible de CMS2 (p=0.70). Concordance del modelo Cox = 0.57 (débil).

Concordancia de clasificación (GSE39582 vs. etiqueta oficial del consorcio): kappa=0.679
("buena"), 77.5% accuracy.

## Lectura

El modelo completo de 4 subtipos ya tiene evidencia externa significativa combinada, pero
la señal está concentrada casi enteramente en el eje CMS4 — que es también el eje más
relevante para el módulo de pronóstico longitudinal (detecta movimiento hacia el atractor
de peor pronóstico). Los otros tres ejes clasifican razonablemente bien pero no muestran,
todavía, valor pronóstico independiente reproducible.

## Simulación de tratamiento (nuevo)

`treatment_perturbation.py` implementa tres mecanismos (inmunoterapia anti-PD1, anti-EGFR,
quimioterapia citotóxica), gateados por biología del paciente, con evidencia clínica citada
(KEYNOTE-177 HR=0.60-0.73 en MSI-H/dMMR; requisito RAS/BRAF wild-type para anti-EGFR).
`prognosis_demo.py` ahora reporta, junto con cada alerta de recurrencia: fuerza de evidencia
del atractor hacia el que se dirige el paciente, y qué tratamientos tienen mecanismo
aplicable a ese estado. **Dirección fundamentada en literatura, magnitud NO calibrada** —
sigue siendo exploración in silico, no una herramienta de decisión clínica.

## Limitaciones

- La magnitud del efecto de tratamiento en `treatment_perturbation.py` no está calibrada
  contra datos reales de "antes/después de tratamiento" — solo la dirección está
  fundamentada en literatura. No existen esos datos en el proyecto todavía (ver más abajo).
- El hazard de `prognosis.py` es ordinal, no probabilidad calibrada (falta dato longitudinal
  real tipo DYNAMIC/GALAXY — ese tipo de dato, además, mide ctDNA vía NGS, no expresión de
  ARN vía qPCR, así que ni siquiera calibraría directamente este panel específico; la ruta
  más realista es un piloto propio con seguimiento post-quirúrgico real)
- GSE17536 se usó iterativamente para decidir genes — su validez como cohorte "externa" en
  sentido estadístico estricto está parcialmente comprometida
- Concordance=0.57 en el Cox combinado es débil — significativo no es lo mismo que buen
  discriminador individual
- El gate de `anti_egfr` usa un proxy débil por RNA (cercanía a CMS3) cuando no hay estatus
  RAS/BRAF real — nunca debe sustituir la prueba de mutación (qPCR alelo-específico/HRM)

## Próximos pasos

1. Sumar más cohortes CRCSC al análisis combinado (`build_external_cohort_generic.py` ya
   soporta esto) antes de tocar el panel otra vez
2. Diseñar un piloto propio con seguimiento post-quirúrgico real (INMEGEN) para calibrar
   `prognosis.py` y `treatment_perturbation.py` contra datos reales, no solo dirección
   fundamentada en literatura
3. Obtener estatus RAS/BRAF real (qPCR alelo-específico/HRM) para reemplazar el proxy débil
   por RNA en el gate de `anti_egfr`
