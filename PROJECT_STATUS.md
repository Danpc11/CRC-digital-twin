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
| GSE17536 | Externa (usada iterativamente para ajustar el panel) | 145 | 0.090 |
| GSE17537 | Externa (nunca usada para ajustar el panel) | 55 | 0.71 |
| GSE14333 | Externa | 126 | 0.59 |
| GSE33113 | Externa (estadio II homogéneo) | 89 | **0.031** |

**Cox estratificado combinando las 4 cohortes externas (GSE17536+GSE17537+GSE14333+GSE33113,
n=415, 100 eventos):**

- Modelo crudo (solo subtipo): test global p=0.012. CMS4 HR=2.24 (p=0.0025), CMS1 HR=1.86
  (p=0.025), CMS3 sin efecto (p=0.51).
- **Modelo ajustado por estadio (n=388, 80 eventos): test global p<0.001.** CMS4 HR=2.06
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

Concordancia de clasificación (GSE39582 vs. etiqueta oficial del consorcio): kappa=0.679
("buena"), 77.5% accuracy.

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

## Simulación de tratamiento

`treatment_perturbation.py` implementa tres mecanismos (inmunoterapia anti-PD1, anti-EGFR,
quimioterapia citotóxica), condicionados por biología del paciente, con evidencia clínica citada
(KEYNOTE-177 HR=0.60-0.73 en MSI-H/dMMR; requisito RAS/BRAF wild-type para anti-EGFR).
`prognosis_demo.py` ahora reporta, junto con cada alerta de recurrencia: fuerza de evidencia
del atractor hacia el que se dirige el paciente, y qué tratamientos tienen mecanismo
aplicable a ese estado. **Dirección fundamentada en literatura, magnitud NO calibrada** —
sigue siendo exploración in silico, no una herramienta de decisión clínica.

## Limitaciones

- El hazard de `prognosis.py` es ordinal, no probabilidad calibrada.
- CMS3 sigue sin evidencia de efecto (p=0.11 ajustado), pero con solo 45% de poder estadístico — no se
  puede concluir la ausencia del efecto, solo que el conjunto de muestra actual no alcanza para verlo.
- El criterio de activación de `anti_egfr` usa un proxy débil por RNA (cercanía a CMS3) cuando no hay estatus
  RAS/BRAF real — no debe sustituir la prueba de mutación (qPCR alelo-específico/HRM)

## Próximos pasos

1. Sumar una quinta cohorte externa para confirmar (o no) el hallazgo nuevo de CMS1, y
   para cerrar la brecha de poder restante en CMS4 (76%→80%) y CMS3 (45%→80%)
2. Obtener estatus RAS/BRAF real (qPCR alelo-específico/HRM) para reemplazar el proxy débil
   por RNA en el criterio de activación de `anti_egfr`
