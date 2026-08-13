# Estado del proyecto

**Última actualización:** agosto 2026. Historial detallado en `CHANGELOG.md`.

## Panel actual

10 genes, todos medibles mediante qPCR con transcripción inversa (RT-qPCR): `MLH1`, `GNLY`,
`USP18` (CMS1) · `MYC`, `AXIN2` (CMS2) ·
`FABP1`, `CPS1`, `SI` (CMS3) · `VIM`, `TGFB1` (CMS4). Congelado — no agregar genes sin
justificación cuantitativa nueva (cada gen tiene costo real en un ensayo RT-qPCR).
CMS corresponde a los subtipos moleculares consensuados de cáncer colorrectal (*Consensus
Molecular Subtypes*).

## Evidencia acumulada

| Cohorte | Rol | n | valor p (log-rank) |
|---|---|---|---|
| GSE39582 | Entrenamiento | 557 | 0.00039 |
| TCGA-COAD/READ | Descartada (sin supervivencia libre de recaída [RFS] curada, solo supervivencia global [OS]) | 558 | 0.33 (ninguno separa — problema del desenlace) |
| GSE17536 | Externa (usada iterativamente para ajustar el panel) | 145 | 0.090 |
| GSE17537 | Externa (nunca usada para ajustar el panel) | 55 | 0.71 |
| GSE14333 | Externa | 126 | 0.59 |
| GSE33113 | Externa (estadio II homogéneo) | 89 | **0.031** |

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

**Pendiente**: esta verificación se hizo solo sobre GSE39582 (entrenamiento) — no se ha
confirmado si el mismo patrón (CMS2 estructuralmente inalcanzable con la dinámica original,
resuelto con la corrección) se sostiene con los centroides calibrados de las cohortes
externas, que pueden tener geometría de correlación distinta entre subtipos.

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

1. Sumar una quinta cohorte externa para confirmar (o no) el hallazgo nuevo de CMS1, y
   para cerrar la brecha de poder restante en CMS4 (76%→80%) y CMS3 (45%→80%)
2. Obtener estatus RAS/BRAF real (qPCR alelo-específico/HRM) para reemplazar el proxy débil
   por ARN en el criterio de activación de `anti_egfr`
3. Verificar si la corrección de Modern Hopfield V2 (umbrales de forzamiento por CMS) se
   sostiene con los centroides de las cohortes externas, no solo con GSE39582
