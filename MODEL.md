# Modelo — fundamento teórico

Este documento describe la formulación matemática del gemelo digital: la dinámica de
atractores, cómo se calibra contra datos reales, y cómo se extiende a pronóstico
longitudinal y simulación de tratamiento. Para el estado empírico del proyecto (qué tan
bien funciona, con qué evidencia), ver `PROJECT_STATUS.md`. Este documento es sobre **qué es
el modelo**, no sobre qué tan bueno es.

## 1. Formulación del espacio de estado

Un paciente se representa como un vector $x \in \mathbb{R}^N$, donde cada componente es la
expresión normalizada (z-score) de uno de los $N$ genes del panel ($N=10$ en la versión
actual: `MLH1`, `GNLY`, `USP18`, `MYC`, `AXIN2`, `FABP1`, `CPS1`, `SI`, `VIM`, `TGFB1`).

El origen $x = \mathbf{0}$ representa un estado de referencia sin señal molecular distintiva
— en el contexto de seguimiento post-quirúrgico, esto se interpreta como ausencia de
enfermedad residual detectable.

Cada uno de los cuatro subtipos moleculares consensuados (CMS1–4) se representa como un
**patrón objetivo** $p^\mu \in \mathbb{R}^N$, $\mu = 1, \dots, 4$ — un atractor del sistema
dinámico descrito abajo.

## 2. Calibración: de dónde salen los patrones

Los patrones **no** son parámetros ajustados por optimización iterativa. Son centroides
empíricos: dado un conjunto de pacientes con etiqueta CMS conocida (la clasificación oficial
del consorcio CMS, aplicada a una cohorte de entrenamiento — GSE39582), el patrón de cada
subtipo es simplemente la media de los vectores de expresión (ya z-scoreados) de los
pacientes con esa etiqueta:

$$p^\mu = \frac{1}{|\mathcal{C}_\mu|} \sum_{i \in \mathcal{C}_\mu} x_i$$

donde $\mathcal{C}_\mu$ es el conjunto de pacientes de entrenamiento con etiqueta $\mu$.

Implementado en `calibration.py::calibrate_patterns_from_data()`. Es la única "fase de
aprendizaje" del sistema — no hay descenso de gradiente, no hay épocas, no hay función de
pérdida. El resultado se guarda en `calibrated_patterns.tsv` y de ahí en adelante el modelo
queda congelado: la validación externa (`external_validation.py`) nunca vuelve a tocar estos
valores.

## 3. La matriz de acoplamiento — regla de proyección, no Hebb

La dinámica (sección 4) necesita una matriz $W \in \mathbb{R}^{N \times N}$ que defina cómo
interactúan las componentes del estado. La elección de cómo construir $W$ a partir de los
patrones es la decisión de diseño central del modelo.

**Por qué no la regla de Hebb clásica.** La construcción "obvia" para una red asociativa
tipo Hopfield es la regla de Hebb, $W = \frac{1}{N}\sum_\mu p^\mu (p^\mu)^\top$. Pero esta
regla tiene un límite de capacidad conocido: solo puede almacenar de forma fiable
$\sim 0.14N$ patrones como puntos fijos antes de que aparezcan mínimos espurios y
diafonía entre patrones. Con $N=10$ y 4 patrones, eso está muy por encima del límite de
capacidad de Hebb — los 4 atractores no quedarían garantizados como puntos fijos estables.

**La regla usada: proyección (Personnaz–Guyon–Dreyfus).** En su lugar, $W$ se construye como
el proyector ortogonal sobre el subespacio generado por los patrones:

$$W = P (P^\top P)^{-1} P^\top, \qquad P = [\,p^1 \mid p^2 \mid p^3 \mid p^4\,] \in \mathbb{R}^{N \times 4}$$

Esta construcción garantiza que **cada patrón sea un punto fijo exacto** del sistema
linealizado ($Wp^\mu = p^\mu$ para todo $\mu$), sin importar cuántos patrones haya, siempre
que sean linealmente independientes — no hay límite de capacidad de Hebb del que
preocuparse. Es simétrica por construcción ($W = W^\top$).

Implementado en `attractor_model.py::projection_weight_matrix()` /
`build_model_from_patterns()`. La verificación de que cada patrón es efectivamente un punto
fijo es un test de regresión explícito (`test_patterns_are_approximate_fixed_points_of_linear_system`).

**Consecuencia práctica**: $W$ nunca se guarda en disco ni se "entrena" — se reconstruye
algebraicamente, en un solo paso de álgebra lineal, cada vez que se cargan los patrones.

## 4. La dinámica

El estado del paciente evoluciona según una ecuación diferencial ordinaria tipo Hopfield
continuo:

$$\frac{dx}{dt} = -x + W \tanh(\beta x) + I(t)$$

- $-x$: término de relajación lineal hacia el origen (decaimiento natural, en ausencia de
  cualquier señal, el sistema vuelve a "sin enfermedad residual")
- $W\tanh(\beta x)$: acoplamiento no lineal entre componentes, con $\beta$ controlando la
  nitidez de la no linealidad ($\beta=2.0$ por default)
- $I(t)$: término de forzamiento externo — representa un proceso biológico que empuja el
  estado en una dirección particular (recaída, tratamiento; ver secciones 6–7)

Integrada numéricamente con `scipy.integrate.solve_ivp` (Runge-Kutta 4(5), tolerancias
$10^{-8}$/$10^{-10}$). Implementado en `attractor_model.py::dynamics()`.

## 5. Clasificación

Dado un vector de estado $x$ (medido o simulado), el subtipo más cercano se determina por
**correlación de Pearson** contra cada patrón, no por distancia euclidiana:

$$\hat\mu = \arg\max_\mu \, \mathrm{corr}(x, p^\mu)$$

Se usa correlación, no distancia, porque es invariante a escala — robusta a diferencias de
normalización entre plataformas (microarreglo vs. RT-qPCR vs. distintos lotes). La
correlación máxima ($r$) se reporta como una confianza de clasificación informal: en la
validación externa, las muestras con confianza baja concentran la mayoría de los errores
(ver el barrido de umbral en `error_analysis.py`, no incluido en `src/` por ser una
herramienta de análisis puntual, no parte del pipeline).

Si $\|x\| \approx 0$ (estado en el origen), no se asigna ningún subtipo — se reporta como
`"none"` ("sin enfermedad residual detectable", no un quinto subtipo).

Implementado en `survival_validation.py::risk_score_from_expression()` y, de forma
equivalente para trayectorias simuladas, `prognosis_demo.py::classify_current_state()`.

## 6. Pronóstico longitudinal: trayectoria y alerta

El seguimiento post-quirúrgico se modela como una serie de mediciones periódicas. Antes de
la recaída, sin forzamiento ($I=0$), el sistema decae al origen. Si el paciente recae, se
introduce un término de sesgo hacia el patrón del subtipo de la recaída, con magnitud
creciente en el tiempo desde el inicio:

$$I_{\text{recaída}}(t) = s(t) \cdot p^{\mu_{\text{recaída}}}, \qquad
s(t) = \min\!\big(0.15 \cdot (t - t_0),\ 0.7\big) \ \text{para } t \geq t_0$$

donde $t_0$ es el mes de inicio de la recaída (simulado; en un caso real sería desconocido —
el objetivo del módulo es justamente inferirlo a partir de las mediciones).

**Riesgo (hazard) ordinal**: la distancia al origen en cada punto de medición,

$$h(t) = \|x(t)\|$$

Es **ordinal, no una probabilidad calibrada** — sirve para ordenar momentos de riesgo dentro
de la trayectoria de un mismo paciente, no para afirmar "X% de probabilidad de recurrencia".
Calibrar eso a probabilidad real requeriría datos longitudinales reales (tipo
DYNAMIC/GALAXY) que el proyecto no tiene todavía.

**Detección de alerta**: comparación contra una ventana basal (los primeros puntos
post-quirúrgicos, donde se espera enfermedad residual mínima), con un umbral de 3
desviaciones estándar sobre la media basal — un criterio de control estadístico de procesos
estándar, no calibrado específicamente para este contexto clínico.

Implementado en `prognosis.py::hazard_from_trajectory()` / `detect_recurrence_signal()`,
orquestado en `prognosis_demo.py::simulate_longitudinal_patient()`.

## 7. Simulación de tratamiento: perturbación gateada

La pieza que distingue a este modelo de un clasificador estático: simular el efecto
contrafactual de una intervención. La decisión de diseño clave (y un error de la primera
versión, corregido — ver `CHANGELOG.md`) es que un **tratamiento efectivo no empuja genes
específicos hacia arriba** — empujar los genes que definen el atractor propio del paciente
lo hundiría más en ese atractor, el opuesto biológico de lo que hace un tratamiento que
funciona. En cambio, un tratamiento efectivo se representa como una fuerza de
amortiguamiento proporcional al estado actual, que jala de vuelta hacia el origen:

 $$I_{\text{tx}}(t) = -\varepsilon(x, \text{mecanismo}) \cdot x(t)$$

donde  $\varepsilon \in [0, \text{base\_strength}]$  es una **eficacia gateada** por la
biología del paciente — no un valor fijo. Tres mecanismos implementados, cada uno con su
propia función de gate y evidencia clínica citada (ver docstring de
`treatment_perturbation.py`):

| Mecanismo | Gate de eficacia | Evidencia |
|---|---|---|
| Inmunoterapia anti-PD1 | $\varepsilon \propto \max(\mathrm{corr}(x, p^{\text{CMS1}}), 0)$ — sin piso basal fuera de CMS1 | KEYNOTE-177: HR=0.60–0.73 en MSI-H/dMMR |
| Anti-EGFR | $\varepsilon = 0$ si RAS/BRAF mutante; completo si wild-type; proxy débil por CMS3 si desconocido | Karapetis 2008, Douillard 2013, Di Nicolantonio 2008 |
| Quimioterapia citotóxica | $\varepsilon$ reducido (no nulo) cerca de CMS4 | Consistente con quimiorresistencia relativa de CMS4 en este proyecto |

**Alcance explícito**: la *dirección* de cada gate está fundamentada en mecanismo de acción
clínico real. La *magnitud* (`base_strength`, los coeficientes de escalamiento) es
arbitraria — no hay datos de "antes/después de tratamiento" en el proyecto contra los
cuales calibrarla. Es una herramienta de exploración in silico y generación de hipótesis,
no un predictor validado de respuesta a tratamiento.

## 8. Resumen de correspondencia código ↔ formalismo

| Símbolo | Significado | Función |
|---|---|---|
| $x$ | Vector de estado (expresión z-score) | — |
| $p^\mu$ | Patrón/atractor del subtipo $\mu$ | `calibrate_patterns_from_data()` |
| $W$ | Matriz de acoplamiento (regla de proyección) | `build_model_from_patterns()` |
| $\frac{dx}{dt} = -x + W\tanh(\beta x) + I$ | Dinámica | `dynamics()` |
| $\hat\mu = \arg\max_\mu \mathrm{corr}(x, p^\mu)$ | Clasificación | `classify_current_state()` / `risk_score_from_expression()` |
| $h(t) = \|x(t)\|$ | Riesgo ordinal | `hazard_from_trajectory()` |
| $I_{\text{tx}} = -\varepsilon(x)\, x$ | Perturbación de tratamiento | `apply_treatment_perturbation()` |

## 9. Lo que este modelo no es

Para que quede explícito, sin necesidad de ir a `PROJECT_STATUS.md`: no es un modelo
entrenado por optimización (no hay pérdida, no hay gradientes); no es una red neuronal en
el sentido de tener pesos aprendidos capa por capa (los "pesos" $W$ son una proyección
algebraica cerrada, determinística dados los patrones); no predice magnitud de respuesta a
tratamiento (solo dirección); y el riesgo que produce es ordinal, no una probabilidad de
supervivencia calibrada. Es, en el sentido más preciso, un **sistema dinámico calibrado
empíricamente contra centroides de datos reales**, con una capa de clasificación por
correlación y una capa de perturbación gateada por biología conocida.
