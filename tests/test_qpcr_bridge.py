"""
qpcr_bridge.py

Puente entre valores de Ct crudos de RT-qPCR y la escala numerica que
espera la app (la misma en la que estan calculadas las estadisticas de
referencia congeladas, _ref_mean/_ref_std, calibradas sobre expresion
log2 de microarreglos de GSE39582).

POR QUE HACE FALTA ESTO
------------------------
El Ct es una escala INVERSA a la expresion (menos ciclos = mas
senal) y en un rango numerico (~15-35) completamente distinto al de
expresion log2 de microarreglos. Subir valores de Ct crudos a la app
NO produce un error -- produce una clasificacion sin sentido, porque
(Ct - ref_mean) / ref_std no es un z-score valido. Este modulo resuelve
dos pasos:

  1. Delta-Ct: normaliza cada gen contra un gen de referencia estable
     (housekeeping), en la MISMA muestra -- corrige diferencias de
     calidad/cantidad de ARN de entrada entre muestras. Signo elegido
     para que aumente con la expresion (igual direccion que expresion
     log2 de microarreglos), no como el Ct crudo.

  2. Recalibracion de escala: el Delta-Ct, aunque ya en la direccion
     correcta, NO esta garantizado en la MISMA escala absoluta (misma
     media, mismo rango) que la referencia de microarreglos -- distinto
     gen de referencia, distinta eficiencia de PCR, distinto diseño de
     primers. Se ajusta una transformacion lineal por gen
     (escala_referencia ~= a * DeltaCt + b) usando muestras ancla.

DOS MODOS DE ANCLAJE (VERIFICADOS AMBOS, CON RIGOR DISTINTO)
--------------------------------------------------------------
  "pareado" (gold standard): las muestras ancla tienen valor conocido
     EN LA ESCALA DE REFERENCIA (ej. porque tambien se midieron por
     microarreglos/RNA-seq). El ajuste recupera la transformacion real.

  "por_centroide" (practico, mas debil): las muestras ancla solo tienen
     un CMS ya conocido por otros medios (no remedicion pareada). Se
     usa el centroide calibrado de esa clase como objetivo aproximado
     -- asume que la muestra ancla se parece a su clase "tipica", lo
     cual NO siempre es cierto. Usar con cautela, con el R2 del ajuste
     como senal de que tan bien esta funcionando.

NO VERIFICADO TODAVIA CON DATOS REALES DE RT-qPCR -- este modulo se
construyo y probo con datos simulados. Antes de confiar en el en una
demo con pacientes reales, correr el protocolo de la seccion "USO"
sobre muestras de referencia con CMS ya conocido y confirmar que la
clasificacion resultante es la esperada.

USO
    from qpcr_bridge import compute_delta_ct, fit_qpcr_bridge, apply_qpcr_bridge

    delta_ct_ancla = {gen: [...] for gen in genes}  # Delta-Ct de N muestras ancla
    cms_ancla = ["CMS1_MSI_immune", "CMS3_metabolic", ...]  # CMS conocido de cada ancla

    bridge = fit_qpcr_bridge(delta_ct_ancla, cms_ancla, patterns, gene_order, modo="por_centroide")
    valor_calibrado = apply_qpcr_bridge(delta_ct_paciente_nuevo, bridge, gene_order)
    # valor_calibrado ya esta en la escala que espera zscore_genes(..., stats=frozen_stats)
"""

import numpy as np


def compute_delta_ct(ct_gene: dict, ct_reference: float) -> dict:
    """
    Delta-Ct con signo invertido: ref - gen, para que AUMENTE con la
    expresion (misma direccion que expresion log2 de microarreglos,
    al contrario que el Ct crudo).

    ct_gene: {nombre_gen: Ct_crudo} para una sola muestra
    ct_reference: Ct del gen de referencia/housekeeping, MISMA muestra
    """
    return {gene: ct_reference - ct for gene, ct in ct_gene.items()}


def fit_qpcr_bridge(
    delta_ct_anchors: dict[str, list[float]], anchor_targets: dict[str, list[float]],
    min_r2: float = 0.5,
) -> dict:
    """
    Ajusta una transformacion lineal (a, b) POR GEN: escala_referencia
    ~= a * DeltaCt + b, usando regresion lineal simple sobre las
    muestras ancla.

    delta_ct_anchors: {gen: [DeltaCt de cada muestra ancla]}
    anchor_targets: {gen: [valor objetivo en escala de referencia,
        mismo orden de muestras]} -- ya sea el valor pareado real, o
        el valor del centroide de la clase conocida de cada ancla
        (ver fit_qpcr_bridge_from_known_cms para ese caso).

    Devuelve {gen: {"a":..., "b":..., "r2":..., "n_anclas":...}}. Avisa
    (no revienta) si el ajuste es pobre (R2 bajo) o hay muy pocas
    anclas -- una recta con 2 puntos siempre da R2=1, eso NO es
    evidencia de que el ajuste generalice.
    """
    resultado = {}
    for gene in delta_ct_anchors:
        x = np.array(delta_ct_anchors[gene], dtype=float)
        y = np.array(anchor_targets[gene], dtype=float)
        n = len(x)

        if n < 2:
            resultado[gene] = {"a": None, "b": None, "r2": None, "n_anclas": n,
                                "aviso": f"solo {n} ancla(s) -- no se puede ajustar una recta"}
            continue

        a, b = np.polyfit(x, y, 1)
        y_pred = a * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

        entrada = {"a": float(a), "b": float(b), "r2": float(r2), "n_anclas": n}
        if n < 4:
            entrada["aviso"] = f"solo {n} anclas -- ajuste fragil, agregar mas si es posible"
        elif not np.isnan(r2) and r2 < min_r2:
            entrada["aviso"] = f"R2={r2:.2f} bajo -- el ajuste lineal no explica bien estas anclas"
        resultado[gene] = entrada
    return resultado


def fit_qpcr_bridge_from_known_cms(
    delta_ct_anchors: dict[str, list[float]], anchor_cms_labels: list[str],
    patterns: dict[str, np.ndarray], gene_order: list[str], min_r2: float = 0.5,
) -> dict:
    """
    Modo "por_centroide": arma los objetivos de anclaje a partir del
    CENTROIDE calibrado de la clase CMS ya conocida de cada muestra
    ancla, en vez de un valor pareado real -- mas debil (asume que la
    muestra ancla se parece a su clase "tipica"), pero es lo unico
    disponible sin remedicion pareada.
    """
    anchor_targets = {gene: [] for gene in gene_order}
    for label in anchor_cms_labels:
        if label not in patterns:
            raise ValueError(f"CMS '{label}' no esta en los patrones calibrados: {list(patterns)}")
        for i, gene in enumerate(gene_order):
            anchor_targets[gene].append(float(patterns[label][i]))

    delta_ct_subset = {gene: delta_ct_anchors[gene] for gene in gene_order}
    return fit_qpcr_bridge(delta_ct_subset, anchor_targets, min_r2=min_r2)


def apply_qpcr_bridge(delta_ct_patient: dict, bridge: dict, gene_order: list[str]) -> np.ndarray:
    """Aplica la transformacion ajustada a un paciente nuevo -- devuelve
    el vector ya en la escala que espera zscore_genes(..., stats=frozen_stats)."""
    valores = []
    for gene in gene_order:
        ajuste = bridge.get(gene)
        if ajuste is None or ajuste.get("a") is None:
            raise ValueError(f"No hay ajuste valido para '{gene}' -- revisar anclas de ese gen.")
        valores.append(ajuste["a"] * delta_ct_patient[gene] + ajuste["b"])
    return np.array(valores)
