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

ESCALAS -- LEER ANTES DE TOCAR ESTE MODULO (bug corregido 2026-09-11)
-----------------------------------------------------------------------
Hay DOS escalas numericas en juego y no son intercambiables:

  * escala CRUDA de referencia: expresion log2 de microarreglos, donde
    viven _ref_mean/_ref_std (calibrated_patterns.tsv). Es la escala que
    espera zscore_genes(..., stats=frozen_stats).
  * escala Z: z-score contra esa referencia. Los CENTROIDES calibrados
    (patterns) estan en escala Z, no en escala cruda.

El modo "por_centroide" usa los centroides como objetivo, asi que sin
mas produce valores en escala Z. Si despues se les aplica
zscore_genes(stats=frozen) se normaliza DOS VECES con estadisticas de
escalas distintas y la clasificacion sale sin sentido (verificado: un
paciente CMS1 sintetico salia CMS2). Por eso
fit_qpcr_bridge_from_known_cms acepta gene_stats y, cuando se le pasan,
convierte los objetivos a escala cruda (z*std+mean) para que la
posterior normalizacion congelada sea correcta. Si NO se pasan
gene_stats, el resultado esta en escala Z y NO debe volverse a
normalizar -- ver apply_qpcr_bridge(..., expected_scale) y
classify_delta_ct(), que resuelve la escala automaticamente.

NO VERIFICADO TODAVIA CON DATOS REALES DE RT-qPCR -- este modulo se
construyo y probo con datos simulados. Antes de confiar en el en una
demo con pacientes reales, correr el protocolo de la seccion "USO"
sobre muestras de referencia con CMS ya conocido y confirmar que la
clasificacion resultante es la esperada.

USO
    from qpcr_bridge import compute_delta_ct, fit_qpcr_bridge_from_known_cms, classify_delta_ct

    delta_ct_ancla = {gen: [...] for gen in genes}  # Delta-Ct de N muestras ancla
    cms_ancla = ["CMS1_MSI_immune", "CMS3_metabolic", ...]  # CMS conocido de cada ancla
    frozen = load_gene_reference_stats("calibrated_patterns.tsv")

    bridge = fit_qpcr_bridge_from_known_cms(
        delta_ct_ancla, cms_ancla, patterns, gene_order, gene_stats=frozen)
    cms, corrs, z = classify_delta_ct(delta_ct_paciente, bridge, gene_order, patterns, frozen)
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
    gene_stats: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """
    Modo "por_centroide": arma los objetivos de anclaje a partir del
    CENTROIDE calibrado de la clase CMS ya conocida de cada muestra
    ancla, en vez de un valor pareado real -- mas debil (asume que la
    muestra ancla se parece a su clase "tipica"), pero es lo unico
    disponible sin remedicion pareada.

    gene_stats: {gen: (ref_mean, ref_std)} de la calibracion. Si se
    pasan, los objetivos se llevan a escala CRUDA (z*std+mean) y el
    puente resultante produce valores listos para
    zscore_genes(..., stats=gene_stats). Si NO se pasan, los objetivos
    quedan en escala Z y el resultado NO debe volverse a normalizar.
    El puente guarda en "_meta" en que escala esta para que
    apply_qpcr_bridge pueda avisar si se usa mal.
    """
    anchor_targets = {gene: [] for gene in gene_order}
    for label in anchor_cms_labels:
        if label not in patterns:
            raise ValueError(f"CMS '{label}' no esta en los patrones calibrados: {list(patterns)}")
        for i, gene in enumerate(gene_order):
            z_target = float(patterns[label][i])
            if gene_stats is not None:
                if gene not in gene_stats:
                    raise ValueError(f"gene_stats no tiene referencia para '{gene}'")
                mu, sigma = gene_stats[gene]
                if sigma == 0 or np.isnan(sigma):
                    raise ValueError(f"sigma de referencia invalido para '{gene}'")
                anchor_targets[gene].append(z_target * sigma + mu)
            else:
                anchor_targets[gene].append(z_target)

    delta_ct_subset = {gene: delta_ct_anchors[gene] for gene in gene_order}
    bridge = fit_qpcr_bridge(delta_ct_subset, anchor_targets, min_r2=min_r2)
    bridge["_meta"] = {"output_scale": "raw" if gene_stats is not None else "z"}
    return bridge


def apply_qpcr_bridge(
    delta_ct_patient: dict, bridge: dict, gene_order: list[str],
    expected_scale: str | None = None,
) -> np.ndarray:
    """Aplica la transformacion ajustada a un paciente nuevo.

    expected_scale: "raw" o "z". Si se indica y no coincide con la escala
    en la que se ajusto el puente (bridge["_meta"]["output_scale"]),
    lanza ValueError -- es la proteccion contra la doble normalizacion.
    """
    meta_scale = bridge.get("_meta", {}).get("output_scale")
    if expected_scale is not None and meta_scale is not None and expected_scale != meta_scale:
        raise ValueError(
            f"El puente produce valores en escala '{meta_scale}' pero se esperaba "
            f"'{expected_scale}'. Si vas a aplicar zscore_genes(stats=frozen) despues, "
            "ajusta el puente con gene_stats; si no, no vuelvas a normalizar.")
    valores = []
    for gene in gene_order:
        ajuste = bridge.get(gene)
        if ajuste is None or ajuste.get("a") is None:
            raise ValueError(f"No hay ajuste valido para '{gene}' -- revisar anclas de ese gen.")
        valores.append(ajuste["a"] * delta_ct_patient[gene] + ajuste["b"])
    return np.array(valores)


def classify_delta_ct(
    delta_ct_patient: dict, bridge: dict, gene_order: list[str],
    patterns: dict[str, np.ndarray],
    gene_stats: dict[str, tuple[float, float]] | None = None,
) -> tuple[str, dict[str, float], np.ndarray]:
    """Camino unico y seguro: Delta-Ct -> escala del modelo -> z -> CMS.

    Resuelve la escala segun como se ajusto el puente, de modo que la
    normalizacion congelada se aplica exactamente UNA vez (o ninguna).
    Devuelve (cms_predicho, correlaciones, vector_z).
    """
    scale = bridge.get("_meta", {}).get("output_scale", "z")
    valores = apply_qpcr_bridge(delta_ct_patient, bridge, gene_order)
    if scale == "raw":
        if gene_stats is None:
            raise ValueError("El puente esta en escala cruda: hacen falta gene_stats para el z-score.")
        z = np.array([(valores[i] - gene_stats[g][0]) / gene_stats[g][1]
                      for i, g in enumerate(gene_order)])
    else:
        z = valores
    if np.linalg.norm(z) < 1e-8 or np.std(z) < 1e-12:
        return "none", {k: float("nan") for k in patterns}, z
    corrs = {label: float(np.corrcoef(z, np.asarray(c, dtype=float))[0, 1])
             for label, c in patterns.items()}
    return max(corrs, key=corrs.get), corrs, z
