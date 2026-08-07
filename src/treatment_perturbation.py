"""
treatment_perturbation.py

Primer termino de perturbacion farmacodinamica del gemelo digital --
la pieza que falta para pasar de "clasificador dinamico" a "simulador
de intervenciones", que es la diferencia conceptual real con
herramientas comerciales tipo Cellworks (que si simulan tratamiento,
pero via NGS completo, no compatible con la restriccion de qPCR de
este proyecto).

============================================================
ALCANCE Y LIMITACIONES -- LEER ANTES DE USAR PARA CUALQUIER COSA
============================================================
Este modulo es un ESQUELETO CONCEPTUAL. La DIRECCION de cada efecto
esta fundamentada en literatura clinica real (citada abajo, por
mecanismo). La MAGNITUD (base_strength, los coeficientes de
escalamiento) es arbitraria -- no existe en este proyecto ningun dato
real de "antes/despues de tratamiento" contra el cual calibrar cuanto
se mueve el estado del paciente. Sin esos datos (tipo ensayo con brazo
de tratamiento vs. observacion, ver conversacion de diseno sobre
DYNAMIC/GALAXY), este modulo sirve para EXPLORACION IN SILICO y para
generar hipotesis, nunca para decisiones clinicas ni para afirmar que
el gemelo "predice respuesta a tratamiento" en sentido validado.

Tres mecanismos implementados, cada uno con evidencia real citada. En
los tres, un tratamiento EFECTIVO se representa como una fuerza de
amortiguamiento que jala el vector de estado de vuelta hacia el origen
("sin enfermedad residual") -- no como un empuje sobre genes
especificos, porque empujar los genes que definen el atractor propio
del paciente (ej. reforzar GNLY/USP18 en un paciente ya CMS1) lo
hundiria mas en ese atractor, el opuesto biologico de lo que hace un
tratamiento que funciona. Los "target_genes" de cada mecanismo abajo
son la base biologica del gate de eficacia, no la direccion de la fuerza:

  1. Inmunoterapia (anti-PD-1, ej. pembrolizumab)
     Eficaz especificamente en tumores MSI-H/dMMR (eje CMS1).
     KEYNOTE-177 (Andre et al. 2020, NEJM; actualizacion a 5 anios,
     Andre et al. 2024, Ann Oncol): HR=0.60-0.73 a favor de
     pembrolizumab vs. quimioterapia en MSI-H/dMMR metastasico.
     Los tumores MSS (microsatelite-estable, fuera de CMS1) son
     "inmunologicamente frios" y NO se benefician de forma
     establecida de checkpoint inhibitors en monoterapia -- este
     modulo NO le da un beneficio basal a pacientes fuera de CMS1,
     porque asumir eso seria una sobreclamacion no respaldada.

  2. Anti-EGFR (cetuximab/panitumumab)
     Eficaz SOLO en tumores RAS (KRAS/NRAS) y BRAF wild-type
     (Karapetis et al. 2008, NEJM; Douillard et al. 2013, NEJM;
     Di Nicolantonio et al. 2008, JCO). Sin beneficio establecido en
     mutantes -- de hecho posible dano en algunos subgrupos BRAF-mutante.
     Este panel de ARN NO mide mutacion de KRAS/BRAF directamente --
     requiere la capa de ADN (qPCR alelo-especifico/HRM) ya descrita
     en el diseno original del proyecto (ver README). Si no hay
     estatus real, se usa la cercania al atractor CMS3 (metabolico,
     enriquecido en tumores KRAS-mutantes segun Yuan et al. 2025,
     Nat Commun) como proxy DEBIL -- explicitamente marcado como tal,
     nunca como sustituto de la prueba de mutacion real.

  3. Quimioterapia citotoxica (ej. FOLFOX)
     Efecto general (mecanismo no CMS-especifico), pero
     consistentemente menos eficaz en tumores CMS4 -- patron
     establecido en literatura CMS y confirmado empiricamente en este
     mismo proyecto (CMS4 es el eje de peor pronostico mas robusto en
     las 3 cohortes de validacion externa, ver PROJECT_STATUS.md).
"""

import numpy as np

TREATMENT_MECHANISMS = {
    "immunotherapy_antiPD1": {
        "target_genes": ["GNLY", "USP18"],
        "gate": "cms1_like",
        "evidence": (
            "KEYNOTE-177 (Andre et al. 2020 NEJM; actualizacion 2024 Ann Oncol): "
            "HR=0.60-0.73 vs quimioterapia en MSI-H/dMMR mCRC. Sin beneficio "
            "establecido fuera de MSI-H/dMMR (tumores MSS 'inmunologicamente frios')."
        ),
    },
    "anti_egfr": {
        "target_genes": ["MYC", "AXIN2"],
        "gate": "ras_braf_wildtype",
        "evidence": (
            "Karapetis et al. 2008 NEJM; Douillard et al. 2013 NEJM; "
            "Di Nicolantonio et al. 2008 JCO: RAS/BRAF mutante = sin beneficio."
        ),
    },
    "cytotoxic_chemo": {
        "target_genes": ["VIM", "TGFB1"],
        "gate": "reduced_efficacy_cms4",
        "evidence": (
            "CMS4 = peor pronostico consistente en las 3 cohortes externas de "
            "este proyecto + literatura CMS general de quimiorresistencia relativa."
        ),
    },
}


def apply_treatment_perturbation(
    x_current: np.ndarray,
    gene_order: list,
    treatment: str,
    patterns: dict,
    base_strength: float = 0.5,
    ras_braf_wildtype: bool | None = None,
) -> np.ndarray:
    """
    Devuelve un termino de forzamiento I_treatment (mismo shape que
    x_current) representando el efecto de un tratamiento EFECTIVO --
    para sumar a la dinamica junto con (o en vez de) el termino de
    recaida.

    DISENO IMPORTANTE: un tratamiento que funciona reduce la carga
    tumoral -- el paciente se mueve de vuelta hacia el origen ("sin
    enfermedad residual"), NO se empuja mas fuerte hacia el atractor
    de su propio subtipo. Por eso el termino es una fuerza de
    amortiguamiento proporcional al estado actual (-efficacy * x),
    no un empuje sobre genes especificos -- empujar genes que definen
    el atractor del paciente (ej. GNLY/USP18 en un paciente CMS1)
    lo hundiria mas en ese atractor en vez de limpiarlo, que es
    biologicamente lo opuesto a lo que hace un tratamiento efectivo.

    La eficacia (cuanto jala hacia el origen) esta gateada por la
    biologia del mecanismo -- ver TREATMENT_MECHANISMS y el docstring
    del modulo para la evidencia de cada gate.

    ras_braf_wildtype: SOLO relevante para 'anti_egfr'.
        None  -> estatus desconocido, se usa CMS3 como proxy DEBIL (con
                 eficacia reducida por incertidumbre)
        True  -> wild-type confirmado (ej. por qPCR alelo-especifico/HRM) -> eficacia completa
        False -> mutante confirmado -> eficacia CERO (sin beneficio establecido)
    """
    if treatment not in TREATMENT_MECHANISMS:
        raise ValueError(f"Tratamiento desconocido: {treatment}. Opciones: {list(TREATMENT_MECHANISMS)}")

    spec = TREATMENT_MECHANISMS[treatment]
    missing = [g for g in spec["target_genes"] if g not in gene_order]
    if missing:
        raise ValueError(
            f"El panel no incluye los genes del mecanismo de '{treatment}' ({missing}) -- "
            "la evidencia clinica de este tratamiento se basa en esos genes, sin ellos "
            "el gate de eficacia no tiene fundamento en este panel."
        )
    norm = np.linalg.norm(x_current)

    if spec["gate"] == "cms1_like":
        # Eficacia proporcional a la cercania al atractor CMS1 -- SIN piso
        # basal para pacientes lejos de CMS1, porque asumir beneficio en
        # MSS seria una sobreclamacion (ver docstring del modulo).
        if norm > 1e-8 and "CMS1_MSI_immune" in patterns:
            corr = np.corrcoef(x_current, patterns["CMS1_MSI_immune"])[0, 1]
            efficacy = base_strength * max(corr, 0.0)
        else:
            efficacy = 0.0

    elif spec["gate"] == "ras_braf_wildtype":
        if ras_braf_wildtype is False:
            efficacy = 0.0
        elif ras_braf_wildtype is True:
            efficacy = base_strength
        else:
            # proxy debil por RNA -- NO sustituye la prueba de mutacion real
            if norm > 1e-8 and "CMS3_metabolic" in patterns:
                corr_cms3 = np.corrcoef(x_current, patterns["CMS3_metabolic"])[0, 1]
                efficacy = base_strength * (1.0 - max(corr_cms3, 0.0)) * 0.5  # penalizado por incertidumbre
            else:
                efficacy = base_strength * 0.25  # sin ninguna informacion -- muy conservador

    elif spec["gate"] == "reduced_efficacy_cms4":
        if norm > 1e-8 and "CMS4_mesenchymal" in patterns:
            corr_cms4 = np.corrcoef(x_current, patterns["CMS4_mesenchymal"])[0, 1]
            efficacy = base_strength * (1.0 - 0.5 * max(corr_cms4, 0.0))  # hasta 50% menos eficaz, nunca cero
        else:
            efficacy = base_strength

    else:
        raise ValueError(f"Gate desconocido: {spec['gate']}")

    # Fuerza de amortiguamiento hacia el origen, proporcional al estado
    # actual -- no un empuje sobre genes especificos.
    return -efficacy * x_current


def describe_treatment(treatment: str) -> str:
    """Descripcion en texto del mecanismo y evidencia de un tratamiento, para reportes."""
    if treatment not in TREATMENT_MECHANISMS:
        raise ValueError(f"Tratamiento desconocido: {treatment}")
    spec = TREATMENT_MECHANISMS[treatment]
    return (
        f"{treatment}: eficacia gateada por biologia asociada a "
        f"{', '.join(spec['target_genes'])} (gate='{spec['gate']}'). "
        f"Si aplica, jala el estado de vuelta hacia el origen (reduce carga tumoral). "
        f"Evidencia: {spec['evidence']}"
    )
