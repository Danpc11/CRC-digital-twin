"""Perfil CMS continuo y evaluacion molecular para investigacion.

Este modulo NO recomienda tratamientos. Separa deliberadamente la
tendencia transcriptomica CMS de los biomarcadores que determinan
elegibilidad clinica. Toda salida requiere revision por oncologia.
"""

from __future__ import annotations

import math
import numpy as np


CMS_ORDER = [
    "CMS1_MSI_immune", "CMS2_canonical_WNT",
    "CMS3_metabolic", "CMS4_mesenchymal",
]

POTENTIALLY_ELIGIBLE = "POTENCIALMENTE_ELEGIBLE"
NOT_ELIGIBLE = "NO_ELEGIBLE_POR_BIOMARCADOR"
INSUFFICIENT = "EVIDENCIA_INSUFICIENT"
REQUIRES_TEST = "REQUIERE_PRUEBA"
TRIAL_ONLY = "SOLO_ENSAYO_CLINICO"
OUT_OF_SCOPE = "FUERA_DEL_ALCANCE"


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature debe ser > 0")
    scaled = values / temperature
    scaled -= np.max(scaled)
    exp = np.exp(scaled)
    return exp / exp.sum()


def cms_continuous_profile(
    expression: np.ndarray,
    patterns: dict[str, np.ndarray],
    temperature: float = 0.25,
    dominant_threshold: float = 0.50,
    margin_threshold: float = 0.15,
) -> dict:
    """Devuelve tendencia CMS continua; las puntuaciones no son probabilidades clínicas."""
    x = np.asarray(expression, dtype=float)
    if np.linalg.norm(x) < 1e-8 or np.std(x) < 1e-12:
        return {
            "scores": {k: 0.0 for k in patterns}, "correlations": {k: math.nan for k in patterns},
            "interpretation": "indeterminado", "primary": "none", "secondary": "none",
            "margin": 0.0, "entropy": math.nan,
        }

    labels, correlations = [], []
    for label in CMS_ORDER:
        if label not in patterns:
            continue
        p = np.asarray(patterns[label], dtype=float)
        corr = float(np.corrcoef(x, p)[0, 1])
        labels.append(label)
        correlations.append(corr if np.isfinite(corr) else -1.0)
    if not labels:
        raise ValueError("No hay patrones CMS reconocidos")

    weights = _softmax(np.asarray(correlations), temperature)
    order = np.argsort(weights)[::-1]
    primary, secondary = labels[order[0]], labels[order[1]] if len(order) > 1 else "none"
    top1 = float(weights[order[0]])
    top2 = float(weights[order[1]]) if len(order) > 1 else 0.0
    interpretation = (
        "dominante" if top1 >= dominant_threshold and (top1 - top2) >= margin_threshold
        else "hibrido"
    )
    entropy = float(-np.sum(weights * np.log(weights + 1e-15)) / np.log(len(weights)))
    return {
        "scores": dict(zip(labels, map(float, weights))),
        "correlations": dict(zip(labels, map(float, correlations))),
        "interpretation": interpretation,
        "primary": primary,
        "secondary": secondary,
        "margin": top1 - top2,
        "entropy": entropy,
    }


def _result(mechanism: str, status: str, reason: str, required_tests=()) -> dict:
    return {
        "mechanism": mechanism, "status": status, "reason": reason,
        "required_tests": list(required_tests),
    }


def assess_molecular_eligibility(biomarkers: dict, clinical: dict | None = None) -> list[dict]:
    """Aplica compuertas moleculares; no selecciona regimen, dosis ni secuencia."""
    clinical = clinical or {}
    metastatic = clinical.get("metastatic")
    if metastatic is not True:
        return [_result(
            "systemic_selection", OUT_OF_SCOPE,
            "Estas reglas se limitan a enfermedad avanzada/metastasica; requiere contexto clinico completo."
        )]

    msi = biomarkers.get("msi_mmr", "unknown")
    ras = biomarkers.get("ras", "unknown")
    braf = biomarkers.get("braf", "unknown")
    her2 = biomarkers.get("her2", "unknown")
    kras_g12c = biomarkers.get("kras_g12c", "unknown")
    ntrk = biomarkers.get("ntrk", "unknown")
    results = []

    if msi == "msi_h_dmmr":
        results.append(_result("immune_checkpoint_inhibition", POTENTIALLY_ELIGIBLE,
                               "MSI-H/dMMR confirmado; verificar linea, contraindicaciones y regulacion local."))
    elif msi == "mss_pmmr":
        results.append(_result("immune_checkpoint_inhibition", NOT_ELIGIBLE,
                               "MSS/pMMR: CMS1 o señal inmune no sustituyen MSI/MMR."))
    else:
        results.append(_result("immune_checkpoint_inhibition", REQUIRES_TEST,
                               "Falta MSI/MMR validado.", ["MSI por PCR/NGS o MMR por IHC"]))

    if ras == "wild_type" and braf == "wild_type":
        side = clinical.get("primary_side", "unknown")
        reason = "RAS/BRAF wild-type confirmado; integrar localizacion primaria, linea y objetivo terapeutico."
        if side == "right":
            reason += " Tumor derecho: el beneficio anti-EGFR depende especialmente del contexto/linea."
        results.append(_result("anti_EGFR", POTENTIALLY_ELIGIBLE, reason))
    elif ras == "mutated" or braf == "v600e":
        results.append(_result("anti_EGFR_monotherapy", NOT_ELIGIBLE,
                               "RAS mutado o BRAF V600E: no usar CMS como sustituto de la compuerta molecular."))
    else:
        missing = []
        if ras == "unknown": missing.append("KRAS/NRAS")
        if braf == "unknown": missing.append("BRAF V600E")
        results.append(_result("anti_EGFR", REQUIRES_TEST, "Falta genotipado RAS/BRAF.", missing))

    if braf == "v600e":
        results.append(_result("BRAF_plus_EGFR", POTENTIALLY_ELIGIBLE,
                               "BRAF V600E confirmado; verificar esquema aprobado, linea y regulacion local."))
    elif braf == "wild_type":
        results.append(_result("BRAF_plus_EGFR", NOT_ELIGIBLE, "BRAF V600E no detectado."))
    else:
        results.append(_result("BRAF_plus_EGFR", REQUIRES_TEST, "Falta BRAF V600E.", ["BRAF V600E"]))

    if her2 == "positive" and ras == "wild_type":
        results.append(_result("HER2_targeted", POTENTIALLY_ELIGIBLE,
                               "HER2 positivo y RAS wild-type; verificar metodo, linea e indicacion local."))
    elif her2 == "unknown":
        results.append(_result("HER2_targeted", REQUIRES_TEST, "Falta evaluacion HER2.", ["HER2 IHC/ISH o ensayo validado"]))
    else:
        results.append(_result("HER2_targeted", NOT_ELIGIBLE,
                               "No cumple simultaneamente HER2 positivo y RAS wild-type."))

    if kras_g12c == "positive":
        results.append(_result("KRAS_G12C_plus_EGFR", POTENTIALLY_ELIGIBLE,
                               "KRAS G12C confirmado; verificar tratamientos previos e indicacion local."))
    elif kras_g12c == "unknown":
        results.append(_result("KRAS_G12C_plus_EGFR", REQUIRES_TEST, "Falta KRAS G12C.", ["KRAS G12C"]))
    else:
        results.append(_result("KRAS_G12C_plus_EGFR", NOT_ELIGIBLE, "KRAS G12C no detectado."))

    if ntrk == "fusion_positive":
        results.append(_result("TRK_inhibitor", POTENTIALLY_ELIGIBLE,
                               "Fusion NTRK confirmada; verificar ensayo confirmatorio e indicacion local."))
    elif ntrk == "unknown":
        results.append(_result("TRK_inhibitor", REQUIRES_TEST, "NTRK no evaluado.", ["Fusion NTRK"]))
    else:
        results.append(_result("TRK_inhibitor", NOT_ELIGIBLE, "Fusion NTRK no detectada."))

    results.append(_result("chemotherapy_or_antiangiogenic_backbone", INSUFFICIENT,
                           "Requiere linea, resecabilidad, ECOG, comorbilidades, toxicidad previa y valoracion oncologica."))
    return results


def hybrid_mechanism_hypotheses(cms_profile: dict) -> list[dict]:
    """Genera prioridades exploratorias ponderadas, nunca combinaciones recomendadas."""
    scores = cms_profile["scores"]
    mapping = {
        "immune_axis": scores.get("CMS1_MSI_immune", 0.0),
        "canonical_EGFR_WNT_axis": scores.get("CMS2_canonical_WNT", 0.0),
        "metabolic_RAS_axis": scores.get("CMS3_metabolic", 0.0),
        "stromal_TGFbeta_axis": scores.get("CMS4_mesenchymal", 0.0),
    }
    return [
        {"hypothesis": k, "weight": float(v), "status": TRIAL_ONLY,
         "reason": "Peso transcriptomico exploratorio; requiere biomarcador y evidencia de combinacion."}
        for k, v in sorted(mapping.items(), key=lambda item: item[1], reverse=True)
    ]
