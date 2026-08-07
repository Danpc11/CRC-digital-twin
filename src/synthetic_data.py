"""
synthetic_data.py

Genera un TSV sintetico con la MISMA estructura que se espera de
GSE39582 + etiquetas CMS de Synapse, para poder probar que el pipeline
completo (calibracion -> validacion de supervivencia) corre de punta a
punta ANTES de tocar datos reales.

Los datos aqui NO representan biologia real -- son ruido gaussiano
alrededor de centroides sinteticos, con un gradiente de supervivencia
inyectado a proposito (CMS4 peor pronostico, CMS1 mejor) para poder
verificar que el pipeline de validacion detecta una senal cuando
efectivamente existe una.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GENES = ["MLH1", "GNLY", "USP18", "MYC", "AXIN2", "FABP1", "CPS1", "SI", "VIM", "TGFB1"]

TRUE_CENTROIDS = {
    "CMS1_MSI_immune":    np.array([-2.0, 2.0, 2.0, -1.0, -1.0, -0.8, -0.8, -0.8, -1.0, -0.8]),
    "CMS2_canonical_WNT": np.array([-1.0, -0.8, -0.8, 2.0, 2.0, -0.8, -0.6, -0.6, -1.0, -0.8]),
    "CMS3_metabolic":     np.array([-0.8, -0.6, -0.6, -0.8, -0.6, 2.0, 2.0, 2.0, -0.8, -0.6]),
    "CMS4_mesenchymal":   np.array([-1.0, -0.8, -0.8, -1.0, -0.8, -0.6, -0.6, -0.6, 2.0, 2.0]),
}

# Parametros de supervivencia sinteticos por subtipo (mediana en meses,
# tasa de evento) -- CMS4 peor pronostico, CMS1 mejor, consistente con
# la literatura real de CMS (usado solo para inyectar senal detectable,
# NO son valores clinicos reales).
SURVIVAL_PARAMS = {
    "CMS1_MSI_immune":    {"median_months": 60, "event_rate": 0.20},
    "CMS2_canonical_WNT": {"median_months": 45, "event_rate": 0.30},
    "CMS3_metabolic":     {"median_months": 40, "event_rate": 0.35},
    "CMS4_mesenchymal":   {"median_months": 25, "event_rate": 0.55},
}


def generate_synthetic_cohort(
    n_per_class: int = 80, noise_sigma: float = 1.2, seed: int = 42
) -> pd.DataFrame:
    """Genera un dataframe sintetico con expresion + subtipo + supervivencia."""
    rng = np.random.default_rng(seed)
    rows = []
    sample_idx = 0

    for label, centroid in TRUE_CENTROIDS.items():
        params = SURVIVAL_PARAMS[label]
        for _ in range(n_per_class):
            expr = centroid + rng.normal(0, noise_sigma, size=len(GENES))
            event = rng.random() < params["event_rate"]
            # tiempo exponencial alrededor de la mediana del subtipo
            duration = rng.exponential(params["median_months"])
            row = {"sample_id": f"SYN-{sample_idx:04d}", "cms_label": label}
            row.update({gene: val for gene, val in zip(GENES, expr)})
            row["relapse_free_months"] = round(duration, 1)
            row["relapse_event"] = int(event)
            rows.append(row)
            sample_idx += 1

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path

    df = generate_synthetic_cohort()
    out_path = Path(__file__).resolve().parents[1] / "data" / "synthetic_cohort.tsv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Cohorte sintetica generada: {out_path} ({len(df)} pacientes)")
