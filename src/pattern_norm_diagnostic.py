"""
pattern_norm_diagnostic.py

Diagnostico de la hipotesis alternativa al "hallazgo CMS2 inalcanzable":
que la dominancia de CMS1 en la dinamica Modern Hopfield es un
artefacto de que los centroides calibrados tienen NORMAS DESIGUALES.

El campo es dx/dt = -x + X softmax(beta X^T x). El termino X^T x
favorece por construccion al patron de mayor norma: si ||p_CMS1|| es
la mayor, cerca del origen su logit crece mas rapido que los otros para
cualquier x, y el softmax lo elige. Ramsauer et al. (2020) discuten
exactamente esto via la separacion Delta_mu y ||p_mu||.

Que hace este script:
  1. Reporta la norma de cada centroide y su cociente respecto al mayor.
  2. Corre find_minimum_forcing_strength (V1, sin estabilizador) para
     los 4 CMS con los centroides TAL CUAL y con los centroides
     NORMALIZADOS a norma unitaria (la direccion se conserva; la
     correlacion, que es la metrica de clasificacion, es invariante a
     esta reescala).
  3. Opcionalmente repite el barrido V1/V2 completo
     (compare_forcing_sweep_v1_v2) con ambos juegos de patrones.

Si con patrones normalizados los umbrales de CMS2/CMS3/CMS4 caen a los
de CMS1, la asimetria era de normas, no de biologia, y la pregunta
metodologica pasa a ser: por que no normalizar X antes de anadir
estabilizador + correccion basal + rampa. Este script NO cambia el
motor de la app; solo produce la evidencia para decidirlo.

USO:
    python3 src/pattern_norm_diagnostic.py \\
        --patterns results_gse39582/calibrated_patterns.tsv \\
        --beta 3.0 --output results_pattern_norm/ [--full-sweep]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration import load_calibrated_patterns
from modern_hopfield import (
    compare_forcing_sweep_v1_v2,
    find_minimum_forcing_strength,
    summarize_forcing_thresholds,
)


def normalize_patterns(patterns: dict[str, np.ndarray], target_norm: float | None = None) -> dict:
    """Reescala cada centroide a norma comun (por defecto la media de las normas,
    para no cambiar la escala global del paisaje, solo igualar entre clases)."""
    norms = {k: float(np.linalg.norm(v)) for k, v in patterns.items()}
    if any(n < 1e-12 for n in norms.values()):
        raise ValueError("Algun centroide tiene norma ~0")
    scale = float(np.mean(list(norms.values()))) if target_norm is None else target_norm
    return {k: np.asarray(v, dtype=float) * (scale / norms[k]) for k, v in patterns.items()}


def norm_table(patterns: dict) -> pd.DataFrame:
    norms = {k: float(np.linalg.norm(v)) for k, v in patterns.items()}
    mx = max(norms.values())
    return pd.DataFrame([
        {"patron": k, "norma": n, "cociente_vs_mayor": n / mx} for k, n in norms.items()
    ]).sort_values("norma", ascending=False)


def v1_thresholds(patterns: dict, beta: float, candidates, corr_threshold: float) -> pd.DataFrame:
    rows = []
    for label in patterns:
        res = find_minimum_forcing_strength(
            patterns, label, beta=beta, strength_candidates=candidates,
            corr_threshold=corr_threshold)
        first = res["detalle"][0] if res["detalle"] else {}
        rows.append({
            "patron_objetivo": label,
            "umbral_minimo_v1": res["umbral_minimo_encontrado"],
            "corr_objetivo_fuerza_minima": first.get("corr_con_objetivo"),
            "termina_pareciendose_a_fuerza_minima": first.get("mas_parecido_a"),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--patterns", required=True)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--corr-threshold", type=float, default=0.9)
    ap.add_argument("--forcing-candidates", type=float, nargs="+",
                    default=[0.7, 1.5, 3.0, 5.0, 8.0, 12.0, 20.0])
    ap.add_argument("--output", default="results_pattern_norm")
    ap.add_argument("--full-sweep", action="store_true",
                    help="Ademas del V1 rapido, correr compare_forcing_sweep_v1_v2 con ambos juegos")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    patterns, genes = load_calibrated_patterns(args.patterns)
    normalized = normalize_patterns(patterns)

    print("=== Normas de los centroides calibrados ===")
    nt = norm_table(patterns)
    print(nt.to_string(index=False))
    nt.to_csv(out / "pattern_norms.tsv", sep="\t", index=False)

    print(f"\n=== Umbral minimo de fuerza (V1, beta={args.beta}) -- centroides tal cual ===")
    raw_t = v1_thresholds(patterns, args.beta, args.forcing_candidates, args.corr_threshold)
    print(raw_t.to_string(index=False))

    print(f"\n=== Umbral minimo de fuerza (V1, beta={args.beta}) -- centroides con norma igualada ===")
    norm_t = v1_thresholds(normalized, args.beta, args.forcing_candidates, args.corr_threshold)
    print(norm_t.to_string(index=False))

    merged = raw_t.merge(norm_t, on="patron_objetivo", suffixes=("_crudo", "_normalizado"))
    merged.to_csv(out / "v1_thresholds_raw_vs_normalized.tsv", sep="\t", index=False)

    if args.full_sweep:
        for tag, pats in (("crudo", patterns), ("normalizado", normalized)):
            print(f"\n=== Barrido V1/V2 completo -- {tag} ===")
            sweep = compare_forcing_sweep_v1_v2(
                pats, beta=args.beta, strength_candidates=args.forcing_candidates,
                corr_threshold=args.corr_threshold)
            sweep.to_csv(out / f"sweep_v1_v2_{tag}.tsv", sep="\t", index=False)
            summary = summarize_forcing_thresholds(sweep)
            print(summary.to_string(index=False))
            summary.to_csv(out / f"thresholds_v1_v2_{tag}.tsv", sep="\t", index=False)

    print(f"\nSalidas en: {out}/")
    print("Lectura: (a) si los umbrales 'normalizado' se igualan entre CMS, la asimetria V1 "
          "era de normas de los centroides. (b) Si al normalizar simplemente CAMBIA que "
          "patron domina (verificado con datos sinteticos: pasa de CMS1 a CMS4 con normas "
          "casi iguales), el mecanismo principal es la deriva pre-recaida: en V1 el origen es "
          "una silla y el estado ya se fue a un atractor antes de que empiece el forzamiento; "
          "ese es el defecto que V2 corrige (correccion basal + k), no las normas. En ambos "
          "casos el 'umbral' V1 no mide biologia.")


if __name__ == "__main__":
    main()
