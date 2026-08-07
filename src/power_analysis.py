"""
power_analysis.py

Analisis de poder estadistico para estudios de supervivencia -- lo que
la guia REMARK (REporting recommendations for tumor MARKer prognostic
studies) pide declarar explicitamente y que casi ningun trabajo reporta.

EL PUNTO CENTRAL
----------------
En analisis de supervivencia el poder NO lo determina el numero de
pacientes, sino el numero de EVENTOS. Un estudio con 1000 pacientes y
10 recaidas tiene menos poder que uno con 200 pacientes y 80 recaidas.

Esto importa para interpretar correctamente un resultado no
significativo: p>0.05 con poder del 28% NO es evidencia de ausencia de
efecto, es evidencia de que la muestra no alcanza para saberlo. La
distincion es la diferencia entre "el marcador no sirve" y "el estudio
no puede responder la pregunta".

FORMULA
-------
Schoenfeld (1983) para comparacion de dos grupos en un modelo de Cox:

    d = (z_{1-alfa/2} + z_{1-beta})^2 / (p1 * p2 * ln(HR)^2)

donde d = eventos requeridos, p1/p2 = proporcion en cada grupo. La
formula asume riesgos proporcionales y censura no informativa.

USO
    python3 src/power_analysis.py --input results_external/scored_external_cohort.tsv
    python3 src/power_analysis.py --input a.tsv --input b.tsv --reference CMS2_canonical_WNT
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

EVENTS_PER_VARIABLE_MIN = 10  # convencion habitual para regresion de Cox


def events_needed(hr: float, p1: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Eventos requeridos (Schoenfeld) para detectar un HR con el poder dado."""
    if hr <= 0 or hr == 1.0 or not (0 < p1 < 1):
        return float("nan")
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    return (z_a + z_b) ** 2 / (p1 * (1 - p1) * np.log(hr) ** 2)


def power_achieved(hr: float, p1: float, n_events: float, alpha: float = 0.05) -> float:
    """Poder real alcanzado dado el numero de eventos disponibles."""
    if hr <= 0 or hr == 1.0 or not (0 < p1 < 1) or n_events <= 0:
        return float("nan")
    z_a = norm.ppf(1 - alpha / 2)
    z_b = np.sqrt(n_events * p1 * (1 - p1) * np.log(hr) ** 2) - z_a
    return float(norm.cdf(z_b))


def detectable_hr(p1: float, n_events: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """HR minimo detectable con el numero de eventos disponible."""
    if not (0 < p1 < 1) or n_events <= 0:
        return float("nan")
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    return float(np.exp((z_a + z_b) / np.sqrt(n_events * p1 * (1 - p1))))


def events_table(df: pd.DataFrame, group_col: str, event_col: str) -> pd.DataFrame:
    """Pacientes, eventos y tasa de eventos por grupo -- tabla obligatoria en REMARK."""
    rows = []
    for grp, sub in df.groupby(group_col, sort=False):
        n = len(sub)
        ev = int(sub[event_col].sum())
        rows.append({"grupo": grp, "n": n, "eventos": ev,
                      "censurados": n - ev,
                      "tasa_eventos": ev / n if n else float("nan")})
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out


def analyze(df: pd.DataFrame, group_col: str, event_col: str, reference: str | None,
             n_covariates: int = 1, observed_hr: dict | None = None) -> dict:
    tab = events_table(df, group_col, event_col)
    total_events = int(df[event_col].sum())
    reference = reference or tab.iloc[0]["grupo"]

    print("=" * 74)
    print("EVENTOS POR GRUPO  (el poder depende de estos, no del n de pacientes)")
    print("=" * 74)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nTotal: {len(df)} pacientes, {total_events} eventos "
          f"({100*total_events/len(df):.1f}% de tasa global)")

    epv = total_events / max(n_covariates, 1)
    print(f"\nEventos por variable (EPV): {epv:.1f} con {n_covariates} covariable(s)")
    if epv < EVENTS_PER_VARIABLE_MIN:
        print(f"  AVISO: por debajo del minimo convencional de {EVENTS_PER_VARIABLE_MIN} EPV. "
              "Las estimaciones pueden ser inestables y los intervalos, poco fiables.")
    else:
        print(f"  Por encima del minimo convencional de {EVENTS_PER_VARIABLE_MIN} EPV.")

    print("\n" + "=" * 74)
    print(f"PODER POR COMPARACION  (cada grupo vs. referencia '{reference}')")
    print("=" * 74)

    ref_row = tab[tab["grupo"] == reference]
    if ref_row.empty:
        print(f"AVISO: la referencia '{reference}' no aparece en los datos.")
        return {"events_table": tab, "power_table": pd.DataFrame()}
    n_ref = int(ref_row.iloc[0]["n"])
    ev_ref = int(ref_row.iloc[0]["eventos"])

    prows = []
    for _, r in tab.iterrows():
        if r["grupo"] == reference:
            continue
        n_g = int(r["n"])
        ev_pair = ev_ref + int(r["eventos"])
        p1 = n_g / (n_g + n_ref)
        hr = (observed_hr or {}).get(r["grupo"])
        min_hr = detectable_hr(p1, ev_pair)
        prow = {"grupo": r["grupo"], "n_grupo": n_g, "n_ref": n_ref,
                 "eventos_en_comparacion": ev_pair,
                 "balance_p1": p1,
                 "HR_min_detectable_80pc": min_hr}
        if hr:
            prow["HR_observado"] = hr
            prow["eventos_necesarios"] = events_needed(hr, p1)
            prow["poder_alcanzado"] = power_achieved(hr, p1, ev_pair)
        prows.append(prow)

    ptab = pd.DataFrame(prows)
    print(ptab.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    if "poder_alcanzado" in ptab.columns:
        print("\nINTERPRETACION:")
        for _, r in ptab.iterrows():
            if pd.isna(r.get("poder_alcanzado")):
                continue
            pw = 100 * r["poder_alcanzado"]
            if pw < 50:
                nota = (f"SUBPOTENCIADO: con {pw:.0f}% de poder, un resultado no significativo "
                        "NO es evidencia de ausencia de efecto")
            elif pw < 80:
                nota = f"poder limitado ({pw:.0f}%), por debajo del 80% convencional"
            else:
                nota = f"poder adecuado ({pw:.0f}%)"
            print(f"  {r['grupo']}: {nota}")
            print(f"    harian falta ~{r['eventos_necesarios']:.0f} eventos para 80% "
                  f"(disponibles: {r['eventos_en_comparacion']:.0f})")

    print("\nEl 'HR minimo detectable' es el efecto mas pequeno que este estudio podria "
          "haber detectado con 80% de poder. Efectos menores a ese quedan fuera del "
          "alcance de la muestra, existan o no.")

    return {"events_table": tab, "power_table": ptab, "epv": epv, "total_events": total_events}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True,
                         help="scored_*.tsv (repetir para combinar cohortes)")
    parser.add_argument("--group-col", default="predicted_cms")
    parser.add_argument("--event-col", default="relapse_event")
    parser.add_argument("--duration-col", default="relapse_free_months")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--n-covariates", type=int, default=1,
                         help="Numero de covariables del modelo, para calcular EPV")
    parser.add_argument("--observed-hr", default=None,
                         help="HR observados para calcular poder retrospectivo, formato "
                              "'CMS4_mesenchymal=1.88,CMS1_MSI_immune=1.67'")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    frames = [pd.read_csv(p, sep="\t") for p in args.input]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[args.duration_col, args.event_col, args.group_col])

    observed_hr = None
    if args.observed_hr:
        observed_hr = {k: float(v) for k, v in
                        (pair.split("=") for pair in args.observed_hr.split(","))}

    res = analyze(df, args.group_col, args.event_col, args.reference,
                   args.n_covariates, observed_hr)

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        res["events_table"].to_csv(out / "events_by_group.tsv", sep="\t", index=False)
        if len(res["power_table"]):
            res["power_table"].to_csv(out / "power_by_comparison.tsv", sep="\t", index=False)
        print(f"\nTablas guardadas en: {out}")


if __name__ == "__main__":
    main()
