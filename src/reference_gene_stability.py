"""
reference_gene_stability.py

Preseleccion in silico de genes de referencia (housekeeping) para el ensayo
RT-qPCR del panel, al estilo geNorm / NormFinder, usando la expresion que ya
tenemos: GSE39582 (Affymetrix U133 Plus 2.0, log2 RMA) y TCGA-CRC (RNA-seq,
log2 RSEM). Un gen de referencia sirve si (a) varia poco entre pacientes,
(b) NO cambia con el subtipo CMS ni con el estadio -- si lo hiciera, la
normalizacion por Delta-Ct borraria parte de la senal que el panel quiere
medir -- y (c) se expresa lo bastante para un Ct estable.

Candidatos por defecto: los 5 genes de referencia de Oncotype DX Colon
(Clark-Langone et al. 2010, BMC Cancer 10:691: ATP5E, GPX1, PGK1, UBB, VDAC2)
mas housekeeping clasicos (ACTB, GAPDH, B2M, HPRT1, RPLP0, TBP). Son genes
de publicacion abierta; la eleccion se justifica empiricamente con datos
propios, no hay problema de propiedad intelectual.

Metricas por gen y cohorte:
  - mean_log2, sd_log2: nivel y dispersion global
  - eta2_CMS / F_CMS / maxdelta_CMS_log2: ANOVA una via a traves de CMS
    (fraccion de varianza explicada por el subtipo; delta maximo entre medias)
  - eta2_stage: idem a traves del estadio armonizado (si la cohorte lo trae)
  - geNorm_M: media de SD(log-ratio con cada otro candidato) (Vandesompele 2002);
    bajo = estable. OJO: genes co-regulados (p. ej. muy abundantes) se
    "protegen" entre si en esta metrica; por eso no se usa sola.
  - NormFinder_rho: version simplificada del modelo de Andersen 2004 sobre
    expresion centrada por muestra: sqrt(var entre grupos CMS + media de var intra)
Ranking por cohorte = promedio de rangos (sd, eta2_CMS, geNorm_M, NormFinder_rho);
ranking combinado = promedio entre cohortes. Se reporta la concordancia de
rangos entre plataformas (Spearman) y la eliminacion por pasos de geNorm.

Resultado 2026-09-11 (ver PROJECT_STATUS.md): recomendados UBB, RPLP0, TBP,
ACTB (+HPRT1); excluir ATP5E (eta2_CMS 0.26-0.28), GAPDH (metabolico,
eta2_CMS 0.14-0.17) y B2M (MHC-I, sube con CMS1). Es preseleccion: la
eleccion final requiere confirmacion en FFPE por RT-qPCR (geNorm/NormFinder
sobre Ct reales; MIQE).

USO
    python3 src/reference_gene_stability.py \\
        --gse-expression data/raw_geo/gse39582_expression_probes.tsv \\
        --gse-annotation data/raw_geo/GPL570.txt \\
        --gse-labels data/gse39582_cms_labeled.tsv \\
        --tcga-expression data/raw_synapse/tcga_rnaseq/TCGACRC_expression-merged.tsv \\
        --tcga-labels data/tcga_rnaseq_cms_labeled.tsv \\
        --output results_reference_genes/
    Cualquiera de las dos cohortes puede omitirse. --candidates cambia la lista.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gse39582_dataset import parse_platform_annotation
from clinical_covariates import harmonize_stage

ONCOTYPE_REFERENCE = ["ATP5E", "GPX1", "PGK1", "UBB", "VDAC2"]
CLASSIC_HOUSEKEEPING = ["ACTB", "GAPDH", "B2M", "HPRT1", "RPLP0", "TBP"]
CMS_LEVELS = ["CMS1_MSI_immune", "CMS2_canonical_WNT", "CMS3_metabolic", "CMS4_mesenchymal"]
RANK_METRICS = ["sd_log2", "eta2_CMS", "geNorm_M", "NormFinder_rho"]


# ----------------------------------------------------------------- metricas
def anova_eta2(x: pd.Series, groups: pd.Series) -> tuple[float, float, float]:
    """(F, eta2, delta maximo entre medias de grupo). NaN si hay <2 grupos."""
    df = pd.DataFrame({"x": x, "g": groups}).dropna()
    if df["g"].nunique() < 2 or len(df) < 3:
        return np.nan, np.nan, np.nan
    parts = [v["x"].to_numpy() for _, v in df.groupby("g")]
    F, _ = stats.f_oneway(*parts)
    grand = df["x"].mean()
    ss_between = sum(len(v) * (v.mean() - grand) ** 2 for v in parts)
    ss_total = ((df["x"] - grand) ** 2).sum()
    means = df.groupby("g")["x"].mean()
    eta2 = float(ss_between / ss_total) if ss_total > 0 else np.nan
    return float(F), eta2, float(means.max() - means.min())


def genorm_M(X: pd.DataFrame) -> pd.Series:
    """M de geNorm: para cada gen, media de SD(log-ratio con cada otro gen)."""
    if X.shape[1] < 2:
        raise ValueError("geNorm necesita al menos 2 genes")
    M = {j: float(np.mean([np.std(X[j] - X[k], ddof=1) for k in X.columns if k != j]))
         for j in X.columns}
    return pd.Series(M, name="geNorm_M")


def genorm_stepwise(X: pd.DataFrame, keep: int = 3) -> tuple[list[str], pd.Series]:
    """Elimina el gen con mayor M y recalcula, hasta quedar `keep` genes.
    Devuelve (orden de eliminacion, M de los finalistas)."""
    cur, order = X.copy(), []
    while cur.shape[1] > keep:
        M = genorm_M(cur)
        worst = M.idxmax()
        order.append(worst)
        cur = cur.drop(columns=[worst])
    return order, genorm_M(cur)


def normfinder_rho(X: pd.DataFrame, groups: pd.Series) -> pd.Series:
    """Estabilidad tipo NormFinder (simplificada): se centra cada muestra por la
    media de los candidatos (quita el efecto de carga), y por gen se combina la
    varianza de las medias entre grupos con la media de las varianzas intra."""
    df = X.sub(X.mean(axis=1), axis=0)
    g = groups.reindex(df.index)
    df, g = df[g.notna()], g[g.notna()]
    if g.nunique() < 2:
        return pd.Series(np.nan, index=X.columns, name="NormFinder_rho")
    rho = {}
    for j in df.columns:
        means = df[j].groupby(g).mean()
        within = df[j].groupby(g).var(ddof=1)
        rho[j] = float(np.sqrt(means.var(ddof=0) + within.mean()))
    return pd.Series(rho, name="NormFinder_rho")


def evaluate_cohort(X: pd.DataFrame, cms: pd.Series, stage: pd.Series | None, cohort: str,
                    source: dict[str, str]) -> pd.DataFrame:
    """Tabla de metricas y rangos para una cohorte. X = muestras x genes (log2)."""
    cms = cms.reindex(X.index).where(lambda s: s.isin(CMS_LEVELS))
    rows = []
    for gname in X.columns:
        F_c, eta_c, d_c = anova_eta2(X[gname], cms)
        F_s, eta_s, d_s = anova_eta2(X[gname], stage.reindex(X.index)) if stage is not None else (np.nan,) * 3
        rows.append({"gene": gname, "cohort": cohort, "source": source.get(gname, "otro"),
                     "mean_log2": float(X[gname].mean()), "sd_log2": float(X[gname].std(ddof=1)),
                     "F_CMS": F_c, "eta2_CMS": eta_c, "maxdelta_CMS_log2": d_c,
                     "F_stage": F_s, "eta2_stage": eta_s, "maxdelta_stage_log2": d_s})
    res = pd.DataFrame(rows).set_index("gene").join(genorm_M(X)).join(normfinder_rho(X, cms))
    for m in RANK_METRICS:
        res[f"rank_{m}"] = res[m].rank()
    res["rank_mean_cohort"] = res[[f"rank_{m}" for m in RANK_METRICS]].mean(axis=1)
    return res


def combine_rankings(per_cohort: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, float | None]:
    """Promedia rank_mean_cohort entre cohortes; Spearman entre las dos primeras si hay 2."""
    ranks = pd.DataFrame({c: r["rank_mean_cohort"] for c, r in per_cohort.items()})
    comb = ranks.add_prefix("rank_")
    comb.insert(0, "source", next(iter(per_cohort.values()))["source"])
    comb["rank_combined"] = ranks.mean(axis=1)
    rho = None
    if ranks.shape[1] == 2:
        rho = float(stats.spearmanr(ranks.iloc[:, 0], ranks.iloc[:, 1]).statistic)
    return comb.sort_values("rank_combined"), rho


# ----------------------------------------------------------------- carga
def select_expressed_probes(probe_means: pd.Series, max_gap_log2: float = 2.0) -> list[str]:
    """Probes 'vivos' de un gen: los que estan a menos de `max_gap_log2` del probe
    mas expresado. Promediar un probe de fondo (p. ej. HPRT1 1565446_at, media
    2.4 vs 9.7 del probe real) reduce la SD del gen a la mitad y lo hace ver
    artificialmente estable -- hallazgo de la revision de codigo 2026-09-11."""
    top = probe_means.max()
    return probe_means[probe_means >= top - max_gap_log2].index.tolist()


def load_affy_gene_matrix(expr_path: Path, annot_path: Path, candidates: list[str],
                          max_gap_log2: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Promedia los probes EXPRESADOS de cada candidato (ver select_expressed_probes),
    excluyendo los controles AFFX. Devuelve (muestras x genes, tabla por probe con
    la marca de si entro al promedio)."""
    annot = parse_platform_annotation(annot_path)
    sym = annot.set_index("ID")["Gene Symbol"].astype(str)
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    gene, per_probe = {}, []
    for g in candidates:
        probes = [p for p in sym[sym == g].index if p in expr.index and not str(p).startswith("AFFX")]
        if not probes:
            print(f"AVISO: {g} sin probes utilizables en la anotacion; se omite")
            continue
        means = expr.loc[probes].mean(axis=1)
        used = select_expressed_probes(means, max_gap_log2)
        dropped = sorted(set(probes) - set(used))
        if dropped:
            print(f"  {g}: {len(dropped)} probe(s) de fondo excluidos del promedio "
                  f"({', '.join(f'{p}={means[p]:.1f}' for p in dropped)}; max={means.max():.1f})")
        gene[g] = expr.loc[used].mean(axis=0)
        for p in probes:
            per_probe.append({"gene": g, "probe": p, "mean_log2": float(means[p]),
                              "sd_log2": float(expr.loc[p].std(ddof=1)), "usado": p in used})
    return pd.DataFrame(gene), pd.DataFrame(per_probe)


def load_gene_rows_matrix(expr_path: Path, candidates: list[str]) -> pd.DataFrame:
    """Matriz genes x muestras (TCGA) -> muestras x candidatos presentes."""
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    present = [g for g in candidates if g in expr.index]
    missing = sorted(set(candidates) - set(present))
    if missing:
        print(f"AVISO: no estan en la matriz: {missing}")
    return expr.loc[present].T


def load_labels(path: Path, stage_col: str | None) -> tuple[pd.Series, pd.Series | None]:
    lab = pd.read_csv(path, sep="\t").set_index("sample_id")
    cms = lab["cms_label"]
    stage = harmonize_stage(lab[stage_col], verbose=False) if stage_col and stage_col in lab.columns else None
    return cms, stage


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gse-expression", help="Matriz probes x muestras (Affymetrix), log2")
    parser.add_argument("--gse-annotation", help="GPL570.txt (registro propio de la plataforma)")
    parser.add_argument("--gse-labels", help="TSV con sample_id, cms_label y estadio (build_gse39582_dataset.py)")
    parser.add_argument("--gse-stage-col", default="stage")
    parser.add_argument("--probe-max-gap", type=float, default=2.0,
                        help="Probes a mas de esta distancia (log2) del probe mas expresado del gen "
                             "se consideran de fondo y no entran al promedio")
    parser.add_argument("--tcga-expression", help="Matriz genes x muestras (RNA-seq), log2")
    parser.add_argument("--tcga-labels", help="TSV con sample_id y cms_label")
    parser.add_argument("--candidates", nargs="*", default=None,
                        help="Lista de genes; por defecto Oncotype (5) + clasicos (6)")
    parser.add_argument("--output", default="results_reference_genes")
    args = parser.parse_args()

    candidates = args.candidates or ONCOTYPE_REFERENCE + CLASSIC_HOUSEKEEPING
    source = {g: "Oncotype" for g in ONCOTYPE_REFERENCE} | {g: "clasico" for g in CLASSIC_HOUSEKEEPING}
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    cols = ["source", "mean_log2", "sd_log2", "eta2_CMS", "maxdelta_CMS_log2", "eta2_stage",
            "geNorm_M", "NormFinder_rho", "rank_mean_cohort"]

    per_cohort: dict[str, pd.DataFrame] = {}
    matrices: dict[str, pd.DataFrame] = {}

    if args.gse_expression:
        if not (args.gse_annotation and args.gse_labels):
            raise ValueError("--gse-expression requiere --gse-annotation y --gse-labels")
        X, probes = load_affy_gene_matrix(Path(args.gse_expression), Path(args.gse_annotation),
                                          candidates, max_gap_log2=args.probe_max_gap)
        cms, stage = load_labels(Path(args.gse_labels), args.gse_stage_col)
        print(f"GSE39582: {X.shape[0]} muestras, {X.shape[1]} genes; con CMS oficial: "
              f"{cms.reindex(X.index).isin(CMS_LEVELS).sum()}; con estadio: "
              f"{stage.reindex(X.index).notna().sum() if stage is not None else 0}")
        res = evaluate_cohort(X, cms, stage, "GSE39582", source)
        per_cohort["GSE39582"], matrices["GSE39582"] = res, X
        probes.to_csv(out_dir / "gse39582_per_probe.tsv", sep="\t", index=False)
        res.to_csv(out_dir / "stability_GSE39582.tsv", sep="\t")
        print("\n=== GSE39582 (Affymetrix, log2 RMA) ===")
        print(res[cols].sort_values("rank_mean_cohort").round(3).to_string())

    if args.tcga_expression:
        if not args.tcga_labels:
            raise ValueError("--tcga-expression requiere --tcga-labels")
        X = load_gene_rows_matrix(Path(args.tcga_expression), candidates)
        cms, _ = load_labels(Path(args.tcga_labels), None)
        print(f"\nTCGA: {X.shape[0]} muestras, {X.shape[1]} genes; con CMS: "
              f"{cms.reindex(X.index).isin(CMS_LEVELS).sum()} (sin estadio)")
        res = evaluate_cohort(X, cms, None, "TCGA", source)
        per_cohort["TCGA"], matrices["TCGA"] = res, X
        res.to_csv(out_dir / "stability_TCGA.tsv", sep="\t")
        print("\n=== TCGA (RNA-seq, log2 RSEM) ===")
        print(res[cols].sort_values("rank_mean_cohort").round(3).to_string())

    if not per_cohort:
        raise ValueError("Hay que dar al menos una cohorte (--gse-expression o --tcga-expression)")

    comb, rho = combine_rankings(per_cohort)
    print("\n=== RANKING COMBINADO (menor = mas estable) ===")
    print(comb.round(3).to_string())
    if rho is not None:
        print(f"Concordancia de rangos entre plataformas: Spearman rho={rho:.2f} "
              "(baja = el orden fino depende de la plataforma; mirar el conjunto, no la posicion)")
    comb.to_csv(out_dir / "stability_combined_ranking.tsv", sep="\t")

    print("\n=== geNorm por pasos: orden de eliminacion -> finalistas ===")
    for name, X in matrices.items():
        order, finals = genorm_stepwise(X, keep=3)
        print(f"{name}: {' > '.join(order)} -> {finals.round(3).to_dict()}")

    print(f"\nSalidas en {out_dir}/")
    print("Preseleccion in silico: confirmar en FFPE por RT-qPCR (geNorm/NormFinder sobre Ct).")


if __name__ == "__main__":
    main()
