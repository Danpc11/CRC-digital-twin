"""
feature_selection.py

Seleccion de genes DATA-DRIVEN para el panel CMS, en vez de elegir
genes por hipotesis biologica uno a la vez (que fue lo que hicimos con
CPS1/SI -- funciono -- y con CKLF -- no funciono, y no hay forma de
saber de antemano cual va a funcionar sin probarlo en wet-lab).

Usa el TRANSCRIPTOMA COMPLETO de GSE39582 (no solo el panel actual de
9 genes) y tres metodos complementarios:

    1. AUC uno-contra-el-resto por gen y por subtipo (vectorizado via
       rangos) -- que tan bien separa cada gen "este subtipo" del
       resto. Es la metrica mas directamente ligada a la logica del
       clasificador actual (correlacion con un patron por subtipo).
    2. ANOVA F-test multiclase -- poder discriminante general entre
       los 4 subtipos (filtro univariado clasico de expresion
       diferencial).
    3. Random Forest (importancia multivariada) -- captura genes que
       son debiles solos pero valiosos en combinacion, algo que los
       metodos univariados (1) y (2) no pueden ver.

IMPORTANTE: el resultado es una lista de CANDIDATOS a verificar contra
literatura y factibilidad de RT-qPCR, no una lista para copiar y pegar
directo al panel -- un gen con AUC alto puede no tener rol biologico
conocido, ser dificil de amplificar por PCR, o ser redundante con
genes ya en el panel. Sigue haciendo falta criterio humano en la
seleccion final.

USO:
    python3 src/feature_selection.py \\
        --expr data/raw_geo/gse39582_expression_probes.tsv \\
        --annotation data/raw_geo/GPL570.txt \\
        --labels data/raw_synapse/tcga_cms_labels/cms_labels_public_all.txt \\
        --dataset gse39582 \\
        --output results_feature_selection/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))

CMS_RENAME = {
    "CMS1": "CMS1_MSI_immune",
    "CMS2": "CMS2_canonical_WNT",
    "CMS3": "CMS3_metabolic",
    "CMS4": "CMS4_mesenchymal",
    "NOLBL": "none",
}
CMS_LABEL_COLUMN = "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples"

# Genes ya en el panel actual (para marcar en el reporte que candidatos son "nuevos")
CURRENT_PANEL = {"MLH1", "GNLY", "USP18", "MYC", "AXIN2", "FABP1", "CPS1", "SI", "VIM", "TGFB1"}


def parse_platform_annotation(path) -> pd.DataFrame:
    table_lines = []
    header = None
    in_table = False
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                in_table = False
                continue
            if in_table:
                if header is None:
                    header = line.split("\t")
                else:
                    table_lines.append(line)
    if header is None:
        raise ValueError(f"No se encontro tabla de plataforma en {path}")
    from io import StringIO
    return pd.read_csv(StringIO("\n".join(table_lines)), sep="\t", names=header)


def build_full_gene_matrix(expr_path: Path, annot_path: Path) -> pd.DataFrame:
    """
    Mapea TODOS los probes a simbolos de gen (no solo el panel actual),
    promediando probes duplicados por gen. Devuelve matriz
    (muestras x genes).
    """
    print("Cargando anotacion de plataforma...")
    annot = parse_platform_annotation(annot_path)
    symbol_col = [c for c in annot.columns if "symbol" in c.lower()][0]
    id_col = "ID" if "ID" in annot.columns else annot.columns[0]
    probe_to_symbol = annot.set_index(id_col)[symbol_col].dropna()
    probe_to_symbol = probe_to_symbol[probe_to_symbol.astype(str).str.strip() != ""]

    print("Cargando expresion (probes)...")
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)

    print(f"Mapeando {len(expr)} probes a simbolos de gen (promediando duplicados)...")
    valid_probes = expr.index.intersection(probe_to_symbol.index)
    expr = expr.loc[valid_probes]
    gene_labels = probe_to_symbol.loc[valid_probes]

    gene_matrix = expr.groupby(gene_labels).mean()  # (n_genes, n_samples)
    gene_matrix = gene_matrix.T  # (n_samples, n_genes)
    print(f"Matriz final: {gene_matrix.shape[0]} muestras x {gene_matrix.shape[1]} genes unicos")
    return gene_matrix


def one_vs_rest_auc(expr: np.ndarray, labels: np.ndarray, classes: list) -> pd.DataFrame:
    """
    AUC uno-contra-el-resto por gen y por clase, vectorizado via
    rangos (equivalente al test de Mann-Whitney U normalizado):
        AUC = (rank_sum_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

    expr: (n_samples, n_genes), ya deberia estar en escala comparable
    (no hace falta z-score para AUC, es invariante a transformaciones
    monotonas, pero no afecta si ya viene z-scoreado).
    """
    ranks = rankdata(expr, axis=0)
    results = {}
    for c in classes:
        pos_mask = labels == c
        n_pos, n_neg = pos_mask.sum(), (~pos_mask).sum()
        rank_sum_pos = ranks[pos_mask].sum(axis=0)
        auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        results[c] = auc
    return pd.DataFrame(results, index=None)


def multiclass_anova(expr: np.ndarray, labels: np.ndarray, classes: list) -> np.ndarray:
    """F-statistic de ANOVA de una via, por gen, entre las clases."""
    groups = [expr[labels == c] for c in classes]
    f_stats = np.zeros(expr.shape[1])
    # f_oneway vectoriza sobre columnas si se le pasan arrays 2D por grupo
    f, p = f_oneway(*groups)
    return f


def pairwise_auc(expr: np.ndarray, labels: np.ndarray, class_a: str, class_b: str) -> pd.Series:
    """
    AUC especifico para separar class_a de class_b (no de "el resto"),
    solo usando las muestras de esas dos clases. Distinto de
    one_vs_rest_auc: un gen puede tener AUC mediocre uno-contra-el-resto
    pero ser excelente separando especificamente dos clases que se
    confunden entre si (que es justo el problema que se quiere resolver
    aqui: CMS3 vs CMS1 y CMS4 vs CMS2 se confunden en la cohorte
    externa, no "CMS3 vs todo lo demas").
    """
    mask = (labels == class_a) | (labels == class_b)
    sub_expr = expr[mask]
    sub_labels = labels[mask]

    ranks = rankdata(sub_expr, axis=0)
    pos_mask = sub_labels == class_a
    n_pos, n_neg = pos_mask.sum(), (~pos_mask).sum()
    rank_sum_pos = ranks[pos_mask].sum(axis=0)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return pd.Series(auc, name=f"auc_{class_a}_vs_{class_b}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", required=True, help="TSV de expresion (probes x muestras)")
    parser.add_argument("--annotation", required=True, help="GPL570.txt (anotacion de plataforma)")
    parser.add_argument("--labels", required=True, help="cms_labels_public_all.txt")
    parser.add_argument("--dataset", default="gse39582", help="Valor de la columna 'dataset' a filtrar")
    parser.add_argument("--output", default="results_feature_selection")
    parser.add_argument("--top-n-report", type=int, default=15, help="Cuantos candidatos mostrar por subtipo")
    parser.add_argument("--rf-candidate-pool", type=int, default=300,
                         help="Cuantos genes (por AUC maximo) pasan al filtro de Random Forest")
    parser.add_argument("--pair", action="append", default=[],
                         help="Par de clases a analizar especificamente, formato 'CMS3_metabolic,CMS1_MSI_immune'. "
                              "Puede pasarse varias veces para varios pares.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_matrix = build_full_gene_matrix(Path(args.expr), Path(args.annotation))

    print("Cargando etiquetas CMS...")
    labels_df = pd.read_csv(args.labels, sep="\t")
    subset = labels_df[labels_df["dataset"] == args.dataset].set_index("sample")
    subset["cms_label"] = subset[CMS_LABEL_COLUMN].replace(CMS_RENAME)
    subset = subset[subset["cms_label"] != "none"]

    common = gene_matrix.index.intersection(subset.index)
    print(f"Muestras con expresion + CMS: {len(common)}")
    gene_matrix = gene_matrix.loc[common]
    cms_labels = subset.loc[common, "cms_label"].to_numpy()
    classes = sorted(set(cms_labels))

    # z-score por gen (consistente con el resto del pipeline, aunque
    # AUC no lo requiere estrictamente)
    print("Normalizando (z-score por gen)...")
    gene_matrix = (gene_matrix - gene_matrix.mean()) / gene_matrix.std().replace(0, np.nan)
    gene_matrix = gene_matrix.dropna(axis=1)  # elimina genes con varianza cero
    expr_arr = gene_matrix.to_numpy()
    gene_names = gene_matrix.columns.to_numpy()

    print(f"\n--- Metodo 1: AUC uno-contra-el-resto ({len(gene_names)} genes) ---")
    auc_df = one_vs_rest_auc(expr_arr, cms_labels, classes)
    auc_df.index = gene_names

    print(f"--- Metodo 2: ANOVA F-test multiclase ---")
    f_stats = multiclass_anova(expr_arr, cms_labels, classes)
    anova_series = pd.Series(f_stats, index=gene_names, name="anova_F")

    print(f"--- Metodo 3: Random Forest (importancia, sobre top {args.rf_candidate_pool} por AUC) ---")
    from sklearn.ensemble import RandomForestClassifier
    max_auc_per_gene = auc_df.max(axis=1)
    top_pool = max_auc_per_gene.sort_values(ascending=False).head(args.rf_candidate_pool).index
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(gene_matrix[top_pool].to_numpy(), cms_labels)
    rf_importance = pd.Series(rf.feature_importances_, index=top_pool, name="rf_importance")

    # Combinar todo en un reporte
    report = auc_df.copy()
    report["anova_F"] = anova_series
    report["rf_importance"] = rf_importance
    report["rf_importance"] = report["rf_importance"].fillna(0.0)
    report["in_current_panel"] = report.index.isin(CURRENT_PANEL)
    report = report.sort_values("anova_F", ascending=False)
    report.to_csv(out_dir / "gene_ranking_full.tsv", sep="\t")
    print(f"\nRanking completo guardado en: {out_dir / 'gene_ranking_full.tsv'}")

    print(f"\n{'='*70}\nTOP {args.top_n_report} CANDIDATOS POR SUBTIPO (por AUC uno-contra-el-resto)\n{'='*70}")
    for c in classes:
        top = auc_df[c].sort_values(ascending=False).head(args.top_n_report)
        print(f"\n{c}:")
        for gene, auc in top.items():
            marker = " [YA EN PANEL]" if gene in CURRENT_PANEL else " [NUEVO]"
            rf_score = rf_importance.get(gene, 0.0)
            print(f"  {gene:12s} AUC={auc:.3f}  RF_importance={rf_score:.4f}{marker}")

    if args.pair:
        print(f"\n{'='*70}\nANALISIS PAREADO (pares de clases especificos)\n{'='*70}")
        pair_results = {}
        for pair_str in args.pair:
            class_a, class_b = [p.strip() for p in pair_str.split(",")]
            if class_a not in classes or class_b not in classes:
                print(f"AVISO: '{class_a}' o '{class_b}' no estan entre las clases disponibles ({classes}), se omite.")
                continue
            pair_auc = pairwise_auc(expr_arr, cms_labels, class_a, class_b)
            pair_auc.index = gene_names
            pair_results[f"{class_a}_vs_{class_b}"] = pair_auc

            print(f"\n--- {class_a} vs {class_b} (AUC alto = mas expresado en {class_a}, bajo = mas en {class_b}) ---")
            # candidatos que separan bien esta pareja especificamente
            # (AUC lejos de 0.5 en cualquier direccion), ordenado por
            # distancia a 0.5 -- no por AUC uno-contra-el-resto, que es
            # una pregunta distinta
            distance = (pair_auc - 0.5).abs().sort_values(ascending=False)
            top_pair = distance.head(args.top_n_report)
            for gene in top_pair.index:
                auc_val = pair_auc[gene]
                direction = f"mas alto en {class_a}" if auc_val > 0.5 else f"mas alto en {class_b}"
                marker = " [YA EN PANEL]" if gene in CURRENT_PANEL else " [NUEVO]"
                print(f"  {gene:12s} AUC={auc_val:.3f} ({direction}){marker}")

        if pair_results:
            pair_df = pd.DataFrame(pair_results)
            pair_df.to_csv(out_dir / "pairwise_gene_ranking.tsv", sep="\t")
            print(f"\nRanking pareado completo guardado en: {out_dir / 'pairwise_gene_ranking.tsv'}")


if __name__ == "__main__":
    main()
