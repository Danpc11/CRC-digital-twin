"""
concordance_analysis.py

Matriz de concordancia entre predicted_cms (reclasificacion del panel
reducido) y cms_label (etiqueta oficial del consorcio), a partir de
scored_cohort.tsv generado por run_pipeline.py.

Util para diagnosticar POR QUE las curvas KM de un subtipo divergen
entre el modelo y la etiqueta oficial (ver conversacion: CMS3 diverge
notablemente en GSE39582) -- si un subtipo tiene baja concordancia,
significa que el panel reducido esta agrupando pacientes distintos
bajo esa etiqueta, lo cual explica directamente por que su curva de
supervivencia se ve distinta.

USO:
    python3 src/concordance_analysis.py --input results_gse39582/scored_cohort.tsv
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    df = df[df["cms_label"] != "none"]

    print(f"n = {len(df)} pacientes con etiqueta oficial (excluyendo 'none')\n")

    # Matriz de confusion: filas = etiqueta oficial, columnas = prediccion del modelo
    confusion = pd.crosstab(df["cms_label"], df["predicted_cms"], margins=True, margins_name="Total")
    print("Matriz de concordancia (filas=oficial, columnas=modelo):")
    print(confusion)

    print("\nConcordancia por subtipo (% de pacientes con etiqueta oficial X que el modelo tambien clasifico como X):")
    for label in sorted(df["cms_label"].unique()):
        subset = df[df["cms_label"] == label]
        match_rate = (subset["predicted_cms"] == label).mean()
        print(f"  {label}: {match_rate:.1%} ({(subset['predicted_cms'] == label).sum()}/{len(subset)})")

    overall_agreement = (df["cms_label"] == df["predicted_cms"]).mean()
    print(f"\nConcordancia global (accuracy simple): {overall_agreement:.1%}")

    # Cohen's kappa -- concordancia corregida por azar, mas informativa
    # que accuracy simple cuando las clases estan desbalanceadas
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(df["cms_label"], df["predicted_cms"])
        print(f"Cohen's kappa: {kappa:.3f} "
              f"({'pobre' if kappa < 0.2 else 'aceptable' if kappa < 0.4 else 'moderada' if kappa < 0.6 else 'buena' if kappa < 0.8 else 'muy buena'})")
    except ImportError:
        print("(instala scikit-learn para ver Cohen's kappa: pip install scikit-learn)")


if __name__ == "__main__":
    main()
