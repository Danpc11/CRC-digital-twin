def test_clinical_covariates_not_mistaken_for_genes():
    """
    Regresion de un bug real: 'stage' (estadio clinico numerico) se
    detectaba como gen, se z-scoreaba y entraba a la calibracion como
    un rasgo mas del panel -- sin ningun error visible, solo p-valores
    que cambiaban sin explicacion.
    """
    import pandas as pd
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from calibration import infer_gene_columns

    df = pd.DataFrame({
        "sample_id": ["A", "B", "C"],
        "cms_label": ["CMS1_MSI_immune"] * 3,
        "GENE1": [1.0, 2.0, 3.0],
        "GENE2": [4.0, 5.0, 6.0],
        "stage": [1, 2, 3],                    # numerico: el caso que fallaba
        "age": [65, 72, 58],
        "relapse_free_months": [10.0, 20.0, 30.0],
        "relapse_event": [0, 1, 0],
        "predicted_cms": ["CMS1_MSI_immune"] * 3,
        "classification_confidence": [0.8, 0.7, 0.9],
    })
    detected = infer_gene_columns(df)
    assert detected == ["GENE1", "GENE2"], f"columnas mal detectadas: {detected}"
    for leaked in ("stage", "age", "predicted_cms", "classification_confidence"):
        assert leaked not in detected, f"'{leaked}' no debe tratarse como gen"
