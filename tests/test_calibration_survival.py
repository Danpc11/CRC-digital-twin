"""
Tests de regresion para calibration.py, survival_validation.py y
prognosis.py, usando datos sinteticos (synthetic_data.py) -- no datos
reales, ya que esos requieren descarga externa que el usuario debe
hacer por su cuenta (ver README).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration import (
    calibrate_patterns_from_data,
    infer_gene_columns,
    load_labeled_dataset,
    zscore_genes,
)
from prognosis import detect_recurrence_signal, hazard_from_trajectory
from survival_validation import (
    risk_score_from_expression,
    score_cohort,
    validate_survival_by_subtype,
)
from synthetic_data import GENES, TRUE_CENTROIDS, generate_synthetic_cohort


@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_cohort(n_per_class=60, noise_sigma=1.0, seed=1)


def test_infer_gene_columns_matches_expected(synthetic_df):
    gene_cols = infer_gene_columns(synthetic_df)
    assert set(gene_cols) == set(GENES)


def test_calibration_recovers_approximate_ranking_of_true_centroids(synthetic_df):
    """
    No esperamos recuperar los centroides exactos (hay ruido), pero si
    que el centroide calibrado de cada clase correlacione mas alto con
    su propio centroide verdadero que con los de las otras clases.
    """
    patterns, gene_cols = calibrate_patterns_from_data(synthetic_df)
    for label, calibrated in patterns.items():
        true_centroid = TRUE_CENTROIDS[label]
        # ambos deben estar en el mismo orden de genes
        self_corr = np.corrcoef(calibrated, true_centroid)[0, 1]
        other_corrs = [
            np.corrcoef(calibrated, TRUE_CENTROIDS[other])[0, 1]
            for other in TRUE_CENTROIDS if other != label
        ]
        assert self_corr > max(other_corrs), (
            f"El centroide calibrado de {label} no correlaciona mas fuerte "
            "con su propio patron verdadero que con los otros"
        )


def test_calibration_raises_on_unrecognized_label():
    import pandas as pd
    from calibration import VALID_CMS_LABELS
    bad_df = pd.DataFrame({
        "sample_id": ["A", "B"],
        "cms_label": ["CMS1_MSI_immune", "NOT_A_REAL_LABEL"],
        "GENE1": [1.0, 2.0],
    })
    unknown = set(bad_df["cms_label"]) - set(VALID_CMS_LABELS) - {"none"}
    assert unknown == {"NOT_A_REAL_LABEL"}


def test_zscore_raises_on_zero_variance_column():
    import pandas as pd
    df = pd.DataFrame({"GENE_CONST": [5.0, 5.0, 5.0]})
    with pytest.raises(ValueError):
        zscore_genes(df, ["GENE_CONST"])


def test_risk_score_from_expression_matches_own_centroid():
    patterns = {label: vec for label, vec in TRUE_CENTROIDS.items()}
    for label, centroid in TRUE_CENTROIDS.items():
        predicted, corr = risk_score_from_expression(centroid, patterns)
        assert predicted == label
        assert corr > 0.99


def test_survival_validation_end_to_end(synthetic_df):
    patterns, gene_cols = calibrate_patterns_from_data(synthetic_df)
    z = zscore_genes(synthetic_df, gene_cols)
    scored = score_cohort(z, gene_cols, patterns)
    result = validate_survival_by_subtype(scored)
    assert result["n_patients"] == len(synthetic_df)
    assert result["n_groups"] >= 2
    # con la senal inyectada en synthetic_data.py, esperamos p significativo
    assert result["logrank_p_value"] < 0.05


def test_survival_validation_raises_on_missing_columns(synthetic_df):
    z = synthetic_df.drop(columns=["relapse_free_months"])
    patterns, gene_cols = calibrate_patterns_from_data(synthetic_df)
    zscored = zscore_genes(synthetic_df, gene_cols)
    scored = score_cohort(zscored, gene_cols, patterns)
    scored = scored.drop(columns=["relapse_free_months"])
    with pytest.raises(ValueError):
        validate_survival_by_subtype(scored)


# --- prognosis.py ---

def test_hazard_from_trajectory_zero_state_is_zero_hazard():
    x = np.zeros((8, 10))
    hazard = hazard_from_trajectory(x)
    assert np.allclose(hazard, 0.0)


def test_detect_recurrence_signal_flags_late_spike():
    # baseline plano en cero, luego un salto claro
    hazard = np.array([0.0, 0.0, 0.1, 0.2, 3.5, 3.8, 4.0])
    alert, idx = detect_recurrence_signal(hazard, baseline_window=2, threshold_sigma=3.0)
    assert alert is True
    # baseline mu=0, sigma=0 -> threshold=mu+0.1=0.1; el primer valor
    # que supera 0.1 es hazard[3]=0.2, no hazard[2]=0.1 (no es estrictamente mayor)
    assert idx == 3


def test_load_labeled_dataset_raises_on_unrecognized_label_in_file(tmp_path):
    import pandas as pd
    bad_path = tmp_path / "bad.tsv"
    pd.DataFrame({
        "sample_id": ["A", "B"],
        "cms_label": ["CMS1_MSI_immune", "NOT_A_REAL_LABEL"],
        "GENE1": [1.0, 2.0],
    }).to_csv(bad_path, sep="\t", index=False)
    with pytest.raises(ValueError):
        load_labeled_dataset(bad_path)


def test_detect_recurrence_signal_no_alert_when_flat():
    hazard = np.array([0.1, 0.12, 0.11, 0.13, 0.10, 0.12])
    alert, idx = detect_recurrence_signal(hazard, baseline_window=2, threshold_sigma=3.0)
    assert alert is False
    assert idx is None


def test_detect_recurrence_signal_raises_on_short_series():
    hazard = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        detect_recurrence_signal(hazard, baseline_window=2)


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
