"""Pruebas del valor incremental de CMS sobre covariables clínicas."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pooled_cox_validation import (
    bootstrap_cindex_increment,
    build_cox_frame,
    nested_model_increment,
)


def _survival_data(seed=12):
    rng = np.random.default_rng(seed)
    rows = []
    labels = ["CMS1_MSI_immune", "CMS2_canonical_WNT", "CMS3_metabolic", "CMS4_mesenchymal"]
    effects = dict(zip(labels, [1.5, 1.0, 1.1, 2.2]))
    for cohort in ["A", "B"]:
        for _ in range(120):
            label = rng.choice(labels)
            stage = int(rng.integers(1, 4))
            rate = effects[label] * (1.5 ** (stage - 1)) / 40
            event_time = rng.exponential(1 / rate)
            censor_time = rng.exponential(55)
            rows.append({
                "cohort": cohort, "predicted_cms": label,
                "stage_harmonized": stage,
                "relapse_free_months": min(event_time, censor_time),
                "relapse_event": int(event_time <= censor_time),
            })
    return pd.DataFrame(rows)


def test_nested_model_reports_joint_cms_increment():
    data = _survival_data()
    reference = "CMS2_canonical_WNT"
    reduced_df = build_cox_frame(
        data, "relapse_free_months", "relapse_event", reference,
        ["stage_harmonized"], include_cms=False)
    full_df = build_cox_frame(
        data, "relapse_free_months", "relapse_event", reference,
        ["stage_harmonized"], include_cms=True)
    reduced = CoxPHFitter().fit(reduced_df, "duration", "event", strata=["cohort"])
    full = CoxPHFitter().fit(full_df, "duration", "event", strata=["cohort"])
    result = nested_model_increment(reduced, full)

    assert result["df"] == 3
    assert 0 <= result["p_incremental"] <= 1
    assert "delta_c_index" in result


def test_bootstrap_cindex_increment_returns_requested_rows():
    data = _survival_data(seed=13)
    result = bootstrap_cindex_increment(
        data, "relapse_free_months", "relapse_event", "CMS2_canonical_WNT",
        iterations=3, seed=4)
    assert len(result) == 3
    assert {"iteration", "delta_c_index"}.issubset(result.columns)
