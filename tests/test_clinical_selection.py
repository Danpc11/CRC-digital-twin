import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_selection import (
    NOT_ELIGIBLE, POTENTIALLY_ELIGIBLE, REQUIRES_TEST,
    assess_molecular_eligibility, cms_continuous_profile,
    hybrid_mechanism_hypotheses,
)


PATTERNS = {
    "CMS1_MSI_immune": np.array([1.0, 0.0, 0.0, -1.0]),
    "CMS2_canonical_WNT": np.array([0.0, 1.0, -1.0, 0.0]),
    "CMS3_metabolic": np.array([-1.0, 0.0, 0.0, 1.0]),
    "CMS4_mesenchymal": np.array([0.0, -1.0, 1.0, 0.0]),
}


def _by_mechanism(results):
    return {r["mechanism"]: r for r in results}


def test_continuous_profile_detects_clear_dominant_state():
    result = cms_continuous_profile(PATTERNS["CMS1_MSI_immune"], PATTERNS)
    assert result["primary"] == "CMS1_MSI_immune"
    assert result["interpretation"] == "dominante"
    assert abs(sum(result["scores"].values()) - 1.0) < 1e-12


def test_continuous_profile_can_report_hybrid():
    x = 0.5 * PATTERNS["CMS1_MSI_immune"] + 0.5 * PATTERNS["CMS2_canonical_WNT"]
    result = cms_continuous_profile(x, PATTERNS)
    assert result["interpretation"] == "hibrido"
    assert {result["primary"], result["secondary"]} == {
        "CMS1_MSI_immune", "CMS2_canonical_WNT"}


def test_cms_signal_never_substitutes_missing_msi_test():
    results = _by_mechanism(assess_molecular_eligibility(
        {"msi_mmr": "unknown"}, {"metastatic": True}))
    assert results["immune_checkpoint_inhibition"]["status"] == REQUIRES_TEST


def test_mss_blocks_checkpoint_even_if_cms1_would_be_high():
    results = _by_mechanism(assess_molecular_eligibility(
        {"msi_mmr": "mss_pmmr"}, {"metastatic": True}))
    assert results["immune_checkpoint_inhibition"]["status"] == NOT_ELIGIBLE


def test_anti_egfr_requires_confirmed_ras_and_braf_wildtype():
    unknown = _by_mechanism(assess_molecular_eligibility({}, {"metastatic": True}))
    assert unknown["anti_EGFR"]["status"] == REQUIRES_TEST
    confirmed = _by_mechanism(assess_molecular_eligibility(
        {"ras": "wild_type", "braf": "wild_type"}, {"metastatic": True}))
    assert confirmed["anti_EGFR"]["status"] == POTENTIALLY_ELIGIBLE


def test_braf_her2_g12c_and_ntrk_rules_are_biomarker_gated():
    results = _by_mechanism(assess_molecular_eligibility({
        "braf": "v600e", "ras": "wild_type", "her2": "positive",
        "kras_g12c": "positive", "ntrk": "fusion_positive",
    }, {"metastatic": True}))
    assert results["BRAF_plus_EGFR"]["status"] == POTENTIALLY_ELIGIBLE
    assert results["HER2_targeted"]["status"] == POTENTIALLY_ELIGIBLE
    assert results["KRAS_G12C_plus_EGFR"]["status"] == POTENTIALLY_ELIGIBLE
    assert results["TRK_inhibitor"]["status"] == POTENTIALLY_ELIGIBLE


def test_hybrid_hypotheses_are_trial_only_not_recommendations():
    profile = cms_continuous_profile(
        0.5 * PATTERNS["CMS1_MSI_immune"] + 0.5 * PATTERNS["CMS2_canonical_WNT"], PATTERNS)
    hypotheses = hybrid_mechanism_hypotheses(profile)
    assert len(hypotheses) == 4
    assert all(h["status"] == "SOLO_ENSAYO_CLINICO" for h in hypotheses)
