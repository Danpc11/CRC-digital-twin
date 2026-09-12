"""Tests de reference_gene_stability.py con expresion sintetica de respuesta
conocida: un gen estable, uno que depende del CMS, uno ruidoso y dos
co-regulados. Se verifica que cada metrica ordene como debe y que el
ranking combinado y el geNorm por pasos se comporten."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reference_gene_stability import (
    anova_eta2,
    combine_rankings,
    evaluate_cohort,
    genorm_M,
    genorm_stepwise,
    normfinder_rho,
)

CMS = ["CMS1_MSI_immune", "CMS2_canonical_WNT", "CMS3_metabolic", "CMS4_mesenchymal"]


def _expression(n=400, seed=11):
    rng = np.random.default_rng(seed)
    cms = pd.Series(rng.choice(CMS, size=n), index=[f"s{i}" for i in range(n)])
    load = rng.normal(0, 0.3, n)  # efecto de carga por muestra (comun a todos)
    X = pd.DataFrame(index=cms.index)
    X["STABLE"] = 12 + load + rng.normal(0, 0.15, n)
    X["CMS_DEP"] = 10 + load + rng.normal(0, 0.15, n) + cms.map({CMS[0]: 1.0, CMS[1]: 0, CMS[2]: 0, CMS[3]: -0.5}).to_numpy()
    X["NOISY"] = 8 + load + rng.normal(0, 0.9, n)
    shared = rng.normal(0, 0.5, n)
    X["COREG_A"] = 11 + load + shared + rng.normal(0, 0.05, n)
    X["COREG_B"] = 11.5 + load + shared + rng.normal(0, 0.05, n)
    stage = pd.Series(rng.integers(1, 4, n).astype(float), index=cms.index)
    return X, cms, stage


def test_anova_eta2_flags_cms_dependent_gene():
    X, cms, _ = _expression()
    _, eta_dep, delta_dep = anova_eta2(X["CMS_DEP"], cms)
    _, eta_stable, _ = anova_eta2(X["STABLE"], cms)
    assert eta_dep > 0.3 > eta_stable
    assert delta_dep > 1.2
    F, eta, d = anova_eta2(pd.Series([1.0, 2.0]), pd.Series(["a", "a"]))
    assert np.isnan(eta)


def test_genorm_M_protects_coregulated_pair_and_penalizes_noise():
    X, _, _ = _expression()
    M = genorm_M(X)
    assert M["NOISY"] == M.max()
    # los co-regulados se protegen entre si: quedan por debajo del ruidoso
    assert M["COREG_A"] < M["NOISY"] and M["COREG_B"] < M["NOISY"]
    order, finals = genorm_stepwise(X, keep=3)
    assert order[0] == "NOISY"
    assert len(finals) == 3


def test_normfinder_penalizes_cms_dependence_more_than_genorm_does():
    X, cms, _ = _expression()
    rho = normfinder_rho(X, cms)
    M = genorm_M(X)
    assert rho["CMS_DEP"] > rho["STABLE"]
    # el gen dependiente de CMS sube mas de rango en NormFinder que en geNorm
    assert rho.rank()["CMS_DEP"] >= M.rank()["CMS_DEP"]


def test_evaluate_and_combine_rankings():
    X, cms, stage = _expression()
    res_a = evaluate_cohort(X, cms, stage, "A", {"STABLE": "clasico"})
    res_b = evaluate_cohort(X * 1.1 + 0.3, cms, None, "B", {"STABLE": "clasico"})
    assert res_a.loc["STABLE", "rank_mean_cohort"] == res_a["rank_mean_cohort"].min()
    assert np.isnan(res_b["eta2_stage"]).all()
    assert res_a.loc["STABLE", "source"] == "clasico" and res_a.loc["NOISY", "source"] == "otro"
    comb, rho = combine_rankings({"A": res_a, "B": res_b})
    assert comb.index[0] == "STABLE"
    assert rho is not None and rho > 0.9  # misma estructura -> rangos concordantes


def test_select_expressed_probes_drops_background_probe():
    from reference_gene_stability import select_expressed_probes
    means = pd.Series({"202854_at": 9.67, "1565446_at": 2.38})  # HPRT1 real
    assert select_expressed_probes(means) == ["202854_at"]
    means2 = pd.Series({"a": 13.0, "b": 12.5, "c": 11.2})
    assert set(select_expressed_probes(means2)) == {"a", "b", "c"}
