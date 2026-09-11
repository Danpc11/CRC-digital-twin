"""Regresiones de la revision del 2026-09-11 (fuera del puente qPCR)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attractor_model import W, N, DRIVER_BIAS, dynamics, simulate_langevin
from calibration import infer_gene_columns
from prognosis import detect_recurrence_signal


def test_infer_gene_columns_excludes_pipeline_outputs():
    df = pd.DataFrame({
        "sample_id": ["a", "b"], "cms_label": ["none", "none"],
        "MYC": [1.0, 2.0], "VIM": [0.5, 0.1],
        "cms1_tendency": [0.3, 0.2], "cms_margin": [0.1, 0.2], "cms_entropy": [0.9, 0.8],
        "modern_hopfield_correlation": [0.9, 0.8], "modern_hopfield_input_margin": [0.2, 0.1],
        "classification_confidence": [0.7, 0.6], "stage": [2, 3],
    })
    assert infer_gene_columns(df) == ["MYC", "VIM"]


def test_dynamics_rejects_noise_in_ode_rhs():
    with pytest.raises(ValueError, match="simulate_langevin"):
        dynamics(0.0, np.zeros(N), W, DRIVER_BIAS["none"], noise_sigma=0.1)


def test_simulate_langevin_reduces_to_deterministic_when_sigma_zero():
    from scipy.integrate import solve_ivp
    x0 = 0.1 * np.ones(N)
    I = DRIVER_BIAS["APC_mut"]
    em = simulate_langevin(W, I, x0, noise_sigma=0.0, t_span=(0, 5), dt=1e-3)
    ref = solve_ivp(dynamics, (0, 5), x0, args=(W, I), rtol=1e-9, atol=1e-11)
    assert np.allclose(em["x"][:, -1], ref.y[:, -1], atol=5e-3)


def test_simulate_langevin_is_reproducible_and_noisy():
    x0 = np.zeros(N)
    a = simulate_langevin(W, DRIVER_BIAS["none"], x0, 0.3, (0, 2), 0.01, seed=1)
    b = simulate_langevin(W, DRIVER_BIAS["none"], x0, 0.3, (0, 2), 0.01, seed=1)
    c = simulate_langevin(W, DRIVER_BIAS["none"], x0, 0.3, (0, 2), 0.01, seed=2)
    assert np.array_equal(a["x"], b["x"])
    assert not np.array_equal(a["x"], c["x"])
    assert np.std(a["x"][:, -1]) > 0


def test_detect_recurrence_warns_with_short_baseline_and_uses_floor():
    hazard = np.array([0.02, 0.05, 0.03, 0.04, 0.30])
    with pytest.warns(UserWarning, match="sigma basal"):
        alert, idx = detect_recurrence_signal(hazard, baseline_window=2, absolute_floor=0.1)
    assert alert and idx == 4
    # con 3 puntos basales y la misma serie, la sigma es estimable pero el piso protege
    alert3, idx3 = detect_recurrence_signal(hazard, baseline_window=3, absolute_floor=0.1)
    assert alert3 and idx3 == 4
