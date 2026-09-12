"""Regression coverage for interval bounds and default baseline behavior."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attractor_model import simulate_langevin
from prognosis import detect_recurrence_signal, summarize_patient_trajectory


@pytest.mark.parametrize("span, dt", [((0, 1), 0.3), ((2, 2.1), 1), ((0, 0.9), 0.3)])
def test_langevin_ends_at_requested_time_and_scales_each_step(span, dt):
    sigma = 0.4
    result = simulate_langevin(np.zeros((1, 1)), np.ones(1), np.zeros(1),
                               sigma, span, dt, seed=42)
    times = result["t"]
    assert times[0] == span[0]
    assert times[-1] == span[1]
    assert np.all(times <= span[1])
    assert np.all(np.diff(times) > 0)
    assert np.all(np.diff(times) <= dt + 1e-14)
    # Independent scalar EM recurrence checks drift and Wiener scaling,
    # including the shortened final step.
    rng = np.random.default_rng(42)
    expected = [0.0]
    for h in np.diff(times):
        expected.append(expected[-1] + (1 - expected[-1]) * h
                        + sigma * np.sqrt(h) * rng.standard_normal())
    np.testing.assert_allclose(result["x"][0], expected)


def test_langevin_zero_duration_returns_initial_state():
    result = simulate_langevin(np.zeros((1, 1)), np.ones(1), np.array([2.0]),
                               0.4, (2, 2), 0.3)
    np.testing.assert_array_equal(result["t"], [2])
    np.testing.assert_array_equal(result["x"], [[2.0]])


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf])
def test_langevin_rejects_invalid_step(dt):
    with pytest.raises(ValueError, match="dt"):
        simulate_langevin(np.zeros((1, 1)), np.zeros(1), np.zeros(1), 0, dt=dt)


@pytest.mark.parametrize("span", [(1, 0), (0, np.inf), (np.nan, 1)])
def test_langevin_rejects_invalid_interval(span):
    with pytest.raises(ValueError, match="t_span"):
        simulate_langevin(np.zeros((1, 1)), np.zeros(1), np.zeros(1), 0, t_span=span)


def test_default_baseline_and_summary_do_not_warn():
    hazard = np.array([0.02, 0.05, 0.03, 0.04, 0.30])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert detect_recurrence_signal(hazard) == (True, 4)
        summary = summarize_patient_trajectory(np.arange(5), hazard[None, :])
    assert summary["alert_timepoint"] == 4


def test_default_baseline_requires_followup_after_three_points():
    with pytest.raises(ValueError, match="mas de 3"):
        detect_recurrence_signal(np.array([0.02, 0.05, 0.03]))
