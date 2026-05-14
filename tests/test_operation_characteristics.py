"""Unit tests for bayesprop.utils.operation_characteristics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scipy.stats import binomtest

from bayesprop.utils.operation_characteristics import (
    grid_fixed_n,
    matched_calibration_alpha,
    simulate_fixed_n,
    simulate_sequential,
    wilson_band,
)


# ── simulate_fixed_n ─────────────────────────────────────────


class TestSimulateFixedN:
    """Tests for the fixed-n OC simulator."""

    def test_summary_keys_and_rate_sum(self) -> None:
        rng = np.random.default_rng(0)
        summary, pvals = simulate_fixed_n(0.7, 0.6, n=100, n_sim=20, rng=rng)
        # Required schema
        for key in (
            "reject",
            "accept",
            "inconclusive",
            "ci_coverage",
            "p_A",
            "p_B",
            "delta",
            "n",
            "n_sim",
        ):
            assert key in summary
        # Three Bayes-decision rates form a valid probability simplex.
        total = summary["reject"] + summary["accept"] + summary["inconclusive"]
        assert abs(total - 1.0) < 1e-12
        # Echoes preserve the inputs.
        assert summary["p_A"] == 0.7
        assert summary["p_B"] == 0.6
        assert abs(summary["delta"] - 0.1) < 1e-12
        assert summary["n"] == 100.0
        assert summary["n_sim"] == 20.0
        # One Fisher p-value per replicate.
        assert pvals.shape == (20,)
        assert np.all((pvals >= 0.0) & (pvals <= 1.0))

    def test_deterministic_given_rng(self) -> None:
        s1, p1 = simulate_fixed_n(0.7, 0.6, n=80, n_sim=15, rng=np.random.default_rng(7))
        s2, p2 = simulate_fixed_n(0.7, 0.6, n=80, n_sim=15, rng=np.random.default_rng(7))
        assert s1 == s2
        np.testing.assert_array_equal(p1, p2)

    def test_clear_effect_rejects_majority(self) -> None:
        rng = np.random.default_rng(1)
        summary, _ = simulate_fixed_n(0.85, 0.55, n=200, n_sim=40, rng=rng)
        # With Δ = 0.30 and n = 200 the BF≥3 rule should reject in
        # essentially every replicate.
        assert summary["reject"] > 0.9

    def test_track_ci_false_yields_nan(self) -> None:
        rng = np.random.default_rng(2)
        summary, _ = simulate_fixed_n(
            0.6, 0.6, n=50, n_sim=10, rng=rng, track_ci=False
        )
        assert np.isnan(summary["ci_coverage"])

    def test_invalid_bf_bounds_raise(self) -> None:
        rng = np.random.default_rng(3)
        with pytest.raises(ValueError, match="bf_lower"):
            simulate_fixed_n(
                0.6, 0.6, n=50, n_sim=5, rng=rng, bf_upper=1.0, bf_lower=2.0
            )


# ── grid_fixed_n ─────────────────────────────────────────────


class TestGridFixedN:
    """Tests for the grid sweep."""

    def test_shapes_and_columns(self) -> None:
        grid = [(0.55, 0.6), (0.6, 0.6), (0.7, 0.6)]
        df, pv = grid_fixed_n(grid, n=80, n_sim=12, seed=0)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 9)
        assert pv.shape == (3, 12)
        for col in ("reject", "accept", "inconclusive", "delta", "ci_coverage"):
            assert col in df.columns

    def test_seed_reproducibility(self) -> None:
        grid = [(0.55, 0.6), (0.65, 0.6)]
        df1, p1 = grid_fixed_n(grid, n=60, n_sim=10, seed=42)
        df2, p2 = grid_fixed_n(grid, n=60, n_sim=10, seed=42)
        pd.testing.assert_frame_equal(df1, df2)
        np.testing.assert_array_equal(p1, p2)


# ── matched_calibration_alpha ────────────────────────────────


class TestMatchedCalibrationAlpha:
    """Tests for the matched-α helper."""

    def test_matches_at_null_grid(self) -> None:
        # Construct a synthetic p-value matrix where the null row is
        # Uniform(0, 1). With bayes_type1_rate=0.05 the empirical
        # 5%-quantile should be ≈ 0.05.
        rng = np.random.default_rng(0)
        p_values = rng.uniform(0, 1, size=(3, 5000))
        alpha = matched_calibration_alpha(
            p_values, bayes_type1_rate=0.05, null_grid_index=0
        )
        assert 0.03 < alpha < 0.07

    def test_zero_rate_returns_zero(self) -> None:
        p_values = np.zeros((1, 100)) + 0.5
        assert matched_calibration_alpha(p_values, 0.0, 0) == 0.0

    def test_invalid_index_raises(self) -> None:
        p_values = np.zeros((2, 10))
        with pytest.raises(ValueError, match="out of range"):
            matched_calibration_alpha(p_values, 0.1, null_grid_index=5)

    def test_requires_2d(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            matched_calibration_alpha(np.zeros(10), 0.1, 0)


# ── simulate_sequential ──────────────────────────────────────


class TestSimulateSequential:
    """Tests for the sequential stopping-time harness."""

    def test_schema_and_quantile_order(self) -> None:
        rng = np.random.default_rng(0)
        out = simulate_sequential(
            0.75, 0.55, n_sim=10, rng=rng, n_min=40, n_max=400, batch_size=40
        )
        for key in (
            "p_A",
            "p_B",
            "delta",
            "median_n",
            "q05",
            "q25",
            "q75",
            "q95",
            "frac_censored",
            "frac_reject",
            "frac_accept",
        ):
            assert key in out
        # Quantiles must be monotone non-decreasing.
        assert out["q05"] <= out["q25"] <= out["median_n"] <= out["q75"] <= out["q95"]

    def test_decisive_effect_mostly_rejects(self) -> None:
        rng = np.random.default_rng(1)
        out = simulate_sequential(
            0.85, 0.55, n_sim=15, rng=rng, n_min=40, n_max=400, batch_size=40
        )
        assert out["frac_reject"] > 0.7
        # Almost no censoring at this effect size.
        assert out["frac_censored"] < 0.2


# ── wilson_band ──────────────────────────────────────────────


class TestWilsonBand:
    """Tests for the Wilson confidence band helper."""

    def test_matches_scipy_reference(self) -> None:
        # The helper is a thin vectoriser around scipy's binomtest +
        # proportion_ci(method="wilson"), so we check it agrees pointwise.
        rates = np.array([0.0, 0.05, 0.5, 0.95, 1.0])
        n_sim = 200
        lo, hi = wilson_band(rates, n_sim=n_sim)
        for r, l, h in zip(rates, lo, hi):
            k = int(round(r * n_sim))
            ref = binomtest(k, n_sim).proportion_ci(method="wilson")
            assert l == pytest.approx(ref.low, abs=1e-12)
            assert h == pytest.approx(ref.high, abs=1e-12)

    def test_endpoints_in_unit_interval(self) -> None:
        # Wilson is bounded — even at the extremes the band must stay
        # inside [0, 1], unlike a textbook Wald interval which would
        # over/undershoot near 0 and 1.
        rates = np.linspace(0.0, 1.0, 21)
        lo, hi = wilson_band(rates, n_sim=50)
        assert np.all(lo >= 0.0)
        assert np.all(hi <= 1.0)
        assert np.all(lo <= rates + 1e-12)
        assert np.all(hi >= rates - 1e-12)

    def test_band_width_shrinks_with_n(self) -> None:
        # Tightening n_sim must shrink the band: a basic sanity check
        # that the helper correctly threads n_sim through to binomtest.
        rate = np.array([0.30])
        lo_small, hi_small = wilson_band(rate, n_sim=50)
        lo_big, hi_big = wilson_band(rate, n_sim=5000)
        assert (hi_big - lo_big) < (hi_small - lo_small)

    def test_confidence_level_widens_band(self) -> None:
        rate = np.array([0.30])
        lo90, hi90 = wilson_band(rate, n_sim=200, confidence=0.90)
        lo99, hi99 = wilson_band(rate, n_sim=200, confidence=0.99)
        assert (hi99 - lo99) > (hi90 - lo90)

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="n_sim"):
            wilson_band(np.array([0.5]), n_sim=0)
        with pytest.raises(ValueError, match="confidence"):
            wilson_band(np.array([0.5]), n_sim=10, confidence=1.5)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            wilson_band(np.array([-0.1, 0.5]), n_sim=10)
