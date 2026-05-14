"""Unit tests for bayesprop.utils.operation_characteristics_paired."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesprop.utils.operation_characteristics_paired import (
    grid_fixed_n_paired,
    matched_calibration_alpha,
    simulate_fixed_n_paired,
    simulate_sequential_paired,
    wilson_band,
)


# ── simulate_fixed_n_paired ──────────────────────────────────


class TestSimulateFixedNPaired:
    """Tests for the paired fixed-n OC simulator."""

    def test_summary_keys_and_rate_sum(self) -> None:
        rng = np.random.default_rng(0)
        summary, pvals = simulate_fixed_n_paired(
            0.65, 0.55, n=150, n_sim=20, rng=rng
        )
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
        total = summary["reject"] + summary["accept"] + summary["inconclusive"]
        assert abs(total - 1.0) < 1e-12
        assert summary["p_A"] == 0.65
        assert summary["p_B"] == 0.55
        assert summary["delta"] == pytest.approx(0.10)
        assert pvals.shape == (20,)
        assert np.all((pvals >= 0.0) & (pvals <= 1.0))

    def test_deterministic_given_rng(self) -> None:
        s1, p1 = simulate_fixed_n_paired(
            0.7, 0.6, n=100, n_sim=15, rng=np.random.default_rng(7)
        )
        s2, p2 = simulate_fixed_n_paired(
            0.7, 0.6, n=100, n_sim=15, rng=np.random.default_rng(7)
        )
        assert s1 == s2
        np.testing.assert_array_equal(p1, p2)

    def test_clear_effect_rejects_majority(self) -> None:
        # A paired design with Δ = 0.2 at n = 200 should reject in
        # almost every replicate — paired data has a much tighter
        # sampling distribution than the equivalent non-paired one.
        rng = np.random.default_rng(1)
        summary, _ = simulate_fixed_n_paired(
            0.75, 0.55, n=200, n_sim=30, rng=rng
        )
        assert summary["reject"] > 0.8

    def test_track_ci_false_yields_nan(self) -> None:
        rng = np.random.default_rng(2)
        summary, _ = simulate_fixed_n_paired(
            0.5, 0.5, n=80, n_sim=10, rng=rng, track_ci=False
        )
        assert np.isnan(summary["ci_coverage"])

    def test_invalid_bf_bounds_raise(self) -> None:
        rng = np.random.default_rng(3)
        with pytest.raises(ValueError, match="bf_lower"):
            simulate_fixed_n_paired(
                0.6, 0.6, n=50, n_sim=5, rng=rng, bf_upper=1.0, bf_lower=2.0
            )


# ── grid_fixed_n_paired ──────────────────────────────────────


class TestGridFixedNPaired:
    """Tests for the paired grid sweep."""

    def test_shapes_and_columns(self) -> None:
        grid = [(0.55, 0.6), (0.6, 0.6), (0.7, 0.6)]
        df, pv = grid_fixed_n_paired(grid, n=120, n_sim=15, seed=0)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 9)
        assert pv.shape == (3, 15)
        for col in ("reject", "accept", "inconclusive", "delta", "ci_coverage"):
            assert col in df.columns

    def test_seed_reproducibility(self) -> None:
        grid = [(0.55, 0.6), (0.65, 0.6)]
        df1, p1 = grid_fixed_n_paired(grid, n=80, n_sim=10, seed=42)
        df2, p2 = grid_fixed_n_paired(grid, n=80, n_sim=10, seed=42)
        pd.testing.assert_frame_equal(df1, df2)
        np.testing.assert_array_equal(p1, p2)


# ── re-exports (matched_calibration_alpha, wilson_band) ──────


class TestPairedReExports:
    """Sanity check the parametrisation-free helpers reach the paired API."""

    def test_matched_alpha_reachable(self) -> None:
        rng = np.random.default_rng(0)
        pv = rng.uniform(0, 1, size=(3, 1000))
        alpha = matched_calibration_alpha(pv, 0.05, 0)
        assert 0.02 < alpha < 0.08

    def test_wilson_band_reachable(self) -> None:
        rates = np.array([0.05, 0.5, 0.95])
        lo, hi = wilson_band(rates, n_sim=200)
        assert lo.shape == hi.shape == rates.shape
        assert np.all(lo >= 0.0)
        assert np.all(hi <= 1.0)


# ── simulate_sequential_paired ───────────────────────────────


class TestSimulateSequentialPaired:
    """Tests for the paired sequential stopping-time harness."""

    def test_schema_and_quantile_order(self) -> None:
        rng = np.random.default_rng(0)
        out = simulate_sequential_paired(
            0.7, 0.55, n_sim=10, rng=rng, n_min=40, n_max=400, batch_size=40
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
        assert out["q05"] <= out["q25"] <= out["median_n"] <= out["q75"] <= out["q95"]

    def test_decisive_effect_mostly_rejects(self) -> None:
        rng = np.random.default_rng(1)
        out = simulate_sequential_paired(
            0.80, 0.55, n_sim=15, rng=rng, n_min=40, n_max=400, batch_size=40
        )
        assert out["frac_reject"] > 0.7
        assert out["frac_censored"] < 0.2
