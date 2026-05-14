"""Unit tests for bayesprop.utils.utils module."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from bayesprop.utils.utils import (
    bf10_to_ph0,
    bfda_power_curve,
    bfda_simulate,
    find_n_for_power,
    fisher_exact_nonpaired_test,
    plot_bfda_power,
    plot_bfda_sensitivity,
)

# ── bf10_to_ph0 ──────────────────────────────────────────────


class TestBf10ToPh0:
    """Tests for the bf10_to_ph0 conversion."""

    def test_equal_evidence(self) -> None:
        result = bf10_to_ph0(1.0, prior_H0=0.5)
        assert abs(result - 0.5) < 1e-10

    def test_strong_h1(self) -> None:
        result = bf10_to_ph0(100.0, prior_H0=0.5)
        assert result < 0.05

    def test_strong_h0(self) -> None:
        result = bf10_to_ph0(0.01, prior_H0=0.5)
        assert result > 0.95

    def test_prior_influence(self) -> None:
        result_low = bf10_to_ph0(1.0, prior_H0=0.1)
        result_high = bf10_to_ph0(1.0, prior_H0=0.9)
        assert result_low < result_high


# ── bfda_simulate ─────────────────────────────────────────────


class TestBfdaSimulate:
    """Tests for the generic BFDA simulation engine."""

    def test_returns_dict(self) -> None:
        def gen(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
            return rng.binomial(1, 0.7, n).astype(float), rng.binomial(1, 0.5, n).astype(float)

        def decide(y_a: np.ndarray, y_b: np.ndarray) -> bool:
            return True  # always decisive

        result = bfda_simulate(gen, decide, sample_sizes=[10, 20], n_sim=10, seed=42)
        assert isinstance(result, dict)
        assert set(result.keys()) == {10, 20}

    def test_power_values_in_range(self) -> None:
        def gen(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
            return rng.binomial(1, 0.7, n).astype(float), rng.binomial(1, 0.5, n).astype(float)

        def decide(y_a: np.ndarray, y_b: np.ndarray) -> bool:
            return False  # never decisive

        result = bfda_simulate(gen, decide, sample_sizes=[10], n_sim=10, seed=42)
        assert 0.0 <= result[10] <= 1.0

    def test_always_decisive(self) -> None:
        def gen(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
            return np.ones(n), np.zeros(n)

        def decide(y_a: np.ndarray, y_b: np.ndarray) -> bool:
            return True

        result = bfda_simulate(gen, decide, sample_sizes=[5], n_sim=20, seed=0)
        assert result[5] == 1.0


# ── bfda_power_curve ──────────────────────────────────────────


class TestBfdaPowerCurve:
    """Tests for the non-paired BFDA power curve."""

    def test_returns_correct_keys(self) -> None:
        sizes = [20, 50]
        result = bfda_power_curve(theta_A_true=0.8, theta_B_true=0.5, sample_sizes=sizes, n_sim=20, seed=42)
        assert set(result.keys()) == set(sizes)

    def test_power_increases_with_n(self) -> None:
        result = bfda_power_curve(
            theta_A_true=0.8,
            theta_B_true=0.5,
            sample_sizes=[20, 100, 500],
            n_sim=50,
            seed=42,
        )
        powers = list(result.values())
        # Power should generally increase (allow non-strict for small n_sim)
        assert powers[-1] >= powers[0]

    def test_no_effect_low_power(self) -> None:
        result = bfda_power_curve(theta_A_true=0.7, theta_B_true=0.7, sample_sizes=[50], n_sim=50, seed=42)
        assert result[50] < 0.5  # should rarely exceed threshold when no effect


# ── bfda_power_curve_ph0 ─────────────────────────────────────


class TestBfdaPowerCurvePh0:
    """Tests for the P(H0)-based BFDA power curve."""

    def test_nonpaired_design(self) -> None:
        result = bfda_power_curve(
            theta_A_true=0.8,
            theta_B_true=0.5,
            sample_sizes=[50],
            n_sim=20,
            seed=42,
            design="nonpaired",
            decision_rule="posterior_null",
        )
        assert 50 in result
        assert 0.0 <= result[50] <= 1.0

    def test_invalid_design_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown design"):
            bfda_power_curve(
                theta_A_true=0.8,
                theta_B_true=0.5,
                sample_sizes=[50],
                n_sim=5,
                design="invalid",
            )


# ── fisher_exact_nonpaired_test ──────────────────────────────


class TestFisherExactNonpaired:
    """Tests for the Fisher exact frequentist baseline."""

    def test_clear_effect_rejects(self) -> None:
        rng = np.random.default_rng(0)
        y_A = rng.binomial(1, 0.85, 300).astype(float)
        y_B = rng.binomial(1, 0.55, 300).astype(float)
        result = fisher_exact_nonpaired_test(y_A, y_B)
        assert result.p_value < 1e-6
        assert result.odds_ratio is not None and result.odds_ratio > 1.0
        assert result.n_A == 300 and result.n_B == 300
        assert result.test == "fisher_exact"

    def test_null_is_uniform_in_p_value(self) -> None:
        """Under H0 the p-values should be (approximately) uniform on (0, 1]."""
        rng = np.random.default_rng(1)
        n_sim = 400
        p_values = np.empty(n_sim)
        for i in range(n_sim):
            y_A = rng.binomial(1, 0.5, 200).astype(float)
            y_B = rng.binomial(1, 0.5, 200).astype(float)
            p_values[i] = fisher_exact_nonpaired_test(y_A, y_B).p_value
        # Fisher is discrete/conservative so we only check the bound.
        assert (p_values < 0.05).mean() <= 0.07

    def test_alternative_arguments(self) -> None:
        rng = np.random.default_rng(2)
        y_A = rng.binomial(1, 0.8, 200).astype(float)
        y_B = rng.binomial(1, 0.5, 200).astype(float)
        two_sided = fisher_exact_nonpaired_test(y_A, y_B, alternative="two-sided")
        greater = fisher_exact_nonpaired_test(y_A, y_B, alternative="greater")
        less = fisher_exact_nonpaired_test(y_A, y_B, alternative="less")
        assert two_sided.alternative == "two-sided"
        assert greater.p_value < two_sided.p_value
        assert less.p_value > 0.5

    def test_rejects_non_binary_input(self) -> None:
        rng = np.random.default_rng(3)
        y_A = rng.uniform(0, 1, 100)
        y_B = rng.binomial(1, 0.5, 100).astype(float)
        with pytest.raises(ValueError, match="0/1"):
            fisher_exact_nonpaired_test(y_A, y_B)


# ── find_n_for_power ─────────────────────────────────────────


class TestFindNForPower:
    """Tests for the find_n_for_power interpolation utility."""

    def test_interpolation(self) -> None:
        curve = {50: 0.3, 100: 0.6, 200: 0.9}
        n = find_n_for_power(curve, target_power=0.80)
        assert n is not None
        assert 100 < n < 200

    def test_target_already_reached(self) -> None:
        curve = {50: 0.9, 100: 0.95}
        n = find_n_for_power(curve, target_power=0.80)
        assert n is not None
        assert n == 50.0

    def test_target_unreachable(self) -> None:
        curve = {50: 0.1, 100: 0.2}
        n = find_n_for_power(curve, target_power=0.80)
        assert n is None


# ── Plotting ──────────────────────────────────────────────────


class TestBfdaPlots:
    """Smoke tests for BFDA plotting functions."""

    def test_plot_bfda_power(self) -> None:
        curve = {50: 0.3, 100: 0.6, 200: 0.9}
        fig = plot_bfda_power(curve, theta_A_true=0.8, theta_B_true=0.5)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_bfda_sensitivity(self) -> None:
        fig = plot_bfda_sensitivity(
            theta_A_true=0.8,
            theta_B_true=0.5,
            sample_sizes=[20, 50],
            thresholds=[3.0],
            n_sim=10,
            seed=42,
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close("all")
