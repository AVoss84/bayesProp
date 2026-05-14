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
    binarize_if_needed,
    find_n_for_power,
    fisher_exact_nonpaired_test,
    mcnemar_paired_test,
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


# ── mcnemar_paired_test ──────────────────────────────────────


class TestMcnemarPaired:
    """Tests for McNemar's exact / chi² paired test."""

    def test_detects_clear_effect(self) -> None:
        # Heavy imbalance toward A succeeding when B fails — should
        # reject H_0 with a small p-value.
        y_A = np.array([1] * 20 + [0] * 5 + [1] * 5 + [0] * 70)
        y_B = np.array([0] * 20 + [1] * 5 + [1] * 5 + [0] * 70)
        result = mcnemar_paired_test(y_A, y_B)
        assert result.p_value < 0.01
        assert result.test == "mcnemar_exact"
        assert result.odds_ratio == 4.0  # b/c = 20/5

    def test_null_distribution_uniform_ish(self) -> None:
        # Under H_0 the p-value distribution should be roughly Uniform.
        rng = np.random.default_rng(1)
        n_reps = 100
        p_values = np.empty(n_reps)
        for i in range(n_reps):
            y_A = rng.binomial(1, 0.4, 200)
            y_B = rng.binomial(1, 0.4, 200)
            p_values[i] = mcnemar_paired_test(y_A, y_B).p_value
        assert 0.02 < float(np.mean(p_values < 0.05)) < 0.20

    def test_alternative_argument_split(self) -> None:
        y_A = np.array([1] * 15 + [0] * 5 + [0] * 80)
        y_B = np.array([0] * 15 + [1] * 5 + [0] * 80)
        two = mcnemar_paired_test(y_A, y_B, alternative="two-sided").p_value
        greater = mcnemar_paired_test(y_A, y_B, alternative="greater").p_value
        less = mcnemar_paired_test(y_A, y_B, alternative="less").p_value
        # b > c so "greater" should reject more strongly than "less".
        assert greater < less
        # Two-sided is bounded by twice the smaller one-sided.
        assert two <= 2.0 * min(greater, less) + 1e-12

    def test_no_discordant_pairs_returns_one(self) -> None:
        y = np.array([1, 0, 1, 0, 1, 0])
        # All concordant -> no information; p must be 1.
        assert mcnemar_paired_test(y, y).p_value == 1.0

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            mcnemar_paired_test([0, 1, 0], [0, 1])

    def test_rejects_non_binary_input(self) -> None:
        rng = np.random.default_rng(2)
        y_A = rng.uniform(0, 1, 50)
        y_B = rng.binomial(1, 0.5, 50).astype(float)
        with pytest.raises(ValueError, match="0/1"):
            mcnemar_paired_test(y_A, y_B)


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


# ── binarize_if_needed ───────────────────────────────────────


class TestBinarizeIfNeeded:
    """Unit tests for the shared binarisation helper.

    The helper must be idempotent on already-binary input, binarise
    continuous values in ``[0, 1]`` at the supplied threshold, and refuse
    inputs that are out of range (positive or negative)."""

    def test_passes_through_binary_input(self) -> None:
        y = np.array([0, 1, 1, 0, 1], dtype=float)
        out = binarize_if_needed(y)
        # Same values, float dtype, untouched.
        assert np.array_equal(out, y)
        assert out.dtype == np.float64

    def test_passes_through_binary_input_int_dtype(self) -> None:
        # Integer 0/1 must also be recognised as already binary.
        y = np.array([0, 1, 0], dtype=np.int64)
        out = binarize_if_needed(y)
        assert np.array_equal(out, y.astype(float))

    def test_binarises_continuous_at_default_threshold(self) -> None:
        y = np.array([0.1, 0.4, 0.5, 0.6, 0.9])
        out = binarize_if_needed(y)
        # ≥ 0.5 → 1, else 0.
        assert out.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]

    def test_binarises_continuous_at_custom_threshold(self) -> None:
        y = np.array([0.1, 0.4, 0.5, 0.6, 0.9])
        out = binarize_if_needed(y, threshold=0.7)
        assert out.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]

    def test_rejects_values_above_one(self) -> None:
        with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
            binarize_if_needed(np.array([0.5, 1.5]))

    def test_rejects_negative_values(self) -> None:
        with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
            binarize_if_needed(np.array([-0.01, 0.5, 0.9]))

    def test_rejects_nan_values(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            binarize_if_needed(np.array([0.3, np.nan, 0.6]))

    def test_empty_input_round_trips(self) -> None:
        # Empty arrays are a valid (no-op) input.
        out = binarize_if_needed(np.array([], dtype=float))
        assert out.size == 0

    def test_verbose_prints_only_when_binarising(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Already-binary → silent even when verbose=True.
        binarize_if_needed(np.array([0.0, 1.0]), verbose=True, name="y_test")
        assert capsys.readouterr().out == ""

        # Continuous → one warning line.
        binarize_if_needed(np.array([0.2, 0.8]), verbose=True, name="y_test")
        captured = capsys.readouterr().out
        assert "y_test" in captured and "threshold" in captured
