"""Unit tests for bayesAB.resources.bayes_nonpaired module."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from bayesAB.resources.bayes_nonpaired import (
    NonPairedBayesPropTest,
    _format_bf,
    beta_diff_pdf,
    descriptive_summary,
)
from bayesAB.resources.data_schemas import (
    HypothesisDecision,
    NonPairedSummary,
    NonPairedTestResult,
    PosteriorProbH0Result,
    PPCStatistic,
    ROPEResult,
    SavageDickeyResult,
)

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible binary data with known effect."""
    rng = np.random.default_rng(42)
    y_a = rng.binomial(1, 0.8, size=80).astype(float)
    y_b = rng.binomial(1, 0.6, size=80).astype(float)
    return y_a, y_b


@pytest.fixture
def fitted_model(binary_data: tuple[np.ndarray, np.ndarray]) -> NonPairedBayesPropTest:
    """Return a fitted NonPairedBayesPropTest model."""
    y_a, y_b = binary_data
    return NonPairedBayesPropTest(seed=42, n_samples=10_000).fit(y_a, y_b)


# ── beta_diff_pdf ─────────────────────────────────────────────


class TestBetaDiffPdf:
    """Tests for the beta_diff_pdf function."""

    def test_integrates_to_one(self) -> None:
        from scipy.integrate import quad

        area, _ = quad(lambda z: beta_diff_pdf(z, 5.0, 3.0, 4.0, 4.0), -0.999, 0.999)
        assert abs(area - 1.0) < 0.01

    def test_zero_outside_range(self) -> None:
        assert beta_diff_pdf(-1.0, 2.0, 2.0, 2.0, 2.0) == 0.0
        assert beta_diff_pdf(1.0, 2.0, 2.0, 2.0, 2.0) == 0.0

    def test_symmetric_prior(self) -> None:
        val_pos = beta_diff_pdf(0.3, 1.0, 1.0, 1.0, 1.0)
        val_neg = beta_diff_pdf(-0.3, 1.0, 1.0, 1.0, 1.0)
        assert abs(val_pos - val_neg) < 1e-10

    def test_positive_density(self) -> None:
        assert beta_diff_pdf(0.0, 5.0, 3.0, 4.0, 4.0) > 0.0


# ── _format_bf ────────────────────────────────────────────────


class TestFormatBf:
    """Tests for the _format_bf helper."""

    def test_normal_range(self) -> None:
        assert _format_bf(3.5) == "3.50"

    def test_large_value(self) -> None:
        result = _format_bf(1e6)
        assert "10^" in result

    def test_small_positive(self) -> None:
        result = _format_bf(1e-6)
        assert "10^" in result


# ── NonPairedBayesPropTest ────────────────────────────────────


class TestNonPairedInit:
    """Tests for NonPairedBayesPropTest initialization."""

    def test_default_params(self) -> None:
        model = NonPairedBayesPropTest()
        assert model.alpha0 == 1.0
        assert model.beta0 == 1.0
        assert model.threshold == 0.7
        assert model.n_samples == 20_000

    def test_custom_params(self) -> None:
        model = NonPairedBayesPropTest(alpha0=0.5, beta0=0.5, threshold=0.8, n_samples=5000)
        assert model.alpha0 == 0.5
        assert model.threshold == 0.8
        assert model.n_samples == 5000

    def test_verbose_init(self, capsys: pytest.CaptureFixture[str]) -> None:
        NonPairedBayesPropTest(verbose=True)
        captured = capsys.readouterr()
        assert "Initialized" in captured.out

    def test_check_fitted_raises(self) -> None:
        model = NonPairedBayesPropTest()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.savage_dickey_test()


class TestNonPairedTest:
    """Tests for NonPairedBayesPropTest.test()."""

    def test_returns_nonpaired_test_result(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_a, y_b = binary_data
        model = NonPairedBayesPropTest(seed=42)
        result = model.test(y_a, y_b)
        assert isinstance(result, NonPairedTestResult)

    def test_posterior_params_correct(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_a, y_b = binary_data
        model = NonPairedBayesPropTest(alpha0=1.0, beta0=1.0, seed=42)
        result = model.test(y_a, y_b)
        k_a = y_a.sum()
        n_a = len(y_a)
        assert result.thetaA_post.alpha == 1.0 + k_a
        assert result.thetaA_post.beta == 1.0 + n_a - k_a

    def test_p_b_greater_a_in_range(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_a, y_b = binary_data
        result = NonPairedBayesPropTest(seed=42).test(y_a, y_b)
        assert 0.0 <= result.P_B_greater_A <= 1.0


class TestNonPairedFit:
    """Tests for NonPairedBayesPropTest.fit()."""

    def test_returns_self(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_a, y_b = binary_data
        model = NonPairedBayesPropTest(seed=42, n_samples=5000)
        result = model.fit(y_a, y_b)
        assert result is model

    def test_binarize_continuous_scores(self) -> None:
        """Non-binary inputs should be binarized at threshold."""
        scores_a = np.array([0.9, 0.8, 0.3, 0.6, 0.75])
        scores_b = np.array([0.4, 0.7, 0.2, 0.5, 0.85])
        model = NonPairedBayesPropTest(threshold=0.7, seed=42, n_samples=1000).fit(scores_a, scores_b)
        assert model.summary is not None

    def test_binarize_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        """verbose=True should log when binarizing."""
        scores = np.array([0.9, 0.3, 0.7, 0.5])
        binary = np.array([1.0, 0.0, 1.0, 0.0])
        model = NonPairedBayesPropTest(threshold=0.7, seed=42, n_samples=1000, verbose=True)
        model.fit(scores, binary)
        captured = capsys.readouterr()
        assert "binarizing" in captured.out.lower()

    def test_summary_is_nonpaired_summary(self, fitted_model: NonPairedBayesPropTest) -> None:
        assert isinstance(fitted_model.summary, NonPairedSummary)

    def test_summary_fields(self, fitted_model: NonPairedBayesPropTest) -> None:
        s = fitted_model.summary
        assert -1.0 <= s.mean_delta <= 1.0
        assert s.ci_95.lower < s.ci_95.upper
        assert 0.0 <= s.p_A_greater_B <= 1.0

    def test_delta_samples_shape(self, fitted_model: NonPairedBayesPropTest) -> None:
        assert fitted_model.delta_samples.shape == (10_000,)

    def test_trace_summary_is_dataframe(self, fitted_model: NonPairedBayesPropTest) -> None:
        assert isinstance(fitted_model.trace_summary, pd.DataFrame)
        assert len(fitted_model.trace_summary) > 0

    def test_known_effect_detected(self) -> None:
        rng = np.random.default_rng(99)
        y_a = rng.binomial(1, 0.9, size=200).astype(float)
        y_b = rng.binomial(1, 0.5, size=200).astype(float)
        model = NonPairedBayesPropTest(seed=99, n_samples=10_000).fit(y_a, y_b)
        assert model.summary.p_A_greater_B > 0.95


class TestNonPairedSavageDickey:
    """Tests for NonPairedBayesPropTest.savage_dickey_test()."""

    def test_returns_savage_dickey_result(self, fitted_model: NonPairedBayesPropTest) -> None:
        result = fitted_model.savage_dickey_test()
        assert isinstance(result, SavageDickeyResult)

    def test_bf_reciprocal(self, fitted_model: NonPairedBayesPropTest) -> None:
        result = fitted_model.savage_dickey_test()
        assert abs(result.BF_01 * result.BF_10 - 1.0) < 0.01

    def test_has_interpretation(self, fitted_model: NonPairedBayesPropTest) -> None:
        result = fitted_model.savage_dickey_test()
        assert len(result.interpretation) > 0
        assert len(result.decision) > 0

    def test_strong_effect_rejects_h0(self) -> None:
        """Large effect → high BF₁₀ → 'Reject H0'."""
        rng = np.random.default_rng(99)
        y_a = rng.binomial(1, 0.95, size=200).astype(float)
        y_b = rng.binomial(1, 0.3, size=200).astype(float)
        m = NonPairedBayesPropTest(seed=99, n_samples=10_000).fit(y_a, y_b)
        bf = m.savage_dickey_test()
        assert bf.BF_10 > 10
        assert bf.decision == "Reject H0"

    def test_no_effect_fails_to_reject(self) -> None:
        """Equal rates → low BF₁₀ → 'Fail to reject H0'."""
        rng = np.random.default_rng(99)
        y_a = rng.binomial(1, 0.5, size=200).astype(float)
        y_b = rng.binomial(1, 0.5, size=200).astype(float)
        m = NonPairedBayesPropTest(seed=99, n_samples=10_000).fit(y_a, y_b)
        bf = m.savage_dickey_test()
        assert bf.decision == "Fail to reject H0"


class TestNonPairedPosteriorProbH0:
    """Tests for NonPairedBayesPropTest.posterior_probability_H0()."""

    def test_returns_posterior_prob_result(self) -> None:
        result = NonPairedBayesPropTest.posterior_probability_H0(1.0, prior_H0=0.5)
        assert isinstance(result, PosteriorProbH0Result)

    def test_equal_bf_equal_prior(self) -> None:
        result = NonPairedBayesPropTest.posterior_probability_H0(1.0, prior_H0=0.5)
        assert abs(result.p_H0 - 0.5) < 1e-6

    def test_strong_evidence_for_h0(self) -> None:
        result = NonPairedBayesPropTest.posterior_probability_H0(100.0, prior_H0=0.5)
        assert result.p_H0 > 0.95
        assert result.decision == "Fail to reject H0"

    def test_strong_evidence_against_h0(self) -> None:
        result = NonPairedBayesPropTest.posterior_probability_H0(0.01, prior_H0=0.5)
        assert result.p_H1 > 0.95
        assert result.decision == "Reject H0"

    def test_undecided(self) -> None:
        result = NonPairedBayesPropTest.posterior_probability_H0(1.0, prior_H0=0.5)
        assert result.decision == "Undecided"


class TestNonPairedPPC:
    """Tests for NonPairedBayesPropTest.ppc_pvalues()."""

    def test_returns_dict_of_ppc_statistics(self, fitted_model: NonPairedBayesPropTest) -> None:
        ppc = fitted_model.ppc_pvalues(seed=42)
        assert isinstance(ppc, dict)
        for stat_name, stat in ppc.items():
            assert isinstance(stat_name, str)
            assert isinstance(stat, PPCStatistic)

    def test_p_values_in_range(self, fitted_model: NonPairedBayesPropTest) -> None:
        ppc = fitted_model.ppc_pvalues(seed=42)
        for stat in ppc.values():
            assert 0.0 <= stat.p_value <= 1.0

    def test_status_is_valid(self, fitted_model: NonPairedBayesPropTest) -> None:
        ppc = fitted_model.ppc_pvalues(seed=42)
        for stat in ppc.values():
            assert stat.status in ("OK", "WARN")


class TestNonPairedPlots:
    """Tests for NonPairedBayesPropTest plotting methods (smoke tests)."""

    def test_plot_posteriors(self, fitted_model: NonPairedBayesPropTest) -> None:
        fitted_model.plot_posteriors()
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_posterior_delta(self, fitted_model: NonPairedBayesPropTest) -> None:
        fitted_model.plot_posterior_delta()
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_savage_dickey(self, fitted_model: NonPairedBayesPropTest) -> None:
        fitted_model.plot_savage_dickey()
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_print_summary(self, fitted_model: NonPairedBayesPropTest, capsys: pytest.CaptureFixture[str]) -> None:
        fitted_model.print_summary()
        captured = capsys.readouterr()
        assert "P(A > B)" in captured.out


class TestNonPairedStatic:
    """Tests for NonPairedBayesPropTest static methods."""

    def test_plot_forest(self, fitted_model: NonPairedBayesPropTest) -> None:
        results = {"metric1": fitted_model}
        NonPairedBayesPropTest.plot_forest(results)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_print_comparison_table(
        self, fitted_model: NonPairedBayesPropTest, capsys: pytest.CaptureFixture[str]
    ) -> None:
        results = {"metric1": fitted_model}
        NonPairedBayesPropTest.print_comparison_table(results)
        captured = capsys.readouterr()
        assert "metric1" in captured.out


# ── rope_test / decide ────────────────────────────────────────


class TestNonPairedRopeTest:
    """Tests for NonPairedBayesPropTest.rope_test method."""

    def test_returns_rope_result(self, fitted_model: NonPairedBayesPropTest) -> None:
        r = fitted_model.rope_test()
        assert isinstance(r, ROPEResult)

    def test_rope_bounds_match_epsilon(self) -> None:
        """Default rope matches rope_epsilon."""
        rng = np.random.default_rng(42)
        y_a = rng.binomial(1, 0.7, size=60).astype(float)
        y_b = rng.binomial(1, 0.7, size=60).astype(float)
        m = NonPairedBayesPropTest(seed=42, rope_epsilon=0.05).fit(y_a, y_b)
        r = m.rope_test()
        assert r.rope_lower == pytest.approx(-0.05)
        assert r.rope_upper == pytest.approx(0.05)

    def test_custom_rope_override(self, fitted_model: NonPairedBayesPropTest) -> None:
        r = fitted_model.rope_test(rope=(-0.10, 0.10))
        assert r.rope_lower == pytest.approx(-0.10)
        assert r.rope_upper == pytest.approx(0.10)

    def test_large_effect_rejects(self) -> None:
        """With a clear effect and tight ROPE, should reject."""
        rng = np.random.default_rng(42)
        y_a = rng.binomial(1, 0.9, size=200).astype(float)
        y_b = rng.binomial(1, 0.5, size=200).astype(float)
        m = NonPairedBayesPropTest(seed=42, n_samples=10_000).fit(y_a, y_b)
        r = m.rope_test(rope=(-0.02, 0.02))
        assert "Reject" in r.decision


class TestNonPairedDecide:
    """Tests for NonPairedBayesPropTest.decide method."""

    def test_returns_hypothesis_decision(self, fitted_model: NonPairedBayesPropTest) -> None:
        d = fitted_model.decide()
        assert isinstance(d, HypothesisDecision)

    def test_all_rule_populates_all(self, fitted_model: NonPairedBayesPropTest) -> None:
        d = fitted_model.decide(rule="all")
        assert d.bayes_factor is not None
        assert d.posterior_null is not None
        assert d.rope is not None
        assert d.rule == "all"

    def test_bf_only(self, fitted_model: NonPairedBayesPropTest) -> None:
        d = fitted_model.decide(rule="bayes_factor")
        assert d.bayes_factor is not None
        assert d.posterior_null is None
        assert d.rope is None

    def test_rope_only(self, fitted_model: NonPairedBayesPropTest) -> None:
        d = fitted_model.decide(rule="rope")
        assert d.rope is not None
        assert d.bayes_factor is None
        assert d.posterior_null is None

    def test_posterior_null_includes_bf(self, fitted_model: NonPairedBayesPropTest) -> None:
        """posterior_null rule needs BF internally, so BF should be populated."""
        d = fitted_model.decide(rule="posterior_null")
        assert d.bayes_factor is not None
        assert d.posterior_null is not None
        assert d.rope is None


# ── descriptive_summary ──────────────────────────────────────


class TestDescriptiveSummary:
    """Tests for the descriptive_summary function."""

    def test_returns_dataframe(self) -> None:
        scores_data = {
            "model_A": "A",
            "model_B": "B",
            "metrics": {
                "test_metric": {
                    "s_A_raw": [0.9, 0.8, 0.7, 0.6, 0.5],
                    "s_B_raw": [0.8, 0.7, 0.6, 0.5, 0.4],
                }
            },
        }
        df = descriptive_summary(scores_data, thresholds=[0.5, 0.7])
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ── DGP recovery tests ───────────────────────────────────────


class TestNonPairedDGPRecovery:
    """Verify that posterior estimates recover the true DGP parameters.

    These tests simulate data from known parameters, fit the model, and
    check that the posterior 95% CI covers the true value and the point
    estimate is close.
    """

    @pytest.mark.parametrize(
        ("theta_A", "theta_B", "N"),
        [
            (0.80, 0.60, 200),
            (0.70, 0.70, 300),  # null effect
            (0.90, 0.50, 150),  # large effect
        ],
    )
    def test_true_delta_in_ci(self, theta_A: float, theta_B: float, N: int) -> None:
        """True Δ = θ_A − θ_B should fall inside the 95% CI."""
        from bayesAB.utils.utils import simulate_nonpaired_scores

        sim = simulate_nonpaired_scores(N=N, theta_A=theta_A, theta_B=theta_B, seed=42)
        model = NonPairedBayesPropTest(seed=42, n_samples=50_000).fit(sim.y_A, sim.y_B)

        true_delta = theta_A - theta_B
        ci = model.summary.ci_95
        assert ci.lower <= true_delta <= ci.upper, (
            f"True Δ={true_delta:.3f} not in 95% CI [{ci.lower:.3f}, {ci.upper:.3f}]"
        )

    def test_theta_posteriors_cover_truth(self) -> None:
        """Individual θ_A and θ_B posterior means should be close to truth."""
        from bayesAB.utils.utils import simulate_nonpaired_scores

        theta_A, theta_B = 0.80, 0.55
        sim = simulate_nonpaired_scores(N=300, theta_A=theta_A, theta_B=theta_B, seed=7)
        model = NonPairedBayesPropTest(seed=7, n_samples=50_000).fit(sim.y_A, sim.y_B)

        assert abs(model.summary.theta_A_mean - theta_A) < 0.08
        assert abs(model.summary.theta_B_mean - theta_B) < 0.08

    def test_null_effect_not_rejected(self) -> None:
        """Under H₀ (θ_A = θ_B), BF should not reject."""
        from bayesAB.utils.utils import simulate_nonpaired_scores

        sim = simulate_nonpaired_scores(N=200, theta_A=0.65, theta_B=0.65, seed=42)
        model = NonPairedBayesPropTest(seed=42, n_samples=50_000).fit(sim.y_A, sim.y_B)
        bf = model.savage_dickey_test()
        assert bf.decision == "Fail to reject H0"

    def test_large_effect_rejected(self) -> None:
        """With a large true effect (Δ=0.35), BF should reject H₀."""
        from bayesAB.utils.utils import simulate_nonpaired_scores

        sim = simulate_nonpaired_scores(N=200, theta_A=0.85, theta_B=0.50, seed=42)
        model = NonPairedBayesPropTest(seed=42, n_samples=50_000).fit(sim.y_A, sim.y_B)
        bf = model.savage_dickey_test()
        assert bf.BF_10 > 10
        assert bf.decision == "Reject H0"

    def test_mean_delta_close_to_truth(self) -> None:
        """Posterior mean Δ should be within 0.08 of the true value."""
        from bayesAB.utils.utils import simulate_nonpaired_scores

        theta_A, theta_B = 0.75, 0.60
        sim = simulate_nonpaired_scores(N=300, theta_A=theta_A, theta_B=theta_B, seed=99)
        model = NonPairedBayesPropTest(seed=99, n_samples=50_000).fit(sim.y_A, sim.y_B)

        true_delta = theta_A - theta_B
        assert abs(model.summary.mean_delta - true_delta) < 0.08
