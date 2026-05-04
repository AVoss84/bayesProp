"""Unit tests for bayesAB.resources.bayes_paired_laplace module."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from bayesAB.resources.bayes_paired_laplace import (
    PairedBayesPropTest,
    _format_bf,
    sigmoid,
)
from bayesAB.resources.data_schemas import (
    HypothesisDecision,
    PairedSummary,
    PosteriorProbH0Result,
    PPCStatistic,
    ROPEResult,
    SavageDickeyResult,
)

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def paired_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible paired binary data."""
    rng = np.random.default_rng(42)
    y_a = rng.binomial(1, 0.8, size=60)
    y_b = rng.binomial(1, 0.6, size=60)
    return y_a, y_b


@pytest.fixture
def fitted_model(paired_data: tuple[np.ndarray, np.ndarray]) -> PairedBayesPropTest:
    """Return a fitted PairedBayesPropTest model."""
    y_a, y_b = paired_data
    return PairedBayesPropTest(seed=42, n_samples=5000).fit(y_a, y_b)


# ── sigmoid ───────────────────────────────────────────────────


class TestSigmoid:
    """Tests for the sigmoid function."""

    def test_midpoint(self) -> None:
        assert abs(sigmoid(0.0) - 0.5) < 1e-10

    def test_extreme_positive(self) -> None:
        assert sigmoid(100.0) > 0.999

    def test_extreme_negative(self) -> None:
        assert sigmoid(-100.0) < 0.001

    def test_array_input(self) -> None:
        result = sigmoid(np.array([-10.0, 0.0, 10.0]))
        assert result.shape == (3,)
        assert result[1] == pytest.approx(0.5)


# ── _format_bf ────────────────────────────────────────────────


class TestFormatBf:
    """Tests for the _format_bf helper."""

    def test_normal_value(self) -> None:
        assert _format_bf(2.5) == "2.50"

    def test_large_value(self) -> None:
        assert "10^" in _format_bf(1e5)


# ── PairedBayesPropTest ──────────────────────────────────────


class TestPairedLaplaceInit:
    """Tests for PairedBayesPropTest initialization."""

    def test_default_params(self) -> None:
        model = PairedBayesPropTest()
        assert model.prior_sigma_delta == 1.0
        assert model.n_samples == 8000

    def test_custom_params(self) -> None:
        model = PairedBayesPropTest(prior_sigma_delta=2.0, seed=99, n_samples=3000)
        assert model.prior_sigma_delta == 2.0
        assert model.seed == 99


class TestPairedLaplaceFit:
    """Tests for PairedBayesPropTest.fit()."""

    def test_returns_self(self, paired_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_a, y_b = paired_data
        model = PairedBayesPropTest(seed=42, n_samples=3000)
        result = model.fit(y_a, y_b)
        assert result is model

    def test_summary_type(self, fitted_model: PairedBayesPropTest) -> None:
        assert isinstance(fitted_model.summary, PairedSummary)

    def test_summary_fields(self, fitted_model: PairedBayesPropTest) -> None:
        s = fitted_model.summary
        assert -1.0 <= s.mean_delta <= 1.0
        assert s.ci_95.lower < s.ci_95.upper
        assert 0.0 <= s.p_A_greater_B <= 1.0

    def test_delta_a_samples_populated(self, fitted_model: PairedBayesPropTest) -> None:
        assert fitted_model.delta_A_samples is not None
        assert len(fitted_model.delta_A_samples) > 0

    def test_laplace_dict_populated(self, fitted_model: PairedBayesPropTest) -> None:
        assert fitted_model.laplace is not None
        assert "map" in fitted_model.laplace
        assert "cov" in fitted_model.laplace

    def test_trace_summary_is_dataframe(self, fitted_model: PairedBayesPropTest) -> None:
        assert isinstance(fitted_model.trace_summary, pd.DataFrame)

    def test_known_effect_detected(self) -> None:
        rng = np.random.default_rng(99)
        y_a = rng.binomial(1, 0.9, size=200)
        y_b = rng.binomial(1, 0.4, size=200)
        model = PairedBayesPropTest(seed=99, n_samples=5000).fit(y_a, y_b)
        assert model.summary.p_A_greater_B > 0.95


class TestPairedLaplaceSavageDickey:
    """Tests for PairedBayesPropTest.savage_dickey_test()."""

    def test_returns_savage_dickey_result(self, fitted_model: PairedBayesPropTest) -> None:
        result = fitted_model.savage_dickey_test()
        assert isinstance(result, SavageDickeyResult)

    def test_bf_reciprocal(self, fitted_model: PairedBayesPropTest) -> None:
        result = fitted_model.savage_dickey_test()
        assert abs(result.BF_01 * result.BF_10 - 1.0) < 0.01


class TestPairedLaplacePosteriorProbH0:
    """Tests for PairedBayesPropTest.posterior_probability_H0()."""

    def test_returns_correct_type(self) -> None:
        result = PairedBayesPropTest.posterior_probability_H0(1.0, prior_H0=0.5)
        assert isinstance(result, PosteriorProbH0Result)

    def test_equal_bf(self) -> None:
        result = PairedBayesPropTest.posterior_probability_H0(1.0, prior_H0=0.5)
        assert abs(result.p_H0 - 0.5) < 1e-6


class TestPairedLaplacePPC:
    """Tests for PairedBayesPropTest.ppc_pvalues()."""

    def test_returns_dict_of_ppc(self, fitted_model: PairedBayesPropTest) -> None:
        ppc = fitted_model.ppc_pvalues(seed=42)
        assert isinstance(ppc, dict)
        for stat in ppc.values():
            assert isinstance(stat, PPCStatistic)
            assert 0.0 <= stat.p_value <= 1.0


class TestPairedLaplacePlots:
    """Smoke tests for PairedBayesPropTest plotting methods."""

    def test_plot_posterior_delta(self, fitted_model: PairedBayesPropTest) -> None:
        fitted_model.plot_posterior_delta()
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_savage_dickey(self, fitted_model: PairedBayesPropTest) -> None:
        fitted_model.plot_savage_dickey()
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_laplace_posterior(self, fitted_model: PairedBayesPropTest) -> None:
        fitted_model.plot_laplace_posterior()
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_print_summary(self, fitted_model: PairedBayesPropTest, capsys: pytest.CaptureFixture[str]) -> None:
        fitted_model.print_summary()
        captured = capsys.readouterr()
        assert "P(A > B)" in captured.out


class TestPairedLaplaceRopeTest:
    """Tests for PairedBayesPropTest.rope_test method."""

    def test_returns_rope_result(self, fitted_model: PairedBayesPropTest) -> None:
        r = fitted_model.rope_test()
        assert isinstance(r, ROPEResult)

    def test_delta_samples_stored(self, fitted_model: PairedBayesPropTest) -> None:
        """fit() should store delta_samples for rope_test."""
        assert fitted_model.delta_samples is not None
        assert len(fitted_model.delta_samples) > 0

    def test_custom_rope(self, fitted_model: PairedBayesPropTest) -> None:
        r = fitted_model.rope_test(rope=(-0.10, 0.10))
        assert r.rope_lower == pytest.approx(-0.10)
        assert r.rope_upper == pytest.approx(0.10)


class TestPairedLaplaceDecide:
    """Tests for PairedBayesPropTest.decide method."""

    def test_returns_hypothesis_decision(self, fitted_model: PairedBayesPropTest) -> None:
        d = fitted_model.decide()
        assert isinstance(d, HypothesisDecision)

    def test_all_rule_populates_all(self, fitted_model: PairedBayesPropTest) -> None:
        d = fitted_model.decide(rule="all")
        assert d.bayes_factor is not None
        assert d.posterior_null is not None
        assert d.rope is not None

    def test_bf_only(self, fitted_model: PairedBayesPropTest) -> None:
        d = fitted_model.decide(rule="bayes_factor")
        assert d.bayes_factor is not None
        assert d.rope is None

    def test_rope_only(self, fitted_model: PairedBayesPropTest) -> None:
        d = fitted_model.decide(rule="rope")
        assert d.rope is not None
        assert d.bayes_factor is None


class TestPairedLaplaceStatic:
    """Tests for PairedBayesPropTest static methods."""

    def test_simulate_paired_scores(self) -> None:
        from bayesAB.utils.utils import simulate_paired_scores

        sim = simulate_paired_scores(N=50, seed=0)
        assert isinstance(sim, dict)
        assert len(sim["y_A"]) == 50
        assert len(sim["y_B"]) == 50

    def test_plot_forest(self, fitted_model: PairedBayesPropTest) -> None:
        results = {"metric1": fitted_model}
        PairedBayesPropTest.plot_forest(results)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_print_comparison_table(
        self, fitted_model: PairedBayesPropTest, capsys: pytest.CaptureFixture[str]
    ) -> None:
        results = {"metric1": fitted_model}
        PairedBayesPropTest.print_comparison_table(results)
        captured = capsys.readouterr()
        assert "metric1" in captured.out
