"""Unit tests for bayesAB.resources.bayes_paired_pg module."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from bayesAB.resources.bayes_paired_pg import (
    PairedBayesPropTestPG,
    _build_design_matrix,
    _format_bf,
    sigmoid,
)
from bayesAB.resources.data_schemas import (
    MCMCDiagnostics,
    PairedSummary,
    PosteriorProbH0Result,
    PPCStatistic,
    SavageDickeyResult,
)

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def paired_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible paired binary data."""
    rng = np.random.default_rng(42)
    y_a = rng.binomial(1, 0.8, size=50)
    y_b = rng.binomial(1, 0.6, size=50)
    return y_a, y_b


@pytest.fixture
def fitted_model(paired_data: tuple[np.ndarray, np.ndarray]) -> PairedBayesPropTestPG:
    """Return a fitted PairedBayesPropTestPG model with small MCMC budget."""
    y_a, y_b = paired_data
    return PairedBayesPropTestPG(seed=42, n_iter=400, burn_in=100, n_chains=2).fit(y_a, y_b)


# ── Module-level functions ────────────────────────────────────


class TestSigmoid:
    """Tests for the numerically stable sigmoid."""

    def test_midpoint(self) -> None:
        assert abs(sigmoid(0.0) - 0.5) < 1e-10

    def test_large_negative(self) -> None:
        result = sigmoid(-500.0)
        assert result >= 0.0  # no overflow
        assert result < 1e-100

    def test_array(self) -> None:
        result = sigmoid(np.array([-1.0, 0.0, 1.0]))
        assert result.shape == (3,)


class TestBuildDesignMatrix:
    """Tests for _build_design_matrix."""

    def test_shapes(self) -> None:
        y_a = np.array([1, 0, 1])
        y_b = np.array([0, 1, 0])
        X, y = _build_design_matrix(y_a, y_b)
        assert X.shape == (6, 2)
        assert y.shape == (6,)

    def test_design_structure(self) -> None:
        y_a = np.array([1, 0])
        y_b = np.array([0, 1])
        X, y = _build_design_matrix(y_a, y_b)
        # First n rows are model A (delta_A column = 1)
        assert X[0, 1] == 1.0
        assert X[1, 1] == 1.0
        # Last n rows are model B (delta_A column = 0)
        assert X[2, 1] == 0.0
        assert X[3, 1] == 0.0


class TestFormatBf:
    """Tests for the _format_bf helper."""

    def test_normal(self) -> None:
        assert _format_bf(5.0) == "5.00"

    def test_large(self) -> None:
        assert "10^" in _format_bf(1e6)


# ── PairedBayesPropTestPG ────────────────────────────────────


class TestPairedPGInit:
    """Tests for PairedBayesPropTestPG initialization."""

    def test_default_params(self) -> None:
        model = PairedBayesPropTestPG()
        assert model.prior_sigma_delta == 1.0
        assert model.prior_sigma_mu == 2.0
        assert model.n_iter == 2000
        assert model.burn_in == 500
        assert model.n_chains == 4

    def test_custom_params(self) -> None:
        model = PairedBayesPropTestPG(
            prior_sigma_delta=0.5,
            prior_sigma_mu=1.0,
            n_iter=500,
            burn_in=100,
            n_chains=2,
        )
        assert model.prior_sigma_delta == 0.5
        assert model.n_chains == 2


class TestPairedPGFit:
    """Tests for PairedBayesPropTestPG.fit()."""

    def test_returns_self(self, paired_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_a, y_b = paired_data
        model = PairedBayesPropTestPG(seed=42, n_iter=300, burn_in=50, n_chains=2)
        result = model.fit(y_a, y_b)
        assert result is model

    def test_summary_type(self, fitted_model: PairedBayesPropTestPG) -> None:
        assert isinstance(fitted_model.summary, PairedSummary)

    def test_summary_fields(self, fitted_model: PairedBayesPropTestPG) -> None:
        s = fitted_model.summary
        assert -1.0 <= s.mean_delta <= 1.0
        assert s.ci_95.lower < s.ci_95.upper
        assert 0.0 <= s.p_A_greater_B <= 1.0

    def test_chains_shape(self, fitted_model: PairedBayesPropTestPG) -> None:
        assert fitted_model.chains is not None
        assert fitted_model.chains.ndim == 3
        assert fitted_model.chains.shape[0] == 2  # n_chains
        assert fitted_model.chains.shape[2] == 2  # mu, delta_A

    def test_delta_a_samples_populated(self, fitted_model: PairedBayesPropTestPG) -> None:
        assert fitted_model.delta_A_samples is not None
        assert len(fitted_model.delta_A_samples) > 0

    def test_trace_summary_is_dataframe(self, fitted_model: PairedBayesPropTestPG) -> None:
        assert isinstance(fitted_model.trace_summary, pd.DataFrame)


class TestPairedPGMCMCDiagnostics:
    """Tests for PairedBayesPropTestPG.mcmc_diagnostics()."""

    def test_returns_mcmc_diagnostics(self, fitted_model: PairedBayesPropTestPG) -> None:
        diag = fitted_model.mcmc_diagnostics()
        assert isinstance(diag, MCMCDiagnostics)

    def test_r_hat_reasonable(self, fitted_model: PairedBayesPropTestPG) -> None:
        diag = fitted_model.mcmc_diagnostics()
        # R-hat should be close to 1 for converged chains
        assert diag.mu.r_hat < 2.0
        assert diag.delta_A.r_hat < 2.0

    def test_ess_positive(self, fitted_model: PairedBayesPropTestPG) -> None:
        diag = fitted_model.mcmc_diagnostics()
        assert diag.mu.ess > 0
        assert diag.delta_A.ess > 0


class TestPairedPGSavageDickey:
    """Tests for PairedBayesPropTestPG.savage_dickey_test()."""

    def test_returns_savage_dickey_result(self, fitted_model: PairedBayesPropTestPG) -> None:
        result = fitted_model.savage_dickey_test()
        assert isinstance(result, SavageDickeyResult)

    def test_bf_reciprocal(self, fitted_model: PairedBayesPropTestPG) -> None:
        result = fitted_model.savage_dickey_test()
        assert abs(result.BF_01 * result.BF_10 - 1.0) < 0.01


class TestPairedPGPosteriorProbH0:
    """Tests for PairedBayesPropTestPG.posterior_probability_H0()."""

    def test_returns_correct_type(self) -> None:
        result = PairedBayesPropTestPG.posterior_probability_H0(1.0, prior_H0=0.5)
        assert isinstance(result, PosteriorProbH0Result)


class TestPairedPGPPC:
    """Tests for PairedBayesPropTestPG.ppc_pvalues()."""

    def test_returns_dict_of_ppc(self, fitted_model: PairedBayesPropTestPG) -> None:
        ppc = fitted_model.ppc_pvalues(seed=42)
        assert isinstance(ppc, dict)
        for stat in ppc.values():
            assert isinstance(stat, PPCStatistic)
            assert 0.0 <= stat.p_value <= 1.0


class TestPairedPGPlots:
    """Smoke tests for PairedBayesPropTestPG plotting methods."""

    def test_plot_trace(self, fitted_model: PairedBayesPropTestPG) -> None:
        fitted_model.plot_trace()
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_posterior_delta(self, fitted_model: PairedBayesPropTestPG) -> None:
        fitted_model.plot_posterior_delta()
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_savage_dickey(self, fitted_model: PairedBayesPropTestPG) -> None:
        fitted_model.plot_savage_dickey()
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_print_summary(self, fitted_model: PairedBayesPropTestPG, capsys: pytest.CaptureFixture[str]) -> None:
        fitted_model.print_summary()
        captured = capsys.readouterr()
        assert "P(A > B)" in captured.out


class TestPairedPGStatic:
    """Tests for PairedBayesPropTestPG static methods."""

    def test_simulate_paired_scores(self) -> None:
        from bayesAB.utils.utils import simulate_paired_scores

        sim = simulate_paired_scores(N=30, seed=0)
        assert isinstance(sim, dict)
        assert len(sim["y_A"]) == 30
        assert len(sim["y_B"]) == 30

    def test_plot_forest(self, fitted_model: PairedBayesPropTestPG) -> None:
        results = {"metric1": fitted_model}
        PairedBayesPropTestPG.plot_forest(results)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_print_comparison_table(
        self, fitted_model: PairedBayesPropTestPG, capsys: pytest.CaptureFixture[str]
    ) -> None:
        results = {"metric1": fitted_model}
        PairedBayesPropTestPG.print_comparison_table(results)
        captured = capsys.readouterr()
        assert "metric1" in captured.out
