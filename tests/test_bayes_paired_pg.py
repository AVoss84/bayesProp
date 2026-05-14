"""Unit tests for bayesprop.resources.bayes_paired_pg module."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from bayesprop.resources.bayes_paired_laplace import PairedBayesPropTest
from bayesprop.resources.bayes_paired_pg import (
    PairedBayesPropTestPG,
    _build_design_matrix,
    _format_bf,
    sigmoid,
)
from bayesprop.resources.data_schemas import (
    HypothesisDecision,
    MCMCDiagnostics,
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

    def test_small_positive(self) -> None:
        result = _format_bf(1e-6)
        assert "10^" in result


# ── PairedBayesPropTestPG ────────────────────────────────────


class TestPairedPGInit:
    """Tests for PairedBayesPropTestPG initialization."""

    def test_default_params(self) -> None:
        model = PairedBayesPropTestPG()
        assert model.prior_sigma_delta == 1.0
        assert model.prior_sigma_mu == 2.0
        assert model.n_iter == 1000
        assert model.burn_in == 200
        assert model.n_chains == 2

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

    def test_check_fitted_raises(self) -> None:
        model = PairedBayesPropTestPG()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.savage_dickey_test()


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

    def test_strong_effect_rejects_h0(self) -> None:
        """Large effect → BF₁₀ > 3 → 'Reject H0'."""
        rng = np.random.default_rng(99)
        y_a = rng.binomial(1, 0.95, size=200)
        y_b = rng.binomial(1, 0.3, size=200)
        m = PairedBayesPropTestPG(seed=99, n_iter=600, burn_in=100, n_chains=2).fit(y_a, y_b)
        bf = m.savage_dickey_test()
        assert bf.BF_10 > 3
        assert bf.decision == "Reject H0"


class TestPairedPGPosteriorProbH0:
    """Tests for PairedBayesPropTestPG.posterior_probability_H0()."""

    def test_returns_correct_type(self) -> None:
        result = PairedBayesPropTestPG.posterior_probability_H0(1.0, prior_H0=0.5)
        assert isinstance(result, PosteriorProbH0Result)

    def test_strong_evidence_for_h0(self) -> None:
        result = PairedBayesPropTestPG.posterior_probability_H0(100.0, prior_H0=0.5)
        assert result.p_H0 > 0.95
        assert result.decision == "Fail to reject H0"

    def test_strong_evidence_against_h0(self) -> None:
        result = PairedBayesPropTestPG.posterior_probability_H0(0.01, prior_H0=0.5)
        assert result.p_H1 > 0.95
        assert result.decision == "Reject H0"

    def test_undecided(self) -> None:
        result = PairedBayesPropTestPG.posterior_probability_H0(1.0, prior_H0=0.5)
        assert result.decision == "Undecided"


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


class TestPairedPGRopeTest:
    """Tests for PairedBayesPropTestPG.rope_test method."""

    def test_returns_rope_result(self, fitted_model: PairedBayesPropTestPG) -> None:
        r = fitted_model.rope_test()
        assert isinstance(r, ROPEResult)

    def test_delta_samples_stored(self, fitted_model: PairedBayesPropTestPG) -> None:
        """fit() should store delta_samples for rope_test."""
        assert fitted_model.delta_samples is not None
        assert len(fitted_model.delta_samples) > 0

    def test_custom_rope(self, fitted_model: PairedBayesPropTestPG) -> None:
        r = fitted_model.rope_test(rope=(-0.10, 0.10))
        assert r.rope_lower == pytest.approx(-0.10)
        assert r.rope_upper == pytest.approx(0.10)


class TestPairedPGDecide:
    """Tests for PairedBayesPropTestPG.decide method."""

    def test_returns_hypothesis_decision(self, fitted_model: PairedBayesPropTestPG) -> None:
        d = fitted_model.decide()
        assert isinstance(d, HypothesisDecision)

    def test_all_rule_populates_all(self, fitted_model: PairedBayesPropTestPG) -> None:
        d = fitted_model.decide(rule="all")
        assert d.bayes_factor is not None
        assert d.posterior_null is not None
        assert d.rope is not None

    def test_bf_only(self, fitted_model: PairedBayesPropTestPG) -> None:
        d = fitted_model.decide(rule="bayes_factor")
        assert d.bayes_factor is not None
        assert d.rope is None

    def test_rope_only(self, fitted_model: PairedBayesPropTestPG) -> None:
        d = fitted_model.decide(rule="rope")
        assert d.rope is not None
        assert d.bayes_factor is None


class TestPairedPGStatic:
    """Tests for PairedBayesPropTestPG static methods."""

    def test_simulate_paired_scores(self) -> None:
        from bayesprop.utils.utils import simulate_paired_scores

        sim = simulate_paired_scores(N=30, seed=0)
        assert hasattr(sim, "y_A")
        assert len(sim.y_A) == 30
        assert len(sim.y_B) == 30

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


# ── DGP recovery tests ───────────────────────────────────────


class TestPairedPGDGPRecovery:
    """Verify that Pólya-Gamma Gibbs posterior estimates recover the true DGP parameters.

    The PG model is a fixed-effects logistic regression:
        y_A ~ Bern(σ(μ + δ_A)),  y_B ~ Bern(σ(μ)).
    We use :func:`simulate_paired_scores` (which now defaults to
    ``sigma_theta=0``, matching the model) for parameter recovery tests.

    For decision-rule tests (BF) we pass ``sigma_theta > 0`` to add
    random effects — those tests verify directional correctness rather
    than point-estimate recovery.
    """

    @pytest.mark.parametrize(
        ("mu", "delta_A", "N"),
        [
            (0.0, 0.5, 500),
            (0.0, 1.0, 400),  # larger effect
            (0.0, 0.0, 500),  # null effect
            (0.5, 0.8, 500),  # non-zero intercept
        ],
    )
    def test_delta_A_posterior_covers_truth(self, mu: float, delta_A: float, N: int) -> None:
        """Logit-scale δ_A posterior samples should cover the true value (95% CI)."""
        from bayesprop.utils.utils import simulate_paired_scores

        sim = simulate_paired_scores(N=N, mu=mu, delta_A=delta_A, seed=42)
        model = PairedBayesPropTestPG(seed=42, n_iter=2000, burn_in=500, n_chains=4).fit(sim.y_A, sim.y_B)

        # CI on the logit-scale δ_A
        lo = float(np.quantile(model.delta_A_samples, 0.025))
        hi = float(np.quantile(model.delta_A_samples, 0.975))
        assert lo <= delta_A <= hi, f"True δ_A={delta_A:.3f} not in 95% CI [{lo:.3f}, {hi:.3f}]"

    def test_mean_delta_A_close_to_truth(self) -> None:
        """Posterior mean of logit-scale δ_A should be close to the true value."""
        from bayesprop.utils.utils import simulate_paired_scores

        delta_A = 0.6
        sim = simulate_paired_scores(N=500, delta_A=delta_A, seed=99)
        model = PairedBayesPropTestPG(seed=99, n_iter=2000, burn_in=500, n_chains=4).fit(sim.y_A, sim.y_B)

        assert abs(model.summary.delta_A_posterior_mean - delta_A) < 0.25

    def test_prob_delta_covers_truth(self) -> None:
        """Probability-scale Δ should cover the true value derived from the DGP."""
        from bayesprop.utils.utils import simulate_paired_scores

        mu, delta_A = 0.0, 0.5
        true_delta_prob = sigmoid(mu + delta_A) - sigmoid(mu)

        sim = simulate_paired_scores(N=500, mu=mu, delta_A=delta_A, seed=42)
        model = PairedBayesPropTestPG(seed=42, n_iter=2000, burn_in=500, n_chains=4).fit(sim.y_A, sim.y_B)

        ci = model.summary.ci_95
        assert ci.lower <= true_delta_prob <= ci.upper

    def test_null_effect_not_rejected(self) -> None:
        """Under H₀ (δ_A = 0), BF should not reject."""
        from bayesprop.utils.utils import simulate_paired_scores

        sim = simulate_paired_scores(N=300, delta_A=0.0, sigma_theta=2.0, seed=42)
        model = PairedBayesPropTestPG(seed=42, n_iter=2000, burn_in=500, n_chains=4).fit(sim.y_A, sim.y_B)
        bf = model.savage_dickey_test()
        assert bf.decision == "Fail to reject H0"

    def test_large_effect_rejected(self) -> None:
        """With a large true effect (δ_A=1.5), BF should reject H₀."""
        from bayesprop.utils.utils import simulate_paired_scores

        sim = simulate_paired_scores(N=300, delta_A=1.5, sigma_theta=2.0, seed=42)
        model = PairedBayesPropTestPG(seed=42, n_iter=2000, burn_in=500, n_chains=4).fit(sim.y_A, sim.y_B)
        bf = model.savage_dickey_test()
        assert bf.BF_10 > 10
        assert bf.decision == "Reject H0"

    def test_chains_converge(self) -> None:
        """R-hat should be close to 1 for a well-behaved DGP with sufficient data."""
        from bayesprop.utils.utils import simulate_paired_scores

        sim = simulate_paired_scores(N=500, delta_A=0.5, seed=42)
        model = PairedBayesPropTestPG(seed=42, n_iter=2000, burn_in=500, n_chains=4).fit(sim.y_A, sim.y_B)

        diag = model.mcmc_diagnostics()
        assert diag.delta_A.r_hat < 1.1, f"R-hat for δ_A = {diag.delta_A.r_hat:.3f}"
        assert diag.mu.r_hat < 1.1, f"R-hat for μ = {diag.mu.r_hat:.3f}"

    def test_pg_and_laplace_agree(self) -> None:
        """PG and Laplace posterior means for δ_A should be within 0.25 of each other."""
        from bayesprop.utils.utils import simulate_paired_scores

        sim = simulate_paired_scores(N=500, delta_A=0.5, seed=42)

        pg_model = PairedBayesPropTestPG(seed=42, n_iter=2000, burn_in=500, n_chains=4).fit(sim.y_A, sim.y_B)

        laplace_model = PairedBayesPropTest(seed=42, n_samples=10_000).fit(sim.y_A, sim.y_B)

        assert abs(pg_model.summary.delta_A_posterior_mean - laplace_model.summary.delta_A_posterior_mean) < 0.25
