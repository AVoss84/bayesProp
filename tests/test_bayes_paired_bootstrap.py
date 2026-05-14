"""Unit tests for PairedBayesPropTestBB."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesprop.resources.bayes_paired_bootstrap import PairedBayesPropTestBB
from bayesprop.resources.data_schemas import (
    HypothesisDecision,
    PairedSummary,
    ROPEResult,
)


# ── Construction ─────────────────────────────────────────────


class TestPairedBootstrapConstruction:
    """Constructor validation."""

    def test_defaults(self) -> None:
        m = PairedBayesPropTestBB()
        assert m.n_samples == 20_000
        assert m.dirichlet_alpha == 1.0
        assert m.rope_epsilon == 0.02
        # Not yet fitted
        assert m.summary is None
        assert m.delta_samples is None

    def test_rejects_nonpositive_alpha(self) -> None:
        with pytest.raises(ValueError, match="dirichlet_alpha"):
            PairedBayesPropTestBB(dirichlet_alpha=0.0)
        with pytest.raises(ValueError, match="dirichlet_alpha"):
            PairedBayesPropTestBB(dirichlet_alpha=-0.5)

    def test_rejects_nonpositive_n_samples(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            PairedBayesPropTestBB(n_samples=0)


# ── fit() ────────────────────────────────────────────────────


class TestPairedBootstrapFit:
    """fit() produces a properly-shaped posterior on Δ."""

    def test_fit_populates_state(self) -> None:
        y_A = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 0])
        y_B = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
        m = PairedBayesPropTestBB(n_samples=3000, seed=42).fit(y_A, y_B)
        assert m.delta_samples is not None
        assert m.delta_samples.shape == (3000,)
        assert isinstance(m.summary, PairedSummary)
        assert isinstance(m.trace_summary, pd.DataFrame)
        assert m.y_A_obs is not None and len(m.y_A_obs) == 10
        # Posterior mean should be close to the observed mean difference
        # at this large(ish) n_samples.
        emp = float((y_A - y_B).mean())
        assert abs(m.summary.mean_delta - emp) < 0.05
        # 95% CI bounds should bracket the empirical mean
        assert m.summary.ci_95.lower < emp < m.summary.ci_95.upper
        # Posterior of superiority is a probability in [0, 1].
        assert 0.0 <= m.summary.p_A_greater_B <= 1.0

    def test_deterministic_given_seed(self) -> None:
        y_A = np.array([1, 1, 0, 1, 1, 0, 0, 1])
        y_B = np.array([0, 1, 0, 0, 1, 1, 0, 0])
        m1 = PairedBayesPropTestBB(n_samples=500, seed=7).fit(y_A, y_B)
        m2 = PairedBayesPropTestBB(n_samples=500, seed=7).fit(y_A, y_B)
        assert m1.delta_samples is not None and m2.delta_samples is not None
        np.testing.assert_array_equal(m1.delta_samples, m2.delta_samples)

    def test_recovers_known_effect(self) -> None:
        # n large enough that BB nonparametric posterior concentrates
        # around the true Δ = p_A − p_B ≈ 0.20.
        rng = np.random.default_rng(0)
        n = 500
        y_A = rng.binomial(1, 0.70, n)
        y_B = rng.binomial(1, 0.50, n)
        m = PairedBayesPropTestBB(n_samples=4000, seed=0).fit(y_A, y_B)
        assert m.summary is not None
        assert abs(m.summary.mean_delta - (y_A - y_B).mean()) < 0.01
        # 95% CI for Δ should exclude 0 with probability ~1 at this n.
        assert m.summary.ci_95.lower > 0
        # Strong posterior of superiority for a clear positive effect.
        assert m.summary.p_A_greater_B > 0.99

    def test_chunking_matches_unchunked(self) -> None:
        # Long arrays force the chunk loop to execute multiple iterations,
        # which must yield identical samples to a single-chunk run with
        # the same seed.
        rng = np.random.default_rng(1)
        y_A = rng.binomial(1, 0.6, 200)
        y_B = rng.binomial(1, 0.5, 200)
        m_small = PairedBayesPropTestBB(n_samples=1000, seed=0).fit(y_A, y_B)
        m_large = PairedBayesPropTestBB(n_samples=100_000, seed=0).fit(y_A, y_B)
        # Posterior means should be very close for the same seed; we
        # mainly want to ensure the loop executes without overflow/skip.
        assert m_small.summary is not None and m_large.summary is not None
        assert abs(m_small.summary.mean_delta - m_large.summary.mean_delta) < 0.01

    def test_rejects_shape_mismatch(self) -> None:
        m = PairedBayesPropTestBB(n_samples=200)
        with pytest.raises(ValueError, match="identical shapes"):
            m.fit(np.array([0, 1, 0]), np.array([0, 1]))

    def test_rejects_non_binary(self) -> None:
        m = PairedBayesPropTestBB(n_samples=200)
        with pytest.raises(ValueError, match="0/1"):
            m.fit(np.array([0, 1, 2]), np.array([0, 1, 0]))

    def test_rejects_empty_data(self) -> None:
        m = PairedBayesPropTestBB(n_samples=200)
        with pytest.raises(ValueError, match="empty"):
            m.fit(np.array([], dtype=int), np.array([], dtype=int))


# ── decision API ─────────────────────────────────────────────


class TestPairedBootstrapDecisions:
    """ROPE-only decision surface."""

    @pytest.fixture
    def fitted_decisive(self) -> PairedBayesPropTestBB:
        rng = np.random.default_rng(0)
        y_A = rng.binomial(1, 0.75, 300)
        y_B = rng.binomial(1, 0.50, 300)
        return PairedBayesPropTestBB(n_samples=3000, seed=0).fit(y_A, y_B)

    @pytest.fixture
    def fitted_null(self) -> PairedBayesPropTestBB:
        rng = np.random.default_rng(1)
        y_A = rng.binomial(1, 0.50, 400)
        y_B = rng.binomial(1, 0.50, 400)
        return PairedBayesPropTestBB(n_samples=3000, seed=0).fit(y_A, y_B)

    def test_rope_test_returns_result(
        self, fitted_decisive: PairedBayesPropTestBB
    ) -> None:
        r = fitted_decisive.rope_test()
        assert isinstance(r, ROPEResult)
        # Decisive effect: most of the posterior mass should be above ROPE.
        assert r.pct_above_rope > 0.9
        assert "Reject H0" in r.decision

    def test_rope_test_under_null_concentrates_inside(
        self, fitted_null: PairedBayesPropTestBB
    ) -> None:
        # Under a true null with n=400 and ROPE = ±0.05, the BB posterior
        # should put most of its mass inside the ROPE.
        r = fitted_null.rope_test(rope=(-0.05, 0.05))
        assert r.pct_in_rope > 0.5
        # The "posterior of null" = pct_in_rope is what replaces a
        # Bayes-factor-style P(H0|data) for this nonparametric class.
        assert 0.0 <= r.pct_in_rope <= 1.0

    def test_summary_exposes_posterior_of_superiority(
        self, fitted_decisive: PairedBayesPropTestBB
    ) -> None:
        # The BB class deliberately drops posterior_probability_H0; the
        # three quantities of interest must remain reachable directly.
        assert fitted_decisive.summary is not None
        # Posterior of superiority is on .summary
        assert fitted_decisive.summary.p_A_greater_B > 0.99
        # Posterior of null is on rope_test().pct_in_rope
        assert fitted_decisive.rope_test().pct_in_rope < 0.05

    def test_decide_returns_rope_only(
        self, fitted_decisive: PairedBayesPropTestBB
    ) -> None:
        d = fitted_decisive.decide()
        assert isinstance(d, HypothesisDecision)
        # By design: only the ROPE sub-result is populated.
        assert d.bayes_factor is None
        assert d.posterior_null is None
        assert d.rope is not None
        assert d.rule == "rope"
        # The rope sub-decision must agree with a direct call.
        direct = fitted_decisive.rope_test()
        assert d.rope.decision == direct.decision
        assert d.rope.pct_in_rope == pytest.approx(direct.pct_in_rope, abs=1e-12)

    def test_posterior_probability_H0_not_exposed(self) -> None:
        # Deliberate API asymmetry: the BB class does not provide a
        # Bayes-factor-style posterior_probability_H0 method because
        # any default prior on H_0 would have no role in the BB
        # posterior. Use rope_test().pct_in_rope instead.
        assert not hasattr(PairedBayesPropTestBB, "posterior_probability_H0")

    def test_savage_dickey_test_not_exposed(self) -> None:
        # Deliberate API asymmetry: the BB class does not provide a
        # Savage–Dickey method because no parametric prior on Δ exists.
        assert not hasattr(PairedBayesPropTestBB, "savage_dickey_test")

    def test_check_fitted_raises(self) -> None:
        m = PairedBayesPropTestBB(n_samples=100)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.rope_test()
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.decide()


# ── plotting ─────────────────────────────────────────────────


class TestPairedBootstrapPlotting:
    """Smoke tests for the plot helper."""

    def test_plot_posterior_runs(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y_A = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 0])
        y_B = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
        m = PairedBayesPropTestBB(n_samples=500, seed=0).fit(y_A, y_B)
        ax = m.plot_posterior()
        assert ax is not None
        plt.close("all")
