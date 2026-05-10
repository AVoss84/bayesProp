"""Unit tests for bayesprop.resources.bayes_paired_laplace module."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from bayesprop.resources.bayes_paired_laplace import (
    PairedBayesPropTest,
    SequentialPairedBayesPropTest,
    _format_bf,
    _paired_laplace_from_counts,
    sigmoid,
)
from bayesprop.resources.data_schemas import (
    HypothesisDecision,
    PairedSummary,
    PosteriorProbH0Result,
    PPCStatistic,
    ROPEResult,
    SavageDickeyResult,
    SequentialLaplaceLookResult,
    SequentialLaplaceState,
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

    def test_small_positive(self) -> None:
        result = _format_bf(1e-6)
        assert "10^" in result


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

    def test_check_fitted_raises(self) -> None:
        model = PairedBayesPropTest()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.savage_dickey_test()


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

    def test_trace_summary_is_dataframe(
        self, fitted_model: PairedBayesPropTest
    ) -> None:
        assert isinstance(fitted_model.trace_summary, pd.DataFrame)

    def test_known_effect_detected(self) -> None:
        rng = np.random.default_rng(99)
        y_a = rng.binomial(1, 0.9, size=200)
        y_b = rng.binomial(1, 0.4, size=200)
        model = PairedBayesPropTest(seed=99, n_samples=5000).fit(y_a, y_b)
        assert model.summary.p_A_greater_B > 0.95


class TestPairedLaplaceSavageDickey:
    """Tests for PairedBayesPropTest.savage_dickey_test()."""

    def test_returns_savage_dickey_result(
        self, fitted_model: PairedBayesPropTest
    ) -> None:
        result = fitted_model.savage_dickey_test()
        assert isinstance(result, SavageDickeyResult)

    def test_bf_reciprocal(self, fitted_model: PairedBayesPropTest) -> None:
        result = fitted_model.savage_dickey_test()
        assert abs(result.BF_01 * result.BF_10 - 1.0) < 0.01

    def test_strong_effect_rejects_h0(self) -> None:
        """Large effect → BF₁₀ > 3 → 'Reject H0'."""
        rng = np.random.default_rng(99)
        y_a = rng.binomial(1, 0.95, size=200)
        y_b = rng.binomial(1, 0.3, size=200)
        m = PairedBayesPropTest(seed=99, n_samples=5000).fit(y_a, y_b)
        bf = m.savage_dickey_test()
        assert bf.BF_10 > 10
        assert bf.decision == "Reject H0"

    def test_no_effect_fails_to_reject(self) -> None:
        """Equal rates → low BF₁₀ → 'Fail to reject H0'."""
        rng = np.random.default_rng(99)
        y_a = rng.binomial(1, 0.5, size=200)
        y_b = rng.binomial(1, 0.5, size=200)
        m = PairedBayesPropTest(seed=99, n_samples=5000).fit(y_a, y_b)
        bf = m.savage_dickey_test()
        assert bf.decision == "Fail to reject H0"


class TestPairedLaplacePosteriorProbH0:
    """Tests for PairedBayesPropTest.posterior_probability_H0()."""

    def test_returns_correct_type(self) -> None:
        result = PairedBayesPropTest.posterior_probability_H0(1.0, prior_H0=0.5)
        assert isinstance(result, PosteriorProbH0Result)

    def test_equal_bf(self) -> None:
        result = PairedBayesPropTest.posterior_probability_H0(1.0, prior_H0=0.5)
        assert abs(result.p_H0 - 0.5) < 1e-6

    def test_strong_evidence_for_h0(self) -> None:
        result = PairedBayesPropTest.posterior_probability_H0(100.0, prior_H0=0.5)
        assert result.p_H0 > 0.95
        assert result.decision == "Fail to reject H0"

    def test_strong_evidence_against_h0(self) -> None:
        result = PairedBayesPropTest.posterior_probability_H0(0.01, prior_H0=0.5)
        assert result.p_H1 > 0.95
        assert result.decision == "Reject H0"

    def test_undecided(self) -> None:
        result = PairedBayesPropTest.posterior_probability_H0(1.0, prior_H0=0.5)
        assert result.decision == "Undecided"


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

    def test_print_summary(
        self, fitted_model: PairedBayesPropTest, capsys: pytest.CaptureFixture[str]
    ) -> None:
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

    def test_returns_hypothesis_decision(
        self, fitted_model: PairedBayesPropTest
    ) -> None:
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
        from bayesprop.utils.utils import simulate_paired_scores

        sim = simulate_paired_scores(N=50, seed=0)
        assert hasattr(sim, "y_A")
        assert len(sim.y_A) == 50
        assert len(sim.y_B) == 50

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


# ── DGP recovery tests ───────────────────────────────────────


class TestPairedLaplaceDGPRecovery:
    """Verify that Laplace posterior estimates recover the true DGP parameters.

    The Laplace model is a fixed-effects logistic regression:
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
    def test_delta_A_posterior_covers_truth(
        self, mu: float, delta_A: float, N: int
    ) -> None:
        """Logit-scale δ_A posterior samples should cover the true value (95% CI)."""
        from bayesprop.utils.utils import simulate_paired_scores

        sim = simulate_paired_scores(N=N, mu=mu, delta_A=delta_A, seed=42)
        model = PairedBayesPropTest(seed=42, n_samples=10_000).fit(sim.y_A, sim.y_B)

        # CI on the logit-scale δ_A
        lo = float(np.quantile(model.delta_A_samples, 0.025))
        hi = float(np.quantile(model.delta_A_samples, 0.975))
        assert (
            lo <= delta_A <= hi
        ), f"True δ_A={delta_A:.3f} not in 95% CI [{lo:.3f}, {hi:.3f}]"

    def test_mean_delta_A_close_to_truth(self) -> None:
        """Posterior mean of logit-scale δ_A should be close to the true value."""
        from bayesprop.utils.utils import simulate_paired_scores

        delta_A = 0.6
        sim = simulate_paired_scores(N=500, delta_A=delta_A, seed=99)
        model = PairedBayesPropTest(seed=99, n_samples=10_000).fit(sim.y_A, sim.y_B)

        assert abs(model.summary.delta_A_posterior_mean - delta_A) < 0.25

    def test_prob_delta_covers_truth(self) -> None:
        """Probability-scale Δ should cover the true value derived from the DGP."""
        from bayesprop.utils.utils import simulate_paired_scores

        mu, delta_A = 0.0, 0.5
        true_delta_prob = sigmoid(mu + delta_A) - sigmoid(mu)

        sim = simulate_paired_scores(N=500, mu=mu, delta_A=delta_A, seed=42)
        model = PairedBayesPropTest(seed=42, n_samples=10_000).fit(sim.y_A, sim.y_B)

        ci = model.summary.ci_95
        assert ci.lower <= true_delta_prob <= ci.upper

    def test_null_effect_not_rejected(self) -> None:
        """Under H₀ (δ_A = 0), BF should not reject."""
        from bayesprop.utils.utils import simulate_paired_scores

        sim = simulate_paired_scores(N=300, delta_A=0.0, sigma_theta=2.0, seed=42)
        model = PairedBayesPropTest(seed=42, n_samples=10_000).fit(sim.y_A, sim.y_B)
        bf = model.savage_dickey_test()
        assert bf.decision == "Fail to reject H0"

    def test_large_effect_rejected(self) -> None:
        """With a large true effect (δ_A=1.5), BF should reject H₀."""
        from bayesprop.utils.utils import simulate_paired_scores

        sim = simulate_paired_scores(N=300, delta_A=1.5, sigma_theta=2.0, seed=42)
        model = PairedBayesPropTest(seed=42, n_samples=10_000).fit(sim.y_A, sim.y_B)
        bf = model.savage_dickey_test()
        assert bf.BF_10 > 10
        assert bf.decision == "Reject H0"

    def test_map_estimate_recovers_delta(self) -> None:
        """Laplace MAP estimate for δ_A should be close to the true value."""
        from bayesprop.utils.utils import simulate_paired_scores

        delta_A = 0.8
        sim = simulate_paired_scores(N=500, delta_A=delta_A, seed=7)
        model = PairedBayesPropTest(seed=7, n_samples=10_000).fit(sim.y_A, sim.y_B)

        map_delta = model.laplace["map"][1]  # index 1 is delta_A
        assert abs(map_delta - delta_A) < 0.25


# ── _paired_laplace_from_counts ───────────────────────────────


class TestPairedLaplaceFromCounts:
    """Tests for the count-based Laplace helper."""

    def test_matches_batch_fit_on_same_counts(self) -> None:
        """Count-based MAP/cov must match a full PairedBayesPropTest.fit()."""
        rng = np.random.default_rng(0)
        y_a = rng.binomial(1, 0.75, size=200)
        y_b = rng.binomial(1, 0.55, size=200)

        batch = PairedBayesPropTest(prior_sigma_delta=1.0, seed=0, n_samples=2_000).fit(
            y_a, y_b
        )

        theta_map, cov, H = _paired_laplace_from_counts(
            n_A=int(len(y_a)),
            k_A=int(y_a.sum()),
            n_B=int(len(y_b)),
            k_B=int(y_b.sum()),
            prior_sigma_delta=1.0,
        )
        assert theta_map[0] == pytest.approx(batch.laplace["map"][0], abs=1e-6)
        assert theta_map[1] == pytest.approx(batch.laplace["map"][1], abs=1e-6)
        assert cov == pytest.approx(batch.laplace["cov"], abs=1e-8)
        assert H == pytest.approx(batch.laplace["H"], abs=1e-8)

    def test_warm_start_is_close(self) -> None:
        """A warm-start near the optimum should still converge."""
        theta_map_a, _, _ = _paired_laplace_from_counts(
            n_A=80, k_A=60, n_B=80, k_B=45, prior_sigma_delta=1.0
        )
        theta_map_b, _, _ = _paired_laplace_from_counts(
            n_A=80,
            k_A=60,
            n_B=80,
            k_B=45,
            prior_sigma_delta=1.0,
            x0=(float(theta_map_a[0]), float(theta_map_a[1])),
        )
        assert theta_map_a == pytest.approx(theta_map_b, abs=1e-4)

    def test_extreme_warm_start_still_converges(self) -> None:
        """Backtracking line search must rescue a wildly-off warm-start."""
        theta_ref, _, _ = _paired_laplace_from_counts(
            n_A=120, k_A=80, n_B=120, k_B=50, prior_sigma_delta=1.0
        )
        theta_far, _, _ = _paired_laplace_from_counts(
            n_A=120,
            k_A=80,
            n_B=120,
            k_B=50,
            prior_sigma_delta=1.0,
            x0=(20.0, -20.0),
        )
        assert theta_far == pytest.approx(theta_ref, abs=1e-5)

    def test_hessian_positive_definite(self) -> None:
        """Returned Hessian must be symmetric positive definite."""
        _, cov, H = _paired_laplace_from_counts(
            n_A=50, k_A=30, n_B=50, k_B=25, prior_sigma_delta=1.0
        )
        assert H[0, 1] == pytest.approx(H[1, 0])
        eigvals = np.linalg.eigvalsh(H)
        assert np.all(eigvals > 0)
        # cov must be the inverse: H @ cov ≈ I.
        assert (H @ cov) == pytest.approx(np.eye(2), abs=1e-10)

    def test_gradient_vanishes_at_map(self) -> None:
        """At the returned MAP the gradient should be (numerically) zero."""
        n_A, k_A, n_B, k_B = 200, 130, 200, 90
        prior_sigma_delta, prior_sigma_mu = 1.0, 2.0
        theta_map, _, _ = _paired_laplace_from_counts(
            n_A=n_A,
            k_A=k_A,
            n_B=n_B,
            k_B=k_B,
            prior_sigma_delta=prior_sigma_delta,
        )
        mu, delta = float(theta_map[0]), float(theta_map[1])
        p_A = 1.0 / (1.0 + np.exp(-(mu + delta)))
        p_B = 1.0 / (1.0 + np.exp(-mu))
        g_mu = (k_A - n_A * p_A) + (k_B - n_B * p_B) - mu / prior_sigma_mu**2
        g_delta = (k_A - n_A * p_A) - delta / prior_sigma_delta**2
        assert abs(g_mu) < 1e-7
        assert abs(g_delta) < 1e-7


# ── SequentialPairedBayesPropTest ─────────────────────────────


class TestSequentialPaired:
    """Tests for SequentialPairedBayesPropTest."""

    def test_initial_state(self) -> None:
        seq = SequentialPairedBayesPropTest(prior_sigma_delta=1.0)
        assert seq.n_A == 0 and seq.n_B == 0
        assert seq.successes_A == 0 and seq.successes_B == 0
        assert seq.history == []
        assert not seq.stopped
        assert seq.stop_reason is None
        assert seq.last_model is None

    def test_invalid_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError):
            SequentialPairedBayesPropTest(bf_upper=5.0, bf_lower=10.0)
        with pytest.raises(ValueError):
            SequentialPairedBayesPropTest(bf_lower=0.0)

    def test_rejects_non_binary_input(self) -> None:
        seq = SequentialPairedBayesPropTest()
        with pytest.raises(ValueError):
            seq.update(np.array([0.5, 1.0]), np.array([0, 1]))

    def test_rejects_unequal_paired_batches(self) -> None:
        seq = SequentialPairedBayesPropTest()
        with pytest.raises(ValueError):
            seq.update(np.array([0, 1, 1]), np.array([0, 1]))

    def test_sequential_matches_batch_fit(self) -> None:
        """Streaming refits must reproduce a single batch fit exactly."""
        rng = np.random.default_rng(11)
        y_a = rng.binomial(1, 0.7, size=90).astype(int)
        y_b = rng.binomial(1, 0.5, size=90).astype(int)

        seq = SequentialPairedBayesPropTest(
            prior_sigma_delta=1.0,
            bf_upper=1e12,
            bf_lower=1e-12,
            n_max=10**6,
            seed=0,
            n_samples=2_000,
        )
        for chunk in range(3):
            sl = slice(chunk * 30, (chunk + 1) * 30)
            seq.update(y_a[sl], y_b[sl])

        last = seq.history[-1]
        batch = PairedBayesPropTest(prior_sigma_delta=1.0, seed=0, n_samples=2_000).fit(
            y_a, y_b
        )
        assert last.posterior_state.mu_map == pytest.approx(
            batch.laplace["map"][0], abs=1e-4
        )
        assert last.posterior_state.delta_A_map == pytest.approx(
            batch.laplace["map"][1], abs=1e-4
        )
        assert np.asarray(last.posterior_state.cov) == pytest.approx(
            batch.laplace["cov"], abs=1e-4
        )

    def test_snapshot_types(self) -> None:
        rng = np.random.default_rng(2)
        ya = rng.binomial(1, 0.6, size=60).astype(int)
        yb = rng.binomial(1, 0.5, size=60).astype(int)
        seq = SequentialPairedBayesPropTest(seed=2, n_samples=1_500)
        snap = seq.update(ya, yb)

        assert isinstance(snap, SequentialLaplaceLookResult)
        assert isinstance(snap.posterior_state, SequentialLaplaceState)
        assert isinstance(snap.decision, HypothesisDecision)
        assert snap.decision.bayes_factor is not None
        assert snap.decision.rope is not None
        assert snap.look == 1
        assert snap.n_A == 60 and snap.n_B == 60
        assert seq.last_model is not None

    def test_stopping_on_bf_upper(self) -> None:
        """A strong effect should trigger BF10 ≥ bf_upper."""
        rng = np.random.default_rng(7)
        batches = [
            (
                rng.binomial(1, 0.85, size=40).astype(int),
                rng.binomial(1, 0.35, size=40).astype(int),
            )
            for _ in range(15)
        ]
        seq = SequentialPairedBayesPropTest(
            prior_sigma_delta=1.0,
            bf_upper=10.0,
            bf_lower=0.1,
            n_min=20,
            seed=7,
            n_samples=2_000,
        )
        final = seq.run(batches)
        assert seq.stopped
        assert "H1" in seq.stop_reason
        assert final.decision.bayes_factor.BF_10 >= 10.0

    def test_stopping_on_n_max(self) -> None:
        rng = np.random.default_rng(3)
        batches = [
            (
                rng.binomial(1, 0.50, size=10).astype(int),
                rng.binomial(1, 0.50, size=10).astype(int),
            )
            for _ in range(20)
        ]
        seq = SequentialPairedBayesPropTest(
            bf_upper=1e9,
            bf_lower=1e-9,
            n_max=40,
            seed=3,
            n_samples=1_000,
        )
        final = seq.run(batches)
        assert seq.stopped
        assert seq.stop_reason == "n_max reached"
        assert final.n_A >= 40 and final.n_B >= 40

    def test_update_after_stop_raises(self) -> None:
        rng = np.random.default_rng(4)
        ya = rng.binomial(1, 0.9, size=200).astype(int)
        yb = rng.binomial(1, 0.3, size=200).astype(int)
        seq = SequentialPairedBayesPropTest(
            bf_upper=3.0,
            n_min=10,
            seed=4,
            n_samples=1_500,
        )
        seq.update(ya, yb)
        assert seq.stopped
        with pytest.raises(RuntimeError):
            seq.update(ya[:5], yb[:5])

    def test_run_empty_batches_raises(self) -> None:
        seq = SequentialPairedBayesPropTest()
        with pytest.raises(ValueError):
            seq.run(iter(()))

    def test_history_frame_columns(self) -> None:
        rng = np.random.default_rng(5)
        ya = rng.binomial(1, 0.6, size=40).astype(int)
        yb = rng.binomial(1, 0.5, size=40).astype(int)
        seq = SequentialPairedBayesPropTest(seed=5, n_samples=1_000)
        seq.update(ya, yb)
        df = seq.history_frame()
        assert {
            "look",
            "n_A",
            "n_B",
            "mu_MAP",
            "delta_A_MAP",
            "P_A_gt_B",
            "BF_10",
            "BF_01",
            "pct_in_rope",
            "stop",
            "stop_reason",
        }.issubset(df.columns)

    def test_plot_trajectory_runs(self) -> None:
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(6)
        seq = SequentialPairedBayesPropTest(seed=6, n_samples=1_000)
        for _ in range(3):
            ya = rng.binomial(1, 0.65, size=30).astype(int)
            yb = rng.binomial(1, 0.45, size=30).astype(int)
            seq.update(ya, yb)
        seq.plot_trajectory()
        plt.close("all")
