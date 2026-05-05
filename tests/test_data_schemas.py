"""Unit tests for bayesprop.resources.data_schemas module."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from bayesprop.resources.data_schemas import (
    BetaParams,
    CredibleInterval,
    HypothesisDecision,
    MCMCDiagnostics,
    MCMCParamDiagnostic,
    NonPairedConfig,
    NonPairedSummary,
    NonPairedTestResult,
    PairedLaplaceConfig,
    PairedPGConfig,
    PairedSummary,
    PosteriorProbH0Result,
    PPCStatistic,
    ROPEResult,
    SavageDickeyResult,
)


class TestCredibleInterval:
    """Tests for CredibleInterval schema."""

    def test_valid(self) -> None:
        ci = CredibleInterval(lower=-0.1, upper=0.3)
        assert ci.lower == -0.1
        assert ci.upper == 0.3

    def test_serialization_roundtrip(self) -> None:
        ci = CredibleInterval(lower=0.0, upper=1.0)
        data = ci.model_dump()
        assert data == {"lower": 0.0, "upper": 1.0}
        assert CredibleInterval(**data) == ci


class TestBetaParams:
    """Tests for BetaParams schema."""

    def test_valid(self) -> None:
        bp = BetaParams(alpha=2.0, beta=3.0)
        assert bp.alpha == 2.0
        assert bp.beta == 3.0

    def test_alpha_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            BetaParams(alpha=0.0, beta=1.0)

    def test_beta_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            BetaParams(alpha=1.0, beta=-1.0)


class TestSavageDickeyResult:
    """Tests for SavageDickeyResult schema."""

    def test_valid(self) -> None:
        sd = SavageDickeyResult(
            BF_01=0.5,
            BF_10=2.0,
            posterior_density_at_0=1.0,
            prior_density_at_0=2.0,
            interpretation="Moderate evidence for H1",
            decision="Reject H0",
        )
        assert sd.BF_01 == 0.5
        assert sd.BF_10 == 2.0
        assert sd.decision == "Reject H0"


class TestPosteriorProbH0Result:
    """Tests for PosteriorProbH0Result schema."""

    def test_alias_construction(self) -> None:
        result = PosteriorProbH0Result(
            **{
                "P(H0|data)": 0.3,
                "P(H1|data)": 0.7,
                "prior_odds": 1.0,
                "posterior_odds": 0.43,
                "decision": "Reject H0",
            }
        )
        assert result.p_H0 == 0.3
        assert result.p_H1 == 0.7
        assert result.decision == "Reject H0"

    def test_field_name_construction(self) -> None:
        result = PosteriorProbH0Result(
            p_H0=0.3,
            p_H1=0.7,
            prior_odds=1.0,
            posterior_odds=0.43,
            decision="Reject H0",
        )
        assert result.p_H0 == 0.3
        assert result.decision == "Reject H0"


class TestPPCStatistic:
    """Tests for PPCStatistic schema."""

    def test_ok_status(self) -> None:
        ppc = PPCStatistic(observed=0.5, p_value=0.3, status="OK")
        assert ppc.status == "OK"

    def test_warn_status(self) -> None:
        ppc = PPCStatistic(observed=0.5, p_value=0.01, status="WARN")
        assert ppc.status == "WARN"

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PPCStatistic(observed=0.5, p_value=0.3, status="BAD")

    def test_p_value_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PPCStatistic(observed=0.5, p_value=1.5, status="OK")
        with pytest.raises(ValidationError):
            PPCStatistic(observed=0.5, p_value=-0.1, status="OK")


class TestNonPairedConfig:
    """Tests for NonPairedConfig schema."""

    def test_defaults(self) -> None:
        cfg = NonPairedConfig()
        assert cfg.alpha0 == 1.0
        assert cfg.beta0 == 1.0
        assert cfg.threshold == 0.7
        assert cfg.n_quad == 100
        assert cfg.seed == 0
        assert cfg.n_samples == 20_000
        assert cfg.verbose is False

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValidationError):
            NonPairedConfig(alpha0=-1.0)


class TestNonPairedTestResult:
    """Tests for NonPairedTestResult schema."""

    def test_valid(self) -> None:
        result = NonPairedTestResult(
            thetaA_post=BetaParams(alpha=10.0, beta=5.0),
            thetaB_post=BetaParams(alpha=8.0, beta=7.0),
            P_B_greater_A=0.35,
        )
        assert result.P_B_greater_A == 0.35
        assert result.thetaA_post.alpha == 10.0

    def test_p_b_greater_a_bounds(self) -> None:
        with pytest.raises(ValidationError):
            NonPairedTestResult(
                thetaA_post=BetaParams(alpha=1.0, beta=1.0),
                thetaB_post=BetaParams(alpha=1.0, beta=1.0),
                P_B_greater_A=1.5,
            )


class TestNonPairedSummary:
    """Tests for NonPairedSummary schema."""

    def test_alias_construction(self) -> None:
        s = NonPairedSummary(
            mean_delta=0.1,
            ci_95=CredibleInterval(lower=-0.05, upper=0.25),
            **{"P(A > B)": 0.85},
            theta_A_mean=0.8,
            theta_B_mean=0.7,
        )
        assert s.p_A_greater_B == 0.85

    def test_field_name_construction(self) -> None:
        s = NonPairedSummary(
            mean_delta=0.1,
            ci_95=CredibleInterval(lower=-0.05, upper=0.25),
            p_A_greater_B=0.85,
            theta_A_mean=0.8,
            theta_B_mean=0.7,
        )
        assert s.p_A_greater_B == 0.85


class TestPairedSummary:
    """Tests for PairedSummary schema."""

    def test_valid(self) -> None:
        s = PairedSummary(
            mean_delta=0.05,
            ci_95=CredibleInterval(lower=-0.1, upper=0.2),
            p_A_greater_B=0.6,
            delta_A_posterior_mean=0.2,
        )
        assert s.delta_A_posterior_mean == 0.2

    def test_p_a_greater_b_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PairedSummary(
                mean_delta=0.05,
                ci_95=CredibleInterval(lower=-0.1, upper=0.2),
                p_A_greater_B=-0.1,
                delta_A_posterior_mean=0.2,
            )


class TestPairedConfigs:
    """Tests for PairedLaplaceConfig and PairedPGConfig schemas."""

    def test_laplace_defaults(self) -> None:
        cfg = PairedLaplaceConfig()
        assert cfg.prior_sigma_delta == 1.0
        assert cfg.seed == 0
        assert cfg.n_samples == 8_000

    def test_pg_defaults(self) -> None:
        cfg = PairedPGConfig()
        assert cfg.prior_sigma_delta == 1.0
        assert cfg.prior_sigma_mu == 2.0
        assert cfg.n_iter == 2_000
        assert cfg.burn_in == 500
        assert cfg.n_chains == 4

    def test_pg_invalid_sigma(self) -> None:
        with pytest.raises(ValidationError):
            PairedPGConfig(prior_sigma_delta=0.0)


class TestMCMCDiagnostics:
    """Tests for MCMCDiagnostics schema."""

    def test_valid(self) -> None:
        diag = MCMCDiagnostics(
            mu=MCMCParamDiagnostic(r_hat=1.001, ess=500.0),
            delta_A=MCMCParamDiagnostic(r_hat=1.01, ess=400.0),
        )
        assert diag.mu.r_hat == 1.001
        assert diag.delta_A.ess == 400.0

    def test_model_dump(self) -> None:
        diag = MCMCDiagnostics(
            mu=MCMCParamDiagnostic(r_hat=1.0, ess=1000.0),
            delta_A=MCMCParamDiagnostic(r_hat=1.0, ess=800.0),
        )
        data = diag.model_dump()
        assert "mu" in data
        assert data["mu"]["r_hat"] == 1.0


class TestROPEResult:
    """Tests for ROPEResult schema and from_samples classmethod."""

    def test_valid_construction(self) -> None:
        r = ROPEResult(
            rope_lower=-0.02,
            rope_upper=0.02,
            ci_lower=0.05,
            ci_upper=0.20,
            pct_in_rope=0.0,
            pct_below_rope=0.0,
            pct_above_rope=1.0,
            decision="Reject H0 — A practically better",
            interpretation="95% CI entirely above ROPE",
        )
        assert r.pct_above_rope == 1.0
        assert r.ci_mass == 0.95  # default

    def test_from_samples_reject_h0_positive(self) -> None:
        """Large positive effect → CI above ROPE → Reject H0."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0.20, 0.03, size=10_000)
        r = ROPEResult.from_samples(samples, rope=(-0.02, 0.02))
        assert r.decision == "Reject H0 — A practically better"
        assert r.ci_lower > 0.02

    def test_from_samples_reject_h0_negative(self) -> None:
        """Large negative effect → CI below ROPE → B better."""
        rng = np.random.default_rng(42)
        samples = rng.normal(-0.20, 0.03, size=10_000)
        r = ROPEResult.from_samples(samples, rope=(-0.02, 0.02))
        assert r.decision == "Reject H0 — B practically better"
        assert r.ci_upper < -0.02

    def test_from_samples_accept_h0(self) -> None:
        """Tiny effect with very tight posterior → CI inside ROPE → Accept H0."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0.0, 0.005, size=10_000)
        r = ROPEResult.from_samples(samples, rope=(-0.02, 0.02))
        assert r.decision == "Accept H0 — practically equivalent"
        assert r.ci_lower >= -0.02
        assert r.ci_upper <= 0.02

    def test_from_samples_undecided(self) -> None:
        """Moderate uncertainty → CI overlaps ROPE → Undecided."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0.02, 0.04, size=10_000)
        r = ROPEResult.from_samples(samples, rope=(-0.02, 0.02))
        assert r.decision == "Undecided — CI overlaps ROPE"

    def test_pct_fractions_sum_to_one(self) -> None:
        """Posterior fractions below / in / above ROPE should sum to ~1."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0.01, 0.05, size=10_000)
        r = ROPEResult.from_samples(samples, rope=(-0.02, 0.02))
        assert abs(r.pct_in_rope + r.pct_below_rope + r.pct_above_rope - 1.0) < 0.01

    def test_custom_ci_mass(self) -> None:
        """Non-default ci_mass is respected."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0.0, 0.05, size=10_000)
        r90 = ROPEResult.from_samples(samples, rope=(-0.02, 0.02), ci_mass=0.90)
        r99 = ROPEResult.from_samples(samples, rope=(-0.02, 0.02), ci_mass=0.99)
        assert r90.ci_mass == 0.90
        assert r99.ci_mass == 0.99
        # 99% CI should be wider than 90% CI
        assert (r99.ci_upper - r99.ci_lower) > (r90.ci_upper - r90.ci_lower)

    def test_pct_in_rope_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ROPEResult(
                rope_lower=-0.02,
                rope_upper=0.02,
                ci_lower=0.0,
                ci_upper=0.1,
                pct_in_rope=1.5,
                pct_below_rope=0.0,
                pct_above_rope=0.0,
                decision="x",
                interpretation="x",
            )


class TestHypothesisDecision:
    """Tests for HypothesisDecision schema."""

    def test_defaults(self) -> None:
        d = HypothesisDecision(rule="all")
        assert d.bayes_factor is None
        assert d.posterior_null is None
        assert d.rope is None

    def test_with_bayes_factor(self) -> None:
        bf = SavageDickeyResult(
            BF_01=0.1,
            BF_10=10.0,
            posterior_density_at_0=0.5,
            prior_density_at_0=5.0,
            interpretation="Strong evidence against H0",
            decision="Reject H0",
        )
        d = HypothesisDecision(bayes_factor=bf, rule="bayes_factor")
        assert d.bayes_factor is not None
        assert d.bayes_factor.BF_10 == 10.0

    def test_serialization_roundtrip(self) -> None:
        d = HypothesisDecision(rule="rope")
        data = d.model_dump()
        assert data["rule"] == "rope"
        assert HypothesisDecision(**data) == d
