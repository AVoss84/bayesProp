"""Unit tests for bayesAB.resources.data_schemas module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bayesAB.resources.data_schemas import (
    BetaParams,
    CredibleInterval,
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
            }
        )
        assert result.p_H0 == 0.3
        assert result.p_H1 == 0.7

    def test_field_name_construction(self) -> None:
        result = PosteriorProbH0Result(p_H0=0.3, p_H1=0.7, prior_odds=1.0, posterior_odds=0.43)
        assert result.p_H0 == 0.3


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
