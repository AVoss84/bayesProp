"""Pydantic data contracts for Bayesian A/B test resources.

Defines request / response schemas for the non-paired Beta-Bernoulli model,
the paired Laplace model, and the paired Pólya-Gamma Gibbs model.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

# ====================================================================== #
#  Decision rule type
# ====================================================================== #

DecisionRuleType = Literal["bayes_factor", "posterior_null", "rope", "all"]

# ====================================================================== #
#  Shared / common schemas
# ====================================================================== #


class CredibleInterval(BaseModel):
    """Symmetric credible (or confidence) interval."""

    lower: float = Field(..., description="Lower bound of the interval.")
    upper: float = Field(..., description="Upper bound of the interval.")


class BetaParams(BaseModel):
    """Parameters of a Beta distribution."""

    alpha: float = Field(..., gt=0, description="Alpha (shape) parameter.")
    beta: float = Field(..., gt=0, description="Beta (shape) parameter.")


class SavageDickeyResult(BaseModel):
    """Result of a Savage-Dickey density-ratio Bayes factor test."""

    BF_01: float = Field(..., description="Bayes factor in favour of H0.")
    BF_10: float = Field(..., description="Bayes factor in favour of H1.")
    posterior_density_at_0: float = Field(
        ..., description="Posterior density evaluated at the null value."
    )
    prior_density_at_0: float = Field(
        ..., description="Prior density evaluated at the null value."
    )
    interpretation: str = Field(
        ..., description="Human-readable interpretation of the evidence."
    )
    decision: str = Field(
        ..., description="Decision string, e.g. 'Reject H0' or 'Fail to reject H0'."
    )


class PosteriorProbH0Result(BaseModel):
    """Posterior probability of H0 under a spike-and-slab prior."""

    p_H0: float = Field(..., alias="P(H0|data)", description="Posterior P(H0).")
    p_H1: float = Field(..., alias="P(H1|data)", description="Posterior P(H1).")
    prior_odds: float = Field(..., description="Prior odds in favour of H0.")
    posterior_odds: float = Field(..., description="Posterior odds in favour of H0.")
    decision: str = Field(
        ...,
        description="Decision: 'Reject H0', 'Fail to reject H0', or 'Undecided'.",
    )

    model_config = {"populate_by_name": True}


class PPCStatistic(BaseModel):
    """Single posterior predictive check statistic."""

    observed: float = Field(..., description="Observed value of the test statistic.")
    p_value: float = Field(..., ge=0, le=1, description="Two-sided PPC p-value.")
    status: Literal["OK", "WARN"] = Field(
        ..., description="OK if p > 0.05, WARN otherwise."
    )


class FrequentistTestResult(BaseModel):
    """Result of a frequentist two-proportion test (e.g. Fisher's exact)."""

    p_value: float = Field(
        ..., ge=0, le=1, description="Two-sided p-value under H0: p_A = p_B."
    )
    odds_ratio: float | None = Field(
        ...,
        description=(
            "Sample odds ratio (p_A/(1-p_A)) / (p_B/(1-p_B)); "
            "``None`` if any cell is zero."
        ),
    )
    successes_A: int = Field(..., ge=0, description="Successes in group A.")
    successes_B: int = Field(..., ge=0, description="Successes in group B.")
    n_A: int = Field(..., ge=0, description="Sample size of group A.")
    n_B: int = Field(..., ge=0, description="Sample size of group B.")
    alternative: Literal["two-sided", "greater", "less"] = Field(
        default="two-sided", description="Alternative hypothesis used."
    )
    test: str = Field(
        default="fisher_exact", description="Identifier of the test that was run."
    )


# ====================================================================== #
#  Simulation result schemas
# ====================================================================== #


class NonPairedTrueParams(BaseModel):
    """True parameters used to generate non-paired simulation data."""

    N: int = Field(..., gt=0, description="Number of observations per group.")
    theta_A: float = Field(
        ..., ge=0, le=1, description="True success rate for model A."
    )
    theta_B: float = Field(
        ..., ge=0, le=1, description="True success rate for model B."
    )


class NonPairedSimResult(BaseModel):
    """Output of :func:`simulate_nonpaired_scores`."""

    model_config = {"arbitrary_types_allowed": True}

    y_A: npt.NDArray[np.float64] = Field(
        ..., description="Binary outcomes for model A."
    )
    y_B: npt.NDArray[np.float64] = Field(
        ..., description="Binary outcomes for model B."
    )
    theta_A: float = Field(
        ..., ge=0, le=1, description="True success rate for model A."
    )
    theta_B: float = Field(
        ..., ge=0, le=1, description="True success rate for model B."
    )
    true_params: NonPairedTrueParams = Field(
        ..., description="True simulation parameters."
    )


class PairedTrueParams(BaseModel):
    """True parameters used to generate paired simulation data."""

    N: int = Field(..., gt=0, description="Number of paired observations.")
    mu: float = Field(default=0.0, description="Shared logit-scale intercept.")
    sigma_theta: float = Field(
        default=0.0,
        ge=0,
        description="SD of the latent item ability (0 = fixed effects).",
    )
    delta_A: float = Field(..., description="Logit-scale treatment effect for model A.")
    delta_B: float = Field(
        default=0.0, description="Logit-scale offset for model B (0 by default)."
    )


class PairedSimResult(BaseModel):
    """Output of :func:`simulate_paired_scores`."""

    model_config = {"arbitrary_types_allowed": True}

    y_A: npt.NDArray[np.float64] = Field(
        ..., description="Binary outcomes for model A."
    )
    y_B: npt.NDArray[np.float64] = Field(
        ..., description="Binary outcomes for model B."
    )
    p_A_true: npt.NDArray[np.float64] = Field(
        ..., description="Item-level probabilities for model A."
    )
    p_B_true: npt.NDArray[np.float64] = Field(
        ..., description="Item-level probabilities for model B."
    )
    theta_true: npt.NDArray[np.float64] = Field(
        ..., description="Latent item abilities."
    )
    true_params: PairedTrueParams = Field(
        ..., description="True simulation parameters."
    )


# ====================================================================== #
#  Non-paired Beta-Bernoulli model
# ====================================================================== #


class NonPairedConfig(BaseModel):
    """Configuration for :class:`NonPairedBayesPropTest`."""

    alpha0: float = Field(default=1.0, gt=0, description="Prior alpha for Beta.")
    beta0: float = Field(default=1.0, gt=0, description="Prior beta for Beta.")
    threshold: float = Field(
        default=0.7, ge=0, le=1, description="Binarization threshold."
    )
    n_quad: int = Field(
        default=100, gt=0, description="Number of Gauss-Legendre quadrature nodes."
    )
    seed: int = Field(default=0, description="Random seed.")
    n_samples: int = Field(
        default=20_000, gt=0, description="Number of Monte Carlo draws."
    )
    verbose: bool = Field(default=False, description="Print diagnostic messages.")


class NonPairedTestResult(BaseModel):
    """Output of :meth:`NonPairedBayesPropTest.test`."""

    thetaA_post: BetaParams = Field(
        ..., description="Posterior Beta parameters for group A."
    )
    thetaB_post: BetaParams = Field(
        ..., description="Posterior Beta parameters for group B."
    )
    P_B_greater_A: float = Field(
        ...,
        ge=0,
        le=1,
        description="Posterior probability of superiority P(theta_B > theta_A).",
    )


class NonPairedSummary(BaseModel):
    """Summary produced by :meth:`NonPairedBayesPropTest.fit`."""

    mean_delta: float = Field(..., description="Posterior mean of Delta = p_A - p_B.")
    ci_95: CredibleInterval = Field(
        ..., description="95 % credible interval for Delta."
    )
    p_A_greater_B: float = Field(
        ..., alias="P(A > B)", ge=0, le=1, description="P(p_A > p_B)."
    )
    theta_A_mean: float = Field(..., description="Posterior mean of theta_A.")
    theta_B_mean: float = Field(..., description="Posterior mean of theta_B.")

    model_config = {"populate_by_name": True}


# ====================================================================== #
#  Paired model (shared summary)
# ====================================================================== #


class PairedSummary(BaseModel):
    """Summary produced by :meth:`PairedBayesPropTest.fit` (Laplace or PG)."""

    mean_delta: float = Field(
        ..., description="Posterior mean of Delta = p_A - p_B (probability scale)."
    )
    ci_95: CredibleInterval = Field(
        ..., description="95 % credible interval for Delta."
    )
    p_A_greater_B: float = Field(
        ..., alias="P(A > B)", ge=0, le=1, description="P(p_A > p_B)."
    )
    delta_A_posterior_mean: float = Field(
        ..., description="Posterior mean of delta_A (logit scale)."
    )

    model_config = {"populate_by_name": True}


# ====================================================================== #
#  Paired Laplace model
# ====================================================================== #


class PairedLaplaceConfig(BaseModel):
    """Configuration for :class:`PairedBayesPropTest` (Laplace approximation)."""

    prior_sigma_delta: float = Field(
        default=1.0, gt=0, description="SD of N(0, sigma) prior on delta_A."
    )
    seed: int = Field(default=0, description="Random seed.")
    n_samples: int = Field(
        default=8_000, gt=0, description="Number of Laplace posterior draws."
    )


# ====================================================================== #
#  Paired Pólya-Gamma model
# ====================================================================== #


class PairedPGConfig(BaseModel):
    """Configuration for :class:`PairedBayesPropTestPG` (PG Gibbs sampler)."""

    prior_sigma_delta: float = Field(
        default=1.0, gt=0, description="SD of N(0, sigma) prior on delta_A."
    )
    prior_sigma_mu: float = Field(
        default=2.0, gt=0, description="SD of N(0, sigma) prior on mu."
    )
    seed: int = Field(default=0, description="Random seed.")
    n_iter: int = Field(
        default=1_000, gt=0, description="Total Gibbs iterations per chain."
    )
    burn_in: int = Field(
        default=200, ge=0, description="Number of warm-up iterations to discard."
    )
    n_chains: int = Field(default=2, gt=0, description="Number of MCMC chains.")


class MCMCParamDiagnostic(BaseModel):
    """MCMC convergence diagnostics for a single parameter."""

    r_hat: float = Field(..., description="Gelman-Rubin R-hat statistic.")
    ess: float = Field(..., description="Effective sample size.")


class MCMCDiagnostics(BaseModel):
    """MCMC diagnostics for all parameters."""

    mu: MCMCParamDiagnostic = Field(..., description="Diagnostics for mu.")
    delta_A: MCMCParamDiagnostic = Field(..., description="Diagnostics for delta_A.")


# ====================================================================== #
#  ROPE decision
# ====================================================================== #


class ROPEResult(BaseModel):
    """Result of a ROPE (Region of Practical Equivalence) analysis.

    Decision rules (Kruschke, 2018):

    - 95% CI entirely **outside** ROPE → Reject H₀
    - 95% CI entirely **inside** ROPE → Accept H₀
    - 95% CI **overlaps** ROPE → Undecided
    """

    rope_lower: float = Field(..., description="Lower bound of the ROPE.")
    rope_upper: float = Field(..., description="Upper bound of the ROPE.")
    ci_lower: float = Field(..., description="Lower bound of the credible interval.")
    ci_upper: float = Field(..., description="Upper bound of the credible interval.")
    ci_mass: float = Field(
        default=0.95, ge=0, le=1, description="Mass of the credible interval."
    )
    pct_in_rope: float = Field(
        ..., ge=0, le=1, description="Fraction of posterior inside ROPE."
    )
    pct_below_rope: float = Field(
        ..., ge=0, le=1, description="Fraction of posterior below ROPE."
    )
    pct_above_rope: float = Field(
        ..., ge=0, le=1, description="Fraction of posterior above ROPE."
    )
    decision: str = Field(
        ...,
        description="Decision: 'Reject H0 — A practically better', 'Accept H0', or 'Undecided'.",
    )
    interpretation: str = Field(
        ..., description="Human-readable interpretation of the ROPE analysis."
    )

    @classmethod
    def from_samples(
        cls,
        samples: npt.NDArray[np.floating],
        rope: tuple[float, float] = (-0.02, 0.02),
        ci_mass: float = 0.95,
    ) -> ROPEResult:
        """Compute ROPE decision from posterior samples.

        Args:
            samples: 1-D array of posterior draws for Δ (e.g. θ_A − θ_B).
            rope: (lower, upper) bounds of the ROPE.
            ci_mass: Credible interval mass (default 95%).

        Returns:
            Populated :class:`ROPEResult`.
        """
        alpha = (1 - ci_mass) / 2
        ci_lower = float(np.quantile(samples, alpha))
        ci_upper = float(np.quantile(samples, 1 - alpha))

        pct_in = float(np.mean((samples >= rope[0]) & (samples <= rope[1])))
        pct_below = float(np.mean(samples < rope[0]))
        pct_above = float(np.mean(samples > rope[1]))

        if ci_lower > rope[1]:
            decision = "Reject H0 — A practically better"
            interpretation = "95% CI entirely above ROPE — meaningful positive effect"
        elif ci_upper < rope[0]:
            decision = "Reject H0 — B practically better"
            interpretation = "95% CI entirely below ROPE — meaningful negative effect"
        elif ci_lower >= rope[0] and ci_upper <= rope[1]:
            decision = "Accept H0 — practically equivalent"
            interpretation = "95% CI entirely inside ROPE — effect is negligible"
        else:
            decision = "Undecided — CI overlaps ROPE"
            interpretation = "95% CI overlaps ROPE boundary — more data needed"

        return cls(
            rope_lower=rope[0],
            rope_upper=rope[1],
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_mass=ci_mass,
            pct_in_rope=pct_in,
            pct_below_rope=pct_below,
            pct_above_rope=pct_above,
            decision=decision,
            interpretation=interpretation,
        )


# ====================================================================== #
#  Composite hypothesis decision
# ====================================================================== #


class HypothesisDecision(BaseModel):
    """Composite hypothesis test result combining multiple decision frameworks.

    Depending on the chosen :attr:`rule`, one or more of the sub-results
    will be populated.
    """

    bayes_factor: SavageDickeyResult | None = Field(
        default=None, description="Savage-Dickey Bayes factor result."
    )
    posterior_null: PosteriorProbH0Result | None = Field(
        default=None, description="Posterior probability of H₀."
    )
    rope: ROPEResult | None = Field(default=None, description="ROPE analysis result.")
    rule: DecisionRuleType = Field(
        default="all", description="Which decision rule was applied."
    )


# ====================================================================== #
#  Sequential / streaming design
# ====================================================================== #


class SequentialPosteriorState(BaseModel):
    """Running Beta posterior state for a sequential non-paired test.

    By Beta-Bernoulli conjugacy this *is* the prior for the next batch.
    """

    alpha_A: float = Field(..., description="Posterior alpha for arm A.")
    beta_A: float = Field(..., description="Posterior beta for arm A.")
    alpha_B: float = Field(..., description="Posterior alpha for arm B.")
    beta_B: float = Field(..., description="Posterior beta for arm B.")


class SequentialLookResult(BaseModel):
    """Snapshot of the sequential test after a single look (batch update)."""

    look: int = Field(..., ge=1, description="1-based index of this look.")
    n_A: int = Field(..., ge=0, description="Cumulative sample size for arm A.")
    n_B: int = Field(..., ge=0, description="Cumulative sample size for arm B.")
    successes_A: int = Field(..., ge=0, description="Cumulative successes for arm A.")
    successes_B: int = Field(..., ge=0, description="Cumulative successes for arm B.")
    posterior_state: SequentialPosteriorState = Field(
        ..., description="Running Beta posterior state after this look."
    )
    P_B_greater_A: float = Field(
        ..., ge=0, le=1, description="Posterior probability P(theta_B > theta_A)."
    )
    decision: HypothesisDecision = Field(
        ..., description="Composite decision result at this look."
    )
    stop: bool = Field(
        ..., description="Whether a sequential stopping rule has triggered."
    )
    stop_reason: str | None = Field(
        default=None, description="Reason for stopping (None if continuing)."
    )


# ====================================================================== #
#  Sequential design — paired Laplace model
# ====================================================================== #


class SequentialLaplaceState(BaseModel):
    """Running Laplace posterior state for the sequential paired model.

    Carries the MAP estimate and 2x2 covariance of (mu, delta_A) on the
    logit scale, recomputed from cumulative sufficient statistics at
    every look.
    """

    mu_map: float = Field(..., description="MAP estimate of mu (logit scale).")
    delta_A_map: float = Field(
        ..., description="MAP estimate of delta_A (logit scale)."
    )
    cov: list[list[float]] = Field(
        ...,
        description="2x2 Laplace covariance matrix for (mu, delta_A).",
    )


class SequentialLaplaceLookResult(BaseModel):
    """Snapshot of the sequential paired Laplace test after a single look."""

    look: int = Field(..., ge=1, description="1-based index of this look.")
    n_A: int = Field(..., ge=0, description="Cumulative sample size for arm A.")
    n_B: int = Field(..., ge=0, description="Cumulative sample size for arm B.")
    successes_A: int = Field(..., ge=0, description="Cumulative successes for arm A.")
    successes_B: int = Field(..., ge=0, description="Cumulative successes for arm B.")
    posterior_state: SequentialLaplaceState = Field(
        ..., description="Running Laplace posterior state after this look."
    )
    P_A_greater_B: float = Field(
        ...,
        ge=0,
        le=1,
        description="Posterior probability P(p_A > p_B) on the probability scale.",
    )
    decision: HypothesisDecision = Field(
        ..., description="Composite decision result at this look."
    )
    stop: bool = Field(
        ..., description="Whether a sequential stopping rule has triggered."
    )
    stop_reason: str | None = Field(
        default=None, description="Reason for stopping (None if continuing)."
    )
