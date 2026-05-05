"""Non-paired Bayesian A/B test using conjugate Beta-Bernoulli model.

This module provides :class:`NonPairedBayesPropTest` for comparing two
independent groups via binarized pass/fail counts with a Beta-Bernoulli
conjugate model, Savage-Dickey Bayes factor on the difference of
proportions, posterior predictive checks, and publication-ready plots.

Also provides :func:`descriptive_summary` for building a combined
descriptive statistics + threshold-sweep summary table.

Typical workflow::

    from ai_eval.resources.bayes_nonpaired import NonPairedBayesPropTest, descriptive_summary

    bb = NonPairedBayesPropTest(threshold=0.7)
    result = bb.test(scores_A, scores_B)

    df = descriptive_summary(scores_dict, thresholds=[0.5, 0.7, 0.8, 0.9, 0.95])
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import betainc, betaln
from scipy.stats import gaussian_kde

from bayesprop.resources.data_schemas import (
    BetaParams,
    CredibleInterval,
    DecisionRuleType,
    HypothesisDecision,
    NonPairedSummary,
    NonPairedTestResult,
    PosteriorProbH0Result,
    PPCStatistic,
    ROPEResult,
    SavageDickeyResult,
)


def _format_bf(value: float) -> str:
    """Format a Bayes Factor for human-readable display."""
    if value > 1e4:
        return f"10^{np.log10(value):.0f}"
    elif value < 1e-4 and value > 0:
        return f"10^{np.log10(value):.0f}"
    else:
        return f"{value:.2f}"


def beta_diff_pdf(
    z: float,
    a1: float,
    b1: float,
    a2: float,
    b2: float,
    n_grid: int = 2000,
) -> float:
    """Evaluate the PDF of Δ = θ_A − θ_B at *z* via log-space convolution.

    Where θ_A ~ Beta(a1, b1) and θ_B ~ Beta(a2, b2) are independent.

    Args:
        z: Point at which to evaluate the density (−1 < z < 1).
        a1: Alpha parameter for the Beta distribution of θ_A.
        b1: Beta parameter for the Beta distribution of θ_A.
        a2: Alpha parameter for the Beta distribution of θ_B.
        b2: Beta parameter for the Beta distribution of θ_B.
        n_grid: Number of quadrature nodes for trapezoidal integration.

    Returns:
        f_Δ(z), the density of the difference at *z*.
    """
    if z <= -1 or z >= 1:
        return 0.0

    # Δ = θ_A − θ_B  ⟹  θ_A = θ_B + z
    # f_Δ(z) = ∫ f_A(x) · f_B(x − z) dx   for x in [max(0,z), min(1,1+z)]
    lower = max(0.0, z)
    upper = min(1.0, 1.0 + z)

    x = np.linspace(lower + 1e-12, upper - 1e-12, n_grid)

    # log Beta PDFs for θ_A at x
    log_fA = (a1 - 1) * np.log(x) + (b1 - 1) * np.log(1 - x) - betaln(a1, b1)
    # θ_B = x − z
    xb = x - z
    log_fB = (a2 - 1) * np.log(xb) + (b2 - 1) * np.log(1 - xb) - betaln(a2, b2)

    integrand = np.exp(log_fA + log_fB)
    return float(np.trapezoid(integrand, x))


class NonPairedBayesPropTest:
    """Non-paired Bayesian A/B test using conjugate Beta-Bernoulli model.

    Workflow:
        1. Binarize continuous scores at a threshold.
        2. Update Beta(alpha0, beta0) prior with observed pass/fail counts.
        3. Compute P(theta_B > theta_A) via Gauss-Legendre quadrature.
    """

    def __init__(
        self,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        threshold: float = 0.7,
        n_quad: int = 100,
        seed: int = 0,
        n_samples: int = 20_000,
        verbose: bool = False,
        decision_rule: DecisionRuleType = "all",
        rope_epsilon: float = 0.02,
    ) -> None:
        """Initialise the Beta-Bernoulli proportion test.

        Args:
            alpha0: Prior alpha parameter for the Beta distribution.
            beta0: Prior beta parameter for the Beta distribution.
            threshold: Binarization threshold for continuous scores.
            n_quad: Number of Gauss-Legendre quadrature nodes.
            seed: Random seed for Monte Carlo sampling.
            n_samples: Number of Monte Carlo draws for difference posterior.
            verbose: If True, print diagnostic messages.
            decision_rule: Default decision framework — one of
                ``"bayes_factor"``, ``"posterior_null"``, ``"rope"``, or ``"all"``.
            rope_epsilon: Half-width of the ROPE interval (default 0.02 = 2 pp).
        """
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.threshold = threshold
        self.n_quad = n_quad
        self.seed = seed
        self.n_samples = n_samples
        self.verbose = verbose
        self.decision_rule = decision_rule
        self.rope_epsilon = rope_epsilon

        if self.verbose:
            print(
                f"Initialized NonPairedBayesPropTest with "
                f"alpha0={alpha0}, beta0={beta0}, threshold={threshold}, n_quad={n_quad}"
            )

        # precompute quadrature nodes/weights (reused across calls)
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        self._x = 0.5 * (nodes + 1)
        self._w = 0.5 * weights

    def _binarize(self, y: np.ndarray) -> np.ndarray:
        """Return y unchanged if already binary, else binarize at self.threshold."""
        unique = np.unique(y)
        if np.all(np.isin(unique, [0.0, 1.0])):
            return y

        if self.verbose:
            print(f"Warning: non-binary scores detected, binarizing at threshold {self.threshold}")
        return (y >= self.threshold).astype(float)

    def prob_greater(self, a1: float, b1: float, a2: float, b2: float) -> float:
        """Compute P(theta1 > theta2) via Gauss-Legendre quadrature.

        Args:
            a1: Alpha parameter of the first Beta distribution.
            b1: Beta parameter of the first Beta distribution.
            a2: Alpha parameter of the second Beta distribution.
            b2: Beta parameter of the second Beta distribution.

        Returns:
            Probability that a draw from Beta(a1, b1) exceeds
            a draw from Beta(a2, b2).
        """
        x, w = self._x, self._w
        log_pdf1 = (a1 - 1) * np.log(x) + (b1 - 1) * np.log(1 - x) - betaln(a1, b1)
        cdf2 = betainc(a2, b2, x)
        return float(np.dot(w, np.exp(log_pdf1) * cdf2))

    def test(self, y_a: npt.ArrayLike, y_b: npt.ArrayLike) -> NonPairedTestResult:
        """Run the Beta-Bernoulli test. Auto-binarizes if scores are not 0/1.

        Args:
            y_a: Scores for group A (continuous or binary).
            y_b: Scores for group B (continuous or binary).

        Returns:
            :class:`NonPairedTestResult` with posterior Beta parameters
            and P(theta_B > theta_A).
        """
        y_a = self._binarize(np.asarray(y_a, dtype=float))
        y_b = self._binarize(np.asarray(y_b, dtype=float))

        N_a, N_b = len(y_a), len(y_b)
        sA, sB = y_a.sum(), y_b.sum()

        a1 = self.alpha0 + sA
        b1 = self.beta0 + (N_a - sA)
        a2 = self.alpha0 + sB
        b2 = self.beta0 + (N_b - sB)

        p = self.prob_greater(a2, b2, a1, b1)  # P(theta_B > theta_A)

        return NonPairedTestResult(
            thetaA_post=BetaParams(alpha=float(a1), beta=float(b1)),
            thetaB_post=BetaParams(alpha=float(a2), beta=float(b2)),
            P_B_greater_A=float(p),
        )

    # ------------------------------------------------------------------ #
    #  Full fit (with difference posterior)
    # ------------------------------------------------------------------ #

    def fit(self, y_a: npt.ArrayLike, y_b: npt.ArrayLike) -> NonPairedBayesPropTest:
        """Fit Beta posteriors and sample difference posterior via Monte Carlo.

        This extends :meth:`test` by also drawing from the individual
        Beta posteriors to build the posterior of Δ = p_A − p_B, which
        enables Savage-Dickey Bayes factors, posterior predictive checks,
        and richer summaries.

        Args:
            y_a: Scores for group A (continuous or binary).
            y_b: Scores for group B (continuous or binary).

        Returns:
            self (for method chaining).
        """
        self.y_A_obs = self._binarize(np.asarray(y_a, dtype=float))
        self.y_B_obs = self._binarize(np.asarray(y_b, dtype=float))

        N_a, N_b = len(self.y_A_obs), len(self.y_B_obs)
        sA, sB = self.y_A_obs.sum(), self.y_B_obs.sum()

        self.a_A = self.alpha0 + sA
        self.b_A = self.beta0 + (N_a - sA)
        self.a_B = self.alpha0 + sB
        self.b_B = self.beta0 + (N_b - sB)

        # P(B > A) via quadrature (exact)
        self.p_B_greater_A = self.prob_greater(self.a_B, self.b_B, self.a_A, self.b_A)

        # Monte Carlo samples from Beta posteriors
        rng = np.random.default_rng(self.seed)
        self.theta_A_samples = rng.beta(self.a_A, self.b_A, size=self.n_samples)
        self.theta_B_samples = rng.beta(self.a_B, self.b_B, size=self.n_samples)
        self.delta_samples = self.theta_A_samples - self.theta_B_samples

        self.summary = NonPairedSummary(
            mean_delta=float(self.delta_samples.mean()),
            ci_95=CredibleInterval(
                lower=float(np.quantile(self.delta_samples, 0.025)),
                upper=float(np.quantile(self.delta_samples, 0.975)),
            ),
            **{"P(A > B)": float((self.delta_samples > 0).mean())},
            theta_A_mean=float(self.theta_A_samples.mean()),
            theta_B_mean=float(self.theta_B_samples.mean()),
        )

        self.trace_summary = pd.DataFrame(
            {
                "mean": [
                    self.theta_A_samples.mean(),
                    self.theta_B_samples.mean(),
                    self.delta_samples.mean(),
                ],
                "sd": [
                    self.theta_A_samples.std(),
                    self.theta_B_samples.std(),
                    self.delta_samples.std(),
                ],
                "hdi_3%": [
                    np.quantile(self.theta_A_samples, 0.03),
                    np.quantile(self.theta_B_samples, 0.03),
                    np.quantile(self.delta_samples, 0.03),
                ],
                "hdi_97%": [
                    np.quantile(self.theta_A_samples, 0.97),
                    np.quantile(self.theta_B_samples, 0.97),
                    np.quantile(self.delta_samples, 0.97),
                ],
            },
            index=["theta_A", "theta_B", "delta"],
        )

        return self

    def _check_fitted(self) -> None:
        """Raise RuntimeError if .fit() has not been called."""
        if not hasattr(self, "delta_samples") or self.delta_samples is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    # ------------------------------------------------------------------ #
    #  Hypothesis testing
    # ------------------------------------------------------------------ #

    def savage_dickey_test(self, null_value: float = 0.0) -> SavageDickeyResult:
        """Savage-Dickey density-ratio Bayes factor for H0: Δ = null_value.

        The prior on Δ = p_A − p_B is induced by the independent
        Beta(α₀, β₀) priors. Densities are computed via exact
        log-space convolution (:func:`beta_diff_pdf`).

        Args:
            null_value: The point null hypothesis value for Δ.

        Returns:
            :class:`SavageDickeyResult` with BF_01, BF_10, densities,
            interpretation, and decision.
        """
        self._check_fitted()

        # Posterior density at null via analytic convolution
        posterior_at_null = beta_diff_pdf(null_value, self.a_A, self.b_A, self.a_B, self.b_B)

        # Prior density at null via analytic convolution
        prior_at_null = beta_diff_pdf(null_value, self.alpha0, self.beta0, self.alpha0, self.beta0)

        BF_01 = posterior_at_null / prior_at_null
        BF_10 = 1.0 / BF_01

        if BF_10 > 100:
            interpretation = "Decisive evidence against H0"
        elif BF_10 > 30:
            interpretation = "Very strong evidence against H0"
        elif BF_10 > 10:
            interpretation = "Strong evidence against H0"
        elif BF_10 > 3:
            interpretation = "Moderate evidence against H0"
        elif BF_10 > 1:
            interpretation = "Anecdotal evidence against H0"
        elif BF_10 == 1:
            interpretation = "No evidence either way"
        elif BF_01 > 100:
            interpretation = "Decisive evidence for H0"
        elif BF_01 > 30:
            interpretation = "Very strong evidence for H0"
        elif BF_01 > 10:
            interpretation = "Strong evidence for H0"
        elif BF_01 > 3:
            interpretation = "Moderate evidence for H0"
        else:
            interpretation = "Anecdotal evidence for H0"

        decision = "Reject H0" if BF_10 > 3 else "Fail to reject H0"

        return SavageDickeyResult(
            BF_01=BF_01,
            BF_10=BF_10,
            posterior_density_at_0=posterior_at_null,
            prior_density_at_0=prior_at_null,
            interpretation=interpretation,
            decision=decision,
        )

    @staticmethod
    def posterior_probability_H0(BF_01: float, prior_H0: float = 0.5) -> PosteriorProbH0Result:
        """Convert BF_01 to posterior probability of H0 (spike-and-slab).

        Args:
            BF_01: Bayes factor in favour of H0.
            prior_H0: Prior probability of H0 (default 0.5).

        Returns:
            :class:`PosteriorProbH0Result` with posterior and prior odds
            and model probabilities.
        """
        prior_odds = prior_H0 / (1 - prior_H0)
        posterior_odds = BF_01 * prior_odds
        P_H0 = posterior_odds / (1 + posterior_odds)
        P_H1 = 1 - P_H0

        if P_H1 > 0.95:
            decision = "Reject H0"
        elif P_H0 > 0.95:
            decision = "Fail to reject H0"
        else:
            decision = "Undecided"

        return PosteriorProbH0Result(
            **{"P(H0|data)": P_H0, "P(H1|data)": P_H1},
            prior_odds=prior_odds,
            posterior_odds=posterior_odds,
            decision=decision,
        )

    # ------------------------------------------------------------------ #
    #  ROPE analysis
    # ------------------------------------------------------------------ #

    def rope_test(
        self,
        rope: tuple[float, float] | None = None,
        ci_mass: float = 0.95,
    ) -> ROPEResult:
        """ROPE analysis on the posterior of Δ = θ_A − θ_B.

        Args:
            rope: (lower, upper) ROPE bounds. Defaults to
                ``(-self.rope_epsilon, +self.rope_epsilon)``.
            ci_mass: Credible interval mass (default 95%).

        Returns:
            :class:`ROPEResult` with CI, ROPE overlap fractions, and
            decision.
        """
        self._check_fitted()
        if rope is None:
            rope = (-self.rope_epsilon, self.rope_epsilon)
        return ROPEResult.from_samples(self.delta_samples, rope=rope, ci_mass=ci_mass)

    # ------------------------------------------------------------------ #
    #  Composite decision
    # ------------------------------------------------------------------ #

    def decide(self, rule: DecisionRuleType | None = None) -> HypothesisDecision:
        """Run the chosen decision framework(s) and return a composite result.

        Args:
            rule: Override the default ``decision_rule``. One of
                ``"bayes_factor"``, ``"posterior_null"``, ``"rope"``,
                or ``"all"``.

        Returns:
            :class:`HypothesisDecision` with the requested sub-results
            populated.
        """
        self._check_fitted()
        rule = rule or self.decision_rule

        bf: SavageDickeyResult | None = None
        pn: PosteriorProbH0Result | None = None
        rp: ROPEResult | None = None

        if rule in ("bayes_factor", "posterior_null", "all"):
            bf = self.savage_dickey_test()
        if rule in ("posterior_null", "all"):
            assert bf is not None  # guaranteed by the branch above
            pn = self.posterior_probability_H0(bf.BF_01)
        if rule in ("rope", "all"):
            rp = self.rope_test()

        return HypothesisDecision(bayes_factor=bf, posterior_null=pn, rope=rp, rule=rule)

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #

    def ppc_pvalues(self, seed: int | None = None) -> dict[str, PPCStatistic]:
        """Posterior predictive p-values for summary statistics.

        Returns:
            Dict mapping statistic name to :class:`PPCStatistic`.
        """
        self._check_fitted()

        rng = np.random.default_rng(seed if seed is not None else self.seed)

        n_A = len(self.y_A_obs)
        n_B = len(self.y_B_obs)

        # Replicate datasets from posterior draws
        y_A_rep = rng.binomial(1, self.theta_A_samples[:, None], size=(self.n_samples, n_A))
        y_B_rep = rng.binomial(1, self.theta_B_samples[:, None], size=(self.n_samples, n_B))

        checks = {
            "mean(y_A)": (self.y_A_obs.mean(), y_A_rep.mean(axis=1)),
            "mean(y_B)": (self.y_B_obs.mean(), y_B_rep.mean(axis=1)),
            "mean(y_A)-mean(y_B)": (
                self.y_A_obs.mean() - self.y_B_obs.mean(),
                y_A_rep.mean(axis=1) - y_B_rep.mean(axis=1),
            ),
        }

        results: dict[str, PPCStatistic] = {}
        for stat_name, (obs_val, rep_vals) in checks.items():
            p_val = min(
                2
                * min(
                    float((rep_vals >= obs_val).mean()),
                    float((rep_vals <= obs_val).mean()),
                ),
                1.0,
            )
            results[stat_name] = PPCStatistic(
                observed=float(obs_val),
                p_value=p_val,
                status="OK" if p_val > 0.05 else "WARN",
            )
        return results

    # ------------------------------------------------------------------ #
    #  Plotting
    # ------------------------------------------------------------------ #

    def plot_posteriors(self, **kwargs: dict) -> None:
        """Two-panel plot: overlaid θ_A / θ_B posteriors and Δ = θ_A − θ_B.

        The left panel shows the analytic Beta posterior densities for
        θ_A and θ_B overlaid in a single axes. The right panel shows the
        Monte Carlo difference posterior Δ = θ_A − θ_B.

        Args:
            **kwargs: Accepts ``figsize`` (default ``(14, 5)``) and
                ``title`` (default ``"Beta-Bernoulli Posteriors (Non-Paired)"``).
        """
        import matplotlib.pyplot as plt
        from scipy.stats import beta as beta_dist

        self._check_fitted()

        figsize = kwargs.pop("figsize", (14, 5))
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Panel 1: θ_A and θ_B overlaid (analytic Beta densities)
        ax = axes[0]
        x = np.linspace(0, 1, 500)
        pdf_A = beta_dist.pdf(x, self.a_A, self.b_A)
        pdf_B = beta_dist.pdf(x, self.a_B, self.b_B)

        ax.plot(
            x,
            pdf_A,
            color="#2196F3",
            linewidth=2,
            label=f"θ_A ~ Beta({self.a_A:.0f}, {self.b_A:.0f})",
        )
        ax.fill_between(x, pdf_A, alpha=0.15, color="#2196F3")
        ax.plot(
            x,
            pdf_B,
            color="#4CAF50",
            linewidth=2,
            label=f"θ_B ~ Beta({self.a_B:.0f}, {self.b_B:.0f})",
        )
        ax.fill_between(x, pdf_B, alpha=0.15, color="#4CAF50")

        ax.axvline(
            self.a_A / (self.a_A + self.b_A),
            color="#2196F3",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )
        ax.axvline(
            self.a_B / (self.a_B + self.b_B),
            color="#4CAF50",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )
        ax.set_xlabel("θ")
        ax.set_ylabel("Density")
        ax.set_title("Beta Posteriors", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Panel 2: Δ = θ_A − θ_B
        ax = axes[1]
        samples = self.delta_samples
        ax.hist(
            samples,
            bins=60,
            density=True,
            alpha=0.6,
            color="#9C27B0",
            edgecolor="white",
        )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(
            samples.mean(),
            color="#9C27B0",
            linewidth=1.5,
            label=f"Mean = {samples.mean():.4f}",
        )
        ax.set_xlabel("Δ = θ_A − θ_B")
        ax.set_ylabel("Density")
        ax.set_title("Difference Posterior", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        fig.suptitle(
            kwargs.pop("title", "Beta-Bernoulli Posteriors (Non-Paired)"),
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

    def plot_posterior_delta(self, color: str = "#9C27B0", **kwargs: dict) -> None:
        """KDE posterior density of Δ = θ_A − θ_B with 95% CI."""
        import matplotlib.pyplot as plt

        self._check_fitted()
        samples = self.delta_samples
        ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
        mean_val = samples.mean()

        kde = gaussian_kde(samples)
        x_grid = np.linspace(samples.min() - 0.05, samples.max() + 0.05, 500)
        density = kde(x_grid)

        figsize = kwargs.pop("figsize", (7, 5))
        _, ax = plt.subplots(figsize=figsize)
        ax.plot(x_grid, density, color=color, linewidth=2)
        ax.fill_between(x_grid, density, alpha=0.15, color=color)
        mask = (x_grid >= ci_low) & (x_grid <= ci_high)
        ax.fill_between(
            x_grid[mask],
            density[mask],
            alpha=0.35,
            color=color,
            label="95% CI",
        )
        ax.axvline(
            mean_val,
            color=color,
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
            label=f"Mean = {mean_val:.4f}",
        )
        ax.axvline(
            0,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label="Δ = 0 (no difference)",
        )
        ax.set_xlabel("Δ = θ_A − θ_B")
        ax.set_ylabel("Density")
        ax.set_title(
            kwargs.pop("title", "Posterior of Δ (Beta-Bernoulli)"),
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_savage_dickey(self, color: str = "#9C27B0", **kwargs: dict) -> None:
        """Posterior vs prior density of Δ with Savage-Dickey BF annotation."""
        import matplotlib.pyplot as plt

        self._check_fitted()
        bf = self.savage_dickey_test()
        samples = self.delta_samples

        x_grid = np.linspace(samples.min() - 0.1, samples.max() + 0.1, 500)

        # Posterior and prior density via analytic convolution
        density_post = np.array([beta_diff_pdf(z, self.a_A, self.b_A, self.a_B, self.b_B) for z in x_grid])
        density_prior = np.array([beta_diff_pdf(z, self.alpha0, self.beta0, self.alpha0, self.beta0) for z in x_grid])

        figsize = kwargs.pop("figsize", (7, 5))
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_grid, density_post, color=color, linewidth=2, label="Posterior")
        ax.fill_between(x_grid, density_post, alpha=0.15, color=color)
        ax.plot(
            x_grid,
            density_prior,
            color="gray",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label=f"Prior Beta({self.alpha0},{self.beta0})",
        )
        ax.plot(
            0,
            bf.posterior_density_at_0,
            "o",
            color="red",
            markersize=10,
            zorder=5,
            label=f"Post. at Δ=0: {bf.posterior_density_at_0:.2f}",
        )
        ax.plot(
            0,
            bf.prior_density_at_0,
            "s",
            color="gray",
            markersize=8,
            zorder=5,
            label=f"Prior at Δ=0: {bf.prior_density_at_0:.2f}",
        )

        bf10_label = _format_bf(bf.BF_10)
        log10_bf = np.log10(bf.BF_10)
        ax.text(
            0.02,
            0.97,
            f"$BF_{{10}}$ = {bf10_label}\n$\\log_{{10}}BF_{{10}}$ = {log10_bf:.1f}\n{bf.decision}",
            fontsize=10,
            fontweight="bold",
            color="darkred",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="lightyellow",
                edgecolor="darkred",
                alpha=0.9,
            ),
        )
        ax.set_xlabel("Δ = θ_A − θ_B")
        ax.set_ylabel("Density")
        ax.set_title(
            kwargs.pop("title", "Savage-Dickey Test (Beta-Bernoulli)"),
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  Summary / reporting
    # ------------------------------------------------------------------ #

    def print_summary(self) -> None:
        """Print posterior summary, Savage-Dickey test, and PPC p-values."""
        self._check_fitted()

        s = self.summary
        n_A, n_B = len(self.y_A_obs), len(self.y_B_obs)
        verdict = "A wins" if s.p_A_greater_B > 0.95 else ("Tied" if s.p_A_greater_B > 0.5 else "B wins")

        print("Beta-Bernoulli posterior summary (Non-Paired)")
        print("=" * 60)
        print(
            f"  θ_A ~ Beta({self.a_A:.0f}, {self.b_A:.0f})  "
            f"mean={s.theta_A_mean:.4f}  "
            f"[n_A={n_A}, k_A={int(self.y_A_obs.sum())}]"
        )
        print(
            f"  θ_B ~ Beta({self.a_B:.0f}, {self.b_B:.0f})  "
            f"mean={s.theta_B_mean:.4f}  "
            f"[n_B={n_B}, k_B={int(self.y_B_obs.sum())}]"
        )
        print(f"  Mean Δ (θ_A − θ_B):  {s.mean_delta:.4f}")
        print(f"  95% CI:              [{s.ci_95.lower:.4f}, {s.ci_95.upper:.4f}]")
        print(f"  P(A > B):            {s.p_A_greater_B:.4f}  (MC)")
        print(f"  P(B > A):            {self.p_B_greater_A:.4f}  (quadrature)")
        print(f"  Verdict:             {verdict}")

        bf = self.savage_dickey_test()
        print()
        print("Savage-Dickey Bayes Factor: H0 (Δ = 0) vs H1 (Δ ≠ 0)")
        print("=" * 60)
        print(f"  Prior  density at Δ=0: {bf.prior_density_at_0:.4f}")
        print(f"  Post.  density at Δ=0: {bf.posterior_density_at_0:.4f}")
        print(f"  BF_01 (for H0):        {_format_bf(bf.BF_01)}")
        print(f"  BF_10 (against H0):    {_format_bf(bf.BF_10)}")
        print(f"  log₁₀(BF_10):          {np.log10(bf.BF_10):.1f}")
        print(f"  → {bf.interpretation}")
        print(f"  → Decision: {bf.decision}")

        post = self.posterior_probability_H0(bf.BF_01)
        print()
        print("Posterior model probabilities (prior P(H0) = 0.5)")
        print("=" * 60)
        print(f"  P(H0|data): {post.p_H0:.2e}")
        print(f"  P(H1|data): {post.p_H1:.6f}")

        ppc = self.ppc_pvalues()
        print()
        print("Posterior Predictive p-values")
        print("=" * 60)
        print(f"  {'Statistic':<25} {'Observed':>10} {'p-value':>10} {'Status':>8}")
        print("  " + "-" * 55)
        for stat, vals in ppc.items():
            print(f"  {stat:<25} {vals.observed:>10.4f} {vals.p_value:>10.3f} {vals.status:>8}")

        print()
        print("Trace summary")
        print("=" * 60)
        print(self.trace_summary.to_string())

    # ------------------------------------------------------------------ #
    #  Multi-metric comparison
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_forest(
        results: dict[str, "NonPairedBayesPropTest"],
        label_A: str = "Model A",
        label_B: str = "Model B",
        **kwargs: Any,
    ) -> None:
        """Forest plot + P(A>B) bar chart for multiple metrics."""
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        metrics = list(results.keys())
        means = [results[m].summary.mean_delta for m in metrics]
        ci_lows = [results[m].summary.ci_95.lower for m in metrics]
        ci_highs = [results[m].summary.ci_95.upper for m in metrics]
        probs = [results[m].summary.p_A_greater_B for m in metrics]

        colors = ["#2196F3" if p > 0.95 else "#FF9800" if p > 0.5 else "#F44336" for p in probs]
        y_pos = np.arange(len(metrics))

        figsize = kwargs.pop("figsize", (14, max(4, 2 * len(metrics))))
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        for i, (m, ci_l, ci_h, col) in enumerate(zip(means, ci_lows, ci_highs, colors, strict=False)):
            ax.plot(
                [ci_l, ci_h],
                [i, i],
                color=col,
                linewidth=2.5,
                solid_capstyle="round",
            )
            ax.plot(m, i, "o", color=col, markersize=8, zorder=5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics, fontsize=11)
        ax.set_xlabel(f"Mean Δ (θ_A − θ_B)\n← {label_B} better | {label_A} better →")
        ax.set_title("Posterior Mean Difference with 95% CI", fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        ax2 = axes[1]
        ax2.barh(y_pos, probs, color=colors, height=0.5, alpha=0.85)
        ax2.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(metrics, fontsize=11)
        ax2.set_xlabel(f"P({label_A} > {label_B})")
        ax2.set_title("Posterior Probability of Superiority", fontweight="bold")
        ax2.set_xlim(0, 1.05)
        ax2.invert_yaxis()
        ax2.grid(axis="x", alpha=0.3)
        for i, p in enumerate(probs):
            ax2.text(
                p + 0.02,
                i,
                f"{p:.2f}",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        legend_elements = [
            mpatches.Patch(color="#2196F3", label="Strong (P > 0.95)"),
            mpatches.Patch(color="#FF9800", label="Moderate (0.5 < P ≤ 0.95)"),
            mpatches.Patch(color="#F44336", label="Reversed (P ≤ 0.5)"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=3,
            fontsize=9,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.suptitle(
            kwargs.pop(
                "title",
                f"{label_A} vs {label_B} — Beta-Bernoulli Comparison (Non-Paired)",
            ),
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_comparison_table(
        results: dict[str, "NonPairedBayesPropTest"],
    ) -> None:
        """Print a formatted comparison table across metrics."""
        print("=" * 80)
        print(f"{'Metric':<25} {'Mean Δ':>8} {'95% CI':>20} {'P(A>B)':>8} {'Verdict':>12}")
        print("=" * 80)
        for m, model in results.items():
            s = model.summary
            verdict = "A wins" if s.p_A_greater_B > 0.95 else ("Tied" if s.p_A_greater_B > 0.5 else "B wins")
            print(
                f"{m:<25} {s.mean_delta:>8.4f} "
                f"[{s.ci_95.lower:>7.4f}, {s.ci_95.upper:>7.4f}] "
                f"{s.p_A_greater_B:>8.4f} {verdict:>12}"
            )
        print("=" * 80)


def descriptive_summary(
    scores_data: dict,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Build a combined descriptive-stats + Beta-Bernoulli threshold-sweep table.

    Parameters
    ----------
    scores_data : dict
        Must contain keys ``"model_A"``, ``"model_B"``, and ``"metrics"``
        where ``metrics[name]`` has ``"s_A_raw"`` and ``"s_B_raw"`` arrays.
    thresholds : list of float, optional
        Binarization thresholds for the Beta-Bernoulli sweep.
        Default: ``[0.5, 0.7, 0.8, 0.9, 0.95]``.

    Returns:
    -------
    pd.DataFrame
        Multi-indexed by (Metric, Model) with descriptive stats and BB results.
    """
    if thresholds is None:
        thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]

    model_A = scores_data["model_A"]
    model_B = scores_data["model_B"]
    metric_names = list(scores_data["metrics"].keys())

    rows: list[dict] = []
    for metric in metric_names:
        s_A = np.array(scores_data["metrics"][metric]["s_A_raw"])
        s_B = np.array(scores_data["metrics"][metric]["s_B_raw"])
        diff = s_A - s_B

        rows.append(
            {
                "Metric": metric,
                "Model": model_A,
                "n": len(s_A),
                "Mean": s_A.mean(),
                "Std": s_A.std(),
                "Min": s_A.min(),
                "Q25": np.quantile(s_A, 0.25),
                "Median": np.median(s_A),
                "Q75": np.quantile(s_A, 0.75),
                "Max": s_A.max(),
            }
        )
        rows.append(
            {
                "Metric": metric,
                "Model": model_B,
                "n": len(s_B),
                "Mean": s_B.mean(),
                "Std": s_B.std(),
                "Min": s_B.min(),
                "Q25": np.quantile(s_B, 0.25),
                "Median": np.median(s_B),
                "Q75": np.quantile(s_B, 0.75),
                "Max": s_B.max(),
            }
        )
        rows.append(
            {
                "Metric": metric,
                "Model": "Δ (A − B)",
                "n": len(diff),
                "Mean": diff.mean(),
                "Std": diff.std(),
                "Min": diff.min(),
                "Q25": np.quantile(diff, 0.25),
                "Median": np.median(diff),
                "Q75": np.quantile(diff, 0.75),
                "Max": diff.max(),
            }
        )

        for tau in thresholds:
            bb_tau = NonPairedBayesPropTest(threshold=tau)
            res = bb_tau.test(s_A, s_B)
            rows.append(
                {
                    "Metric": metric,
                    "Model": f"BB (τ={tau})",
                    "n": len(s_A),
                    "P(B>A)": res.P_B_greater_A,
                    "θ_A posterior": f"Beta({res.thetaA_post.alpha}, {res.thetaA_post.beta})",
                    "θ_B posterior": f"Beta({res.thetaB_post.alpha}, {res.thetaB_post.beta})",
                }
            )

    df = pd.DataFrame(rows).set_index(["Metric", "Model"])

    n_sample = len(np.array(scores_data["metrics"][metric_names[0]]["s_A_raw"]))
    print(f"Paired LLM scores: {model_A} vs {model_B}  (n={n_sample} per model)")
    print(f"  Model A = {model_A}")
    print(f"  Model B = {model_B}")
    print("  Δ = A − B")
    print("  BB (τ=x) = Beta-Bernoulli test binarized at threshold τ")

    return df
