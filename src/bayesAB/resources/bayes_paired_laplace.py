"""Pooled Bernoulli logistic regression for paired A/B model comparison (Laplace).

This module provides :class:`PairedBayesPropTest`, a self-contained class
for fitting a pooled Bayesian Bernoulli logistic model to paired binary
scores via Laplace approximation (MAP + analytical Hessian), performing
hypothesis testing via the Savage-Dickey density ratio, running
posterior-predictive diagnostics, and generating publication-ready plots.

For exact MCMC inference via Pólya-Gamma data augmentation, see
:mod:`ai_eval.resources.bayes_paired_pg`.

Typical workflow::

    from ai_eval.resources.bayes_paired_laplace import PairedBayesPropTest

    model = PairedBayesPropTest(seed=42).fit(y_A, y_B)
    model.print_summary()
    model.plot_posterior_delta()
    model.plot_savage_dickey()

For multi-metric comparisons::

    results = {"Relevancy": model_rel, "Faithfulness": model_faith}
    PairedBayesPropTest.plot_forest(results, label_A="v2", label_B="v1")
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, norm

from bayesAB.resources.data_schemas import (
    CredibleInterval,
    DecisionRuleType,
    HypothesisDecision,
    PairedSummary,
    PosteriorProbH0Result,
    PPCStatistic,
    ROPEResult,
    SavageDickeyResult,
)


def sigmoid(x: npt.ArrayLike) -> np.ndarray:
    """Element-wise sigmoid (logistic) function."""
    return 1.0 / (1.0 + np.exp(-x))


def _format_bf(value: float) -> str:
    """Format a Bayes Factor for human-readable display."""
    if value > 1e4:
        return f"10^{np.log10(value):.0f}"
    elif value < 1e-4 and value > 0:
        return f"10^{np.log10(value):.0f}"
    else:
        return f"{value:.2f}"


class PairedBayesPropTest:
    """Pooled Bernoulli logistic model for paired A/B comparison.

    Uses Laplace approximation (MAP + Hessian) instead of full MCMC
    for fast, analytic posterior inference on binarized scores.

    Generative model::

        μ      ~ N(0, 2)              (overall intercept)
        δ_A    ~ N(0, σ_δ)            (model-A advantage)
        y_A,i  ~ Bernoulli(σ(μ + δ_A))
        y_B,i  ~ Bernoulli(σ(μ))

    Only 2 parameters — no confounding between item effects and model
    effect when >90% of pairs are concordant.

    The prior width on ``delta_A`` is fixed (not learned) so the
    Savage-Dickey density ratio remains exactly consistent.

    Attributes:
        laplace: Dict with MAP estimate, covariance, Hessian, and
            posterior samples (``None`` before :meth:`fit`).
        summary: Dict with ``mean_delta``, ``ci_95``, ``P(A > B)``,
            and ``delta_A_posterior_mean`` on the probability scale.
        trace_summary: ``pandas.DataFrame`` with posterior summary
            for ``delta_A`` and ``mu``.
        delta_A_samples: 1-D array of posterior draws for ``delta_A``
            (logit scale), shape ``(n_samples,)``.
        y_A_obs: Observed binary scores for model A (set by :meth:`fit`).
        y_B_obs: Observed binary scores for model B (set by :meth:`fit`).
    """

    def __init__(
        self,
        prior_sigma_delta: float = 1.0,
        seed: int = 0,
        n_samples: int = 8000,
        decision_rule: DecisionRuleType = "all",
        rope_epsilon: float = 0.02,
    ) -> None:
        """Initialise model configuration.

        Args:
            prior_sigma_delta: Standard deviation of the N(0, σ) prior on
                ``delta_A`` (logit scale).
            seed: Random seed for reproducibility.
            n_samples: Number of draws from the Laplace posterior.
            decision_rule: Default decision framework — one of
                ``"bayes_factor"``, ``"posterior_null"``, ``"rope"``, or ``"all"``.
            rope_epsilon: Half-width of the ROPE interval (default 0.02 = 2 pp).
        """
        self.prior_sigma_delta: float = prior_sigma_delta
        self.seed: int = seed
        self.n_samples: int = n_samples
        self.decision_rule: DecisionRuleType = decision_rule
        self.rope_epsilon: float = rope_epsilon

        # --- Populated by .fit() ---
        self.laplace: dict[str, Any] | None = None
        self.summary: dict[str, Any] | None = None
        self.trace_summary: pd.DataFrame | None = None
        self.delta_A_samples: np.ndarray | None = None
        self.delta_samples: np.ndarray | None = None
        self.y_A_obs: np.ndarray | None = None
        self.y_B_obs: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  Fitting
    # ------------------------------------------------------------------ #

    def fit(self, y_A_obs: np.ndarray, y_B_obs: np.ndarray) -> PairedBayesPropTest:
        """Fit the pooled Bernoulli model via Laplace approximation.

        Analytically computes the MAP via L-BFGS-B on the negative
        log-posterior, then evaluates the closed-form Hessian at the
        MAP to obtain the Laplace covariance: Σ = H⁻¹.

        Log-posterior (up to constant)::

            log p(μ, δ|y) = Σᵢ [y_Aᵢ log σ(μ+δ) + (1-y_Aᵢ) log(1-σ(μ+δ))]
                          + Σᵢ [y_Bᵢ log σ(μ)   + (1-y_Bᵢ) log(1-σ(μ))]
                          - μ²/(2σ_μ²) - δ²/(2σ_δ²)

        Gradient::

            ∂/∂μ = (k_A - n_A·p_A) + (k_B - n_B·p_B) - μ/σ_μ²
            ∂/∂δ = (k_A - n_A·p_A)                    - δ/σ_δ²

        Hessian of the *negative* log-posterior (observed information)::

            H[0,0] = n_A·w_A + n_B·w_B + 1/σ_μ²
            H[1,1] = n_A·w_A           + 1/σ_δ²
            H[0,1] = H[1,0] = n_A·w_A

        where w_A = p_A(1-p_A), w_B = p_B(1-p_B), evaluated at MAP.

        Args:
            y_A_obs: Binary observed scores for model A (0 or 1).
            y_B_obs: Binary observed scores for model B (0 or 1).

        Returns:
            self (for method chaining).
        """
        self.y_A_obs = np.asarray(y_A_obs, dtype=int)
        self.y_B_obs = np.asarray(y_B_obs, dtype=int)

        rng = np.random.default_rng(self.seed)

        n_A = len(self.y_A_obs)
        k_A = int(self.y_A_obs.sum())
        n_B = len(self.y_B_obs)
        k_B = int(self.y_B_obs.sum())
        sigma_mu = 2.0
        sigma_delta = self.prior_sigma_delta

        def neg_log_post(params: np.ndarray) -> float:
            mu, delta = params
            p_A = sigmoid(mu + delta)
            p_B = sigmoid(mu)
            # Clip to avoid log(0)
            eps = 1e-15
            p_A = np.clip(p_A, eps, 1 - eps)
            p_B = np.clip(p_B, eps, 1 - eps)
            ll = k_A * np.log(p_A) + (n_A - k_A) * np.log(1 - p_A)
            ll += k_B * np.log(p_B) + (n_B - k_B) * np.log(1 - p_B)
            lp = -(mu**2) / (2 * sigma_mu**2) - delta**2 / (2 * sigma_delta**2)
            return -(ll + lp)

        def neg_log_post_grad(params: np.ndarray) -> np.ndarray:
            mu, delta = params
            p_A = sigmoid(mu + delta)
            p_B = sigmoid(mu)
            g_mu = (k_A - n_A * p_A) + (k_B - n_B * p_B) - mu / sigma_mu**2
            g_delta = (k_A - n_A * p_A) - delta / sigma_delta**2
            return -np.array([g_mu, g_delta])

        result = minimize(
            neg_log_post,
            x0=np.array([0.0, 0.0]),
            jac=neg_log_post_grad,
            method="L-BFGS-B",
        )
        mu_map, delta_map = result.x

        # Analytical Hessian of negative log-posterior at MAP
        p_A_map = sigmoid(mu_map + delta_map)
        p_B_map = sigmoid(mu_map)
        w_A = p_A_map * (1 - p_A_map)
        w_B = p_B_map * (1 - p_B_map)

        H = np.array(
            [
                [n_A * w_A + n_B * w_B + 1 / sigma_mu**2, n_A * w_A],
                [n_A * w_A, n_A * w_A + 1 / sigma_delta**2],
            ]
        )
        cov = np.linalg.inv(H)

        # Sample from Gaussian posterior
        samples = rng.multivariate_normal([mu_map, delta_map], cov, size=self.n_samples)
        mu_s = samples[:, 0]
        delta_A_s = samples[:, 1]

        pA_s = sigmoid(mu_s + delta_A_s)
        pB_s = sigmoid(mu_s)
        Delta_s = pA_s - pB_s

        self.delta_A_samples = delta_A_s
        self.delta_samples = Delta_s

        self.summary = PairedSummary(
            mean_delta=float(Delta_s.mean()),
            ci_95=CredibleInterval(
                lower=float(np.quantile(Delta_s, 0.025)),
                upper=float(np.quantile(Delta_s, 0.975)),
            ),
            **{"P(A > B)": float((Delta_s > 0).mean())},
            delta_A_posterior_mean=float(delta_A_s.mean()),
        )

        self.trace_summary = pd.DataFrame(
            {
                "mean": [delta_A_s.mean(), mu_s.mean()],
                "sd": [delta_A_s.std(), mu_s.std()],
                "hdi_3%": [
                    np.quantile(delta_A_s, 0.03),
                    np.quantile(mu_s, 0.03),
                ],
                "hdi_97%": [
                    np.quantile(delta_A_s, 0.97),
                    np.quantile(mu_s, 0.97),
                ],
                "MAP": [delta_map, mu_map],
            },
            index=["delta_A", "mu"],
        )

        self.laplace = {
            "map": np.array([mu_map, delta_map]),
            "cov": cov,
            "H": H,
            "mu_samples": mu_s,
            "delta_A_samples": delta_A_s,
            "n_A": len(self.y_A_obs),
            "k_A": int(self.y_A_obs.sum()),
            "n_B": len(self.y_B_obs),
            "k_B": int(self.y_B_obs.sum()),
            "prior_sigma_delta": self.prior_sigma_delta,
        }

        return self

    def _check_fitted(self) -> None:
        """Raise RuntimeError if the model has not been fitted yet."""
        if self.laplace is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    # ------------------------------------------------------------------ #
    #  Hypothesis testing
    # ------------------------------------------------------------------ #

    def savage_dickey_test(self, null_value: float = 0.0) -> SavageDickeyResult:
        """Savage-Dickey density-ratio Bayes factor for H0: delta_A = *null_value*.

        Args:
            null_value: The point null hypothesis value for delta_A.

        Returns:
            :class:`SavageDickeyResult` with BF_01, BF_10, densities,
            interpretation, and decision.
        """
        self._check_fitted()

        kde = gaussian_kde(self.delta_A_samples)
        posterior_at_null = float(kde(null_value)[0])
        prior_at_null = float(norm.pdf(null_value, 0, self.prior_sigma_delta))

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
        """ROPE analysis on the posterior of Δ = p_A − p_B (probability scale).

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
            assert bf is not None  # noqa: S101
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

        mu_s = self.laplace["mu_samples"]
        delta_s = self.laplace["delta_A_samples"]
        n = len(self.y_A_obs)

        p_A_s = sigmoid(mu_s + delta_s)
        p_B_s = sigmoid(mu_s)
        y_A_rep = (rng.random((len(p_A_s), n)) < p_A_s[:, None]).astype(int)
        y_B_rep = (rng.random((len(p_B_s), n)) < p_B_s[:, None]).astype(int)

        checks = {
            "mean(y_A)": (self.y_A_obs.mean(), y_A_rep.mean(axis=1)),
            "mean(y_B)": (self.y_B_obs.mean(), y_B_rep.mean(axis=1)),
            "mean(y_A-y_B)": (
                (self.y_A_obs - self.y_B_obs).mean(),
                (y_A_rep - y_B_rep).mean(axis=1),
            ),
            "std(y_A-y_B)": (
                (self.y_A_obs - self.y_B_obs).std(),
                (y_A_rep - y_B_rep).std(axis=1),
            ),
            "n_disagree": (
                float(np.sum(self.y_A_obs != self.y_B_obs)),
                np.sum(y_A_rep != y_B_rep, axis=1).astype(float),
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

    def plot_laplace_posterior(self, **kwargs: Any) -> None:
        """Two-panel posterior plot: overlaid p_A / p_B and Δ = p_A − p_B.

        The implied success probabilities ``p_A = σ(μ + δ_A)`` and
        ``p_B = σ(μ)`` are computed from the Laplace posterior samples
        and displayed as overlaid KDE densities in the left panel.
        The right panel shows the difference Δ = p_A − p_B.

        Args:
            **kwargs: Accepts ``figsize`` (default ``(14, 5)``) and
                ``title`` (default ``"Laplace Posterior (Pooled Binomial)"``).
        """
        import matplotlib.pyplot as plt

        self._check_fitted()
        assert self.laplace is not None
        mu_s = self.laplace["mu_samples"]
        delta_s = self.laplace["delta_A_samples"]

        p_A_s = sigmoid(mu_s + delta_s)
        p_B_s = sigmoid(mu_s)
        Delta_s = p_A_s - p_B_s

        figsize = kwargs.pop("figsize", (14, 5))
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Panel 1: p_A and p_B overlaid
        ax = axes[0]
        kde_A = gaussian_kde(p_A_s)
        kde_B = gaussian_kde(p_B_s)
        lo = min(p_A_s.min(), p_B_s.min())
        hi = max(p_A_s.max(), p_B_s.max())
        x = np.linspace(max(0, lo - 0.05), min(1, hi + 0.05), 500)

        pdf_A = kde_A(x)
        pdf_B = kde_B(x)
        ax.plot(
            x,
            pdf_A,
            color="#2196F3",
            linewidth=2,
            label=f"p_A = σ(μ+δ_A)  mean={p_A_s.mean():.3f}",
        )
        ax.fill_between(x, pdf_A, alpha=0.15, color="#2196F3")
        ax.plot(
            x,
            pdf_B,
            color="#4CAF50",
            linewidth=2,
            label=f"p_B = σ(μ)  mean={p_B_s.mean():.3f}",
        )
        ax.fill_between(x, pdf_B, alpha=0.15, color="#4CAF50")

        ax.axvline(p_A_s.mean(), color="#2196F3", linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(p_B_s.mean(), color="#4CAF50", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Success probability")
        ax.set_ylabel("Density")
        ax.set_title("Implied Probability Posteriors", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Panel 2: Δ = p_A − p_B
        ax = axes[1]
        ax.hist(
            Delta_s,
            bins=60,
            density=True,
            alpha=0.6,
            color="#9C27B0",
            edgecolor="white",
        )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(
            Delta_s.mean(),
            color="#9C27B0",
            linewidth=1.5,
            label=f"Mean = {Delta_s.mean():.4f}",
        )
        ax.set_xlabel("Δ = p_A − p_B")
        ax.set_ylabel("Density")
        ax.set_title("Difference Posterior", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        fig.suptitle(
            kwargs.pop("suptitle", kwargs.pop("title", "Laplace Posterior (Pooled Binomial)")),
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

    def plot_posterior_delta(self, color: str = "#2196F3", **kwargs) -> None:
        """KDE posterior density of delta_A (logit scale) with 95% CI."""
        import matplotlib.pyplot as plt

        self._check_fitted()
        samples = self.delta_A_samples
        ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
        mean_val = samples.mean()

        kde = gaussian_kde(samples)
        x_grid = np.linspace(samples.min() - 0.5, samples.max() + 0.5, 500)
        density = kde(x_grid)

        figsize = kwargs.pop("figsize", (7, 5))
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_grid, density, color=color, linewidth=2)
        ax.fill_between(x_grid, density, alpha=0.15, color=color)
        mask = (x_grid >= ci_low) & (x_grid <= ci_high)
        ax.fill_between(x_grid[mask], density[mask], alpha=0.35, color=color, label="95% CI")
        ax.axvline(
            mean_val,
            color=color,
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
            label=f"Mean = {mean_val:.3f}",
        )
        ax.axvline(
            0,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label="\u03b4_A = 0 (no difference)",
        )
        ax.set_xlabel(kwargs.pop("xlabel", "\u03b4_A (logit scale)"), fontsize=11)
        ax.set_ylabel(kwargs.pop("ylabel", "Density"), fontsize=11)
        ax.set_title(
            kwargs.pop("title", "Posterior of \u03b4_A (Binomial)"),
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_savage_dickey(self, color: str = "#2196F3", **kwargs) -> None:
        """Posterior vs prior density with Savage-Dickey BF annotation."""
        import matplotlib.pyplot as plt

        bf = self.savage_dickey_test()
        samples = self.delta_A_samples

        kde = gaussian_kde(samples)
        x_grid = np.linspace(samples.min() - 0.5, samples.max() + 0.5, 500)
        density = kde(x_grid)
        prior_density = norm.pdf(x_grid, 0, self.prior_sigma_delta)

        figsize = kwargs.pop("figsize", (7, 5))
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_grid, density, color=color, linewidth=2, label="Posterior")
        ax.fill_between(x_grid, density, alpha=0.15, color=color)
        ax.plot(
            x_grid,
            prior_density,
            color="gray",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label=f"Prior N(0,{self.prior_sigma_delta})",
        )
        ax.plot(
            0,
            bf.posterior_density_at_0,
            "o",
            color="red",
            markersize=10,
            zorder=5,
            label=f"Post. at \u03b4=0: {bf.posterior_density_at_0:.2e}",
        )
        ax.plot(
            0,
            bf.prior_density_at_0,
            "s",
            color="gray",
            markersize=8,
            zorder=5,
            label=f"Prior at \u03b4=0: {bf.prior_density_at_0:.3f}",
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
        ax.set_xlabel(kwargs.pop("xlabel", "\u03b4_A (logit scale)"), fontsize=11)
        ax.set_ylabel(kwargs.pop("ylabel", "Density"), fontsize=11)
        ax.set_title(
            kwargs.pop("title", "Savage-Dickey Test (Binomial)"),
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_ppc(self, seed: int | None = None, **kwargs) -> None:
        """Three-column PPC plot: P(perfect) A, P(perfect) B, rate difference."""
        import matplotlib.pyplot as plt

        self._check_fitted()
        rng = np.random.default_rng(seed if seed is not None else self.seed)

        mu_s = self.laplace["mu_samples"]
        delta_s = self.laplace["delta_A_samples"]
        n = len(self.y_A_obs)

        p_A_s = sigmoid(mu_s + delta_s)
        p_B_s = sigmoid(mu_s)
        y_A_rep = (rng.random((len(p_A_s), n)) < p_A_s[:, None]).astype(int)
        y_B_rep = (rng.random((len(p_B_s), n)) < p_B_s[:, None]).astype(int)

        figsize = kwargs.pop("figsize", (18, 5))
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # P(perfect) Model A
        ax = axes[0]
        frac_A_rep = y_A_rep.mean(axis=1)
        frac_A_obs = self.y_A_obs.mean()
        ax.hist(
            frac_A_rep,
            bins=40,
            density=True,
            color="#2196F3",
            alpha=0.6,
            edgecolor="white",
            label="Replicated",
        )
        ax.axvline(
            frac_A_obs,
            color="#E53935",
            linewidth=2.5,
            label=f"Observed = {frac_A_obs:.3f}",
            zorder=10,
        )
        ax.set_xlabel("Fraction perfect (y=1)")
        ax.set_ylabel("Density")
        ax.set_title("PPC: P(perfect) Model A", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # P(perfect) Model B
        ax = axes[1]
        frac_B_rep = y_B_rep.mean(axis=1)
        frac_B_obs = self.y_B_obs.mean()
        ax.hist(
            frac_B_rep,
            bins=40,
            density=True,
            color="#4CAF50",
            alpha=0.6,
            edgecolor="white",
            label="Replicated",
        )
        ax.axvline(
            frac_B_obs,
            color="#E53935",
            linewidth=2.5,
            label=f"Observed = {frac_B_obs:.3f}",
            zorder=10,
        )
        ax.set_xlabel("Fraction perfect (y=1)")
        ax.set_ylabel("Density")
        ax.set_title("PPC: P(perfect) Model B", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Rate difference
        ax = axes[2]
        diff_rep = frac_A_rep - frac_B_rep
        diff_obs = frac_A_obs - frac_B_obs
        ax.hist(
            diff_rep,
            bins=40,
            density=True,
            color="#9C27B0",
            alpha=0.6,
            edgecolor="white",
            label="Replicated",
        )
        ax.axvline(
            diff_obs,
            color="#E53935",
            linewidth=2.5,
            label=f"Observed = {diff_obs:.3f}",
            zorder=10,
        )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("P(perfect)_A \u2212 P(perfect)_B")
        ax.set_ylabel("Density")
        ax.set_title("PPC: Rate Difference (A \u2212 B)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        fig.suptitle(
            kwargs.pop(
                "suptitle",
                kwargs.pop("title", "Posterior Predictive Checks (Laplace Binomial)"),
            ),
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

    def plot_sensitivity(self, prior_H0: float = 0.5, **kwargs) -> None:
        """Two-panel sensitivity: P(H0|data) vs prior P(H0), and slab-width sweep."""
        import matplotlib.pyplot as plt

        bf = self.savage_dickey_test()

        figsize = kwargs.pop("figsize", (14, 5))
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Left: P(H0|data) vs prior P(H0)
        ax = axes[0]
        prior_grid = np.linspace(0.01, 0.99, 200)
        p_h0_grid = [self.posterior_probability_H0(bf.BF_01, p).p_H0 for p in prior_grid]
        bf10 = bf.BF_10
        bf_label = f"log\u2081\u2080BF\u2081\u2080={np.log10(bf10):.0f}" if bf10 > 1e4 else f"BF\u2081\u2080={bf10:.1f}"
        ax.plot(prior_grid, p_h0_grid, linewidth=2, label=bf_label)
        ax.axhline(0.05, color="red", linestyle="--", alpha=0.5, label="P(H\u2080)=0.05")
        ax.axvline(prior_H0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Prior P(H\u2080)")
        ax.set_ylabel("Posterior P(H\u2080 | data)")
        ax.set_title("Sensitivity: P(H\u2080 | data) vs Prior P(H\u2080)", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Right: BF_10 vs slab width sigma_s
        ax2 = axes[1]
        sigma_grid = np.linspace(0.25, 5.0, 100)
        kde = gaussian_kde(self.delta_A_samples)
        post_at_0 = float(kde(0.0)[0])
        bf10_vals = [norm.pdf(0, 0, s) / post_at_0 for s in sigma_grid]
        ax2.plot(sigma_grid, bf10_vals, linewidth=2)
        ax2.axhline(3, color="red", linestyle="--", alpha=0.5, label="BF\u2081\u2080 = 3")
        ax2.axhline(1, color="gray", linestyle=":", alpha=0.5, label="BF\u2081\u2080 = 1")
        ax2.axvline(
            self.prior_sigma_delta,
            color="gray",
            linestyle="--",
            alpha=0.3,
            label=f"\u03c3_s = {self.prior_sigma_delta} (used)",
        )
        ax2.set_xlabel("Slab width \u03c3_s")
        ax2.set_ylabel("BF\u2081\u2080")
        ax2.set_title("Sensitivity: BF\u2081\u2080 vs Slab Width", fontweight="bold")
        ax2.set_yscale("log")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        fig.suptitle(
            kwargs.pop(
                "suptitle",
                kwargs.pop("title", "Jeffreys-Lindley Sensitivity (Binomial)"),
            ),
            fontsize=13,
            fontweight="bold",
            y=1.04,
        )
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  Summary / reporting
    # ------------------------------------------------------------------ #

    def print_summary(self) -> None:
        """Print posterior summary, Savage-Dickey test, and PPC p-values."""
        self._check_fitted()

        mu_map, delta_map = self.laplace["map"]
        cov = self.laplace["cov"]

        # Laplace posterior info
        s = self.summary
        verdict = "A wins" if s.p_A_greater_B > 0.95 else ("Tied" if s.p_A_greater_B > 0.5 else "B wins")
        print("Laplace posterior summary")
        print("=" * 60)
        print(f"  MAP: \u03bc={mu_map:.4f}, \u03b4_A={delta_map:.4f}")
        print(f"  Posterior sd: \u03bc={np.sqrt(cov[0, 0]):.4f}, \u03b4_A={np.sqrt(cov[1, 1]):.4f}")
        print(f"  Correlation: {cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]):.3f}")
        print(f"  Mean \u0394 (prob scale):  {s.mean_delta:.4f}")
        print(f"  95% CI:               [{s.ci_95.lower:.4f}, {s.ci_95.upper:.4f}]")
        print(f"  P(A > B):             {s.p_A_greater_B:.4f}")
        print(f"  \u03b4_A (logit scale):    {s.delta_A_posterior_mean:.4f}")
        print(f"  Verdict:              {verdict}")

        # Savage-Dickey
        bf = self.savage_dickey_test()
        print()
        print("Savage-Dickey Bayes Factor: H0 (\u03b4_A = 0) vs H1 (\u03b4_A \u2260 0)")
        print("=" * 60)
        print(f"  Prior  density at \u03b4=0: {bf.prior_density_at_0:.6f}")
        print(f"  Post.  density at \u03b4=0: {bf.posterior_density_at_0:.2e}")
        print(f"  BF_01 (for H0):        {_format_bf(bf.BF_01)}")
        print(f"  BF_10 (against H0):    {_format_bf(bf.BF_10)}")
        print(f"  log\u2081\u2080(BF_10):          {np.log10(bf.BF_10):.1f}")
        print(f"  \u2192 {bf.interpretation}")
        print(f"  \u2192 Decision: {bf.decision}")

        # Posterior probability of H0
        post = self.posterior_probability_H0(bf.BF_01)
        print()
        print("Posterior model probabilities (prior P(H0) = 0.5)")
        print("=" * 60)
        print(f"  P(H0|data): {post.p_H0:.2e}")
        print(f"  P(H1|data): {post.p_H1:.6f}")

        # PPC p-values
        ppc = self.ppc_pvalues()
        print()
        print("Posterior Predictive p-values")
        print("=" * 60)
        print(f"  {'Statistic':<20} {'Observed':>10} {'p-value':>10} {'Status':>8}")
        print("  " + "-" * 50)
        for stat, vals in ppc.items():
            print(f"  {stat:<20} {vals.observed:>10.4f} {vals.p_value:>10.3f} {vals.status:>8}")

        # Trace summary
        print()
        print("Laplace trace diagnostics")
        print("=" * 60)
        print(self.trace_summary.to_string())

    # ------------------------------------------------------------------ #
    #  Multi-model comparison (class method)
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_forest(
        results: dict[str, "PairedBayesPropTest"],
        label_A: str = "Model A",
        label_B: str = "Model B",
        **kwargs,
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
            ax.plot([ci_l, ci_h], [i, i], color=col, linewidth=2.5, solid_capstyle="round")
            ax.plot(m, i, "o", color=col, markersize=8, zorder=5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics, fontsize=11)
        ax.set_xlabel(f"Mean \u0394 P(perfect)\n\u2190 {label_B} better | {label_A} better \u2192")
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
            ax2.text(p + 0.02, i, f"{p:.2f}", va="center", fontsize=10, fontweight="bold")

        legend_elements = [
            mpatches.Patch(color="#2196F3", label="Strong (P > 0.95)"),
            mpatches.Patch(color="#FF9800", label="Moderate (0.5 < P \u2264 0.95)"),
            mpatches.Patch(color="#F44336", label="Reversed (P \u2264 0.5)"),
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
                "suptitle",
                kwargs.pop("title", f"{label_A} vs {label_B} \u2014 Pooled Binomial Comparison"),
            ),
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_comparison_table(results: dict[str, "PairedBayesPropTest"]) -> None:
        """Print a formatted comparison table across metrics."""
        print("=" * 80)
        print(f"{'Metric':<25} {'Mean \u0394':>8} {'95% CI':>20} {'P(A>B)':>8} {'Verdict':>12}")
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
