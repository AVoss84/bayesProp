"""Pooled Bernoulli logistic regression for paired A/B model comparison (Pólya-Gamma).

This module provides :class:`PairedBayesPropTestPG`, a class for fitting
a pooled Bayesian Bernoulli logistic model to paired binary scores using
exact Pólya-Gamma data augmentation (Gibbs sampling), performing hypothesis
testing via the Savage-Dickey density ratio, running posterior-predictive
diagnostics, and generating publication-ready plots.

Compared to the Laplace approximation in :mod:`ai_eval.resources.bayes_paired_laplace`,
the PG sampler provides exact (up to MCMC error) posterior inference and
multi-chain MCMC diagnostics (R-hat, ESS).

Typical workflow::

    from ai_eval.resources.bayes_paired_pg import PairedBayesPropTestPG

    model = PairedBayesPropTestPG(seed=42).fit(y_A, y_B)
    model.print_summary()
    model.plot_trace()
    model.plot_posterior_delta()

For multi-metric comparisons::

    results = {"Relevancy": model_rel, "Faithfulness": model_faith}
    PairedBayesPropTestPG.plot_forest(results, label_A="v2", label_B="v1")
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.linalg import solve
from polyagamma import random_polyagamma
from scipy.stats import gaussian_kde, norm

from bayesAB.resources.data_schemas import (
    CredibleInterval,
    MCMCDiagnostics,
    MCMCParamDiagnostic,
    PairedSummary,
    PosteriorProbH0Result,
    PPCStatistic,
    SavageDickeyResult,
)


def sigmoid(x: npt.ArrayLike) -> np.ndarray:
    """Numerically stable element-wise sigmoid function."""
    x = np.asarray(x, dtype=float)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _format_bf(value: float) -> str:
    """Format a Bayes Factor for human-readable display."""
    if value > 1e4:
        return f"10^{np.log10(value):.0f}"
    elif value < 1e-4 and value > 0:
        return f"10^{np.log10(value):.0f}"
    else:
        return f"{value:.2f}"


def _build_design_matrix(y_A: np.ndarray, y_B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stack paired binary outcomes into a design matrix for logistic regression.

    Returns:
        (X, y) where X is (2n, 2) with columns [intercept, d_A]
        and y is (2n,) stacked binary outcomes.
    """
    n = len(y_A)
    X_A = np.column_stack([np.ones(n), np.ones(n)])  # intercept=1, d_A=1
    X_B = np.column_stack([np.ones(n), np.zeros(n)])  # intercept=1, d_A=0
    X = np.vstack([X_A, X_B])
    y = np.concatenate([y_A, y_B])
    return X, y


class PairedBayesPropTestPG:
    """Pooled Bernoulli logistic model for paired A/B comparison (PG Gibbs).

    Uses Pólya-Gamma data augmentation for exact Gibbs sampling instead
    of Laplace approximation.

    Generative model (identical to the Laplace version)::

        μ      ~ N(0, σ_μ)            (overall intercept)
        δ_A    ~ N(0, σ_δ)            (model-A advantage)
        y_A,i  ~ Bernoulli(σ(μ + δ_A))
        y_B,i  ~ Bernoulli(σ(μ))

    Inference proceeds by augmenting with Pólya-Gamma latent variables
    ω_i ~ PG(1, x_i'β), which yields conjugate Gaussian conditionals
    for β = [μ, δ_A].

    Multiple independent chains are run for MCMC diagnostics (R-hat, ESS).

    Attributes:
        chains: Array of shape ``(n_chains, n_samples, 2)`` with posterior
            draws for ``[mu, delta_A]`` per chain (``None`` before :meth:`fit`).
        samples: Pooled posterior draws, shape ``(n_total, 2)``.
        summary: Dict with ``mean_delta``, ``ci_95``, ``P(A > B)``,
            and ``delta_A_posterior_mean`` on the probability scale.
        trace_summary: ``pandas.DataFrame`` with posterior summary
            for ``delta_A`` and ``mu`` including R-hat and ESS.
        delta_A_samples: 1-D array of pooled posterior draws for ``delta_A``
            (logit scale).
        y_A_obs: Observed binary scores for model A (set by :meth:`fit`).
        y_B_obs: Observed binary scores for model B (set by :meth:`fit`).
    """

    def __init__(
        self,
        prior_sigma_delta: float = 1.0,
        prior_sigma_mu: float = 2.0,
        seed: int = 0,
        n_iter: int = 2000,
        burn_in: int = 500,
        n_chains: int = 4,
    ) -> None:
        """Initialise model configuration.

        Args:
            prior_sigma_delta: Standard deviation of the N(0, σ) prior
                on ``delta_A`` (logit scale).
            prior_sigma_mu: Standard deviation of the N(0, σ) prior
                on ``mu`` (logit scale).
            seed: Random seed for reproducibility.
            n_iter: Total Gibbs iterations per chain (including burn-in).
            burn_in: Number of warm-up iterations to discard per chain.
            n_chains: Number of independent MCMC chains.
        """
        self.prior_sigma_delta: float = prior_sigma_delta
        self.prior_sigma_mu: float = prior_sigma_mu
        self.seed: int = seed
        self.n_iter: int = n_iter
        self.burn_in: int = burn_in
        self.n_chains: int = n_chains

        # --- Populated by .fit() ---
        self.chains: np.ndarray | None = None
        self.samples: np.ndarray | None = None
        self.summary: dict[str, Any] | None = None
        self.trace_summary: pd.DataFrame | None = None
        self.delta_A_samples: np.ndarray | None = None
        self.y_A_obs: np.ndarray | None = None
        self.y_B_obs: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  Internal sampler
    # ------------------------------------------------------------------ #

    def _run_single_chain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Run one PG Gibbs chain.

        Returns:
            Array of shape ``(n_iter - burn_in, 2)`` with posterior draws.
        """
        n, p = X.shape
        kappa = y - 0.5

        # Prior precision
        B0_inv = np.diag(
            [
                1.0 / self.prior_sigma_mu**2,
                1.0 / self.prior_sigma_delta**2,
            ]
        )
        b0 = np.zeros(p)

        beta = np.zeros(p)
        samples = []

        for it in range(self.n_iter):
            eta = X @ beta

            # Pólya-Gamma step: ω_i ~ PG(1, η_i)
            omega = random_polyagamma(1.0, eta, random_state=rng)

            # Posterior conditional: β | ω, y
            A = X.T @ np.diag(omega) @ X + B0_inv
            rhs = X.T @ kappa + B0_inv @ b0
            Sigma = solve(A, np.eye(p))
            mu_post = Sigma @ rhs

            beta = rng.multivariate_normal(mu_post, Sigma)

            if it >= self.burn_in:
                samples.append(beta.copy())

        return np.array(samples)

    # ------------------------------------------------------------------ #
    #  Fitting
    # ------------------------------------------------------------------ #

    def fit(self, y_A_obs: np.ndarray, y_B_obs: np.ndarray) -> PairedBayesPropTestPG:
        """Fit the model via PG Gibbs sampling with multiple chains.

        Args:
            y_A_obs: Binary observed scores for model A (0 or 1).
            y_B_obs: Binary observed scores for model B (0 or 1).

        Returns:
            self (for method chaining).
        """
        self.y_A_obs = np.asarray(y_A_obs, dtype=int)
        self.y_B_obs = np.asarray(y_B_obs, dtype=int)

        X, y = _build_design_matrix(self.y_A_obs, self.y_B_obs)

        # Run multiple chains with different seeds
        seed_seq = np.random.SeedSequence(self.seed)
        child_seeds = seed_seq.spawn(self.n_chains)

        chain_list = []
        for i in range(self.n_chains):
            rng = np.random.default_rng(child_seeds[i])
            chain_samples = self._run_single_chain(X, y, rng)
            chain_list.append(chain_samples)

        self.chains = np.array(chain_list)  # (n_chains, n_samples, 2)
        self.samples = self.chains.reshape(-1, 2)  # pooled

        mu_s = self.samples[:, 0]
        delta_A_s = self.samples[:, 1]
        self.delta_A_samples = delta_A_s

        pA_s = sigmoid(mu_s + delta_A_s)
        pB_s = sigmoid(mu_s)
        Delta_s = pA_s - pB_s

        self.summary = PairedSummary(
            mean_delta=float(Delta_s.mean()),
            ci_95=CredibleInterval(
                lower=float(np.quantile(Delta_s, 0.025)),
                upper=float(np.quantile(Delta_s, 0.975)),
            ),
            **{"P(A > B)": float((Delta_s > 0).mean())},
            delta_A_posterior_mean=float(delta_A_s.mean()),
        )

        # MCMC diagnostics
        diag = self.mcmc_diagnostics()

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
                "R-hat": [diag.delta_A.r_hat, diag.mu.r_hat],
                "ESS": [diag.delta_A.ess, diag.mu.ess],
            },
            index=["delta_A", "mu"],
        )

        return self

    def _check_fitted(self) -> None:
        """Raise RuntimeError if the model has not been fitted yet."""
        if self.samples is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    # ------------------------------------------------------------------ #
    #  MCMC diagnostics
    # ------------------------------------------------------------------ #

    @staticmethod
    def _r_hat(chains: np.ndarray) -> float:
        """Gelman-Rubin R-hat for a single parameter.

        Args:
            chains: ``(n_chains, n_samples)`` array for one parameter.
        """
        m, n = chains.shape
        chain_means = chains.mean(axis=1)
        grand_mean = chain_means.mean()
        B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
        W = np.mean(chains.var(axis=1, ddof=1))
        var_hat = (n - 1) / n * W + B / n
        return float(np.sqrt(var_hat / W)) if W > 0 else float("nan")

    @staticmethod
    def _ess(chains: np.ndarray) -> float:
        """Effective sample size (FFT-based autocorrelation estimate).

        Args:
            chains: ``(n_chains, n_samples)`` array for one parameter.
        """
        m, n = chains.shape
        total = 0.0
        for chain in chains:
            x = chain - chain.mean()
            # FFT-based autocorrelation
            f = np.fft.fft(x, n=2 * n)
            acf = np.fft.ifft(f * np.conj(f)).real[:n]
            acf /= acf[0]
            # Sum pairs until negative
            for t in range(1, n - 1, 2):
                rho_pair = acf[t] + (acf[t + 1] if t + 1 < n else 0.0)
                if rho_pair < 0:
                    break
                total += rho_pair
            total += 1.0  # for lag 0
        avg_tau = total / m
        return float(m * n / (2 * avg_tau - 1)) if avg_tau > 0.5 else float(m * n)

    def mcmc_diagnostics(self) -> MCMCDiagnostics:
        """Compute R-hat and ESS for each parameter.

        Returns:
            :class:`MCMCDiagnostics` with R-hat and ESS per parameter.
        """
        self._check_fitted()
        param_names = ["mu", "delta_A"]
        param_diags: dict[str, MCMCParamDiagnostic] = {}
        for i, name in enumerate(param_names):
            param_chains = self.chains[:, :, i]
            param_diags[name] = MCMCParamDiagnostic(
                r_hat=self._r_hat(param_chains),
                ess=self._ess(param_chains),
            )
        return MCMCDiagnostics(**param_diags)

    # ------------------------------------------------------------------ #
    #  Hypothesis testing
    # ------------------------------------------------------------------ #

    def savage_dickey_test(self, null_value: float = 0.0) -> SavageDickeyResult:
        """Savage-Dickey density-ratio Bayes factor for H0: delta_A = *null_value*.

        Args:
            null_value: The point null hypothesis value for delta_A.

        Returns:
            Dict with keys ``BF_01``, ``BF_10``, ``posterior_density_at_0``,
            ``prior_density_at_0``, ``interpretation``, and ``decision``.
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
            Dict with keys ``P(H0|data)``, ``P(H1|data)``,
            ``prior_odds``, and ``posterior_odds``.
        """
        prior_odds = prior_H0 / (1 - prior_H0)
        posterior_odds = BF_01 * prior_odds
        P_H0 = posterior_odds / (1 + posterior_odds)
        return PosteriorProbH0Result(
            **{"P(H0|data)": P_H0, "P(H1|data)": 1 - P_H0},
            prior_odds=prior_odds,
            posterior_odds=posterior_odds,
        )

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #

    def ppc_pvalues(self, seed: int | None = None) -> dict[str, PPCStatistic]:
        """Posterior predictive p-values for summary statistics.

        Returns:
            Dict mapping statistic name to ``{observed, p_value, status}``.
        """
        self._check_fitted()

        rng = np.random.default_rng(seed if seed is not None else self.seed)

        mu_s = self.samples[:, 0]
        delta_s = self.samples[:, 1]
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

    def plot_trace(self, **kwargs) -> None:
        """Trace plots and autocorrelation for all chains."""
        import matplotlib.pyplot as plt

        self._check_fitted()
        param_names = ["\u03bc", "\u03b4_A"]

        figsize = kwargs.pop("figsize", (14, 8))
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        for row, (name, idx) in enumerate(zip(param_names, [0, 1], strict=True)):
            # Trace plot
            ax = axes[row, 0]
            for c in range(self.n_chains):
                ax.plot(
                    self.chains[c, :, idx],
                    alpha=0.6,
                    linewidth=0.5,
                    label=f"Chain {c + 1}",
                )
            ax.set_xlabel("Iteration (post burn-in)")
            ax.set_ylabel(name)
            ax.set_title(f"Trace: {name}", fontweight="bold")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.3)

            # ACF
            ax2 = axes[row, 1]
            max_lag = min(100, self.chains.shape[1] // 2)
            for c in range(self.n_chains):
                x = self.chains[c, :, idx]
                x = x - x.mean()
                acf_full = np.correlate(x, x, mode="full")
                acf_full = acf_full[len(x) - 1 :]
                acf_full /= acf_full[0]
                ax2.plot(acf_full[:max_lag], alpha=0.6, linewidth=0.8)
            ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax2.set_xlabel("Lag")
            ax2.set_ylabel("ACF")
            ax2.set_title(f"Autocorrelation: {name}", fontweight="bold")
            ax2.grid(alpha=0.3)

        fig.suptitle(
            kwargs.pop("title", "MCMC Diagnostics (PG Gibbs)"),
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

    def plot_posteriors(self, **kwargs: Any) -> None:
        """Two-panel posterior plot: overlaid p_A / p_B and Δ = p_A − p_B.

        The implied success probabilities ``p_A = σ(μ + δ_A)`` and
        ``p_B = σ(μ)`` are computed from the pooled MCMC posterior
        samples and displayed as overlaid KDE densities in the left
        panel.  The right panel shows the difference Δ = p_A − p_B.

        Args:
            **kwargs: Accepts ``figsize`` (default ``(14, 5)``) and
                ``title`` (default ``"PG Gibbs Posterior (Pooled Binomial)"``).
        """
        import matplotlib.pyplot as plt

        self._check_fitted()

        mu_s = self.samples[:, 0]
        delta_A_s = self.samples[:, 1]
        p_A_s = sigmoid(mu_s + delta_A_s)
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
            kwargs.pop("title", "PG Gibbs Posterior (Pooled Binomial)"),
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
            kwargs.pop("title", "Posterior of \u03b4_A (PG Gibbs)"),
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
            kwargs.pop("title", "Savage-Dickey Test (PG Gibbs)"),
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

        mu_s = self.samples[:, 0]
        delta_s = self.samples[:, 1]
        n = len(self.y_A_obs)

        p_A_s = sigmoid(mu_s + delta_s)
        p_B_s = sigmoid(mu_s)
        y_A_rep = (rng.random((len(p_A_s), n)) < p_A_s[:, None]).astype(int)
        y_B_rep = (rng.random((len(p_B_s), n)) < p_B_s[:, None]).astype(int)

        figsize = kwargs.pop("figsize", (18, 5))
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        frac_A_rep = y_A_rep.mean(axis=1)
        frac_A_obs = self.y_A_obs.mean()
        ax = axes[0]
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

        frac_B_rep = y_B_rep.mean(axis=1)
        frac_B_obs = self.y_B_obs.mean()
        ax = axes[1]
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

        diff_rep = frac_A_rep - frac_B_rep
        diff_obs = frac_A_obs - frac_B_obs
        ax = axes[2]
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
            kwargs.pop("title", "Posterior Predictive Checks (PG Gibbs)"),
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  Summary / reporting
    # ------------------------------------------------------------------ #

    def print_summary(self) -> None:
        """Print posterior summary, MCMC diagnostics, Savage-Dickey test, and PPC."""
        self._check_fitted()

        s = self.summary
        diag = self.mcmc_diagnostics()
        verdict = "A wins" if s.p_A_greater_B > 0.95 else ("Tied" if s.p_A_greater_B > 0.5 else "B wins")

        print("PG Gibbs posterior summary")
        print("=" * 60)
        print(f"  Chains: {self.n_chains}, Iterations: {self.n_iter}, Burn-in: {self.burn_in}")
        print(f"  Total post-warmup samples: {len(self.delta_A_samples)}")
        print(f"  \u03b4_A posterior mean:  {s.delta_A_posterior_mean:.4f}")
        print(f"  Mean \u0394 (prob scale): {s.mean_delta:.4f}")
        print(f"  95% CI:              [{s.ci_95.lower:.4f}, {s.ci_95.upper:.4f}]")
        print(f"  P(A > B):            {s.p_A_greater_B:.4f}")
        print(f"  Verdict:             {verdict}")

        print()
        print("MCMC diagnostics")
        print("=" * 60)
        for name, d in diag.model_dump().items():
            r_hat = d["r_hat"]
            ess = d["ess"]
            status = "OK" if 0.99 <= r_hat <= 1.05 else "WARN"
            print(f"  {name}: R-hat={r_hat:.4f}, ESS={ess:.0f}  {status}")

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
        print(f"  {'Statistic':<20} {'Observed':>10} {'p-value':>10} {'Status':>8}")
        print("  " + "-" * 50)
        for stat, vals in ppc.items():
            print(f"  {stat:<20} {vals.observed:>10.4f} {vals.p_value:>10.3f} {vals.status:>8}")

        print()
        print("Trace summary")
        print("=" * 60)
        print(self.trace_summary.to_string())

    # ------------------------------------------------------------------ #
    #  Multi-model comparison
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_forest(
        results: dict[str, "PairedBayesPropTestPG"],
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
                "title",
                f"{label_A} vs {label_B} \u2014 PG Gibbs Comparison",
            ),
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_comparison_table(
        results: dict[str, "PairedBayesPropTestPG"],
    ) -> None:
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
