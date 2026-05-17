"""Unified facade for all paired Bayesian A/B testing methods.

This module exposes :class:`PairedBayesPropTest`, a single entry point
that dispatches to one of three inference backends based on the
``method`` parameter:

============  ====================================================
Method        Backend class
============  ====================================================
``"laplace"`` :class:`~bayesprop.resources.bayes_paired_laplace._PairedLaplace`
``"pg"``      :class:`~bayesprop.resources.bayes_paired_pg.PairedBayesPropTestPG`
``"bootstrap"`` :class:`~bayesprop.resources.bayes_paired_bootstrap.PairedBayesPropTestBB`
============  ====================================================

Typical usage::

    from bayesprop.resources.bayes_paired import PairedBayesPropTest

    # Laplace (default — fast)
    model = PairedBayesPropTest(seed=42).fit(y_A, y_B)

    # Exact PG Gibbs
    model = PairedBayesPropTest(method="pg", n_iter=2000).fit(y_A, y_B)

    # Nonparametric bootstrap
    model = PairedBayesPropTest(method="bootstrap").fit(y_A, y_B)
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt

from bayesprop.resources.base import BaseBayesPropTest
from bayesprop.resources.data_schemas import (
    DecisionRuleType,
    HypothesisDecision,
    PosteriorProbH0Result,
    ROPEResult,
)

_METHOD_TYPE = str  # Literal["laplace", "pg", "bootstrap"] at runtime


class PairedBayesPropTest(BaseBayesPropTest):
    """Unified paired A/B test — dispatches to the chosen inference backend.

    This facade provides a single entry point for all paired Bayesian
    A/B testing methods.  The ``method`` parameter selects the
    inference engine:

    ============  ====================================================
    Method        Backend class
    ============  ====================================================
    ``"laplace"`` :class:`~bayesprop.resources.bayes_paired_laplace._PairedLaplace` — MAP + Hessian (fast, default)
    ``"pg"``      :class:`~bayesprop.resources.bayes_paired_pg.PairedBayesPropTestPG` — exact PG Gibbs
    ``"bootstrap"`` :class:`~bayesprop.resources.bayes_paired_bootstrap.PairedBayesPropTestBB` — nonparametric
    ============  ====================================================

    All common attributes (``summary``, ``delta_A_samples``, etc.) and
    methods (``decide()``, ``plot_posteriors()``, etc.) are forwarded
    transparently.  Method-specific features (e.g.
    ``plot_trace()`` for PG, ``laplace`` dict for Laplace) are accessible
    via the facade's attribute forwarding.

    Args:
        method: Inference method — ``"laplace"`` (default), ``"pg"``,
            or ``"bootstrap"``.
        seed: Random seed for reproducibility.
        prior_sigma_delta: Std-dev of the N(0, σ) prior on δ_A
            (logit scale).  Ignored for ``"bootstrap"``.
        decision_rule: Default decision framework.
        rope_epsilon: Half-width of the ROPE interval.
        threshold: Binarisation cutoff for continuous inputs.
        verbose: Emit notices when inputs are binarised.
        hyperprior_mu: ``(a, b)`` IG hyperprior on σ²_μ (Laplace / PG).
        hyperprior_delta: ``(a, b)`` IG hyperprior on σ²_δ (Laplace / PG).
        n_samples: Posterior draws (Laplace / Bootstrap).
        n_iter: Gibbs iterations per chain (PG only).
        burn_in: Burn-in iterations per chain (PG only).
        n_chains: Number of MCMC chains (PG only).
        prior_sigma_mu: Std-dev of the N(0, σ) prior on μ (PG only).
        dirichlet_alpha: Dirichlet concentration (Bootstrap only).

    Example::

        # Laplace (default — fast)
        model = PairedBayesPropTest(seed=42).fit(y_A, y_B)

        # Exact PG Gibbs
        model = PairedBayesPropTest(method="pg", n_iter=2000, n_chains=4).fit(y_A, y_B)

        # Nonparametric bootstrap
        model = PairedBayesPropTest(method="bootstrap").fit(y_A, y_B)
    """

    _METHODS = frozenset({"laplace", "pg", "bootstrap"})

    def __init__(
        self,
        method: _METHOD_TYPE = "laplace",
        *,
        # ── Common ────────────────────────────────────────────────
        seed: int = 0,
        prior_sigma_delta: float = 1.0,
        decision_rule: DecisionRuleType = "all",
        rope_epsilon: float = 0.02,
        threshold: float = 0.5,
        verbose: bool = False,
        hyperprior_mu: tuple[float, float] | None = None,
        hyperprior_delta: tuple[float, float] | None = None,
        # ── Laplace-specific ──────────────────────────────────────
        n_samples: int = 8000,
        # ── PG-specific ──────────────────────────────────────────
        n_iter: int = 1000,
        burn_in: int = 200,
        n_chains: int = 2,
        prior_sigma_mu: float = 2.0,
        # ── Bootstrap-specific ───────────────────────────────────
        dirichlet_alpha: float = 1.0,
    ) -> None:
        """Initialise the facade and create the appropriate backend."""
        if method not in self._METHODS:
            raise ValueError(
                f"Unknown method {method!r}. Choose from {sorted(self._METHODS)}."
            )
        self.method: str = method

        if method == "laplace":
            from bayesprop.resources.bayes_paired_laplace import _PairedLaplace

            self._backend: BaseBayesPropTest = _PairedLaplace(
                prior_sigma_delta=prior_sigma_delta,
                seed=seed,
                n_samples=n_samples,
                decision_rule=decision_rule,
                rope_epsilon=rope_epsilon,
                threshold=threshold,
                verbose=verbose,
                hyperprior_mu=hyperprior_mu,
                hyperprior_delta=hyperprior_delta,
            )
        elif method == "pg":
            from bayesprop.resources.bayes_paired_pg import PairedBayesPropTestPG

            self._backend = PairedBayesPropTestPG(
                prior_sigma_delta=prior_sigma_delta,
                prior_sigma_mu=prior_sigma_mu,
                seed=seed,
                n_iter=n_iter,
                burn_in=burn_in,
                n_chains=n_chains,
                decision_rule=decision_rule,
                rope_epsilon=rope_epsilon,
                threshold=threshold,
                verbose=verbose,
                hyperprior_mu=hyperprior_mu,
                hyperprior_delta=hyperprior_delta,
            )
        elif method == "bootstrap":
            from bayesprop.resources.bayes_paired_bootstrap import (
                PairedBayesPropTestBB,
            )

            self._backend = PairedBayesPropTestBB(
                seed=seed,
                n_samples=n_samples,
                rope_epsilon=rope_epsilon,
                dirichlet_alpha=dirichlet_alpha,
                threshold=threshold,
                verbose=verbose,
            )

    # ------------------------------------------------------------------ #
    #  Representation
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        """Return an informative string representation."""
        return f"PairedBayesPropTest(method={self.method!r}) → {self._backend!r}"

    # ------------------------------------------------------------------ #
    #  Abstract method implementations — delegate to backend
    # ------------------------------------------------------------------ #

    def fit(
        self, y_A_obs: npt.ArrayLike, y_B_obs: npt.ArrayLike
    ) -> "PairedBayesPropTest":
        """Fit the model using the selected inference backend.

        Args:
            y_A_obs: Observed scores for group A.
            y_B_obs: Observed scores for group B.

        Returns:
            ``self`` for method chaining.
        """
        self._backend.fit(y_A_obs, y_B_obs)
        return self

    def decide(
        self, rule: DecisionRuleType | None = None, **kwargs: Any
    ) -> HypothesisDecision:
        """Run the chosen decision framework(s).

        Args:
            rule: Override the default ``decision_rule``.
            **kwargs: Forwarded to the backend.

        Returns:
            :class:`HypothesisDecision` with the requested sub-results.
        """
        if rule is not None:
            return self._backend.decide(rule=rule, **kwargs)
        return self._backend.decide(**kwargs)

    def rope_test(
        self,
        rope: tuple[float, float] | None = None,
        ci_mass: float = 0.95,
    ) -> ROPEResult:
        """ROPE analysis on the posterior of Δ = p_A − p_B.

        Args:
            rope: ``(lower, upper)`` ROPE bounds.
            ci_mass: Credible-interval mass.

        Returns:
            :class:`ROPEResult`.
        """
        return self._backend.rope_test(rope=rope, ci_mass=ci_mass)

    def plot_posteriors(self, **kwargs: Any) -> None:
        """Plot overlaid KDE posteriors of θ_A and θ_B."""
        self._backend.plot_posteriors(**kwargs)

    def plot_posterior_delta(self, **kwargs: Any) -> None:
        """Plot the posterior of Δ = θ_A − θ_B with 95% CI."""
        self._backend.plot_posterior_delta(**kwargs)

    def print_summary(self) -> None:
        """Print a human-readable summary of the fitted model."""
        self._backend.print_summary()

    # ------------------------------------------------------------------ #
    #  Static helpers — delegate to _PairedLaplace
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_forest(
        results: dict[str, BaseBayesPropTest],
        label_A: str = "Group A",
        label_B: str = "Group B",
        **kwargs: Any,
    ) -> None:
        """Forest plot + P(A>B) bar chart for multiple metrics."""
        from bayesprop.resources.bayes_paired_laplace import _PairedLaplace

        _PairedLaplace.plot_forest(results, label_A=label_A, label_B=label_B, **kwargs)

    @staticmethod
    def print_comparison_table(results: dict[str, BaseBayesPropTest]) -> None:
        """Print a formatted comparison table across metrics."""
        from bayesprop.resources.bayes_paired_laplace import _PairedLaplace

        _PairedLaplace.print_comparison_table(results)

    @staticmethod
    def posterior_probability_H0(
        BF_01: float, prior_H0: float = 0.5
    ) -> PosteriorProbH0Result:
        """Convert BF_01 to posterior probability of H0 (spike-and-slab).

        Args:
            BF_01: Bayes factor in favour of H0.
            prior_H0: Prior probability of H0 (default 0.5).

        Returns:
            :class:`PosteriorProbH0Result`.
        """
        from bayesprop.resources.bayes_paired_laplace import _PairedLaplace

        return _PairedLaplace.posterior_probability_H0(BF_01, prior_H0)

    # ------------------------------------------------------------------ #
    #  Attribute forwarding
    # ------------------------------------------------------------------ #

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute lookups to the backend.

        This allows transparent access to backend-specific attributes
        (e.g. ``model.summary``, ``model.laplace``, ``model.plot_trace()``)
        """
        # Avoid infinite recursion when _backend itself hasn't been set yet
        # (e.g. during __init__ before _backend is assigned).
        if name == "_backend":
            raise AttributeError(name)
        return getattr(self._backend, name)
