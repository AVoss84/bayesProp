"""Abstract base class defining the common API for all BayesProp models.

Every concrete model class — :class:`NonPairedBayesPropTest`,
:class:`PairedBayesPropTest`, :class:`PairedBayesPropTestPG`, and
:class:`PairedBayesPropTestBB` — inherits from
:class:`BaseBayesPropTest` so that the core *fit → summarise → decide →
plot* workflow is guaranteed to exist with identical method names
regardless of the underlying inference engine.

Methods that are only meaningful for a subset of models (e.g.
``savage_dickey_test`` for parametric models, ``plot_trace`` for MCMC
models) live on the concrete classes and are **not** declared here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt

from bayesprop.resources.data_schemas import HypothesisDecision, ROPEResult


class BaseBayesPropTest(ABC):
    """Abstract base for all BayesProp A/B test models.

    Subclasses must implement the abstract methods below. The type
    annotations on ``fit`` use :class:`npt.ArrayLike` so that both raw
    lists and NumPy arrays are accepted.
    """

    @abstractmethod
    def fit(
        self,
        y_A: npt.ArrayLike,
        y_B: npt.ArrayLike,
    ) -> "BaseBayesPropTest":
        """Fit the model to observed data and populate ``self.summary``.

        Args:
            y_A: Observed outcomes for arm A.
            y_B: Observed outcomes for arm B.

        Returns:
            ``self`` for method chaining.
        """

    @abstractmethod
    def decide(self, **kwargs: Any) -> HypothesisDecision:
        """Run the composite decision framework.

        Returns:
            A :class:`HypothesisDecision` with the applicable sub-results
            populated (Bayes factor, posterior null, ROPE).
        """

    @abstractmethod
    def rope_test(
        self,
        rope: tuple[float, float] | None = None,
        ci_mass: float = 0.95,
    ) -> ROPEResult:
        """ROPE analysis on the posterior of the treatment effect.

        Args:
            rope: ``(lower, upper)`` ROPE bounds. Defaults to
                ``(-rope_epsilon, +rope_epsilon)``.
            ci_mass: Credible-interval mass (default 95 %).

        Returns:
            Populated :class:`ROPEResult`.
        """

    # ------------------------------------------------------------------ #
    #  Plotting
    # ------------------------------------------------------------------ #

    @abstractmethod
    def plot_posteriors(self, **kwargs: Any) -> None:
        """Plot the posterior distributions of θ_A and θ_B overlaid.

        Single-panel plot showing both marginal posteriors on the
        probability scale.
        """

    @abstractmethod
    def plot_posterior_delta(self, **kwargs: Any) -> None:
        """Plot the posterior of Δ = θ_A − θ_B (probability scale).

        Single-panel KDE (or histogram) of the treatment-effect
        posterior with 95 % credible interval.
        """

    # ------------------------------------------------------------------ #
    #  Summary
    # ------------------------------------------------------------------ #

    @abstractmethod
    def print_summary(self) -> None:
        """Print a human-readable summary of the fitted model."""
