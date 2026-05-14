r"""Bayesian-bootstrap paired-proportions test.

Nonparametric paired A/B test using Rubin's (1981) **Bayesian bootstrap**.
For paired binary observations ``(y_A_i, y_B_i)`` we form the per-pair
differences ``D_i = y_A_i - y_B_i ∈ {-1, 0, +1}`` and place a flat
Dirichlet(1, …, 1) "prior" over the simplex of weights on the empirical
distribution. Each posterior draw of the average treatment effect is

.. math::

    \\Delta^{(s)} = \\sum_{i=1}^n w_i^{(s)} D_i,
    \\qquad \\mathbf{w}^{(s)} \\sim \\text{Dirichlet}(\\alpha,\\dots,\\alpha)

with ``α = 1`` the standard noninformative choice.

The procedure produces a full posterior on ``Δ = p_A − p_B`` *without*
any parametric likelihood and *without* any latent ``δ_A`` on the logit
scale. Decisions are driven by the **three** quantities that are
well-defined directly under the BB posterior:

* **Posterior of null** — ``P(Δ ∈ ROPE | data)``, exposed via
  :meth:`PairedBayesPropTestBB.rope_test` as ``ROPEResult.pct_in_rope``.
* **Posterior of superiority** — ``P(p_A > p_B | data)``, exposed via
  ``model.summary.p_A_greater_B``.
* **ROPE decision** — composite ROPE call returning the full
  :class:`ROPEResult`.

Things deliberately **not** exposed by this class:

* ``savage_dickey_test`` — the BB has no parametric prior on ``Δ`` to
  evaluate at the null. Use one of the parametric paired classes
  (Laplace or Pólya–Gamma) if you need a point-null BF.
* ``posterior_probability_H0`` — for the parametric classes this is a
  Bayes-factor-style conversion from ``BF_01`` to ``P(H_0 | data)``;
  under the BB the same quantity is just ``ROPEResult.pct_in_rope``
  read off the posterior directly. Adding a thin wrapper would force
  the user to commit to a prior on ``H_0`` that has no role in the BB
  posterior itself, and any default flat-prior choice would be
  reparametrisation-non-invariant (Lindley–Jeffreys).

Use cases where this is the right tool:

* Sample size is large enough that a nonparametric posterior is
  trustworthy (≳ 100 paired observations).
* The user wants to sidestep prior elicitation entirely.
* The user wants robustness against model misspecification of the
  underlying paired logistic likelihood.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from bayesprop.resources.data_schemas import (
    CredibleInterval,
    HypothesisDecision,
    PairedSummary,
    ROPEResult,
)


class PairedBayesPropTestBB:
    """Bayesian-bootstrap paired A/B test for binary outcomes.

    Nonparametric counterpart of
    :class:`bayesprop.resources.bayes_paired_laplace.PairedBayesPropTest`
    and
    :class:`bayesprop.resources.bayes_paired_pg.PairedBayesPropTestPG`.

    Generative model (Rubin, 1981)::

        D_i = y_A_i - y_B_i  ∈  {-1, 0, +1}
        w   ~ Dirichlet(α · 1_n)             (α = 1 = standard BB prior)
        Δ   = Σ_i w_i · D_i                  ∈  [-1, +1]

    The class deliberately omits ``savage_dickey_test`` because the
    Bayesian bootstrap has no parametric prior on ``Δ`` to evaluate at
    the null. All decisions are routed through ROPE / posterior mass.

    Attributes:
        n_samples: Number of Bayesian-bootstrap posterior draws.
        seed: Random seed for reproducibility.
        rope_epsilon: Half-width of the default ROPE (default 0.02 = 2 pp).
        dirichlet_alpha: Concentration of the Dirichlet weights
            (default 1.0 = standard noninformative BB).
        y_A_obs: Observed binary outcomes for arm A (set by :meth:`fit`).
        y_B_obs: Observed binary outcomes for arm B (set by :meth:`fit`).
        delta_samples: Posterior draws of ``Δ = p_A - p_B`` (probability
            scale), shape ``(n_samples,)``.
        summary: :class:`PairedSummary` populated by :meth:`fit`.
            Exposes ``mean_delta``, ``ci_95``, and ``p_A_greater_B``
            (the posterior probability of superiority).
        trace_summary: ``pandas.DataFrame`` with posterior summary
            statistics on ``Δ``.
    """

    def __init__(
        self,
        seed: int = 0,
        n_samples: int = 20_000,
        rope_epsilon: float = 0.02,
        dirichlet_alpha: float = 1.0,
    ) -> None:
        """Initialise configuration.

        Args:
            seed: Random seed.
            n_samples: Number of Bayesian-bootstrap posterior draws.
            rope_epsilon: Half-width of the default ROPE on ``Δ``.
            dirichlet_alpha: Concentration of the Dirichlet weights.
                The standard Bayesian bootstrap uses ``1.0``; values
                ``< 1`` concentrate posterior mass on a small number of
                observations (sharper, more bootstrap-like), values
                ``> 1`` smooth toward the empirical mean.

        Raises:
            ValueError: If ``dirichlet_alpha <= 0`` or ``n_samples <= 0``.
        """
        if dirichlet_alpha <= 0:
            raise ValueError(
                f"dirichlet_alpha must be > 0; got {dirichlet_alpha}"
            )
        if n_samples <= 0:
            raise ValueError(f"n_samples must be > 0; got {n_samples}")

        self.seed: int = seed
        self.n_samples: int = n_samples
        self.rope_epsilon: float = rope_epsilon
        self.dirichlet_alpha: float = dirichlet_alpha

        # Populated by .fit().
        self.y_A_obs: np.ndarray | None = None
        self.y_B_obs: np.ndarray | None = None
        self.delta_samples: np.ndarray | None = None
        self.summary: PairedSummary | None = None
        self.trace_summary: pd.DataFrame | None = None
        # Internal cache (mirrors the .laplace dict of the parametric classes).
        self._fitted_state: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    #  Fitting
    # ------------------------------------------------------------------ #

    def fit(
        self,
        y_A_obs: npt.ArrayLike,
        y_B_obs: npt.ArrayLike,
    ) -> PairedBayesPropTestBB:
        """Draw the Bayesian-bootstrap posterior on ``Δ = p_A − p_B``.

        Vectorised: a single ``rng.dirichlet`` call produces all weight
        vectors at once, then ``W @ D`` gives every posterior draw in one
        matmul. For large ``n_samples × n`` the Dirichlet draws are
        chunked to keep peak memory bounded (~400 MB).

        Args:
            y_A_obs: Binary outcomes (0/1) for arm A. Length ``n``.
            y_B_obs: Binary outcomes (0/1) for arm B. Length ``n``,
                aligned with ``y_A_obs`` (paired design).

        Returns:
            ``self`` (for method chaining).

        Raises:
            ValueError: If shapes mismatch or values are non-binary.
        """
        arr_A = np.asarray(y_A_obs, dtype=np.int64)
        arr_B = np.asarray(y_B_obs, dtype=np.int64)
        if arr_A.shape != arr_B.shape:
            raise ValueError(
                f"y_A_obs and y_B_obs must have identical shapes; got "
                f"{arr_A.shape} and {arr_B.shape}"
            )
        for name, arr in (("y_A_obs", arr_A), ("y_B_obs", arr_B)):
            if arr.size and not np.all((arr == 0) | (arr == 1)):
                raise ValueError(
                    f"{name} must contain only 0/1 values; got non-binary entries"
                )

        n = int(arr_A.size)
        if n == 0:
            raise ValueError("Cannot fit on empty data; need at least one pair.")

        self.y_A_obs = arr_A
        self.y_B_obs = arr_B

        # Paired differences in {-1, 0, +1}.
        differences = (arr_A - arr_B).astype(np.float64)

        # Vectorised Bayesian-bootstrap draws. We chunk the (n_samples, n)
        # weights matrix so peak memory stays bounded — at large n the
        # full matrix would be ~8·n·n_samples bytes (e.g. 1 GB for n=5k,
        # n_samples=25k).
        rng = np.random.default_rng(self.seed)
        alpha_vec = np.full(n, self.dirichlet_alpha, dtype=np.float64)
        chunk = max(1, int(5e7 // max(n, 1)))  # ~400 MB chunks
        delta_samples = np.empty(self.n_samples, dtype=np.float64)
        for start in range(0, self.n_samples, chunk):
            stop = min(start + chunk, self.n_samples)
            # rng.dirichlet returns shape (stop-start, n).
            weights = rng.dirichlet(alpha_vec, size=stop - start)
            # Weighted mean differences for every draw in this chunk.
            delta_samples[start:stop] = weights @ differences

        # Batched quantiles for the credible interval and trace summary —
        # one partition per probability set rather than four.
        delta_lo, delta_hi = np.quantile(delta_samples, [0.025, 0.975])
        hdi_lo, hdi_hi = np.quantile(delta_samples, [0.03, 0.97])

        self.delta_samples = delta_samples
        self.summary = PairedSummary(
            mean_delta=float(delta_samples.mean()),
            ci_95=CredibleInterval(lower=float(delta_lo), upper=float(delta_hi)),
            **{"P(A > B)": float((delta_samples > 0).mean())},
            # No latent δ_A in the BB model; report posterior-mean Δ on
            # the probability scale instead so the schema stays populated.
            delta_A_posterior_mean=float(delta_samples.mean()),
        )
        self.trace_summary = pd.DataFrame(
            {
                "mean": [float(delta_samples.mean())],
                "sd": [float(delta_samples.std(ddof=0))],
                "hdi_3%": [float(hdi_lo)],
                "hdi_97%": [float(hdi_hi)],
            },
            index=["delta"],
        )
        self._fitted_state = {
            "n": n,
            "n_pos": int(np.sum(differences > 0)),
            "n_neg": int(np.sum(differences < 0)),
            "n_zero": int(np.sum(differences == 0)),
            "dirichlet_alpha": self.dirichlet_alpha,
        }
        return self

    def _check_fitted(self) -> None:
        """Raise if :meth:`fit` has not been called yet."""
        if self._fitted_state is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    # ------------------------------------------------------------------ #
    #  Hypothesis testing
    # ------------------------------------------------------------------ #

    def rope_test(
        self,
        rope: tuple[float, float] | None = None,
        ci_mass: float = 0.95,
    ) -> ROPEResult:
        """ROPE analysis on the Bayesian-bootstrap posterior of ``Δ``.

        Args:
            rope: ``(lower, upper)`` bounds. Defaults to
                ``(-rope_epsilon, +rope_epsilon)``.
            ci_mass: Credible-interval mass (default 95%).

        Returns:
            Populated :class:`ROPEResult`.
        """
        self._check_fitted()
        if rope is None:
            rope = (-self.rope_epsilon, self.rope_epsilon)
        assert self.delta_samples is not None
        return ROPEResult.from_samples(self.delta_samples, rope=rope, ci_mass=ci_mass)

    # ------------------------------------------------------------------ #
    #  Composite decision
    # ------------------------------------------------------------------ #

    def decide(self) -> HypothesisDecision:
        """Run the ROPE-based composite decision.

        The Bayesian-bootstrap class deliberately ships only one
        decision sub-result — the ROPE analysis — because the three
        quantities of interest are already directly available from the
        BB posterior:

        * **Posterior of null**, ``P(Δ ∈ ROPE | data)`` ⇒
          ``rope_test().pct_in_rope`` (also reachable on the returned
          :class:`HypothesisDecision` via ``decide().rope.pct_in_rope``).
        * **Posterior of superiority**, ``P(p_A > p_B | data)`` ⇒
          ``model.summary.p_A_greater_B``.
        * **ROPE decision** (reject / accept / undecided) ⇒
          ``rope_test().decision``.

        The ``bayes_factor`` and ``posterior_null`` sub-fields of the
        returned :class:`HypothesisDecision` are always ``None`` for
        this class — the BB has no parametric prior on ``Δ``, so a
        Savage–Dickey BF and a prior-weighted ``P(H_0 | data)`` are
        both undefined. Adding a thin Bayes-factor-style wrapper on
        top of the ROPE mass would require committing to a prior on
        ``H_0`` that has no role in the BB posterior itself.

        Returns:
            :class:`HypothesisDecision` with only the ``rope`` field
            populated; ``rule`` is fixed to ``"rope"``.
        """
        self._check_fitted()
        return HypothesisDecision(
            bayes_factor=None,
            posterior_null=None,
            rope=self.rope_test(),
            rule="rope",
        )

    # ------------------------------------------------------------------ #
    #  Plotting
    # ------------------------------------------------------------------ #

    def plot_posterior(
        self,
        rope: tuple[float, float] | None = None,
        bins: int = 80,
        figsize: tuple[float, float] = (9, 5),
        ax: Any = None,
    ) -> Any:
        """Histogram of the posterior on ``Δ`` with the 95 % CI and ROPE.

        Args:
            rope: ``(lower, upper)`` ROPE bounds to overlay. Defaults to
                ``(-rope_epsilon, +rope_epsilon)``.
            bins: Histogram bin count.
            figsize: Figure size if a new figure is created.
            ax: Existing axes to draw on (creates a new figure if None).

        Returns:
            The matplotlib axes used for plotting.
        """
        import matplotlib.pyplot as plt

        self._check_fitted()
        assert self.delta_samples is not None and self.summary is not None
        if rope is None:
            rope = (-self.rope_epsilon, self.rope_epsilon)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.hist(
            self.delta_samples,
            bins=bins,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
        )
        ax.axvline(
            self.summary.mean_delta,
            color="darkred",
            linestyle="--",
            linewidth=2,
            label=f"Posterior mean = {self.summary.mean_delta:.4f}",
        )
        ci = self.summary.ci_95
        ax.axvspan(
            ci.lower,
            ci.upper,
            alpha=0.15,
            color="orange",
            label=f"95% CI = [{ci.lower:.4f}, {ci.upper:.4f}]",
        )
        ax.axvspan(
            rope[0],
            rope[1],
            alpha=0.25,
            color="red",
            label=f"ROPE [{rope[0]}, {rope[1]}]",
        )
        ax.set_xlabel(r"$\Delta$ = p_A $-$ p_B", fontsize=12)
        ax.set_ylabel("Posterior density", fontsize=12)
        ax.set_title(
            "Bayesian-bootstrap posterior on the paired treatment effect",
            fontsize=13,
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        return ax
