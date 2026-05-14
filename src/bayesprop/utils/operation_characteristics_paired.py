"""Operating-characteristic / BFDA simulation harness for the paired-Laplace model.

This module mirrors :mod:`bayesprop.utils.operation_characteristics`
but evaluates the *paired* Bayesian procedure built around
:class:`bayesprop.resources.bayes_paired_laplace.PairedBayesPropTest`
and its sequential variant
:class:`SequentialPairedBayesPropTest`.

A paired binary design has the same three frequentist concerns as the
non-paired one — Type-I rate, three-way decision rates, credible-interval
coverage — plus a sequential variant. The analysis model is
``y_A ~ Bern(σ(μ + δ_A))``, ``y_B ~ Bern(σ(μ))``, with prior
``δ_A ~ N(0, σ_δ²)``. The Savage–Dickey BF tests ``H_0: δ_A = 0``,
which is equivalent to ``H_0: p_A = p_B``.

For an apples-to-apples comparison with the non-paired analysis (and
with the user's intuition), the public API takes the true marginal
probabilities ``(p_A, p_B)`` on the *probability* scale and inverts
the sigmoid internally to recover the analysis-model intercept
``μ = logit(p_B)`` and effect ``δ_A = logit(p_A) − logit(p_B)``. This
parametrisation is the one expected by
:func:`simulate_paired_scores` when ``δ_B = 0`` and ``σ_θ = 0``.

Three public entry points cover the workflow:

* :func:`simulate_fixed_n_paired` — one ``(p_A, p_B, n)`` cell of the
  OC grid, returning the Bayes three-way decision rates, the 95 % CI
  coverage of ``Δ = p_A − p_B``, and the per-replicate McNemar exact
  p-values needed for a frequentist baseline.
* :func:`grid_fixed_n_paired` — sweeps :func:`simulate_fixed_n_paired`
  over an arbitrary list of ``(p_A, p_B)`` pairs.
* :func:`simulate_sequential_paired` — empirical stopping-time
  distribution of :class:`SequentialPairedBayesPropTest` at a single
  ``(p_A, p_B)`` point.

The matched-α helper :func:`matched_calibration_alpha` and the Wilson
band helper :func:`wilson_band` are parametrisation-free and are
re-exported from the non-paired module so callers don't have to
import from two places.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from bayesprop.resources.bayes_paired_laplace import (
    PairedBayesPropTest,
    SequentialPairedBayesPropTest,
)
from bayesprop.resources.bayes_nonpaired import classify_bf
from bayesprop.utils.operation_characteristics import (
    matched_calibration_alpha,
    wilson_band,
)
from bayesprop.utils.utils import mcnemar_paired_test, simulate_paired_scores

# Re-export parametrisation-free helpers so callers can import the
# full paired OC toolbox from a single module.
__all__ = [
    "simulate_fixed_n_paired",
    "grid_fixed_n_paired",
    "simulate_sequential_paired",
    "matched_calibration_alpha",
    "wilson_band",
]

# Decision categories returned by :func:`classify_bf`.
BFDecision = Literal["reject", "accept", "inconclusive"]

# Numerical guard for ``logit(p)`` at the boundaries — keeps the
# DGP well-defined when the user requests ``p ∈ {0, 1}``.
_LOGIT_EPS = 1e-6


def _logit(p: float) -> float:
    """Numerically guarded logit; clips the input to ``[eps, 1-eps]``.

    The OC grid can hit ``p = 0`` or ``p = 1`` at the corners; those
    would explode the logit, so we clip just enough to keep the
    Laplace fit well-conditioned without distorting the simulation.
    """
    p_clipped = min(max(p, _LOGIT_EPS), 1.0 - _LOGIT_EPS)
    return float(np.log(p_clipped / (1.0 - p_clipped)))


# ====================================================================== #
#  Fixed-n simulation
# ====================================================================== #


def simulate_fixed_n_paired(
    p_A: float,
    p_B: float,
    n: int,
    n_sim: int,
    rng: np.random.Generator,
    *,
    prior_sigma_delta: float = 1.0,
    bf_upper: float = 3.0,
    bf_lower: float = 1.0 / 3.0,
    n_samples_mc: int = 4000,
    track_ci: bool = True,
) -> tuple[dict[str, float], np.ndarray]:
    """Monte-Carlo operating characteristics at a single ``(p_A, p_B, n)`` cell.

    For each of ``n_sim`` replicates we

    1. generate one paired dataset of size ``n`` via
       :func:`simulate_paired_scores` with ``μ = logit(p_B)`` and
       ``δ_A = logit(p_A) − logit(p_B)`` (``δ_B = 0``, ``σ_θ = 0``);
    2. fit :class:`PairedBayesPropTest` with prior
       ``δ_A ~ N(0, prior_sigma_delta²)``;
    3. compute the Savage–Dickey BF on ``δ_A = 0`` and classify the
       result with :func:`classify_bf` (configurable ``bf_upper`` /
       ``bf_lower``);
    4. optionally check whether the 95 % credible interval on
       ``Δ = p_A − p_B`` covers the true effect (``track_ci=True``);
    5. run :func:`mcnemar_paired_test` on the *same* simulated data
       so a frequentist baseline can be derived later.

    Args:
        p_A: True success probability for arm A.
        p_B: True success probability for arm B (= the analysis-model
            intercept on the probability scale).
        n: Number of paired observations.
        n_sim: Number of Monte-Carlo replicates.
        rng: Pre-seeded NumPy generator. Threaded through the entire
            loop so the harness is fully deterministic given the seed.
        prior_sigma_delta: Standard deviation of the ``N(0, σ_δ²)``
            prior on ``δ_A`` (logit scale).
        bf_upper: Threshold above which the Bayes rule rejects ``H_0``.
        bf_lower: Threshold below which the Bayes rule accepts ``H_0``.
            Must satisfy ``0 < bf_lower < bf_upper``.
        n_samples_mc: Posterior Laplace draws per replicate. Affects
            the 95 % CI estimate; the BF is computed analytically.
        track_ci: If False, skip the CI coverage check
            (``ci_coverage`` will be ``NaN``). Useful for pure null
            sweeps where coverage is not the quantity of interest.

    Returns:
        Tuple ``(summary, freq_pvalues)``. ``summary`` is a dict with
        keys ``"reject"`` / ``"accept"`` / ``"inconclusive"`` (Bayes
        decision rates, summing to 1), ``"ci_coverage"`` (empirical
        95 % CI coverage of ``Δ``, or ``NaN`` if ``track_ci=False``),
        and the echoes ``"p_A"``, ``"p_B"``, ``"delta"``, ``"n"``,
        ``"n_sim"``. ``freq_pvalues`` is a 1-D ``np.ndarray`` of
        length ``n_sim`` holding the per-replicate McNemar p-values.

    Raises:
        ValueError: If ``bf_lower >= bf_upper`` or ``bf_lower <= 0``
            (validated inside :func:`classify_bf`).
    """
    counts: dict[str, int] = {"reject": 0, "accept": 0, "inconclusive": 0}
    ci_covered = 0
    delta_true = p_A - p_B
    pvals = np.empty(n_sim, dtype=float)

    # Translate the user-facing (p_A, p_B) into the analysis-model
    # parametrisation expected by simulate_paired_scores.
    mu = _logit(p_B)
    delta_A = _logit(p_A) - mu

    for i in range(n_sim):
        sim = simulate_paired_scores(
            N=n,
            mu=mu,
            delta_A=delta_A,
            delta_B=0.0,
            sigma_theta=0.0,
            rng=rng,
        )

        bb = PairedBayesPropTest(
            prior_sigma_delta=prior_sigma_delta,
            n_samples=n_samples_mc,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        bb.fit(sim.y_A, sim.y_B)
        bf10 = bb.savage_dickey_test().BF_10
        counts[classify_bf(bf10, bf_upper, bf_lower)] += 1

        if track_ci:
            # PairedBayesPropTest.summary.ci_95 is already on the
            # probability scale (Δ = σ(μ+δ_A) − σ(μ)), so we can
            # compare directly to delta_true.
            ci = bb.summary.ci_95
            if ci.lower <= delta_true <= ci.upper:
                ci_covered += 1

        # Frequentist baseline on the same simulated data.
        freq = mcnemar_paired_test(sim.y_A, sim.y_B)
        pvals[i] = freq.p_value

    summary: dict[str, float] = {k: v / n_sim for k, v in counts.items()}
    summary["ci_coverage"] = ci_covered / n_sim if track_ci else float("nan")
    summary["p_A"] = p_A
    summary["p_B"] = p_B
    summary["delta"] = delta_true
    summary["n"] = float(n)
    summary["n_sim"] = float(n_sim)
    return summary, pvals


def grid_fixed_n_paired(
    grid: list[tuple[float, float]],
    n: int,
    n_sim: int,
    seed: int,
    *,
    prior_sigma_delta: float = 1.0,
    bf_upper: float = 3.0,
    bf_lower: float = 1.0 / 3.0,
    n_samples_mc: int = 4000,
    track_ci: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sweep :func:`simulate_fixed_n_paired` over an arbitrary ``(p_A, p_B)`` grid.

    A single deterministic ``np.random.Generator`` is created from
    ``seed`` and threaded through every grid cell, so a re-run with the
    same ``seed`` produces bit-identical results.

    Args:
        grid: List of ``(p_A, p_B)`` pairs to evaluate.
        n: Number of paired observations (constant across grid cells).
        n_sim: Replicates per grid cell.
        seed: Top-level random seed.
        prior_sigma_delta: See :func:`simulate_fixed_n_paired`.
        bf_upper: See :func:`simulate_fixed_n_paired`.
        bf_lower: See :func:`simulate_fixed_n_paired`.
        n_samples_mc: See :func:`simulate_fixed_n_paired`.
        track_ci: See :func:`simulate_fixed_n_paired`.

    Returns:
        Tuple ``(rates_df, p_values)``. ``rates_df`` is a
        ``pandas.DataFrame`` with one row per grid cell and columns
        ``reject`` / ``accept`` / ``inconclusive`` / ``ci_coverage`` /
        ``p_A`` / ``p_B`` / ``delta`` / ``n`` / ``n_sim``. ``p_values``
        is a ``(len(grid), n_sim)`` ``np.ndarray`` of McNemar p-values
        — needed to derive the matched calibration α via
        :func:`matched_calibration_alpha`.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    pv_stack: list[np.ndarray] = []
    for p_A, p_B in grid:
        summary, pvals = simulate_fixed_n_paired(
            p_A,
            p_B,
            n,
            n_sim,
            rng,
            prior_sigma_delta=prior_sigma_delta,
            bf_upper=bf_upper,
            bf_lower=bf_lower,
            n_samples_mc=n_samples_mc,
            track_ci=track_ci,
        )
        rows.append(summary)
        pv_stack.append(pvals)
    return pd.DataFrame(rows), np.stack(pv_stack)


# ====================================================================== #
#  Sequential simulation
# ====================================================================== #


def simulate_sequential_paired(
    p_A: float,
    p_B: float,
    n_sim: int,
    rng: np.random.Generator,
    *,
    prior_sigma_delta: float = 1.0,
    bf_upper: float = 3.0,
    bf_lower: float = 1.0 / 3.0,
    n_min: int = 50,
    n_max: int = 600,
    batch_size: int = 50,
    n_samples_mc: int = 2000,
) -> dict[str, float]:
    """Empirical stopping-time distribution of the sequential paired-Laplace test.

    For each of ``n_sim`` replicates we run a fresh
    :class:`SequentialPairedBayesPropTest`, stream batches of size
    ``batch_size`` of paired Bernoulli observations from
    :func:`simulate_paired_scores`, and record the per-arm sample
    size at which the procedure stops. Trials that hit ``n_max``
    without the BF crossing either threshold are *right-censored* and
    classified as ``"inconclusive"``.

    Args:
        p_A: True success probability for arm A.
        p_B: True success probability for arm B.
        n_sim: Number of independent sequential trajectories.
        rng: Pre-seeded NumPy generator.
        prior_sigma_delta: Prior SD on ``δ_A`` (logit scale).
        bf_upper: Stop for ``H_1`` when ``BF_10 ≥ bf_upper``.
        bf_lower: Stop for ``H_0`` when ``BF_10 ≤ bf_lower``.
        n_min: Minimum per-arm sample size before any BF-based stop
            is allowed.
        n_max: Hard cap on per-arm sample size — the trajectory is
            terminated and the trial reported as censored.
        batch_size: Per-arm batch size delivered to each ``update()``.
        n_samples_mc: Posterior Laplace draws per look.

    Returns:
        Dict with the per-trial stopping-time statistics. Keys
        ``"p_A"``, ``"p_B"``, ``"delta"`` echo the inputs. Keys
        ``"median_n"``, ``"q05"``, ``"q25"``, ``"q75"``, ``"q95"``
        are quantiles of the per-arm stopping sample size.
        ``"frac_censored"`` is the fraction of trials that hit
        ``n_max``. ``"frac_reject"`` / ``"frac_accept"`` are the
        fractions of trials whose final classification (via
        :func:`classify_bf`) is ``"reject"`` / ``"accept"``.
    """
    mu = _logit(p_B)
    delta_A = _logit(p_A) - mu
    max_looks = n_max // batch_size

    stop_n: list[int] = []
    censored: list[bool] = []
    decisions: list[BFDecision] = []

    for _ in range(n_sim):
        seq = SequentialPairedBayesPropTest(
            prior_sigma_delta=prior_sigma_delta,
            bf_upper=bf_upper,
            bf_lower=bf_lower,
            n_min=n_min,
            n_max=n_max,
            decision_rule="bayes_factor",
            n_samples=n_samples_mc,
        )
        last = None
        for _look in range(max_looks):
            batch = simulate_paired_scores(
                N=batch_size,
                mu=mu,
                delta_A=delta_A,
                delta_B=0.0,
                sigma_theta=0.0,
                rng=rng,
            )
            last = seq.update(batch.y_A, batch.y_B)
            if seq.stopped:
                break
        assert last is not None, "max_looks=0 would skip the inner loop"

        stop_n.append(min(last.n_A, last.n_B))
        censored.append("n_max" in (last.stop_reason or "n_max reached"))

        bf10_final = (
            last.decision.bayes_factor.BF_10 if last.decision.bayes_factor else 1.0
        )
        decisions.append(classify_bf(bf10_final, bf_upper, bf_lower))

    arr = np.asarray(stop_n)
    decision_arr = np.asarray(decisions)
    return {
        "p_A": p_A,
        "p_B": p_B,
        "delta": p_A - p_B,
        "median_n": float(np.median(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "frac_censored": float(np.mean(censored)),
        "frac_reject": float(np.mean(decision_arr == "reject")),
        "frac_accept": float(np.mean(decision_arr == "accept")),
    }
