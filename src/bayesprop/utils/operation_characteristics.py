"""Operating-characteristic / BFDA simulation harness for the non-paired model.

This module evaluates the *frequentist* operating characteristics of the
Bayesian decision procedure built around
:class:`bayesprop.resources.bayes_nonpaired.NonPairedBayesPropTest`.

A Bayesian model has no Type-I error by itself, but the moment we wrap
it in a decision rule (``BF_10 ≥ bf_upper`` → reject, ``BF_10 ≤ bf_lower``
→ accept, otherwise inconclusive) the rule is a function from data to a
decision and therefore has well-defined frequentist operating
characteristics. Estimating those by Monte-Carlo simulation is the
standard "calibrated Bayes" check (Rubin 1984, Little 2006).

Three public entry points cover the workflow:

* :func:`simulate_fixed_n` — one ``(p_A, p_B, n)`` cell of the OC grid,
  returning the Bayes three-way decision rates, the 95 % CI coverage of
  ``Δ``, and the per-replicate Fisher exact p-values needed for a
  frequentist baseline.
* :func:`grid_fixed_n` — sweeps :func:`simulate_fixed_n` over an arbitrary
  list of ``(p_A, p_B)`` pairs and returns a tidy ``pandas.DataFrame``
  plus the stacked ``(n_grid, n_sim)`` p-value matrix.
* :func:`simulate_sequential` — empirical stopping-time distribution of
  :class:`SequentialNonPairedBayesPropTest` at a single
  ``(p_A, p_B)`` point.

The small helper :func:`matched_calibration_alpha` derives the
frequentist significance level that empirically matches the Bayes rule's
Type-I rate at the null, so the two procedures can be compared at equal
false-positive rates.

All quantities are obtained through the public API of the codebase
(``NonPairedBayesPropTest.fit``, ``savage_dickey_test``, ``classify_bf``,
``fisher_exact_nonpaired_test``, ``SequentialNonPairedBayesPropTest``),
so this module also serves as an end-to-end integration test of those
APIs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from bayesprop.resources.bayes_nonpaired import (
    NonPairedBayesPropTest,
    SequentialNonPairedBayesPropTest,
    classify_bf,
)
from bayesprop.utils.utils import (
    fisher_exact_nonpaired_test,
    simulate_nonpaired_scores,
)

# Decision categories returned by :func:`classify_bf`. Re-exported for
# downstream type hints so callers don't have to reach into the
# ``bayes_nonpaired`` module just for the type.
BFDecision = Literal["reject", "accept", "inconclusive"]


# ====================================================================== #
#  Fixed-n simulation
# ====================================================================== #


def simulate_fixed_n(
    p_A: float,
    p_B: float,
    n: int,
    n_sim: int,
    rng: np.random.Generator,
    *,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    bf_upper: float = 3.0,
    bf_lower: float = 1.0 / 3.0,
    n_samples_mc: int = 4000,
    track_ci: bool = True,
) -> tuple[dict[str, float], np.ndarray]:
    """Monte-Carlo operating characteristics at a single ``(p_A, p_B, n)`` cell.

    For each of ``n_sim`` replicates we

    1. generate one dataset of size ``n`` per arm via
       :func:`simulate_nonpaired_scores`,
    2. fit :class:`NonPairedBayesPropTest` with prior ``Beta(α₀, β₀)``,
    3. compute the Savage–Dickey BF on ``Δ = 0`` and classify the
       result with :func:`classify_bf` (configurable ``bf_upper`` /
       ``bf_lower``),
    4. optionally check whether the 95 % credible interval on
       ``Δ`` covers the true effect ``p_A − p_B`` (``track_ci=True``),
    5. run :func:`fisher_exact_nonpaired_test` on the *same* simulated
       data so a frequentist baseline can be derived later.

    Args:
        p_A: True success probability for group A.
        p_B: True success probability for group B.
        n: Per-arm sample size.
        n_sim: Number of Monte-Carlo replicates.
        rng: Pre-seeded NumPy generator. Threaded through the entire
            loop so the harness is fully deterministic given the seed.
        alpha0: Prior alpha for both Beta(α₀, β₀) priors.
        beta0: Prior beta for both priors.
        bf_upper: Threshold above which the Bayes rule rejects ``H₀``.
        bf_lower: Threshold below which the Bayes rule accepts ``H₀``.
            Must satisfy ``0 < bf_lower < bf_upper``.
        n_samples_mc: Posterior Monte-Carlo draws per replicate. Only
            affects the 95 % CI estimate; the BF is computed analytically.
        track_ci: If False, skip the CI coverage check
            (``ci_coverage`` will be ``NaN``). Useful when CI coverage
            is not the quantity of interest (e.g. a pure null sweep).

    Returns:
        Tuple ``(summary, freq_pvalues)``. ``summary`` is a dict with
        keys ``"reject"`` / ``"accept"`` / ``"inconclusive"`` (empirical
        fractions of the three Bayes decisions, summing to 1),
        ``"ci_coverage"`` (empirical 95 % CI coverage of ``Δ``, or
        ``NaN`` if ``track_ci=False``), and the echoes ``"p_A"``,
        ``"p_B"``, ``"delta"``, ``"n"``, ``"n_sim"``. ``freq_pvalues``
        is a 1-D ``np.ndarray`` of length ``n_sim`` holding the
        per-replicate Fisher exact p-values.

    Raises:
        ValueError: If ``bf_lower >= bf_upper`` or ``bf_lower <= 0``
            (validated inside :func:`classify_bf`).
    """
    counts: dict[str, int] = {"reject": 0, "accept": 0, "inconclusive": 0}
    ci_covered = 0
    delta_true = p_A - p_B
    pvals = np.empty(n_sim, dtype=float)

    for i in range(n_sim):
        # Same simulator the rest of the codebase uses — keeps OC
        # analysis honest if the DGP is ever changed.
        sim = simulate_nonpaired_scores(N=n, theta_A=p_A, theta_B=p_B, rng=rng)

        # Bayesian arm. Each replicate gets its own seed drawn from the
        # outer RNG so the posterior MC draws are independent across
        # replicates while still reproducible from the top-level seed.
        bb = NonPairedBayesPropTest(
            alpha0=alpha0,
            beta0=beta0,
            n_samples=n_samples_mc,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        bb.fit(sim.y_A, sim.y_B)
        bf10 = bb.savage_dickey_test().BF_10
        counts[classify_bf(bf10, bf_upper, bf_lower)] += 1

        if track_ci:
            # Frequentist coverage of the 95 % credible interval on Δ.
            ci = bb.summary.ci_95
            if ci.lower <= delta_true <= ci.upper:
                ci_covered += 1

        # Frequentist baseline on the *same* dataset — required for the
        # matched-α calibration of the OC plot.
        freq = fisher_exact_nonpaired_test(sim.y_A, sim.y_B)
        pvals[i] = freq.p_value

    summary: dict[str, float] = {k: v / n_sim for k, v in counts.items()}
    summary["ci_coverage"] = ci_covered / n_sim if track_ci else float("nan")
    summary["p_A"] = p_A
    summary["p_B"] = p_B
    summary["delta"] = delta_true
    summary["n"] = float(n)
    summary["n_sim"] = float(n_sim)
    return summary, pvals


def grid_fixed_n(
    grid: list[tuple[float, float]],
    n: int,
    n_sim: int,
    seed: int,
    *,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    bf_upper: float = 3.0,
    bf_lower: float = 1.0 / 3.0,
    n_samples_mc: int = 4000,
    track_ci: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sweep :func:`simulate_fixed_n` over an arbitrary ``(p_A, p_B)`` grid.

    A single deterministic ``np.random.Generator`` is created from
    ``seed`` and threaded through every grid cell, so a re-run with the
    same ``seed`` produces bit-identical results.

    Args:
        grid: List of ``(p_A, p_B)`` pairs to evaluate.
        n: Per-arm sample size (constant across grid cells).
        n_sim: Replicates per grid cell.
        seed: Top-level random seed.
        alpha0: See :func:`simulate_fixed_n`.
        beta0: See :func:`simulate_fixed_n`.
        bf_upper: See :func:`simulate_fixed_n`.
        bf_lower: See :func:`simulate_fixed_n`.
        n_samples_mc: See :func:`simulate_fixed_n`.
        track_ci: See :func:`simulate_fixed_n`.

    Returns:
        Tuple ``(rates_df, p_values)``. ``rates_df`` is a
        ``pandas.DataFrame`` with one row per grid cell and columns
        ``reject`` / ``accept`` / ``inconclusive`` / ``ci_coverage`` /
        ``p_A`` / ``p_B`` / ``delta`` / ``n`` / ``n_sim``. ``p_values``
        is a ``(len(grid), n_sim)`` ``np.ndarray`` of Fisher exact
        p-values — needed to derive the matched calibration α via
        :func:`matched_calibration_alpha`.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    pv_stack: list[np.ndarray] = []
    for p_A, p_B in grid:
        summary, pvals = simulate_fixed_n(
            p_A,
            p_B,
            n,
            n_sim,
            rng,
            alpha0=alpha0,
            beta0=beta0,
            bf_upper=bf_upper,
            bf_lower=bf_lower,
            n_samples_mc=n_samples_mc,
            track_ci=track_ci,
        )
        rows.append(summary)
        pv_stack.append(pvals)
    return pd.DataFrame(rows), np.stack(pv_stack)


# ====================================================================== #
#  Frequentist matched calibration
# ====================================================================== #


def matched_calibration_alpha(
    p_values: np.ndarray,
    bayes_type1_rate: float,
    null_grid_index: int,
) -> float:
    """Empirical α that matches a frequentist test's Type-I rate to the Bayes rule's.

    Given a matrix of Fisher (or other) p-values produced under a grid
    of true effects, the row indexed by ``null_grid_index`` corresponds
    to the null case ``Δ = 0``. We pick the p-value cutoff ``α`` whose
    empirical rejection rate on that row equals
    ``bayes_type1_rate``, by reading the corresponding quantile of the
    null p-value distribution. The resulting ``α`` is the fairest
    like-for-like calibration when overlaying the frequentist and
    Bayesian power curves.

    Args:
        p_values: ``(n_grid, n_sim)`` matrix of p-values from a grid
            simulation. Typically the second return of
            :func:`grid_fixed_n`.
        bayes_type1_rate: Empirical ``P(reject H₀ | Δ = 0)`` of the
            Bayes rule on that same grid (i.e.
            ``rates_df.iloc[null_grid_index]["reject"]``).
        null_grid_index: Row index into ``p_values`` corresponding to
            the null grid cell ``Δ = 0``.

    Returns:
        Matched-calibration α, clipped to ``[0, 1]``. Returns ``0.0``
        if ``bayes_type1_rate <= 0`` (the Bayes rule never rejects at
        the null in this simulation), since no positive α can match.

    Raises:
        ValueError: If ``null_grid_index`` is out of bounds or
            ``p_values`` is not 2-D.
    """
    if p_values.ndim != 2:
        raise ValueError(
            f"p_values must be 2-D (n_grid, n_sim); got shape {p_values.shape}"
        )
    if not 0 <= null_grid_index < p_values.shape[0]:
        raise ValueError(
            f"null_grid_index={null_grid_index} out of range for "
            f"p_values of shape {p_values.shape}"
        )
    if bayes_type1_rate <= 0.0:
        return 0.0

    # The empirical α-quantile of the null p-value distribution is by
    # definition the cutoff whose tail mass equals bayes_type1_rate.
    alpha = float(np.quantile(p_values[null_grid_index], bayes_type1_rate))
    return float(np.clip(alpha, 0.0, 1.0))


def wilson_band(
    rates: np.ndarray,
    n_sim: int,
    *,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson confidence band for a vector of Monte-Carlo binomial rates.

    Each entry of ``rates`` is treated as an empirical binomial proportion
    ``k / n_sim`` with ``k = round(rate * n_sim)`` (the number of successes
    among the ``n_sim`` simulation replicates). The Wilson score interval
    is computed pointwise via
    :func:`scipy.stats.binomtest`'s ``proportion_ci(method="wilson")``.

    Unlike the Wald interval, Wilson stays inside ``[0, 1]`` and retains
    its nominal coverage near the boundaries (``rate ≈ 0`` or ``≈ 1``),
    which is exactly where OC curves spend most of their interesting
    structure. See Brown, Cai & DasGupta (2001).

    Args:
        rates: 1-D array of empirical rates in ``[0, 1]`` (e.g. one
            column of an OC ``DataFrame``).
        n_sim: Number of simulation replicates each rate was averaged
            over. Constant across the grid in this codebase.
        confidence: Nominal coverage, e.g. ``0.95`` for a 95 % band.
            Must be in ``(0, 1)``.

    Returns:
        Tuple ``(lower, upper)`` of 1-D ``np.ndarray`` matching the shape
        of ``rates``. Both endpoints are clipped to ``[0, 1]`` by
        construction of the Wilson interval.

    Raises:
        ValueError: If ``n_sim < 1``, ``confidence`` is not in ``(0, 1)``,
            or any ``rate`` lies outside ``[0, 1]``.
    """
    rates = np.asarray(rates, dtype=float)
    if n_sim < 1:
        raise ValueError(f"n_sim must be >= 1; got {n_sim}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    if np.any((rates < 0.0) | (rates > 1.0)):
        raise ValueError("All rates must lie in [0, 1].")

    # binomtest works on integer counts, so we recover k from each rate.
    # round() is the right inverse here: rates produced by the harness
    # are themselves k / n_sim for integer k.
    counts = np.rint(rates * n_sim).astype(int)
    lower = np.empty_like(rates)
    upper = np.empty_like(rates)
    for i, k in enumerate(counts):
        ci = binomtest(int(k), n_sim).proportion_ci(
            confidence_level=confidence, method="wilson"
        )
        lower[i] = ci.low
        upper[i] = ci.high
    return lower, upper


# ====================================================================== #
#  Sequential simulation
# ====================================================================== #


def simulate_sequential(
    p_A: float,
    p_B: float,
    n_sim: int,
    rng: np.random.Generator,
    *,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    bf_upper: float = 3.0,
    bf_lower: float = 1.0 / 3.0,
    n_min: int = 50,
    n_max: int = 600,
    batch_size: int = 50,
    n_samples_mc: int = 2000,
) -> dict[str, float]:
    """Empirical stopping-time distribution of the sequential BF procedure.

    For each of ``n_sim`` replicates we run a fresh
    :class:`SequentialNonPairedBayesPropTest`, stream batches of size
    ``batch_size`` of independent Bernoulli observations from
    :func:`simulate_nonpaired_scores`, and record the per-arm sample
    size at which the procedure stops. Trials that hit ``n_max``
    without the BF crossing either threshold are *right-censored* and
    classified as ``"inconclusive"``.

    Args:
        p_A: True success probability for group A.
        p_B: True success probability for group B.
        n_sim: Number of independent sequential trajectories.
        rng: Pre-seeded NumPy generator.
        alpha0: Prior alpha for both arms.
        beta0: Prior beta for both arms.
        bf_upper: Stop for ``H₁`` when ``BF_10 ≥ bf_upper``.
        bf_lower: Stop for ``H₀`` when ``BF_10 ≤ bf_lower``.
        n_min: Minimum per-arm sample size before any BF-based stop is
            allowed. Guards against unstable early BFs.
        n_max: Hard cap on per-arm sample size — the trajectory is
            terminated and the trial reported as censored.
        batch_size: Per-arm batch size delivered to each ``update()``.
        n_samples_mc: Posterior MC draws per look (only affects the
            sequential test's internal ROPE / coverage estimates;
            the BF is computed analytically).

    Returns:
        Dict with the per-trial stopping-time statistics. Keys ``"p_A"``,
        ``"p_B"``, ``"delta"`` echo the inputs. Keys ``"median_n"``,
        ``"q05"``, ``"q25"``, ``"q75"``, ``"q95"`` are quantiles of the
        per-arm stopping sample size. ``"frac_censored"`` is the
        fraction of trials that hit ``n_max``. ``"frac_reject"`` /
        ``"frac_accept"`` are the fractions of trials whose final
        classification (via :func:`classify_bf`) is ``"reject"`` /
        ``"accept"``.
    """
    # How many looks fit inside the hard cap. We deliberately iterate
    # over an integer range (rather than ``while not seq.stopped``) so
    # we always terminate even if the stop logic is bypassed by an
    # external constraint.
    max_looks = n_max // batch_size

    stop_n: list[int] = []
    censored: list[bool] = []
    decisions: list[BFDecision] = []

    for _ in range(n_sim):
        seq = SequentialNonPairedBayesPropTest(
            alpha0=alpha0,
            beta0=beta0,
            bf_upper=bf_upper,
            bf_lower=bf_lower,
            n_min=n_min,
            n_max=n_max,
            decision_rule="bayes_factor",
            n_samples=n_samples_mc,
        )
        last = None
        for _look in range(max_looks):
            # Re-use the codebase's DGP for the streaming batches so the
            # sequential and fixed-n harnesses agree by construction.
            batch = simulate_nonpaired_scores(
                N=batch_size, theta_A=p_A, theta_B=p_B, rng=rng
            )
            last = seq.update(batch.y_A, batch.y_B)
            if seq.stopped:
                break
        assert last is not None, "max_looks=0 would skip the inner loop"

        stop_n.append(min(last.n_A, last.n_B))
        censored.append("n_max" in (last.stop_reason or "n_max reached"))

        # Use the *same* classifier the stopping rule uses internally,
        # so the sequential decision is reported through the public
        # ``classify_bf`` contract instead of being parsed out of the
        # ``stop_reason`` string.
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
