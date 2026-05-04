"""Bayes Factor Design Analysis (BFDA) utilities for sample-size planning.

Supports both **non-paired** (independent Beta-Bernoulli) and **paired**
(logistic Polya-Gamma) A/B test designs via a generic simulation engine
with pluggable BF computation backends.

References:
    1. Schoenbrodt, F. D. & Wagenmakers, E.-J. (2018). Bayes factor design
       analysis: Planning for compelling evidence. *Psychonomic Bulletin &
       Review*, 25(1), 128-142.
    2. Stefan, A. M., Gronau, Q. F., Schoenbrodt, F. D., & Wagenmakers, E.-J.
       (2019). A tutorial on Bayes Factor Design Analysis using an informed
       prior. *Behavior Research Methods*, 51(3), 1042-1058.

Typical usage::

    from bayesAB.utils.utils import (
        bfda_power_curve,
        bfda_power_curve_paired,
        bfda_power_curve_ph0,
        find_n_for_power,
        plot_bfda_power,
        plot_bfda_sensitivity,
    )

    # Non-paired
    curve = bfda_power_curve(theta_A_true=0.98, theta_B_true=0.93,
                             sample_sizes=[50, 100, 200, 500])

    # Paired
    curve_p = bfda_power_curve_paired(theta_A_true=0.98, theta_B_true=0.93,
                                      sample_sizes=[50, 100, 200, 500])
"""

from __future__ import annotations

from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from bayesAB.resources.bayes_nonpaired import beta_diff_pdf

# ======================================================================
#  Standalone data simulation utilities
# ======================================================================


def simulate_nonpaired_scores(
    N: int = 200,
    theta_A: float = 0.75,
    theta_B: float = 0.60,
    seed: int = 0,
) -> dict[str, Any]:
    """Simulate independent binary outcomes for a non-paired A/B test.

    Each group is sampled independently from a Bernoulli distribution
    with the specified success probability.

    Args:
        N: Number of observations per group.
        theta_A: True success probability for model A.
        theta_B: True success probability for model B.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys ``y_A``, ``y_B`` (binary arrays),
        ``theta_A``, ``theta_B`` (true rates), and ``true_params``.
    """
    rng = np.random.default_rng(seed)
    y_A = rng.binomial(1, theta_A, size=N).astype(float)
    y_B = rng.binomial(1, theta_B, size=N).astype(float)
    return {
        "y_A": y_A,
        "y_B": y_B,
        "theta_A": theta_A,
        "theta_B": theta_B,
        "true_params": {
            "N": N,
            "theta_A": theta_A,
            "theta_B": theta_B,
        },
    }


def simulate_paired_scores(
    N: int = 200,
    sigma_theta: float = 2.0,
    delta_A: float = 0.5,
    delta_B: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Simulate paired binary outcomes from a logistic random-effects DGP.

    Each item *i* has a latent ability ``theta_i ~ N(0, sigma_theta)``.
    The success probabilities are ``sigmoid(theta_i + delta_A)`` and
    ``sigmoid(theta_i + delta_B)`` for models A and B respectively.

    Args:
        N: Number of paired observations.
        sigma_theta: Standard deviation of the latent item ability.
        delta_A: Logit-scale offset for model A.
        delta_B: Logit-scale offset for model B.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys ``y_A``, ``y_B`` (binary arrays),
        ``p_A_true``, ``p_B_true`` (item-level probabilities),
        ``theta_true`` (latent abilities), and ``true_params``.
    """
    rng = np.random.default_rng(seed)
    theta_true = rng.normal(0.0, sigma_theta, size=N)
    p_A = _sigmoid(theta_true + delta_A)
    p_B = _sigmoid(theta_true + delta_B)
    y_A = rng.binomial(1, p_A)
    y_B = rng.binomial(1, p_B)
    return {
        "y_A": y_A,
        "y_B": y_B,
        "p_A_true": p_A,
        "p_B_true": p_B,
        "theta_true": theta_true,
        "true_params": {
            "N": N,
            "sigma_theta": sigma_theta,
            "delta_A": delta_A,
            "delta_B": delta_B,
        },
    }


def _sigmoid(x: npt.ArrayLike) -> np.ndarray:
    """Element-wise sigmoid (logistic) function."""
    return 1.0 / (1.0 + np.exp(-x))


# ======================================================================
#  Generic simulation engine
# ======================================================================


def bfda_simulate(
    data_generator: Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]],
    bf10_computer: Callable[[np.ndarray, np.ndarray], float],
    sample_sizes: list[int],
    bf_threshold: float = 3.0,
    n_sim: int = 500,
    seed: int = 42,
) -> dict[int, float]:
    """Generic BFDA engine -- works with any data-generating process and BF method.

    Args:
        data_generator: Callable(rng, n) -> (y_A, y_B). Generates one simulated
            dataset of size *n* per group using the provided RNG.
        bf10_computer: Callable(y_A, y_B) -> BF_10. Computes the Bayes factor
            in favour of H1 for a single dataset.
        sample_sizes: List of per-group sample sizes to evaluate.
        bf_threshold: BF_10 threshold for "decisive" evidence.
        n_sim: Number of simulated datasets per sample size.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping sample size -> P(BF_10 > threshold).
    """
    rng = np.random.default_rng(seed)

    power: dict[int, float] = {}
    for n in sample_sizes:
        decisive_count = 0
        for _ in range(n_sim):
            y_A, y_B = data_generator(rng, n)
            bf_10 = bf10_computer(y_A, y_B)
            if bf_10 > bf_threshold:
                decisive_count += 1  # Count how many simulations exceed the BF threshold / reject H0
        power[n] = decisive_count / n_sim

    return power


# ======================================================================
#  Non-paired (independent Beta-Bernoulli)
# ======================================================================


def _make_nonpaired_generator(
    theta_A_true: float, theta_B_true: float
) -> Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]]:
    """Create a data generator for independent Bernoulli groups."""

    def generator(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        y_A = rng.binomial(1, theta_A_true, size=n).astype(float)
        y_B = rng.binomial(1, theta_B_true, size=n).astype(float)
        return y_A, y_B

    return generator


def _make_nonpaired_bf_computer(alpha0: float = 1.0, beta0: float = 1.0) -> Callable[[np.ndarray, np.ndarray], float]:
    """Create a BF_10 computer for the non-paired Beta-Bernoulli model.

    Uses exact Savage-Dickey via ``beta_diff_pdf``.
    """
    prior_at_zero = beta_diff_pdf(0.0, alpha0, beta0, alpha0, beta0)

    def compute_bf10(y_A: np.ndarray, y_B: np.ndarray) -> float:
        n_A, n_B = len(y_A), len(y_B)
        k_A, k_B = y_A.sum(), y_B.sum()
        a_A = alpha0 + k_A
        b_A = beta0 + (n_A - k_A)
        a_B = alpha0 + k_B
        b_B = beta0 + (n_B - k_B)

        post_at_zero = beta_diff_pdf(0.0, a_A, b_A, a_B, b_B)
        return prior_at_zero / max(post_at_zero, 1e-300)

    return compute_bf10


def bfda_power_curve(
    theta_A_true: float,
    theta_B_true: float,
    sample_sizes: list[int],
    alpha0: float = 1.0,
    beta0: float = 1.0,
    bf_threshold: float = 3.0,
    n_sim: int = 500,
    seed: int = 42,
) -> dict[int, float]:
    """BFDA for **non-paired** Beta-Bernoulli model (exact Savage-Dickey).

    Simulates independent Bernoulli datasets and computes the proportion
    of simulations where BF_10 exceeds the decision threshold.

    Args:
        theta_A_true: Assumed true success rate for model A.
        theta_B_true: Assumed true success rate for model B.
        sample_sizes: List of per-group sample sizes to evaluate.
        alpha0: Prior Beta alpha parameter.
        beta0: Prior Beta beta parameter.
        bf_threshold: BF_10 threshold for "decisive" evidence.
        n_sim: Number of simulated datasets per sample size.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping sample size -> P(BF_10 > threshold).
    """
    return bfda_simulate(
        data_generator=_make_nonpaired_generator(theta_A_true, theta_B_true),
        bf10_computer=_make_nonpaired_bf_computer(alpha0, beta0),
        sample_sizes=sample_sizes,
        bf_threshold=bf_threshold,
        n_sim=n_sim,
        seed=seed,
    )


# ======================================================================
#  Paired (logistic Polya-Gamma)
# ======================================================================


def _make_paired_generator(
    theta_A_true: float, theta_B_true: float
) -> Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]]:
    """Create a data generator for paired Bernoulli observations.

    Same items are scored by both models -- generates paired binary data.
    """

    def generator(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        y_A = rng.binomial(1, theta_A_true, size=n).astype(float)
        y_B = rng.binomial(1, theta_B_true, size=n).astype(float)
        return y_A, y_B

    return generator


def _make_paired_bf_computer(
    prior_sigma_delta: float = 1.0,
    prior_sigma_mu: float = 2.0,
    n_iter: int = 1000,
    burn_in: int = 300,
    n_chains: int = 2,
    seed: int = 42,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Create a BF_10 computer for the paired logistic (PG Gibbs) model.

    Uses Savage-Dickey via KDE on delta_A posterior vs N(0, sigma_delta) prior.
    """

    def compute_bf10(y_A: np.ndarray, y_B: np.ndarray) -> float:
        from bayesAB.resources.bayes_paired_pg import PairedBayesPropTestPG

        model = PairedBayesPropTestPG(
            prior_sigma_delta=prior_sigma_delta,
            prior_sigma_mu=prior_sigma_mu,
            seed=seed,
            n_iter=n_iter,
            burn_in=burn_in,
            n_chains=n_chains,
        ).fit(y_A, y_B)

        bf = model.savage_dickey_test()
        return bf.BF_10

    return compute_bf10


def bfda_power_curve_paired(
    theta_A_true: float,
    theta_B_true: float,
    sample_sizes: list[int],
    prior_sigma_delta: float = 1.0,
    prior_sigma_mu: float = 2.0,
    bf_threshold: float = 3.0,
    n_sim: int = 200,
    n_iter: int = 1000,
    burn_in: int = 300,
    n_chains: int = 2,
    seed: int = 42,
) -> dict[int, float]:
    """BFDA for **paired** logistic model (Polya-Gamma Gibbs + Savage-Dickey).

    Simulates paired Bernoulli datasets and fits a PG logistic model to
    compute BF_10 via KDE-based Savage-Dickey on delta_A.

    Note:
        Computationally more expensive than the non-paired version because
        each simulated dataset requires MCMC. Use smaller ``n_sim`` and
        ``n_iter`` for design exploration; increase for final analysis.

    Args:
        theta_A_true: Assumed true success rate for model A.
        theta_B_true: Assumed true success rate for model B.
        sample_sizes: List of per-group sample sizes to evaluate.
        prior_sigma_delta: SD of N(0,sigma) prior on delta_A (logit scale).
        prior_sigma_mu: SD of N(0,sigma) prior on mu (logit scale).
        bf_threshold: BF_10 threshold for "decisive" evidence.
        n_sim: Number of simulated datasets per sample size.
        n_iter: Total Gibbs iterations per chain.
        burn_in: Warm-up iterations per chain.
        n_chains: Number of MCMC chains per dataset.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping sample size -> P(BF_10 > threshold).
    """
    return bfda_simulate(
        data_generator=_make_paired_generator(theta_A_true, theta_B_true),
        bf10_computer=_make_paired_bf_computer(
            prior_sigma_delta=prior_sigma_delta,
            prior_sigma_mu=prior_sigma_mu,
            n_iter=n_iter,
            burn_in=burn_in,
            n_chains=n_chains,
            seed=seed,
        ),
        sample_sizes=sample_sizes,
        bf_threshold=bf_threshold,
        n_sim=n_sim,
        seed=seed,
    )


# ======================================================================
#  P(H0) formulation (works with both designs)
# ======================================================================


def bf10_to_ph0(bf_10: float, prior_H0: float = 0.5) -> float:
    """Convert BF_10 to posterior probability of H0.

    Args:
        bf_10: Bayes factor in favour of H1.
        prior_H0: Prior probability of H0.

    Returns:
        P(H0 | data).
    """
    bf_01 = 1.0 / max(bf_10, 1e-300)
    return (bf_01 * prior_H0) / (bf_01 * prior_H0 + (1 - prior_H0))


def bfda_power_curve_ph0(
    theta_A_true: float,
    theta_B_true: float,
    sample_sizes: list[int],
    ph0_threshold: float = 0.05,
    prior_H0: float = 0.5,
    n_sim: int = 500,
    seed: int = 42,
    design: str = "nonpaired",
    alpha0: float = 1.0,
    beta0: float = 1.0,
    **paired_kwargs: Any,
) -> dict[int, float]:
    """BFDA using P(H0|data) < threshold as the decisiveness criterion.

    Works for both paired and non-paired designs.

    Args:
        theta_A_true: Assumed true success rate for model A.
        theta_B_true: Assumed true success rate for model B.
        sample_sizes: List of per-group sample sizes to evaluate.
        ph0_threshold: Reject H0 when P(H0|data) falls below this.
        prior_H0: Prior probability of H0.
        n_sim: Number of simulated datasets per sample size.
        seed: Random seed for reproducibility.
        design: ``"nonpaired"`` or ``"paired"``.
        alpha0: Prior Beta alpha (non-paired only).
        beta0: Prior Beta beta (non-paired only).
        **paired_kwargs: Extra arguments passed to the paired BF computer
            (e.g., ``prior_sigma_delta``, ``n_iter``, ``burn_in``, ``n_chains``).

    Returns:
        Dictionary mapping sample size -> P(P(H0|data) < threshold).
    """
    if design == "nonpaired":
        data_gen = _make_nonpaired_generator(theta_A_true, theta_B_true)
        bf_computer = _make_nonpaired_bf_computer(alpha0, beta0)
    elif design == "paired":
        data_gen = _make_paired_generator(theta_A_true, theta_B_true)
        bf_computer = _make_paired_bf_computer(seed=seed, **paired_kwargs)
    else:
        raise ValueError(f"Unknown design: {design!r}. Use 'nonpaired' or 'paired'.")

    rng = np.random.default_rng(seed)

    power: dict[int, float] = {}
    for n in sample_sizes:
        decisive_count = 0
        for _ in range(n_sim):
            y_A, y_B = data_gen(rng, n)
            bf_10 = bf_computer(y_A, y_B)
            p_h0 = bf10_to_ph0(bf_10, prior_H0)
            if p_h0 < ph0_threshold:
                decisive_count += 1
        power[n] = decisive_count / n_sim

    return power


# ======================================================================
#  Utility functions
# ======================================================================


def find_n_for_power(
    power_curve: dict[int, float],
    target_power: float = 0.80,
) -> float | None:
    """Interpolate the sample size needed to achieve a target power level.

    Args:
        power_curve: Dictionary mapping sample size -> power (from BFDA).
        target_power: Desired power level (default 0.80).

    Returns:
        Interpolated sample size, or ``None`` if target is outside the range.
    """
    ns = list(power_curve.keys())
    ps = list(power_curve.values())

    if min(ps) >= target_power:
        return float(ns[0])
    if max(ps) < target_power:
        return None

    f_interp = interp1d(ps, ns, kind="linear")
    return float(f_interp(target_power))


# ======================================================================
#  Plotting
# ======================================================================


def plot_bfda_power(
    power_curve: dict[int, float],
    theta_A_true: float,
    theta_B_true: float,
    bf_threshold: float = 3.0,
    target_power: float = 0.80,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a BFDA power curve with 80%/95% reference lines.

    Args:
        power_curve: Dictionary mapping sample size -> power.
        theta_A_true: Assumed true rate for model A (for title).
        theta_B_true: Assumed true rate for model B (for title).
        bf_threshold: BF_10 threshold used (for y-axis label).
        target_power: Power level to highlight via interpolation.
        title: Optional custom title.
        ax: Optional matplotlib Axes to plot on.

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.get_figure()

    ns = list(power_curve.keys())
    ps = list(power_curve.values())

    ax.plot(ns, ps, "o-", color="#2196F3", linewidth=2, markersize=7)
    ax.axhline(0.80, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="80% power")
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="95% power")

    n_target = find_n_for_power(power_curve, target_power)
    if n_target is not None:
        ax.axvline(
            n_target,
            color="#E91E63",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.7,
            label=f"n ~ {n_target:.0f} for {target_power:.0%} power",
        )

    delta = theta_A_true - theta_B_true
    if title is None:
        title = f"BFDA Power Curve -- Delta = {delta:.3f} (theta_A={theta_A_true:.2f}, theta_B={theta_B_true:.2f})"
    ax.set_xlabel("Sample size per group (n)")
    ax.set_ylabel(f"P(BF_10 > {bf_threshold:.0f})")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig


def plot_bfda_sensitivity(
    theta_A_true: float,
    theta_B_true: float,
    sample_sizes: list[int],
    thresholds: list[float] | None = None,
    n_sim: int = 500,
    seed: int = 42,
    design: str = "nonpaired",
    title: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Figure:
    """Plot BFDA power curves for multiple BF_10 thresholds.

    Works for both paired and non-paired designs.

    Args:
        theta_A_true: Assumed true success rate for model A.
        theta_B_true: Assumed true success rate for model B.
        sample_sizes: List of per-group sample sizes to evaluate.
        thresholds: BF_10 thresholds to compare (default: [3, 6, 10]).
        n_sim: Number of simulated datasets per sample size.
        seed: Random seed for reproducibility.
        design: ``"nonpaired"`` or ``"paired"``.
        title: Optional custom title.
        ax: Optional matplotlib Axes to plot on.
        **kwargs (Any): Extra arguments for the specific design (e.g. alpha0, beta0,
            prior_sigma_delta, n_iter, etc.).

    Returns:
        The matplotlib Figure.
    """
    if thresholds is None:
        thresholds = [3.0, 6.0, 10.0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.get_figure()

    colors = ["#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#2196F3"]

    for thresh, col in zip(thresholds, colors, strict=False):
        if design == "nonpaired":
            curve = bfda_power_curve(
                theta_A_true=theta_A_true,
                theta_B_true=theta_B_true,
                sample_sizes=sample_sizes,
                bf_threshold=thresh,
                n_sim=n_sim,
                seed=seed,
                alpha0=kwargs.get("alpha0", 1.0),
                beta0=kwargs.get("beta0", 1.0),
            )
        elif design == "paired":
            paired_kw = {k: v for k, v in kwargs.items() if k not in ("alpha0", "beta0")}
            curve = bfda_power_curve_paired(
                theta_A_true=theta_A_true,
                theta_B_true=theta_B_true,
                sample_sizes=sample_sizes,
                bf_threshold=thresh,
                n_sim=n_sim,
                seed=seed,
                **paired_kw,
            )
        else:
            raise ValueError(f"Unknown design: {design!r}")

        ax.plot(
            list(curve.keys()),
            list(curve.values()),
            "o-",
            color=col,
            linewidth=2,
            markersize=6,
            label=f"BF_10 > {thresh:.0f}",
        )

    ax.axhline(0.80, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Sample size per group (n)")
    ax.set_ylabel("P(BF_10 > threshold)")
    if title is None:
        title = f"BFDA Sensitivity -- {design.title()} Design"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig
