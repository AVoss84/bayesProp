from __future__ import annotations

from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from bayesAB.resources.bayes_nonpaired import beta_diff_pdf
from bayesAB.resources.data_schemas import (
    DecisionRuleType,
    NonPairedSimResult,
    NonPairedTrueParams,
    PairedSimResult,
    PairedTrueParams,
)

# ======================================================================
#  Standalone data simulation utilities
# ======================================================================


def simulate_nonpaired_scores(
    N: int = 200,
    theta_A: float = 0.75,
    theta_B: float = 0.60,
    seed: int = 0,
    rng: np.random.Generator | None = None,
) -> NonPairedSimResult:
    """Simulate independent binary outcomes for a non-paired A/B test.

    Each group is sampled independently from a Bernoulli distribution
    with the specified success probability.

    Args:
        N: Number of observations per group.
        theta_A: True success probability for model A.
        theta_B: True success probability for model B.
        seed: Random seed for reproducibility.
        rng: Optional pre-seeded RNG; if provided, *seed* is ignored.

    Returns:
        :class:`NonPairedSimResult` with fields ``y_A``, ``y_B``,
        ``theta_A``, ``theta_B``, and ``true_params``.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    y_A = rng.binomial(1, theta_A, size=N).astype(float)
    y_B = rng.binomial(1, theta_B, size=N).astype(float)
    return NonPairedSimResult(
        y_A=y_A,
        y_B=y_B,
        theta_A=theta_A,
        theta_B=theta_B,
        true_params=NonPairedTrueParams(N=N, theta_A=theta_A, theta_B=theta_B),
    )


def simulate_paired_scores(
    N: int = 200,
    mu: float = 0.0,
    delta_A: float = 0.5,
    delta_B: float = 0.0,
    sigma_theta: float = 0.0,
    seed: int = 0,
    rng: np.random.Generator | None = None,
) -> PairedSimResult:
    """Simulate paired binary outcomes from a logistic DGP.

    Matches the paired model: ``y_A ~ Bern(σ(μ + δ_A))``,
    ``y_B ~ Bern(σ(μ))``.

    When ``sigma_theta > 0`` each item *i* additionally receives a
    random effect ``ε_i ~ N(0, sigma_theta)`` so that
    ``θ_i = μ + ε_i`` (useful for more realistic BFDA simulations).

    Args:
        N: Number of paired observations.
        mu: Shared logit-scale intercept.
        delta_A: Logit-scale treatment effect for model A.
        delta_B: Logit-scale offset for model B (0 by default).
        sigma_theta: SD of optional per-item random effects
            (``0`` = fixed effects matching the model).
        seed: Random seed for reproducibility.
        rng: Optional pre-seeded RNG; if provided, *seed* is ignored.

    Returns:
        :class:`PairedSimResult` with fields ``y_A``, ``y_B``,
        ``p_A_true``, ``p_B_true``, ``theta_true``, and ``true_params``.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    theta_true = mu + rng.normal(0.0, sigma_theta, size=N)
    p_A = _sigmoid(theta_true + delta_A)
    p_B = _sigmoid(theta_true + delta_B)
    y_A = rng.binomial(1, p_A)
    y_B = rng.binomial(1, p_B)
    return PairedSimResult(
        y_A=y_A,
        y_B=y_B,
        p_A_true=p_A,
        p_B_true=p_B,
        theta_true=theta_true,
        true_params=PairedTrueParams(N=N, mu=mu, sigma_theta=sigma_theta, delta_A=delta_A, delta_B=delta_B),
    )


def _sigmoid(x: npt.ArrayLike) -> np.ndarray:
    """Element-wise sigmoid (logistic) function."""
    return 1.0 / (1.0 + np.exp(-x))


# ======================================================================
#  Generic simulation engine
# ======================================================================


def bfda_simulate(
    data_generator: Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]],
    decision_fn: Callable[[np.ndarray, np.ndarray], bool],
    sample_sizes: list[int],
    n_sim: int = 500,
    seed: int = 42,
) -> dict[int, float]:
    """Generic BFDA engine -- works with any data-generating process and decision rule.

    Args:
        data_generator: Callable(rng, n) -> (y_A, y_B). Generates one simulated
            dataset of size *n* per group using the provided RNG.
        decision_fn: Callable(y_A, y_B) -> bool. Returns ``True`` when the
            simulated dataset produces a "decisive" result (i.e. rejects H0).
        sample_sizes: List of per-group sample sizes to evaluate.
        n_sim: Number of simulated datasets per sample size.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping sample size -> P(decisive outcome).
    """
    rng = np.random.default_rng(seed)

    power: dict[int, float] = {}
    for n in sample_sizes:
        decisive_count = sum(decision_fn(*data_generator(rng, n)) for _ in range(n_sim))
        power[n] = decisive_count / n_sim

    return power


# ======================================================================
#  Data generators (private)
# ======================================================================


def _make_nonpaired_generator(
    theta_A_true: float, theta_B_true: float
) -> Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]]:
    """Create a data generator for independent Bernoulli groups.

    Delegates to :func:`simulate_nonpaired_scores`.
    """

    def generator(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        sim = simulate_nonpaired_scores(N=n, theta_A=theta_A_true, theta_B=theta_B_true, rng=rng)
        return sim.y_A, sim.y_B

    return generator


def _make_paired_generator(
    theta_A_true: float,
    theta_B_true: float,
    sigma_theta: float = 0.0,
) -> Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]]:
    """Create a data generator for paired Bernoulli observations.

    Delegates to :func:`simulate_paired_scores`.  ``theta_A_true`` /
    ``theta_B_true`` are converted to the model's ``(mu, delta_A)``
    parameterisation: ``mu = logit(theta_B_true)`` and
    ``delta_A = logit(theta_A_true) - logit(theta_B_true)``.

    Note:
        Due to Jensen's inequality the realised marginal rates will
        differ slightly from the nominal values when ``sigma_theta > 0``.

    Args:
        theta_A_true: Target marginal success rate for model A.
        theta_B_true: Target marginal success rate for model B.
        sigma_theta: SD of optional per-item random effects
            (``0`` = fixed effects matching the model).
    """
    mu = np.log(theta_B_true / (1.0 - theta_B_true))  # logit(p_B)
    delta_A = np.log(theta_A_true / (1.0 - theta_A_true)) - mu  # logit(p_A) - logit(p_B)

    def generator(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        sim = simulate_paired_scores(N=n, mu=mu, delta_A=delta_A, sigma_theta=sigma_theta, rng=rng)
        return sim.y_A, sim.y_B

    return generator


# ======================================================================
#  Decision functions (private)
# ======================================================================


def _make_decision_fn(
    design: str,
    decision_rule: DecisionRuleType,
    *,
    # BF thresholds
    bf_threshold: float = 3.0,
    # P(H0) thresholds
    ph0_threshold: float = 0.05,
    prior_H0: float = 0.5,
    # ROPE thresholds
    rope: tuple[float, float] = (-0.02, 0.02),
    ci_mass: float = 0.95,
    # Non-paired model kwargs
    alpha0: float = 1.0,
    beta0: float = 1.0,
    # Paired model kwargs
    prior_sigma_delta: float = 1.0,
    prior_sigma_mu: float = 2.0,
    n_iter: int = 1000,
    burn_in: int = 300,
    n_chains: int = 2,
    seed: int = 42,
) -> Callable[[np.ndarray, np.ndarray], bool]:
    """Build a decision function for a given design × decision-rule combination.

    Returns a callable ``(y_A, y_B) -> bool`` that is ``True`` when the
    simulated dataset produces a decisive outcome (rejects H0).

    Args:
        design: ``"nonpaired"`` or ``"paired"``.
        decision_rule: ``"bayes_factor"``, ``"posterior_null"``, or ``"rope"``.
        bf_threshold: BF_10 threshold for "decisive" evidence (bayes_factor rule).
        ph0_threshold: Reject H0 when P(H0|data) falls below this (posterior_null rule).
        prior_H0: Prior probability of H0 (posterior_null rule).
        rope: (lower, upper) bounds of the ROPE interval (rope rule).
        ci_mass: Credible interval mass for ROPE analysis (rope rule).
        alpha0: Prior Beta alpha parameter (non-paired only).
        beta0: Prior Beta beta parameter (non-paired only).
        prior_sigma_delta: SD of N(0, σ) prior on delta_A (paired only).
        prior_sigma_mu: SD of N(0, σ) prior on mu (paired only).
        n_iter: Total Gibbs iterations per chain (paired only).
        burn_in: Warm-up iterations per chain (paired only).
        n_chains: Number of MCMC chains per dataset (paired only).
        seed: Random seed for reproducibility (paired only).
    """
    if decision_rule == "all":
        raise ValueError(
            "decision_rule='all' is not supported for BFDA. Choose one of 'bayes_factor', 'posterior_null', or 'rope'."
        )

    # ── Non-paired ────────────────────────────────────────────────────
    if design == "nonpaired":
        if decision_rule in ("bayes_factor", "posterior_null"):
            # Fast analytical BF via Savage-Dickey (no model fitting)
            prior_at_zero = beta_diff_pdf(0.0, alpha0, beta0, alpha0, beta0)

            def _nonpaired_bf(y_A: np.ndarray, y_B: np.ndarray) -> bool:
                n_A, n_B = len(y_A), len(y_B)
                a_A = alpha0 + y_A.sum()
                b_A = beta0 + (n_A - y_A.sum())
                a_B = alpha0 + y_B.sum()
                b_B = beta0 + (n_B - y_B.sum())
                post_at_zero = beta_diff_pdf(0.0, a_A, b_A, a_B, b_B)
                bf_10 = prior_at_zero / max(post_at_zero, 1e-300)

                if decision_rule == "bayes_factor":
                    return bf_10 > bf_threshold
                # posterior_null
                return bf10_to_ph0(bf_10, prior_H0) < ph0_threshold

            return _nonpaired_bf

        # ROPE — need model fitting for delta_samples
        def _nonpaired_rope(y_A: np.ndarray, y_B: np.ndarray) -> bool:
            from bayesAB.resources.bayes_nonpaired import NonPairedBayesPropTest

            model = NonPairedBayesPropTest(alpha0=alpha0, beta0=beta0, seed=seed).fit(y_A, y_B)
            result = model.rope_test(rope=rope, ci_mass=ci_mass)
            return result.decision.startswith("Reject H0")

        return _nonpaired_rope

    # ── Paired ────────────────────────────────────────────────────────
    if design == "paired":

        def _paired_decide(y_A: np.ndarray, y_B: np.ndarray) -> bool:
            from bayesAB.resources.bayes_paired_pg import PairedBayesPropTestPG

            model = PairedBayesPropTestPG(
                prior_sigma_delta=prior_sigma_delta,
                prior_sigma_mu=prior_sigma_mu,
                seed=seed,
                n_iter=n_iter,
                burn_in=burn_in,
                n_chains=n_chains,
            ).fit(y_A, y_B)

            if decision_rule == "bayes_factor":
                return model.savage_dickey_test().BF_10 > bf_threshold
            if decision_rule == "posterior_null":
                bf_10 = model.savage_dickey_test().BF_10
                return bf10_to_ph0(bf_10, prior_H0) < ph0_threshold
            # rope
            result = model.rope_test(rope=rope, ci_mass=ci_mass)
            return result.decision.startswith("Reject H0")

        return _paired_decide

    raise ValueError(f"Unknown design: {design!r}. Use 'nonpaired' or 'paired'.")


# ======================================================================
#  Unified BFDA power curve
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


def bfda_power_curve(
    theta_A_true: float,
    theta_B_true: float,
    sample_sizes: list[int],
    design: str = "nonpaired",
    decision_rule: DecisionRuleType = "bayes_factor",
    *,
    # BF thresholds
    bf_threshold: float = 3.0,
    # P(H0) thresholds
    ph0_threshold: float = 0.05,
    prior_H0: float = 0.5,
    # ROPE thresholds
    rope: tuple[float, float] = (-0.02, 0.02),
    ci_mass: float = 0.95,
    # Simulation
    n_sim: int = 500,
    seed: int = 42,
    # Non-paired model priors
    alpha0: float = 1.0,
    beta0: float = 1.0,
    # Paired DGP
    sigma_theta: float = 2.0,
    # Paired model priors / MCMC
    prior_sigma_delta: float = 1.0,
    prior_sigma_mu: float = 2.0,
    n_iter: int = 1000,
    burn_in: int = 300,
    n_chains: int = 2,
) -> dict[int, float]:
    """Unified Bayes Factor Design Analysis for any design × decision-rule.

    Simulates datasets under a known effect and estimates the probability
    that a given Bayesian decision rule will reject H0 as a function of
    sample size (i.e. Bayesian "power").

    Supported combinations:

    +----------------+----------------+--------------------+--------+
    | design         | decision_rule  | key threshold      | fast?  |
    +================+================+====================+========+
    | ``nonpaired``  | bayes_factor   | ``bf_threshold``   | yes    |
    | ``nonpaired``  | posterior_null | ``ph0_threshold``  | yes    |
    | ``nonpaired``  | rope           | ``rope``           | medium |
    | ``paired``     | bayes_factor   | ``bf_threshold``   | slow   |
    | ``paired``     | posterior_null | ``ph0_threshold``  | slow   |
    | ``paired``     | rope           | ``rope``           | slow   |
    +----------------+----------------+--------------------+--------+

    Args:
        theta_A_true: Assumed true success rate for model A.
        theta_B_true: Assumed true success rate for model B.
        sample_sizes: List of per-group sample sizes to evaluate.
        design: ``"nonpaired"`` or ``"paired"``.
        decision_rule: ``"bayes_factor"``, ``"posterior_null"``, or ``"rope"``.
        bf_threshold: BF_10 threshold for decisive evidence (``bayes_factor``).
        ph0_threshold: Reject H0 when P(H0|data) < this (``posterior_null``).
        prior_H0: Prior probability of H0 (``posterior_null``).
        rope: (lower, upper) bounds of the ROPE (``rope``).
        ci_mass: Credible interval mass for ROPE analysis (``rope``).
        n_sim: Number of simulated datasets per sample size.
        seed: Random seed for reproducibility.
        alpha0: Prior Beta alpha parameter (non-paired only).
        beta0: Prior Beta beta parameter (non-paired only).
        sigma_theta: SD of the shared latent item effect (paired DGP).
        prior_sigma_delta: SD of N(0, σ) prior on delta_A (paired only).
        prior_sigma_mu: SD of N(0, σ) prior on mu (paired only).
        n_iter: Total Gibbs iterations per chain (paired only).
        burn_in: Warm-up iterations per chain (paired only).
        n_chains: Number of MCMC chains per dataset (paired only).

    Returns:
        Dictionary mapping sample size -> P(decisive outcome).
    """
    # Build data generator
    if design == "nonpaired":
        data_gen = _make_nonpaired_generator(theta_A_true, theta_B_true)
    elif design == "paired":
        data_gen = _make_paired_generator(theta_A_true, theta_B_true, sigma_theta=sigma_theta)
    else:
        raise ValueError(f"Unknown design: {design!r}. Use 'nonpaired' or 'paired'.")

    # Build decision function
    decide = _make_decision_fn(
        design=design,
        decision_rule=decision_rule,
        bf_threshold=bf_threshold,
        ph0_threshold=ph0_threshold,
        prior_H0=prior_H0,
        rope=rope,
        ci_mass=ci_mass,
        alpha0=alpha0,
        beta0=beta0,
        prior_sigma_delta=prior_sigma_delta,
        prior_sigma_mu=prior_sigma_mu,
        n_iter=n_iter,
        burn_in=burn_in,
        n_chains=n_chains,
        seed=seed,
    )

    return bfda_simulate(
        data_generator=data_gen,
        decision_fn=decide,
        sample_sizes=sample_sizes,
        n_sim=n_sim,
        seed=seed,
    )


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
    **kwargs: Any,
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
        **kwargs: Extra arguments forwarded to :func:`bfda_power_curve`
            (e.g. ``alpha0``, ``beta0``, ``sigma_theta``,
            ``prior_sigma_delta``, ``n_iter``, etc.).

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
        curve = bfda_power_curve(
            theta_A_true=theta_A_true,
            theta_B_true=theta_B_true,
            sample_sizes=sample_sizes,
            design=design,
            decision_rule="bayes_factor",
            bf_threshold=thresh,
            n_sim=n_sim,
            seed=seed,
            **kwargs,
        )

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
