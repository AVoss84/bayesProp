# Bayes Factor Design Analysis (BFDA)

## Overview

BFDA is the Bayesian analog of frequentist power analysis
(Schönbrodt & Wagenmakers, 2018). In classical power analysis one asks:
*"How large must the sample be so that a frequentist test rejects
$H_0$ with probability $\geq 1-\beta$?"* — BFDA asks the analogous
Bayesian question: *"How large must the sample be so that the Bayes
factor reaches decisive evidence with high probability?"*

### Motivation

Unlike $p$-values, Bayes factors quantify evidence on a continuous scale
and can favour either $H_0$ or $H_1$. However, an experiment can still
yield an **inconclusive** Bayes factor (e.g. $BF_{10} \approx 1$) if
the sample is too small. BFDA addresses this by estimating, for each
candidate sample size $n$, the probability of obtaining a Bayes factor
that exceeds a pre-specified evidence threshold — ensuring that the
experiment is adequately powered to be informative.

### Definition

Given assumed true success rates $\theta_A$ and $\theta_B$ and a
decisiveness threshold $\gamma$ (e.g. $\gamma = 3$), BFDA estimates
**Bayesian power** at sample size $n$ as:

$$
\text{Power}(n) = P\!\left(BF_{10} > \gamma \;\middle|\; \theta_A, \theta_B, n\right)
$$

Because no closed-form expression exists for the distribution of
$BF_{10}$ under the alternative, the probability is estimated via
**Monte Carlo simulation**: for each $n$, many datasets are generated
from the assumed DGP, each is analysed with the chosen Bayesian model,
and the fraction of replications where $BF_{10} > \gamma$ is the
estimated power.

### Algorithm

For each candidate sample size $n$ and for $s = 1, \dots, S$ simulation
replications:

1. **Generate** synthetic data
   $y_A^{(s)} \sim \text{Bern}(\theta_A)^n$,
   $y_B^{(s)} \sim \text{Bern}(\theta_B)^n$
   under the assumed true rates.
2. **Fit** the chosen Bayesian model (non-paired or paired) to
   $(y_A^{(s)}, y_B^{(s)})$.
3. **Compute** the Bayes factor $BF_{10}^{(s)}$ via the Savage-Dickey
   density ratio.
4. **Record** whether $BF_{10}^{(s)} > \gamma$.

The estimated power is $\widehat{\text{Power}}(n) = S^{-1}\sum_{s} \mathbf{1}\!\bigl[BF_{10}^{(s)} > \gamma\bigr]$.

### Alternative decision rules

Besides the Bayes factor threshold, BFDA also supports the **posterior
null** criterion $P(H_0 \mid D) < \alpha$ as the decisiveness condition,
which can be more intuitive for practitioners who think in terms of
posterior probabilities rather than evidence ratios.

!!! tip "After choosing `n`, verify the procedure"
    BFDA picks `n` to hit a target *power*. To check that the resulting
    decision rule is also well-calibrated across the parameter space —
    three-way OC curves, Type-I sweeps, CI coverage, sequential ESS,
    matched-α Fisher baselines — see the
    [Frequentist Evaluation guide](frequentist_evaluation.md).

## Non-paired design

```python
from bayesprop.utils.utils import (
    bfda_power_curve,
    find_n_for_power,
    plot_bfda_power,
    plot_bfda_sensitivity,
)

curve = bfda_power_curve(
    theta_A_true=0.85,
    theta_B_true=0.70,
    sample_sizes=[20, 50, 100, 200, 500],
    bf_threshold=3.0,
    n_sim=500,
    seed=42,
)

# Find required sample size for 80% power
n_needed = find_n_for_power(curve, target_power=0.80)
print(f"Need ~{n_needed:.0f} observations per group for 80% power")

# Plot
plot_bfda_power(curve, theta_A_true=0.85, theta_B_true=0.70)
```

## Paired design

```python
from bayesprop.utils.utils import bfda_power_curve

curve_paired = bfda_power_curve(
    theta_A_true=0.85,
    theta_B_true=0.70,
    sample_sizes=[20, 50, 100, 200],
    design="paired",
    decision_rule="bayes_factor",
    n_sim=100,   # fewer sims — each requires MCMC
    seed=42,
)
```

!!! note
    The paired design is computationally more expensive because each
    simulated dataset requires MCMC. Use smaller `n_sim` for exploration.

## Sensitivity to BF threshold

Compare power across different evidence thresholds:

```python
plot_bfda_sensitivity(
    theta_A_true=0.85,
    theta_B_true=0.70,
    sample_sizes=[20, 50, 100, 200, 500],
    thresholds=[3.0, 6.0, 10.0],
    n_sim=500,
)
```

## P(H₀) formulation

Instead of a BF threshold, you can use $P(H_0 \mid \text{data}) < \alpha$
as the decisiveness criterion:

```python
from bayesprop.utils.utils import bfda_power_curve

curve_ph0 = bfda_power_curve(
    theta_A_true=0.85,
    theta_B_true=0.70,
    sample_sizes=[50, 100, 200],
    design="nonpaired",
    decision_rule="posterior_null",
    ph0_threshold=0.05,
    n_sim=500,
)
```

## References

1. **Schönbrodt & Wagenmakers** (2018). Bayes factor design analysis. *Psychonomic Bulletin & Review*, 25(1), 128–142.
2. **Stefan et al.** (2019). A tutorial on BFDA using an informed prior. *Behavior Research Methods*, 51(3), 1042–1058.

## API

See [API Reference — Utilities](../api/bfda_utils.md) for full function documentation.
