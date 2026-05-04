# Bayes Factor Design Analysis (BFDA)

## Overview

BFDA is the Bayesian analog of frequentist power analysis. Given a
hypothesised true effect \\(\Delta = \theta_A - \theta_B\\), it estimates
the probability of obtaining **decisive** evidence (e.g. \\(BF_{10} > 3\\))
at each sample size via simulation.

\\[
\text{Bayesian Power}(n) = P\!\left(BF_{10} > \text{threshold} \;\middle|\; \theta_A, \theta_B, n\right)
\\]

## Non-paired design

```python
from bayesAB.utils.utils import (
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
from bayesAB.utils.utils import bfda_power_curve_paired

curve_paired = bfda_power_curve_paired(
    theta_A_true=0.85,
    theta_B_true=0.70,
    sample_sizes=[20, 50, 100, 200],
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

Instead of a BF threshold, you can use \\(P(H_0 \mid \text{data}) < \alpha\\)
as the decisiveness criterion:

```python
from bayesAB.utils.utils import bfda_power_curve_ph0

curve_ph0 = bfda_power_curve_ph0(
    theta_A_true=0.85,
    theta_B_true=0.70,
    sample_sizes=[50, 100, 200],
    ph0_threshold=0.05,
    design="nonpaired",
    n_sim=500,
)
```

## References

1. **Schönbrodt & Wagenmakers** (2018). Bayes factor design analysis. *Psychonomic Bulletin & Review*, 25(1), 128–142.
2. **Stefan et al.** (2019). A tutorial on BFDA using an informed prior. *Behavior Research Methods*, 51(3), 1042–1058.

## API

See [API Reference — Utilities](../api/bfda_utils.md) for full function documentation.
