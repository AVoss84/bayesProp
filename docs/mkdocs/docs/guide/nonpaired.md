# Non-Paired Beta-Bernoulli Model

## Overview

The non-paired model compares two **independent** groups using a conjugate
Beta-Bernoulli model. Each group has its own success probability
\\(\theta_A\\) and \\(\theta_B\\), estimated independently from binarized
pass/fail data.

## Generative model

\\[
\theta_A \sim \text{Beta}(\alpha_0, \beta_0) \qquad
\theta_B \sim \text{Beta}(\alpha_0, \beta_0)
\\]
\\[
y_{A,i} \sim \text{Bernoulli}(\theta_A) \qquad
y_{B,i} \sim \text{Bernoulli}(\theta_B)
\\]

The posterior is available in closed form:

\\[
\theta_A \mid y_A \sim \text{Beta}(\alpha_0 + k_A,\; \beta_0 + n_A - k_A)
\\]

where \\(k_A = \sum y_{A,i}\\) is the number of successes.

## Difference posterior

The distribution of \\(\Delta = \theta_A - \theta_B\\) is computed via
**exact log-space convolution** (`beta_diff_pdf`), not KDE. This avoids
bandwidth selection issues and gives deterministic, reproducible results.

## Savage-Dickey Bayes Factor

The hypothesis test \\(H_0: \Delta = 0\\) vs \\(H_1: \Delta \neq 0\\) uses
the Savage-Dickey density ratio:

\\[
BF_{01} = \frac{f_\Delta^{\text{post}}(0)}{f_\Delta^{\text{prior}}(0)}
\\]

Both densities are computed via exact convolution.

## Workflow

```python
from bayesAB.resources.bayes_nonpaired import NonPairedBayesPropTest

model = NonPairedBayesPropTest(
    alpha0=1.0,    # uniform prior
    beta0=1.0,
    seed=42,
    n_samples=50_000,
).fit(y_A, y_B)

# Summary statistics
print(f"Mean Δ: {model.summary.mean_delta:.4f}")
print(f"95% CI: [{model.summary.ci_95.lower:.4f}, {model.summary.ci_95.upper:.4f}]")
print(f"P(A > B): {model.summary.p_A_greater_B:.4f}")

# Hypothesis test
bf = model.savage_dickey_test()
print(f"BF₁₀ = {bf.BF_10:.2f}  →  {bf.decision}")

# ROPE analysis
rope = model.rope_test()
print(f"ROPE: {rope.decision}  ({rope.pct_in_rope:.1%} in ROPE)")

# Or use the unified decide() for all three frameworks at once
d = model.decide()
print(f"BF: {d.bayes_factor.decision}")
print(f"P(H₀): {d.posterior_null.decision}")
print(f"ROPE: {d.rope.decision}")

# Posterior predictive checks
ppc = model.ppc_pvalues(seed=42)
for stat_name, stat in ppc.items():
    print(f"  {stat_name}: p={stat.p_value:.3f} ({stat.status})")

# Plots
model.plot_posteriors()
model.plot_savage_dickey()
```

## Prior sensitivity

Test how results change with different priors:

```python
for name, a0, b0 in [("Uniform", 1, 1), ("Jeffreys", 0.5, 0.5), ("Informative", 2, 2)]:
    m = NonPairedBayesPropTest(alpha0=a0, beta0=b0, seed=42).fit(y_A, y_B)
    bf = m.savage_dickey_test()
    print(f"{name}: BF₁₀={bf.BF_10:.2f}, P(A>B)={m.summary.p_A_greater_B:.4f}")
```

## API

See [API Reference — Non-Paired Model](../api/bayes_nonpaired.md) for full method documentation.
