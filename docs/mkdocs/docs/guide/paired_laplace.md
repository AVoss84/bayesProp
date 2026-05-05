# Paired Model — Laplace Approximation

## Overview

The paired model is used when **both** models are evaluated on the **same**
items. It uses a pooled Bernoulli logistic regression with a Laplace
approximation (MAP + analytical Hessian) for fast, analytic posterior inference.

## Generative model

\\[
\mu \sim \mathcal{N}(0, 2) \qquad
\delta_A \sim \mathcal{N}(0, \sigma_\delta)
\\]
\\[
y_{A,i} \sim \text{Bernoulli}(\sigma(\mu + \delta_A)) \qquad
y_{B,i} \sim \text{Bernoulli}(\sigma(\mu))
\\]

where \\(\sigma(\cdot)\\) is the logistic sigmoid function. The parameter
\\(\delta_A\\) captures model A's advantage on the logit scale.

### Directed Acyclic Graph (DAG)

```mermaid
graph TD
    sigma_mu(["σ_μ"]) --> mu["μ"]
    sigma_delta(["σ_δ"]) --> delta_A["δ_A"]

    mu --> pA["p_A = σ(μ + δ_A)"]
    delta_A --> pA
    mu --> pB["p_B = σ(μ)"]

    pA --> yA(["y_A,i"])
    pB --> yB(["y_B,i"])

    style sigma_mu fill:#e0e0e0,stroke:#757575
    style sigma_delta fill:#e0e0e0,stroke:#757575
    style mu fill:#bbdefb,stroke:#1565c0
    style delta_A fill:#bbdefb,stroke:#1565c0
    style pA fill:#c8e6c9,stroke:#2e7d32
    style pB fill:#c8e6c9,stroke:#2e7d32
    style yA fill:#fff9c4,stroke:#f9a825
    style yB fill:#fff9c4,stroke:#f9a825
```

<small>**Legend:** grey = hyperparameters, blue = latent parameters, green = deterministic,
yellow = observed data.</small>

## When to use

- **Fast inference** — no MCMC, results in milliseconds
- **Moderate sample sizes** — works well with \\(n \geq 30\\)
- **Exploratory analysis** — quick iteration before committing to full MCMC

For exact posterior inference with convergence diagnostics, see
[Paired Model (Pólya-Gamma)](paired_pg.md).

## Workflow

```python
from bayesAB.resources.bayes_paired_laplace import PairedBayesPropTest

model = PairedBayesPropTest(
    prior_sigma_delta=1.0,
    seed=42,
    n_samples=8000,
).fit(y_A, y_B)

# Summary
print(f"Mean Δ: {model.summary.mean_delta:.4f}")
print(f"P(A > B): {model.summary.p_A_greater_B:.4f}")
print(f"δ_A posterior mean: {model.summary.delta_A_posterior_mean:.4f}")

# Hypothesis test
bf = model.savage_dickey_test()
print(f"BF₁₀ = {bf.BF_10:.2f}  →  {bf.decision}")

# Unified decision (BF + P(H₀) + ROPE)
d = model.decide()
print(f"BF:   {d.bayes_factor.decision}")
print(f"P(H₀): {d.posterior_null.decision}")
print(f"ROPE: {d.rope.decision}  ({d.rope.pct_in_rope:.1%} in ROPE)")

# Plots
model.plot_posterior_delta()
model.plot_savage_dickey()
model.plot_laplace_posterior()
```

## Simulating paired data

For testing and validation, generate synthetic paired data:

```python
from bayesAB.utils.utils import simulate_paired_scores

sim = simulate_paired_scores(
    N=200, delta_A=0.5, seed=42
)
model = PairedBayesPropTest(seed=42).fit(sim.y_A, sim.y_B)
```

## Prior sensitivity analysis

```python
model.plot_sensitivity(prior_H0=0.5)
```

## API

See [API Reference — Paired Model (Laplace)](../api/bayes_paired_laplace.md) for full method documentation.
