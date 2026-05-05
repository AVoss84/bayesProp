# bayesprop

**Bayesian A/B testing for proportions** — a Python package for comparing two models (or treatments) using binarized pass/fail scores with fully Bayesian inference, Bayes factor hypothesis testing, and sample-size planning.

## Features

- **Non-paired Beta-Bernoulli model** — conjugate posterior with exact Savage-Dickey Bayes factor via log-space convolution (no KDE)
- **Paired logistic model (Laplace)** — fast MAP + Hessian approximation for paired binary data
- **Paired logistic model (Pólya-Gamma)** — exact MCMC via Gibbs sampling with convergence diagnostics (R-hat, ESS)
- **Unified decision framework** — `model.decide()` runs Bayes factor, posterior null probability, and ROPE analysis in a single call
- **ROPE analysis** — Region of Practical Equivalence testing with automatic CI-based decisions (Kruschke, 2018)
- **Bayes Factor Design Analysis (BFDA)** — Bayesian sample-size planning with power curves
- **Pydantic data contracts** — typed, validated return values for all inference results
- **Publication-ready plots** — posteriors, Savage-Dickey visualisation, forest plots, comparison tables

## Quick example

```python
from bayesprop.resources.bayes_nonpaired import NonPairedBayesPropTest
from bayesprop.utils.utils import simulate_nonpaired_scores

sim = simulate_nonpaired_scores(N=100, theta_A=0.85, theta_B=0.70, seed=42)
y_A, y_B = sim.y_A, sim.y_B

model = NonPairedBayesPropTest(seed=42).fit(y_A, y_B)
model.print_summary()

# Unified decision (BF + P(H₀) + ROPE in one call)
d = model.decide()
print(f"BF₁₀ = {d.bayes_factor.BF_10:.2f}  →  {d.bayes_factor.decision}")
print(f"ROPE: {d.rope.decision}  ({d.rope.pct_in_rope:.1%} in ROPE)")
```

## Models at a glance

| Model | Module | Design | Inference |
|-------|--------|--------|-----------|
| `NonPairedBayesPropTest` | `bayes_nonpaired` | Independent groups | Conjugate Beta-Bernoulli |
| `PairedBayesPropTest` | `bayes_paired_laplace` | Paired observations | Laplace approximation |
| `PairedBayesPropTestPG` | `bayes_paired_pg` | Paired observations | Pólya-Gamma Gibbs sampler |

## Navigation

- [Getting Started](getting_started.md) — installation and first steps
- [User Guide](guide/nonpaired.md) — detailed walkthroughs for each model
- [Decision Rules](guide/decision_rules.md) — ROPE, Bayes factor, and the unified `decide()` API
- [API Reference](api/index.md) — full module documentation
