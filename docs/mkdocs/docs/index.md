# bayesAB

**Bayesian A/B testing for proportions** — a Python package for comparing two models (or treatments) using binarized pass/fail scores with fully Bayesian inference, Bayes factor hypothesis testing, and sample-size planning.

## Features

- **Non-paired Beta-Bernoulli model** — conjugate posterior with exact Savage-Dickey Bayes factor via log-space convolution (no KDE)
- **Paired logistic model (Laplace)** — fast MAP + Hessian approximation for paired binary data
- **Paired logistic model (Pólya-Gamma)** — exact MCMC via Gibbs sampling with convergence diagnostics (R-hat, ESS)
- **Bayes Factor Design Analysis (BFDA)** — Bayesian sample-size planning with power curves
- **Pydantic data contracts** — typed, validated return values for all inference results
- **Publication-ready plots** — posteriors, Savage-Dickey visualisation, forest plots, comparison tables

## Quick example

```python
import numpy as np
from bayesAB.resources.bayes_nonpaired import NonPairedBayesPropTest

rng = np.random.default_rng(42)
y_A = rng.binomial(1, 0.85, size=100).astype(float)
y_B = rng.binomial(1, 0.70, size=100).astype(float)

model = NonPairedBayesPropTest(seed=42).fit(y_A, y_B)
model.print_summary()

bf = model.savage_dickey_test()
print(f"BF₁₀ = {bf.BF_10:.2f}  →  {bf.decision}")
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
- [API Reference](api/index.md) — full module documentation
