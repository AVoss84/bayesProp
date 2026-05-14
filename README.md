<p align="center">
  <img src="https://raw.githubusercontent.com/AVoss84/bayesProp/main/docs/mkdocs/docs/images/package_logo.PNG" alt="BayesProp Logo" width="300">
</p>

# Bayesian A/B Testing for Proportions

[![PyPI](https://img.shields.io/pypi/v/BayesProp?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/BayesProp/)
[![Downloads](https://static.pepy.tech/badge/BayesProp)](https://pepy.tech/projects/bayesprop)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/AVoss84/bayesProp/actions/workflows/tests.yml/badge.svg)](https://github.com/AVoss84/bayesProp/actions/workflows/tests.yml)
[![codecov](https://raw.githubusercontent.com/AVoss84/bayesProp/badges/coverage-badge.svg)](https://github.com/AVoss84/bayesProp/actions/workflows/tests.yml)

A Python package for **Bayesian hypothesis testing** of success-rate differences in any Bernoulli-like experiment,
using analytic and approximate inference methods.
Input data can be **binary** (0/1) or **real-valued on (0, 1)** — continuous scores are automatically binarized at a configurable threshold.
Typical applications include comparing treatments, groups, items, model variants, or any two conditions whose outcomes can be expressed as proportions.
Please check out our [Getting Started](https://avoss84.github.io/bayesProp/getting_started/) guide for installation and quick examples.

## Features

- **Effect-size inference for proportions** — estimate and test the difference in success rates for both **paired** and **non-paired** samples
- **Savage–Dickey Bayes Factor** — test a point-null hypothesis ('treatment effect / difference is zero') without fitting a separate null model
- **Posterior of the null & ROPE** — quantify the posterior mass inside a Region of Practical Equivalence for nuanced decisions beyond simple reject/accept
- **Posterior predictive checks** — assess model fit by comparing observed data to data simulated from the posterior
- **Bayes Factor Design Analysis (BFDA)** — plan sample sizes to reach a target level of evidence *before* running the experiment
- **Sequential / streaming design** — update the posterior batch-by-batch as data arrive and stop early once the Bayes factor crosses an upper or lower threshold (`SequentialNonPairedBayesPropTest`, `SequentialPairedBayesPropTest`)
- **Operating-characteristic analysis** — *calibrated-Bayes* frequentist evaluation of the chosen decision rule: three-way decision rates (`reject` / `accept` / `inconclusive`), Type-I sweep over the baseline rate, 95 % credible-interval coverage, and the sequential stopping-time distribution, with matched-α **Fisher's exact** (non-paired) or **McNemar exact** (paired) baselines overlaid. Pre-built Monte-Carlo harness in `bayesprop.utils.operation_characteristics` and `…_paired`, plus turnkey notebooks for both designs
- **Publication-ready plots** — posterior distributions, predictive checks, Savage–Dickey density-ratio plots, BFDA power curves, sequential BF₁₀ trajectories, and OC diagnostic plots (with Wilson Monte-Carlo bands) out of the box

## Models

| Model | Class | Method | When to use |
|---|---|---|---|
| **Non-paired Beta–Bernoulli** | `NonPairedBayesPropTest` | Conjugate Beta posteriors per arm; P(B>A) by quadrature, Δ summaries by Monte Carlo | Independent groups, exact & fast |
| **Paired Logistic (Laplace)** | `PairedBayesPropTest` | MAP + Laplace approximation | Paired scores, large *n*, fast iteration |
| **Paired Logistic (Pólya–Gamma)** | `PairedBayesPropTestPG` | Exact Gibbs sampling | Paired scores, small *n*, exact posterior |
| **Paired Bayesian Bootstrap** | `PairedBayesPropTestBB` | Nonparametric — Dirichlet weights on the empirical distribution of paired differences | Paired scores, no prior elicitation, robustness to likelihood misspecification (ROPE-driven; no Savage–Dickey BF) |

## Quick start

```python
import numpy as np
from bayesprop.resources.bayes_nonpaired import NonPairedBayesPropTest

# Binary or [0,1]-valued data:
y_A = np.array([1,1,0,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,1])     # 16/20 = 0.80
y_B = np.array([0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,0,0])     #  6/20 = 0.30

# Fit posterior & summarise
model = NonPairedBayesPropTest(seed=42).fit(y_A, y_B)

s = model.summary
print(f"\nMean Δ (θ_A − θ_B) = {s.mean_delta:+.4f}")
print(f"95% CI = [{s.ci_95.lower:.4f}, {s.ci_95.upper:.4f}]")
print(f"P(A > B) = {s.p_A_greater_B:.4f}")

# ── Unified decision ─────────────────────────────────────────────────
d = model.decide()
bf = d.bayes_factor

print("\n--- Unified Decision ---")
print(f"  Bayes Factor: BF₁₀ = {bf.BF_10:.2f}  → {bf.decision}")
print(f"  Posterior Null: P(H₀|D) = {d.posterior_null.p_H0:.4f}  → {d.posterior_null.decision}")
print(f"  ROPE: {d.rope.decision} ({d.rope.pct_in_rope:.1%} in ROPE)")

# Plots
model.plot_posteriors()
model.plot_savage_dickey()
```

## Installation

```bash
pip install BayesProp
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add BayesProp
```

For development (from source):

```bash
git clone https://github.com/AVoss84/bayesProp.git
cd bayesprop
uv venv --python 3.13
uv sync
source .venv/bin/activate
```

## Dependencies

- Python ≥ 3.13
- numpy, scipy, matplotlib, pandas
- pydantic (v2)
- polyagamma

## References

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman & Hall/CRC.
- Kruschke, J. K. (2018). Rejecting or accepting parameter values in Bayesian estimation. *Advances in Methods and Practices in Psychological Science*, 1(2), 270–280.
- Polson, N. G., Scott, J. G. & Windle, J. (2013). Bayesian inference for logistic models using Pólya–Gamma latent variables. *JASA*, 108(504), 1339–1349.
- Rubin, D. B. (1981). The Bayesian Bootstrap. *The Annals of Statistics*, 9(1), 130–134.
- Schönbrodt, F. D. & Wagenmakers, E.-J. (2018). Bayes factor design analysis: Planning for compelling evidence. *Psychonomic Bulletin & Review*, 25(1), 128–142.


