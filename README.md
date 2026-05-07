# Bayesian A/B Testing for Proportions

[![PyPI](https://img.shields.io/pypi/v/BayesProp?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/BayesProp/)
[![TestPyPI](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Ftest.pypi.org%2Fpypi%2FBayesProp%2Fjson&query=%24.info.version&prefix=v&logo=pypi&logoColor=white&label=TestPyPI)](https://test.pypi.org/project/BayesProp/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://avoss84.github.io/bayesProp/)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://github.com/AVoss84/bayesProp/actions/workflows/tests.yml/badge.svg)](https://github.com/AVoss84/bayesProp/actions/workflows/tests.yml)
[![codecov](https://raw.githubusercontent.com/AVoss84/bayesProp/badges/coverage-badge.svg)](https://github.com/AVoss84/bayesProp/actions/workflows/tests.yml)

A Python package for **Bayesian hypothesis testing** of success-rate differences in any Bernoulli-like experiment,
using analytic and approximate inference methods.
Input data can be **binary** (0/1) or **real-valued on (0, 1)** — continuous scores are automatically binarized at a configurable threshold.
Typical applications include comparing treatments, groups, items, model variants, or any two conditions whose outcomes can be expressed as proportions.
Please check out our [Getting Started](https://avoss84.github.io/bayesProp/getting_started/) guide for installation and quick examples.

## Features

- **Effect-size inference for proportions** — estimate and test the difference in success rates for both **paired** and **non-paired** samples
- **Savage–Dickey Bayes Factor** — test a point-null hypothesis ($\delta = 0$) without fitting a separate null model
- **Posterior of the null & ROPE** — quantify the posterior mass inside a Region of Practical Equivalence for nuanced decisions beyond simple reject/accept
- **Posterior predictive checks** — assess model fit by comparing observed data to data simulated from the posterior
- **Bayes Factor Design Analysis (BFDA)** — plan sample sizes to reach a target level of evidence *before* running the experiment
- **Publication-ready plots** — posterior distributions, predictive checks, Savage–Dickey density-ratio plots, and BFDA power curves out of the box

## Models

| Model | Class | Method | When to use |
|---|---|---|---|
| **Non-paired Beta–Bernoulli** | `NonPairedBayesPropTest` | Conjugate Beta posterior | Independent groups, exact & fast |
| **Paired Logistic (Laplace)** | `PairedBayesPropTest` | MAP + Laplace approximation | Paired scores, large *n*, fast iteration |
| **Paired Logistic (Pólya–Gamma)** | `PairedBayesPropTestPG` | Exact Gibbs sampling | Paired scores, small *n*, exact posterior |

## Quick start

```python
from bayesprop.resources.bayes_nonpaired import NonPairedBayesPropTest
from bayesprop.utils.utils import simulate_nonpaired_scores

# Simulate independent binary data
sim = simulate_nonpaired_scores(N=200, theta_A=0.75, theta_B=0.60, seed=42)
y_A, y_B = sim.y_A, sim.y_B

# Fit & summarise
model = NonPairedBayesPropTest(seed=42).fit(y_A, y_B)
print(model.summary)           # NonPairedSummary with mean_delta, ci_95, P(A>B), …

# Hypothesis test
bf = model.savage_dickey_test() # SavageDickeyResult with BF_10, decision, …

# Plots
model.plot_posteriors()
model.plot_savage_dickey()
```

## Package structure

```
├── pyproject.toml
├── justfile                   # task runner (just <recipe>)
├── .pre-commit-config.yaml    # ruff format + lint hooks
├── data/                      # evaluation datasets
├── docs/                      # model derivations & MkDocs site
├── src
│   ├── bayesprop
│   │   ├── config/            # global_config, YAML configs
│   │   ├── resources/
│   │   │   ├── bayes_nonpaired.py      # NonPairedBayesPropTest
│   │   │   ├── bayes_paired_laplace.py # PairedBayesPropTest
│   │   │   ├── bayes_paired_pg.py      # PairedBayesPropTestPG
│   │   │   ├── bfda_utils.py           # BFDA helpers
│   │   │   └── data_schemas.py         # Pydantic models
│   │   ├── services/
│   │   │   └── file.py                 # CSV / JSON / YAML / XLSX I/O
│   │   └── utils/
│   │       └── utils.py                # simulate, BFDA power curves, plots
│   └── notebooks/
│       ├── bayesian_AB_model_comparison_nonpaired.ipynb
│       ├── bayesian_AB_model_comparison_paired_laplace.ipynb
│       └── bayesian_AB_model_comparison_paired_gibbs.ipynb
└── tests/
    ├── test_bayes_nonpaired.py
    ├── test_bayes_paired_laplace.py
    ├── test_bayes_paired_pg.py
    ├── test_bfda_utils.py
    ├── test_data_schemas.py
    └── test_file_services.py
```

## Installation

```bash
pip install BayesProp
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install BayesProp
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
- Schönbrodt, F. D. & Wagenmakers, E.-J. (2018). Bayes factor design analysis: Planning for compelling evidence. *Psychonomic Bulletin & Review*, 25(1), 128–142.


