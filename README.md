# bayesAB — Bayesian A/B Testing for Proportions

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://avoss84.github.io/bayesAB/)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://github.com/AVoss84/bayesAB/actions/workflows/tests.yml/badge.svg)](https://github.com/AVoss84/bayesAB/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/AVoss84/bayesAB/branch/main/graph/badge.svg)](https://codecov.io/gh/AVoss84/bayesAB)

A Python package for **Bayesian hypothesis testing** of binary (pass/fail) outcomes in A/B experiments.
It provides three complementary models, Savage–Dickey Bayes factors, posterior predictive checks,
Bayes Factor Design Analysis (BFDA) for sample-size planning, and publication-ready plots.
Please check out our [Getting Started](https://avoss84.github.io/bayesAB/getting_started/) guide for installation and quick examples.

## Models

| Model | Class | Method | When to use |
|---|---|---|---|
| **Non-paired Beta–Bernoulli** | `NonPairedBayesPropTest` | Conjugate Beta posterior | Independent groups, exact & fast |
| **Paired Logistic (Laplace)** | `PairedBayesPropTest` | MAP + Laplace approximation | Paired scores, large *n*, fast iteration |
| **Paired Logistic (Pólya–Gamma)** | `PairedBayesPropTestPG` | Exact Gibbs sampling | Paired scores, small *n*, exact posterior |

All models return **Pydantic data contracts** (`PairedSummary`, `SavageDickeyResult`, `PPCStatistic`, etc.)
for type-safe downstream use.

## Quick start

```python
from bayesAB.resources.bayes_paired_pg import PairedBayesPropTestPG
from bayesAB.utils.utils import simulate_paired_scores

# Simulate paired binary data
sim = simulate_paired_scores(N=200, delta_A=0.5, seed=42)
y_A, y_B = sim.y_A, sim.y_B

# Fit & summarise
model = PairedBayesPropTestPG(seed=42, n_iter=2000, burn_in=500, n_chains=4).fit(y_A, y_B)
print(model.summary)           # PairedSummary with mean_delta, ci_95, P(A>B), …

# Hypothesis test
bf = model.savage_dickey_test() # SavageDickeyResult with BF_10, decision, …

# Plots
model.plot_posteriors()
model.plot_ppc(seed=42)
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
│   ├── bayesAB
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
# curl -LsSf https://astral.sh/uv/install.sh | sh   # optional: install uv
uv venv --python 3.13
uv sync
source .venv/bin/activate
```

## Development

The project uses [just](https://github.com/casey/just) as a task runner and [pre-commit](https://pre-commit.com/) hooks (ruff format + ruff lint).

```bash
just test            # run pytest
just test-cov        # pytest with coverage
just format          # ruff format + fix
just lint            # format + mypy
just docs-serve      # local MkDocs preview
just pre-commit-all  # run all pre-commit hooks
```

## Documentation

Serve the MkDocs site locally:

```bash
just docs-serve      # http://127.0.0.1:8000
```

Mathematical derivations are available in `docs/`:
- [Beta–Beta posterior inference](docs/beta_beta_posterior_inference.md)
- [Hierarchical Beta regression model](docs/hierarchical_beta_regression_model.md)
- [Hierarchical logistic–normal model](docs/hierarchical_logistic_normal_model.md)

## Dependencies

- Python ≥ 3.13
- numpy, scipy, matplotlib, pandas
- pydantic (v2)
- polyagamma

## References

- Polson, N. G., Scott, J. G. & Windle, J. (2013). Bayesian inference for logistic models using Pólya–Gamma latent variables. *JASA*, 108(504), 1339–1349.
- Schönbrodt, F. D. & Wagenmakers, E.-J. (2018). Bayes factor design analysis: Planning for compelling evidence. *Psychonomic Bulletin & Review*, 25(1), 128–142.


