# Getting Started

## Installation

Clone the repository and install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/AVoss84/bayesianAB.git
cd bayesianAB
uv venv --python 3.13
uv sync
source .venv/bin/activate
```

## Your first A/B test

### Non-paired design (independent groups)

Use this when model A and model B are evaluated on **different** items.

```python
import numpy as np
from bayesAB.resources.bayes_nonpaired import NonPairedBayesPropTest

# Simulate binary outcomes
rng = np.random.default_rng(42)
y_A = rng.binomial(1, 0.85, size=100).astype(float)
y_B = rng.binomial(1, 0.70, size=100).astype(float)

# Fit the model
model = NonPairedBayesPropTest(seed=42, n_samples=20_000).fit(y_A, y_B)

# Print summary
model.print_summary()

# Hypothesis test
bf = model.savage_dickey_test()
print(f"BF₁₀ = {bf.BF_10:.2f}  →  {bf.decision}")

# Visualise
model.plot_posteriors()
model.plot_savage_dickey()
```

### Paired design (same items, two models)

Use this when **both** models are evaluated on the **same** items.

```python
from bayesAB.resources.bayes_paired_laplace import PairedBayesPropTest

# y_A[i] and y_B[i] refer to the same item
model = PairedBayesPropTest(seed=42).fit(y_A, y_B)
model.print_summary()
model.plot_savage_dickey()
```

For exact MCMC inference with convergence diagnostics:

```python
from bayesAB.resources.bayes_paired_pg import PairedBayesPropTestPG

model = PairedBayesPropTestPG(seed=42, n_iter=2000, burn_in=500, n_chains=4)
model.fit(y_A, y_B)
model.print_summary()

# MCMC diagnostics
diag = model.mcmc_diagnostics()
print(f"R-hat (δ_A): {diag.delta_A.r_hat:.3f}")
print(f"ESS (δ_A):   {diag.delta_A.ess:.0f}")
```

## Multi-metric comparison

All three model classes support forest plots and comparison tables:

```python
results = {}
for metric_name, y_a, y_b in metric_data:
    m = NonPairedBayesPropTest(seed=42).fit(y_a, y_b)
    results[metric_name] = m

NonPairedBayesPropTest.plot_forest(results, label_A="Model v2", label_B="Model v1")
NonPairedBayesPropTest.print_comparison_table(results)
```

## Sample-size planning

Use Bayes Factor Design Analysis (BFDA) to determine how many observations you need:

```python
from bayesAB.utils.utils import bfda_power_curve, plot_bfda_power

curve = bfda_power_curve(
    theta_A_true=0.85, theta_B_true=0.70,
    sample_sizes=[20, 50, 100, 200, 500],
    n_sim=500, seed=42,
)
plot_bfda_power(curve, theta_A_true=0.85, theta_B_true=0.70)
```

## Return types

All inference methods return **Pydantic models** with typed, validated fields:

| Method | Return type |
|--------|-------------|
| `model.test()` | `NonPairedTestResult` |
| `model.fit()` → `model.summary` | `NonPairedSummary` / `PairedSummary` |
| `model.savage_dickey_test()` | `SavageDickeyResult` |
| `model.posterior_probability_H0()` | `PosteriorProbH0Result` |
| `model.ppc_pvalues()` | `dict[str, PPCStatistic]` |
| `model.mcmc_diagnostics()` | `MCMCDiagnostics` |

See [Data Schemas](api/data_schemas.md) for full field documentation.
