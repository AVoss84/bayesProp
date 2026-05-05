# Getting Started

## Installation

```bash
pip install bayesAB
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install bayesAB
```

For development (from source):

```bash
git clone https://github.com/AVoss84/bayesAB.git
cd bayesAB
uv venv --python 3.13
uv sync
source .venv/bin/activate
```

## Your first A/B test

### Non-paired design (independent groups)

Use this when model A and model B are evaluated on **different** items.

```python
from bayesAB.resources.bayes_nonpaired import NonPairedBayesPropTest
from bayesAB.utils.utils import simulate_nonpaired_scores

# Simulate binary outcomes
sim = simulate_nonpaired_scores(N=100, theta_A=0.85, theta_B=0.70, seed=42)
y_A, y_B = sim.y_A, sim.y_B

# Fit the model
model = NonPairedBayesPropTest(seed=42, n_samples=20_000).fit(y_A, y_B)

# Print summary
model.print_summary()

# Unified decision (Bayes factor + P(H₀) + ROPE)
d = model.decide()
print(f"BF₁₀ = {d.bayes_factor.BF_10:.2f}  →  {d.bayes_factor.decision}")
print(f"ROPE: {d.rope.decision}  ({d.rope.pct_in_rope:.1%} in ROPE)")

# Visualise
model.plot_posteriors()
model.plot_savage_dickey()
```

### Paired design (same items, two models)

Use this when **both** models are evaluated on the **same** items.

```python
from bayesAB.resources.bayes_paired_laplace import PairedBayesPropTest
from bayesAB.utils.utils import simulate_paired_scores

# Simulate paired binary data (y_A[i] and y_B[i] refer to the same item)
sim = simulate_paired_scores(N=100, delta_A=0.5, seed=42)
y_A, y_B = sim.y_A, sim.y_B

model = PairedBayesPropTest(seed=42).fit(y_A, y_B)
model.print_summary()

# Unified decision
d = model.decide()
print(f"BF₁₀ = {d.bayes_factor.BF_10:.2f}  →  {d.bayes_factor.decision}")
print(f"ROPE: {d.rope.decision}")

model.plot_savage_dickey()
```

For exact MCMC inference with convergence diagnostics:

```python
from bayesAB.resources.bayes_paired_pg import PairedBayesPropTestPG

# Reuse paired data from above
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
| `model.decide()` | `HypothesisDecision` |
| `model.savage_dickey_test()` | `SavageDickeyResult` |
| `model.posterior_probability_H0()` | `PosteriorProbH0Result` |
| `model.rope_test()` | `ROPEResult` |
| `model.ppc_pvalues()` | `dict[str, PPCStatistic]` |
| `model.mcmc_diagnostics()` | `MCMCDiagnostics` |

See [Data Schemas](api/data_schemas.md) for full field documentation.
