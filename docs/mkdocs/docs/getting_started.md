# Getting Started

## Installation

```bash
pip install bayesprop
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install bayesprop
```

For development (from source):

```bash
git clone https://github.com/AVoss84/bayesProp.git
cd bayesprop
uv venv --python 3.13
uv sync
source .venv/bin/activate
```

## Your first A/B test

### Non-paired design (independent groups)

Use this when group A and group B consist of **independent** observations (i.e. different items or subjects in each group).
Input arrays can be **binary** (0/1) or **real-valued on (0, 1)** — continuous scores are automatically binarized at a configurable threshold.

```python
from bayesprop.resources.bayes_nonpaired import NonPairedBayesPropTest
from bayesprop.utils.utils import simulate_nonpaired_scores

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

### Paired design (same items, two conditions)

Use this when **both** conditions (e.g. treatment A and treatment B, or version A and version B) are evaluated on the **same** items or subjects.

```python
from bayesprop.resources.bayes_paired_laplace import PairedBayesPropTest
from bayesprop.utils.utils import simulate_paired_scores

# Simulate paired binary data (y_A[i] and y_B[i] refer to the same item/subject)
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
from bayesprop.resources.bayes_paired_pg import PairedBayesPropTestPG

# Reuse paired data from above
model = PairedBayesPropTestPG(seed=42, n_iter=2000, burn_in=500, n_chains=4)
model.fit(y_A, y_B)
model.print_summary()

# MCMC diagnostics
diag = model.mcmc_diagnostics()
print(f"R-hat (δ_A): {diag.delta_A.r_hat:.3f}")
print(f"ESS (δ_A):   {diag.delta_A.ess:.0f}")
```

## Sample-size planning

Use Bayes Factor Design Analysis (BFDA) to determine how many observations you need:

```python
from bayesprop.utils.utils import bfda_power_curve, plot_bfda_power

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
