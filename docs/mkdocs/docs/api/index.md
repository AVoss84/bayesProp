# API Reference

Full API documentation for all `bayesprop` modules, auto-generated
from source code docstrings.

## Modules

| Module | Description |
|--------|-------------|
| [Data Schemas](data_schemas.md) | Pydantic data contracts — `HypothesisDecision`, `ROPEResult`, `SavageDickeyResult`, configs |
| [Non-Paired Model](bayes_nonpaired.md) | Independent Beta-Bernoulli A/B test |
| [Paired Model (Laplace)](bayes_paired_laplace.md) | Paired logistic model with Laplace approximation |
| [Paired Model (Pólya-Gamma)](bayes_paired_pg.md) | Paired logistic model with PG Gibbs sampler |
| [Paired Model (Bayesian Bootstrap)](bayes_paired_bootstrap.md) | Nonparametric paired test via Dirichlet weights on the empirical distribution |
| [Sequential designs](sequential.md) | Warm-started sequential variants of the non-paired and paired-Laplace models |
| [Utilities](bfda_utils.md) | DGPs, frequentist baselines, BFDA, decision helpers |
| [Operating Characteristics](operation_characteristics.md) | Monte-Carlo OC simulation harness for the non-paired model (fixed-n + sequential) |
| [Operating Characteristics (Paired)](operation_characteristics_paired.md) | Monte-Carlo OC simulation harness for the paired-Laplace model (fixed-n + sequential) |

## Shared decision-rule interface

All four model classes inherit from `BaseBayesPropTest` and expose the same core workflow:

### Fit → Summarise → Decide

| Method | Return type | Description |
|--------|-------------|-------------|
| `model.fit(y_A, y_B)` | `self` | Fit the model to observed data (method chaining) |
| `model.summary` | `PairedSummary` / `NonPairedSummary` | Posterior summary (θ means, Δ, CI, P(A > B)) |
| `model.decide(rule=None)` | `HypothesisDecision` | Run BF + P(H₀) + ROPE in a single call |
| `model.rope_test(rope=None, ci_mass=0.95)` | `ROPEResult` | ROPE analysis on the difference posterior |
| `model.print_summary()` | `None` | Print a human-readable summary to stdout |

### Plotting

| Method | Description |
|--------|-------------|
| `model.plot_posteriors()` | Single-panel overlay of θ_A and θ_B posteriors |
| `model.plot_posterior_delta()` | Single-panel KDE of Δ = θ_A − θ_B (probability scale) with 95 % CI |

### Parametric-only methods

The following are available on the three parametric models (`NonPairedBayesPropTest`, `PairedBayesPropTest`, `PairedBayesPropTestPG`) but **not** on the Bayesian bootstrap (`PairedBayesPropTestBB`), which has no parametric prior on Δ:

| Method | Return type | Description |
|--------|-------------|-------------|
| `model.savage_dickey_test(null_value=0.0)` | `SavageDickeyResult` | Savage-Dickey Bayes factor at the point null |
| `Model.posterior_probability_H0(BF_01, prior_H0=0.5)` | `PosteriorProbH0Result` | Static: convert a BF₀₁ to posterior P(H₀ ∣ D) |
| `model.plot_savage_dickey()` | `None` | Prior vs posterior density with BF annotation |
