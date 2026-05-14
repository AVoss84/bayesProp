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

All three model classes expose the same methods for hypothesis testing:

| Method | Return type | Description |
|--------|-------------|-------------|
| `model.decide(rule=None)` | `HypothesisDecision` | Run BF + P(H₀) + ROPE in a single call |
| `model.savage_dickey_test(null_value=0.0)` | `SavageDickeyResult` | Savage-Dickey Bayes factor at the null |
| `Model.posterior_probability_H0(BF_01, prior_H0=0.5)` | `PosteriorProbH0Result` | Static: convert a BF₀₁ to posterior P(H₀ ∣ D) |
| `model.rope_test(rope=None, ci_mass=0.95)` | `ROPEResult` | ROPE analysis on the difference posterior |
