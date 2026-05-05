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
| [Utilities](bfda_utils.md) | BFDA, simulation & sample-size planning |

## Shared decision-rule interface

All three model classes expose the same methods for hypothesis testing:

| Method | Return type | Description |
|--------|-------------|-------------|
| `model.decide()` | `HypothesisDecision` | Run BF + P(H₀) + ROPE in a single call |
| `model.savage_dickey_test()` | `SavageDickeyResult` | Savage-Dickey Bayes factor |
| `model.posterior_probability_H0()` | `PosteriorProbH0Result` | Posterior null probability |
| `model.rope_test()` | `ROPEResult` | ROPE analysis |
