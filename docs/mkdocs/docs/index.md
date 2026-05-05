# bayesprop

**Bayesian A/B testing for proportions** — a Python package for **Bayesian hypothesis testing** of binary (pass/fail) outcomes in A/B experiments
using analytic and approximate inference methods —
particularly relevant when comparing features in **software engineering** or evaluating model changes in **AI/MLOps** pipelines.

## Features

- **Effect-size inference for proportions** — estimate and test the difference in success rates for both **paired** and **non-paired** samples
- **Savage–Dickey Bayes Factor** — test a point-null hypothesis ($\delta = 0$) without fitting a separate null model
- **Posterior of the null & ROPE** — quantify the posterior mass inside a Region of Practical Equivalence for nuanced decisions beyond simple reject/accept
- **Posterior predictive checks** — assess model fit by comparing observed data to data simulated from the posterior
- **Bayes Factor Design Analysis (BFDA)** — plan sample sizes to reach a target level of evidence *before* running the experiment
- **Publication-ready plots** — posterior distributions, predictive checks, Savage–Dickey density-ratio plots, and BFDA power curves out of the box

## Quick example

```python
from bayesprop.resources.bayes_nonpaired import NonPairedBayesPropTest
from bayesprop.utils.utils import simulate_nonpaired_scores

sim = simulate_nonpaired_scores(N=100, theta_A=0.85, theta_B=0.70, seed=42)
y_A, y_B = sim.y_A, sim.y_B

model = NonPairedBayesPropTest(seed=42).fit(y_A, y_B)
model.print_summary()

# Unified decision (BF + P(H₀) + ROPE in one call)
d = model.decide()
print(f"BF₁₀ = {d.bayes_factor.BF_10:.2f}  →  {d.bayes_factor.decision}")
print(f"ROPE: {d.rope.decision}  ({d.rope.pct_in_rope:.1%} in ROPE)")
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
- [Decision Rules](guide/decision_rules.md) — ROPE, Bayes factor, and the unified `decide()` API
- [API Reference](api/index.md) — full module documentation
