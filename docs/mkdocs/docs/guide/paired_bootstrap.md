# Paired Model — Bayesian Bootstrap

## Overview

The paired Bayesian-bootstrap test ([Rubin, 1981](#references)) is a
**nonparametric** alternative to
[`PairedBayesPropTest`](paired_laplace.md) (Laplace) and
[`PairedBayesPropTestPG`](paired_pg.md) (Pólya–Gamma Gibbs). Instead of
specifying a parametric likelihood or a prior on a logit-scale effect,
it places a flat Dirichlet "prior" over the empirical distribution of
paired differences and reads the posterior on the average treatment
effect off the resulting draws.

Use it when you want to:

* sidestep prior elicitation entirely;
* protect against model misspecification of the paired logistic
  likelihood;
* generate a posterior on $\Delta = p_A - p_B$ with no latent
  $\delta_A$ on the logit scale.

The price you pay is that **no Savage–Dickey BF is available** — there
is no parametric prior on $\Delta$ to evaluate at the null. Decisions
are routed through the ROPE / posterior-mass framework, and the
decision surface is intentionally lean: three quantities are enough.

## Generative model

For paired binary observations $(y_{A,i}, y_{B,i})$ form the per-pair
differences

$$
D_i = y_{A,i} - y_{B,i} \in \{-1, 0, +1\}.
$$

Each posterior draw of the average treatment effect is

$$
\Delta^{(s)} = \sum_{i=1}^n w_i^{(s)} D_i,
\qquad
\mathbf{w}^{(s)} \sim \text{Dirichlet}(\alpha, \dots, \alpha)
$$

with $\alpha = 1$ the standard noninformative choice. The class
exposes `dirichlet_alpha` as a configuration knob: values $< 1$
concentrate posterior mass on a small number of observations (sharper,
more bootstrap-like); values $> 1$ smooth toward the empirical mean.

## Quick start

```python
import numpy as np
from bayesprop.resources.bayes_paired_bootstrap import PairedBayesPropTestBB

# Paired binary outcomes (any 0/1 arrays of equal length).
y_A = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 0])
y_B = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])

model = PairedBayesPropTestBB(n_samples=20_000, seed=42).fit(y_A, y_B)

s = model.summary
print(f"Posterior mean Δ = {s.mean_delta:+.4f}")
print(f"95% CI = [{s.ci_95.lower:.4f}, {s.ci_95.upper:.4f}]")
```

## Decision API

The class ships only the three quantities that are well-defined
*directly* under the BB posterior — no Bayes-factor machinery, no
synthetic prior on $H_0$:

| Quantity | Where to read it |
|---|---|
| **Posterior of null** — $P(\Delta \in \text{ROPE} \mid \text{data})$ | `model.rope_test().pct_in_rope` |
| **Posterior of superiority** — $P(p_A > p_B \mid \text{data})$ | `model.summary.p_A_greater_B` |
| **ROPE decision** (reject / accept / undecided) | `model.rope_test().decision` |

```python
# (1) Posterior of null + full ROPE result
r = model.rope_test(rope=(-0.05, 0.05))
print(f"P(Δ ∈ ROPE) = {r.pct_in_rope:.3f}  → {r.decision}")
print(f"95% CI for Δ = [{r.ci_lower:.3f}, {r.ci_upper:.3f}]")

# (2) Posterior of superiority — straight off the fitted summary
print(f"P(p_A > p_B | data) = {model.summary.p_A_greater_B:.3f}")

# (3) Composite decision (bayes_factor and posterior_null are None by design)
d = model.decide()
assert d.bayes_factor is None
assert d.posterior_null is None
assert d.rule == "rope"
print(d.rope.decision)
```

What is intentionally **not** exposed:

- `savage_dickey_test()` — no parametric prior on $\Delta$ to evaluate
  at the null.
- `posterior_probability_H0()` — under the BB this is just
  `rope_test().pct_in_rope` read off the posterior directly. Wrapping
  it would force the user to commit to a prior on $H_0$ that has no
  role in the BB posterior itself, and any default flat-prior choice
  would be reparametrisation-non-invariant (Lindley–Jeffreys).

If you want a prior-dependent posterior probability of $H_0$ that
*does* react to evidence in a Bayes-factor sense, use a parametric
paired model (Laplace or Pólya–Gamma) with a Savage–Dickey BF — see
[Paired Laplace](paired_laplace.md) and
[Paired Pólya–Gamma](paired_pg.md).

## When to prefer this over the parametric paired classes

| Question | Use |
|---|---|
| Need a Savage–Dickey BF for a point null | `PairedBayesPropTest` (Laplace) or `PairedBayesPropTestPG` (Pólya–Gamma) |
| Need sequential / early-stopping support | `SequentialPairedBayesPropTest` |
| Worried about likelihood misspecification | **`PairedBayesPropTestBB`** |
| Sample size ≤ 30 and prior elicitation is acceptable | `PairedBayesPropTestPG` |
| Sample size ≫ 100 and want a prior-free posterior on $\Delta$ | **`PairedBayesPropTestBB`** |
| Want frequentist OC analysis with a McNemar baseline | `PairedBayesPropTest` (see [Frequentist Evaluation — Paired Laplace](frequentist_evaluation_paired.md)) |

## Plotting

```python
model.plot_posterior(rope=(-0.05, 0.05))
```

Shows the BB posterior histogram with the 95 % CI band, posterior mean,
and ROPE overlay.

## Performance notes

The implementation is fully vectorised — a single
`rng.dirichlet(α·1_n, size=S)` produces all $S$ weight vectors at once,
followed by one matmul `W @ D` for the posterior draws. On the
notebook's `n=10`, `S=50 000` example this is ~200× faster than a
per-draw Python loop. For large $n$ the weight matrix is chunked
internally to keep peak memory below ~400 MB.

## References

1. **Rubin** (1981). The Bayesian Bootstrap. *The Annals of Statistics*,
   9(1), 130–134.
2. **Kruschke** (2018). Rejecting or accepting parameter values in
   Bayesian estimation. *Advances in Methods and Practices in
   Psychological Science*, 1(2), 270–280. (ROPE-based decision making.)

## API

See [API Reference — Paired Model (Bayesian Bootstrap)](../api/bayes_paired_bootstrap.md)
for full method-level documentation.
