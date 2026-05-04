# Decision Rules

## Overview

After fitting a model, you need to **decide** whether model A differs from model B.
`bayesAB` provides three complementary decision frameworks, all accessible via a
single `model.decide()` call.

| Framework | What it tests | Key output |
|-----------|--------------|------------|
| **Bayes Factor** | Evidence for/against \(H_0: \Delta = 0\) | \(BF_{10}\), categorical decision |
| **Posterior Null** | Posterior probability \(P(H_0 \mid D)\) | \(P(H_0)\), threshold decision |
| **ROPE** | Practical equivalence | CI vs ROPE overlap, categorical decision |

## The `decide()` API

All three model classes share the same interface:

```python
model = SomeModel(seed=42).fit(y_A, y_B)

# Run all three frameworks at once
d = model.decide()

# Access individual results
print(d.bayes_factor.BF_10)       # Savage-Dickey BF₁₀
print(d.bayes_factor.decision)    # e.g. "Strong evidence against H0"

print(d.posterior_null.p_H0)      # P(H₀|data)
print(d.posterior_null.decision)  # e.g. "Reject H0"

print(d.rope.decision)            # e.g. "Reject H0 — A practically better"
print(d.rope.pct_in_rope)         # Fraction of posterior in ROPE
```

### Running a single framework

```python
# Only Bayes factor
d = model.decide(rule="bayes_factor")

# Only ROPE
d = model.decide(rule="rope")

# Only posterior null (requires BF internally)
d = model.decide(rule="posterior_null")
```

### Setting the default rule

```python
model = NonPairedBayesPropTest(
    decision_rule="rope",     # default for decide()
    rope_epsilon=0.03,        # half-width of ROPE (default: 0.02)
    seed=42,
).fit(y_A, y_B)

d = model.decide()  # runs only ROPE
```

## Framework 1: Bayes Factor (Savage-Dickey)

The Savage-Dickey density ratio tests \(H_0: \Delta = 0\):

\[
BF_{01} = \frac{f_\Delta^{\text{post}}(0)}{f_\Delta^{\text{prior}}(0)}
\]

- For the non-paired model, both densities are computed via **exact log-space
  convolution** of two Beta distributions.
- For the paired models, they are estimated with Gaussian KDE on posterior
  samples of \(\Delta = p_A - p_B\).

### Decision thresholds

| \(BF_{10}\) | Interpretation |
|-------------|---------------|
| > 100 | Decisive evidence against \(H_0\) |
| 30 – 100 | Very strong evidence against \(H_0\) |
| 10 – 30 | Strong evidence against \(H_0\) |
| 3 – 10 | Moderate evidence against \(H_0\) |
| 1 – 3 | Anecdotal evidence against \(H_0\) |
| 1 | No evidence |
| 1/3 – 1 | Anecdotal evidence for \(H_0\) |
| < 1/3 | Moderate to decisive evidence for \(H_0\) |

```python
bf = model.savage_dickey_test()
print(f"BF₁₀ = {bf.BF_10:.2f}")
print(f"Decision: {bf.decision}")
```

## Framework 2: Posterior Probability of \(H_0\)

Under a spike-and-slab prior, the posterior probability of the null hypothesis is:

\[
P(H_0 \mid D) = \frac{BF_{01} \cdot \pi_0}{BF_{01} \cdot \pi_0 + (1 - \pi_0)}
\]

### Decision thresholds

| Condition | Decision |
|-----------|----------|
| \(P(H_1 \mid D) > 0.95\) | Reject \(H_0\) |
| \(P(H_0 \mid D) > 0.95\) | Fail to reject \(H_0\) |
| Otherwise | Undecided |

```python
pn = model.posterior_probability_H0(bf_01=bf.BF_01)
print(f"P(H₀|D) = {pn.p_H0:.4f}  →  {pn.decision}")
```

## Framework 3: ROPE Analysis

The **Region of Practical Equivalence** (Kruschke, 2018) defines a range of
effect sizes \([-\varepsilon, +\varepsilon]\) considered "practically zero".
The decision is based on where the 95% credible interval falls relative to the ROPE:

| CI position | Decision |
|-------------|----------|
| 95% CI entirely **above** ROPE | Reject \(H_0\) — A practically better |
| 95% CI entirely **below** ROPE | Reject \(H_0\) — B practically better |
| 95% CI entirely **inside** ROPE | Accept \(H_0\) — practically equivalent |
| 95% CI **overlaps** ROPE boundary | Undecided — more data needed |

```python
rope = model.rope_test()
print(f"ROPE [{rope.rope_lower:.2f}, {rope.rope_upper:.2f}]")
print(f"95% CI: [{rope.ci_lower:.4f}, {rope.ci_upper:.4f}]")
print(f"% in ROPE: {rope.pct_in_rope:.1%}")
print(f"Decision: {rope.decision}")
```

### Choosing \(\varepsilon\)

The default `rope_epsilon=0.02` defines a ±2 percentage point ROPE on the
probability scale (\(\Delta = p_A - p_B\)).  Adjust based on your domain:

```python
# Tighter ROPE (±1 pp)
rope = model.rope_test(rope=(-0.01, 0.01))

# Wider ROPE (±5 pp)
rope = model.rope_test(rope=(-0.05, 0.05))
```

## Comparing all three frameworks

```python
d = model.decide()

print(f"Bayes Factor:    {d.bayes_factor.decision}")
print(f"Posterior Null:   {d.posterior_null.decision}")
print(f"ROPE:            {d.rope.decision}")
```

When the three frameworks agree, you can be confident in the conclusion.
When they disagree, examine why — typically due to different sensitivity
to effect size vs. statistical significance.

## Return types

| Type | Description |
|------|-------------|
| `HypothesisDecision` | Composite result from `decide()` |
| `SavageDickeyResult` | Bayes factor with BF₁₀, BF₀₁, decision |
| `PosteriorProbH0Result` | P(H₀), P(H₁), prior/posterior odds, decision |
| `ROPEResult` | CI bounds, ROPE overlap fractions, decision |

See [Data Schemas](../api/data_schemas.md) for full field documentation.

## References

1. **Kruschke, J. K.** (2018). Rejecting or accepting parameter values in Bayesian estimation. *Advances in Methods and Practices in Psychological Science*, 1(2), 270–280.
2. **Kass, R. E. & Raftery, A. E.** (1995). Bayes factors. *Journal of the American Statistical Association*, 90(430), 773–795.
