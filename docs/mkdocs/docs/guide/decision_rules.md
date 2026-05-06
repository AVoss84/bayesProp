# Decision Rules

## Overview

After fitting a model, you need to **decide** whether group A differs from group B.
`bayesprop` provides three complementary decision frameworks, all accessible via a
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

### Spike-and-slab prior

The standard continuous prior on \(\Delta\) assigns zero probability to
the point \(\Delta = 0\) because a single point has Lebesgue measure
zero. To give \(H_0\!: \Delta = 0\) a non-zero prior probability we
use a **spike-and-slab** prior — a mixture of a point mass (Dirac
delta) and a continuous density:

\[
p(\Delta) = \pi_0\;\delta_0(\Delta) \;+\; (1 - \pi_0)\;g(\Delta)
\]

where:

- \(\delta_0\) is the **Dirac measure** (the "spike") concentrated at
  \(\Delta = 0\),
- \(g(\Delta)\) is the continuous **slab** density (e.g. the prior
  on \(\Delta\) under \(H_1\)),
- \(\pi_0 = P(H_0)\) is the prior probability of the null.

In measure-theoretic terms the prior is a mixture of a discrete and a
continuous component: \(\pi_0\,\delta_{\{0\}}\) lives on the singleton
\(\{0\}\) while \((1-\pi_0)\,g\) is absolutely continuous with respect
to Lebesgue measure on \(\mathbb{R}\). The total prior is therefore
**neither** purely discrete nor purely continuous — it is a mixed
measure on \((\mathbb{R},\,\mathcal{B}(\mathbb{R}))\).

### Posterior model probability

Under this prior, Bayes' theorem yields the posterior probability of the
null model:

\[
P(H_0 \mid D)
= \frac{p(D \mid H_0)\;\pi_0}
       {p(D \mid H_0)\;\pi_0 + p(D \mid H_1)\;(1 - \pi_0)}
= \frac{BF_{01}\;\pi_0}
       {BF_{01}\;\pi_0 + (1 - \pi_0)}
\]

where \(BF_{01} = p(D \mid H_0)/p(D \mid H_1)\) is the Bayes factor
in favour of the null. Equivalently in odds form:

\[
\underbrace{\frac{P(H_0 \mid D)}{P(H_1 \mid D)}}_{\text{posterior odds}}
= BF_{01} \;\cdot\;
  \underbrace{\frac{\pi_0}{1 - \pi_0}}_{\text{prior odds}}
\]

This makes clear that the Bayes factor converts prior odds into
posterior odds, and the choice of \(\pi_0\) acts as a calibration knob.
The default \(\pi_0 = 0.5\) gives equal prior odds, so
\(P(H_0 \mid D)\) is determined entirely by the data through
\(BF_{01}\).

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

### Motivation

Both the Bayes factor and the posterior null test a **point null**
\(H_0\!: \Delta = 0\). In practice, a tiny but non-zero effect (say
\(\Delta = 0.001\)) is often irrelevant for decision-making. The
**Region of Practical Equivalence** (ROPE; Kruschke, 2018) addresses
this by replacing the point null with an **interval null**: any effect
inside \([-\varepsilon, +\varepsilon]\) is treated as practically
equivalent to zero.

This shifts the question from *"Is the effect exactly zero?"* to the
more useful *"Is the effect small enough to ignore?"*.

### Definition

Let \(\text{CI}_{95}\) denote the 95% highest-density credible interval
of the posterior of \(\Delta = \theta_A - \theta_B\) (or
\(\Delta = p_A - p_B\) for the paired models). The ROPE is the
symmetric interval:

\[
\text{ROPE} = [-\varepsilon,\; +\varepsilon]
\]

The decision rule compares the position of \(\text{CI}_{95}\) relative
to the ROPE:

| CI position | Decision |
|-------------|----------|
| 95% CI entirely **above** ROPE | Reject \(H_0\) — A practically better |
| 95% CI entirely **below** ROPE | Reject \(H_0\) — B practically better |
| 95% CI entirely **inside** ROPE | Accept \(H_0\) — practically equivalent |
| 95% CI **overlaps** ROPE boundary | Undecided — more data needed |

Additionally, the fraction of the posterior mass inside the ROPE is
reported:

\[
\rho = P(\Delta \in \text{ROPE} \mid D)
= \int_{-\varepsilon}^{+\varepsilon} p(\Delta \mid D)\;\mathrm{d}\Delta
\]

A high \(\rho\) (close to 1) indicates that most of the posterior
supports practical equivalence; a low \(\rho\) (close to 0) indicates
that the effect is clearly outside the ROPE.

### Relationship to the Bayes factor

The ROPE and the Bayes factor answer **different** questions. The Bayes
factor evaluates evidence for a point null (\(\Delta = 0\)); the ROPE
evaluates whether the entire posterior is consistent with a
**range** of negligible effects. In particular:

- A large \(BF_{10}\) (strong evidence against \(H_0\)) **and** 95% CI
  inside the ROPE can co-occur when the posterior is tightly
  concentrated at a small but non-zero \(\Delta\) — statistically
  detectable, but practically irrelevant.
- Conversely, a wide posterior may overlap the ROPE boundary
  (undecided) even when \(BF_{10} \approx 1\) (inconclusive).

Using both frameworks together guards against mistaking statistical
significance for practical importance.

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
3. **Mitchell, T. J. & Beauchamp, J. J.** (1988). Bayesian variable selection in linear regression. *Journal of the American Statistical Association*, 83(404), 1023–1032.
