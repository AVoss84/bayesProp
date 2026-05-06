# Non-Paired Beta-Bernoulli Model

## Overview

The non-paired model compares two **independent** groups using a conjugate
Beta-Bernoulli model. Each group has its own success probability
$\theta_A$ and $\theta_B$, estimated independently from binarized
pass/fail data.

Use this model when group A and group B consist of **independent** observations
(i.e. different items or subjects in each group).

Input arrays can be **binary** (0/1) or **real-valued on (0, 1)** — continuous
scores are automatically binarized at a configurable threshold.

## Generative model

$$
\theta_A \sim \text{Beta}(\alpha_0, \beta_0) \qquad
\theta_B \sim \text{Beta}(\alpha_0, \beta_0)
\qquad (\text{independent draws})
$$

$$
y_{A,i} \sim \text{Bernoulli}(\theta_A), \quad i = 1, \dots, n_A \qquad
y_{B,j} \sim \text{Bernoulli}(\theta_B), \quad j = 1, \dots, n_B
$$

Here $\alpha_0$ and $\beta_0$ are **fixed** hyperparameters (user-specified
constants, not random variables). Although both groups share the same prior
family, $\theta_A$ and $\theta_B$ are drawn **independently**, so the two
groups are fully independent:

$$
p(y_A, y_B \mid \alpha_0, \beta_0)
= p(y_A \mid \alpha_0, \beta_0)\;p(y_B \mid \alpha_0, \beta_0)
$$

Dependence would only arise in a hierarchical model where $\alpha_0, \beta_0$
are themselves random with a shared hyperprior. In this model they are fixed
constants, so the DAG edges from $\alpha_0, \beta_0$ to both $\theta_A$ and
$\theta_B$ encode the same prior specification — not a probabilistic
dependence path.

The posterior is available in closed form via conjugacy:

$$
\theta_A \mid y_A \sim \text{Beta}(\alpha_0 + k_A,\; \beta_0 + n_A - k_A)
$$

where $k_A = \sum_{i} y_{A,i}$ is the number of successes (and analogously for group B).

### Directed Acyclic Graph (DAG)

```mermaid
graph TD
    alpha0(["α₀"]) --> thetaA["θ_A"]
    beta0(["β₀"]) --> thetaA
    alpha0 --> thetaB["θ_B"]
    beta0 --> thetaB

    thetaA --> yA(["y_A,i"])
    thetaB --> yB(["y_B,j"])

    style alpha0 fill:#e0e0e0,stroke:#757575
    style beta0 fill:#e0e0e0,stroke:#757575
    style thetaA fill:#bbdefb,stroke:#1565c0
    style thetaB fill:#bbdefb,stroke:#1565c0
    style yA fill:#fff9c4,stroke:#f9a825
    style yB fill:#fff9c4,stroke:#f9a825
```

<small>**Legend:** grey = hyperparameters, blue = latent parameters,
yellow = observed data.</small>

## Posterior probability of superiority

A key quantity of interest is the probability that group A has a higher
success rate than group B:

$$
P(\theta_A > \theta_B \mid y)
= \int_0^1 f_{\theta_A \mid y}(x)\;
  F_{\theta_B \mid y}(x)\;\mathrm{d}x
$$

where $f_{\theta_A \mid y}$ is the posterior **density** of $\theta_A$
and $F_{\theta_B \mid y}$ is the posterior **CDF** of $\theta_B$.

### Derivation

Because $\theta_A$ and $\theta_B$ are independent a posteriori:

$$
P(\theta_A > \theta_B \mid y)
= \int_0^1 \int_0^x
    f_{\theta_A \mid y}(x)\;f_{\theta_B \mid y}(t)
  \;\mathrm{d}t\;\mathrm{d}x
= \int_0^1 f_{\theta_A \mid y}(x)
  \underbrace{\int_0^x f_{\theta_B \mid y}(t)\;\mathrm{d}t}_{
    F_{\theta_B \mid y}(x)}\;\mathrm{d}x
$$

Substituting the conjugate posteriors
$\theta_A \mid y \sim \text{Beta}(a_A, b_A)$ and
$\theta_B \mid y \sim \text{Beta}(a_B, b_B)$:

$$
P(\theta_A > \theta_B \mid y)
= \int_0^1
    \frac{x^{a_A - 1}(1-x)^{b_A - 1}}{B(a_A, b_A)}
    \;I_x(a_B, b_B)
  \;\mathrm{d}x
$$

where $I_x(a, b) = B(a,b)^{-1}\int_0^x t^{a-1}(1-t)^{b-1}\,\mathrm{d}t$
is the **regularised incomplete Beta function**.

### Numerical evaluation

The integral is computed via **Gauss-Legendre quadrature** with $n_q$
nodes on $[0, 1]$. The Beta density is evaluated in log-space for
numerical stability:

$$
P(\theta_A > \theta_B \mid y)
\approx \sum_{j=1}^{n_q} w_j \;\exp\!\bigl[
  (a_A\!-\!1)\log x_j + (b_A\!-\!1)\log(1\!-\!x_j) - \log B(a_A, b_A)
\bigr]\;I_{x_j}(a_B, b_B)
$$

where $(x_j, w_j)$ are the transformed quadrature nodes and weights on
$[0, 1]$. This gives a **deterministic, exact** result (up to
floating-point precision) — no Monte Carlo noise. The implementation
is in `prob_greater` in `bayesprop.resources.bayes_nonpaired`.

## Difference posterior (exact convolution)

### Distribution of a difference of independent random variables

Let $X$ and $Y$ be independent continuous random variables with densities
$f_X$ and $f_Y$. The density of $Z = X - Y$ is the **convolution** of
$f_X$ with the reflection of $f_Y$:

$$
f_Z(z) = \int_{-\infty}^{\infty} f_X(x)\;f_Y(x - z)\;\mathrm{d}x
$$

This follows directly from the CDF:

$$
P(Z \leq z)
= P(X - Y \leq z)
= \int\!\!\int_{\{(x,y):\,x - y \leq z\}}
  f_X(x)\,f_Y(y)\;\mathrm{d}y\;\mathrm{d}x
$$

Substituting $y = x - z'$ and differentiating with respect to $z$ yields
the convolution integral above.

### Application to the Beta posteriors

In our model the two posteriors are independent:

$$
\theta_A \mid y_A \sim \text{Beta}(a_A,\, b_A), \qquad
\theta_B \mid y_B \sim \text{Beta}(a_B,\, b_B)
$$

with $a_A = \alpha_0 + k_A$, $b_A = \beta_0 + n_A - k_A$ (and
analogously for group B). Because both $\theta_A$ and $\theta_B$ have
support $[0, 1]$, the difference $\Delta = \theta_A - \theta_B$ has
support $(-1, 1)$, and the integration limits tighten to:

$$
f_{\Delta \mid y}(z)
= \int_{\max(0,\,z)}^{\min(1,\,1+z)}
    f_{\theta_A \mid y}(x) \;\cdot\; f_{\theta_B \mid y}(x - z)
  \;\mathrm{d}x
$$

The lower limit $\max(0, z)$ ensures $x \in [0,1]$; the upper limit
$\min(1, 1+z)$ ensures $x - z \in [0,1]$.

Substituting the Beta densities:

$$
f_{\Delta \mid y}(z)
= \frac{1}{B(a_A, b_A)\, B(a_B, b_B)}
  \int_{\max(0,\,z)}^{\min(1,\,1+z)}
    x^{a_A - 1}(1 - x)^{b_A - 1}
    (x - z)^{a_B - 1}(1 - x + z)^{b_B - 1}
  \;\mathrm{d}x
$$

where $B(a, b) = \Gamma(a)\Gamma(b)/\Gamma(a+b)$ is the Beta function.

### Closed form and numerical evaluation

Pham-Gia & Turkkan (1993) showed that the convolution integral admits a
**closed-form** expression in terms of **Appell's first hypergeometric
function** $F_1(a;\,b_1,b_2;\,c;\,x,y)$, split by the sign of $z$.
However, the $F_1$ arguments leave the double-series convergence region
near $z = 0$, requiring analytic continuation. In practice it is simpler
(and equally exact) to evaluate the convolution integral **directly** via
trapezoidal quadrature with the integrand computed in **log-space** for
numerical stability:

$$
\log f_{\Delta}(z)
= \log\!\int \exp\!\bigl[
    (a_A\!-\!1)\log x + (b_A\!-\!1)\log(1\!-\!x)
  + (a_B\!-\!1)\log(x\!-\!z) + (b_B\!-\!1)\log(1\!-\!x\!+\!z)
\bigr]\,\mathrm{d}x
\;-\; \log B(a_A, b_A) - \log B(a_B, b_B)
$$

This avoids underflow that would occur with direct multiplication of
many small values when the Beta parameters are large. The
implementation is in `beta_diff_pdf` in
`bayesprop.resources.bayes_nonpaired`.

!!! note "Reference"
    Pham-Gia, T. & Turkkan, N. (1993). Bayesian analysis of the
    difference of two proportions. *Communications in Statistics —
    Theory and Methods*, **22**(6), 1755–1771.

### Properties

- **Deterministic** — no random sampling, so repeated calls yield
  identical results.
- **Exact** — no KDE bandwidth selection or MC noise; the only
  approximation is floating-point quadrature error (negligible in
  practice).
- **Fast** — evaluating $f_\Delta(z)$ on a grid of 500 points takes
  a few milliseconds on modern hardware.

## Savage-Dickey Bayes Factor

The hypothesis test $H_0\!: \Delta = 0$ vs $H_1\!: \Delta \neq 0$ uses
the Savage-Dickey density ratio:

$$
BF_{01} = \frac{f_\Delta^{\text{post}}(0)}{f_\Delta^{\text{prior}}(0)}
\qquad\Longrightarrow\qquad
BF_{10} = \frac{1}{BF_{01}}
$$

Both densities are computed via exact convolution (no KDE needed), so the
Bayes factor is fully deterministic.

## Step-by-step example

### 1. Simulate data

```python
from bayesprop.utils.utils import simulate_nonpaired_scores
from bayesprop.resources.bayes_nonpaired import NonPairedBayesPropTest

sim = simulate_nonpaired_scores(N=150, theta_A=0.80, theta_B=0.60, seed=42)

print(f"True θ_A = {sim.theta_A:.2f},  θ_B = {sim.theta_B:.2f}")
print(f"True Δ   = {sim.theta_A - sim.theta_B:.2f}")
print(f"Observed rates: A = {sim.y_A.mean():.3f},  B = {sim.y_B.mean():.3f}")
```

### 2. Fit the model

```python
model = NonPairedBayesPropTest(
    alpha0=1.0,      # Beta(1,1) = uniform prior
    beta0=1.0,
    seed=42,
    n_samples=50_000,
).fit(sim.y_A, sim.y_B)

s = model.summary
print(f"Mean Δ (θ_A − θ_B) = {s.mean_delta:+.4f}")
print(f"95% CI = [{s.ci_95.lower:.4f}, {s.ci_95.upper:.4f}]")
print(f"P(A > B) = {s.p_A_greater_B:.4f}")
```

### 3. Unified decision (BF + P(H₀) + ROPE)

```python
d = model.decide()

print(f"Bayes Factor:  BF₁₀ = {d.bayes_factor.BF_10:.2f}  → {d.bayes_factor.decision}")
print(f"Posterior Null: P(H₀|D) = {d.posterior_null.p_H0:.4f}  → {d.posterior_null.decision}")
print(f"ROPE:          {d.rope.decision}  ({d.rope.pct_in_rope:.1%} in ROPE)")
```

### 4. Plot posteriors

```python
model.plot_posteriors(title="Beta-Bernoulli Posteriors")
```

![Posterior distributions of θ_A, θ_B, and Δ](../images/non-paired/posterior_distributions.png)

### 5. Exact convolution vs Monte Carlo

Visualise the exact density of $\Delta$ alongside MC samples:

```python
import numpy as np
import matplotlib.pyplot as plt
from bayesprop.resources.bayes_nonpaired import beta_diff_pdf

z_grid = np.linspace(-0.5, 0.8, 500)

post_density = np.array([
    beta_diff_pdf(z, model.a_A, model.b_A, model.a_B, model.b_B)
    for z in z_grid
])
prior_density = np.array([
    beta_diff_pdf(z, 1.0, 1.0, 1.0, 1.0)
    for z in z_grid
])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(z_grid, post_density, color="#9C27B0", linewidth=2, label="Posterior (exact)")
ax.fill_between(z_grid, post_density, alpha=0.15, color="#9C27B0")
ax.plot(z_grid, prior_density, color="gray", linewidth=1.5, linestyle="--",
        alpha=0.7, label="Prior (Beta(1,1) diff)")
ax.hist(model.delta_samples, bins=80, density=True, alpha=0.25, color="#9C27B0",
        edgecolor="white", label="MC samples")
ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Δ = θ_A − θ_B")
ax.set_ylabel("Density")
ax.set_title("Exact Convolution vs MC Histogram")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

![Exact convolution vs MC histogram](../images/non-paired/exact_convolution_vs_mc.png)

### 6. Savage-Dickey plot

```python
model.plot_savage_dickey(title="Savage-Dickey (exact convolution)")
```

![Savage-Dickey Bayes factor plot](../images/non-paired/savage_dickey_bayes_factor.png)

### 7. Posterior predictive checks

```python
ppc = model.ppc_pvalues(seed=42)

print(f"{'Statistic':<25} {'Observed':>10} {'p-value':>10} {'Status':>8}")
print("-" * 55)
for stat, vals in ppc.items():
    print(f"{stat:<25} {vals.observed:>10.4f} {vals.p_value:>10.3f} {vals.status:>8}")
```

A p-value < 0.05 flags that the observed statistic is extreme under the
fitted model (potential misfit); p-value > 0.05 means the model reproduces
that aspect of the data adequately.

## Multi-metric comparison

When you have multiple metrics (e.g. Faithfulness, Answer Relevancy), fit
a separate model for each and compare them side-by-side:

```python
results = {}
for metric_name, y_a, y_b in metric_data:
    m = NonPairedBayesPropTest(seed=42, n_samples=50_000).fit(y_a, y_b)
    results[metric_name] = m
    d_m = m.decide()
    print(f"{metric_name:<22} Δ={m.summary.mean_delta:+.4f}  "
          f"P(A>B)={m.summary.p_A_greater_B:.4f}  "
          f"BF₁₀={d_m.bayes_factor.BF_10:.2f}  {d_m.bayes_factor.decision}")
```

### Forest plot

```python
NonPairedBayesPropTest.plot_forest(
    results,
    label_A="Model v2",
    label_B="Model v1",
    title="Model v2 vs v1 — Non-Paired Beta-Bernoulli",
)
```

![Forest plot of all metrics](../images/non-paired/forest_plot_all_metrics.png)

### Comparison table

```python
NonPairedBayesPropTest.print_comparison_table(results)
```

## Prior sensitivity analysis

Test how results change with different priors to check robustness:

```python
priors = [
    ("Uniform Beta(1,1)",      1.0, 1.0),
    ("Jeffreys Beta(0.5,0.5)", 0.5, 0.5),
    ("Informative Beta(2,2)",  2.0, 2.0),
    ("Strong Beta(5,5)",       5.0, 5.0),
]

print(f"{'Prior':<28} {'BF₁₀':>8} {'BF Decision':<20} {'ROPE Decision':<20}")
print("=" * 80)

for name, a0, b0 in priors:
    m = NonPairedBayesPropTest(alpha0=a0, beta0=b0, seed=42, n_samples=50_000).fit(y_A, y_B)
    d_i = m.decide()
    print(f"{name:<28} {d_i.bayes_factor.BF_10:>8.2f} "
          f"{d_i.bayes_factor.decision:<20} {d_i.rope.decision:<20}")
```

If the conclusion is stable across priors, you can be confident the result is
not an artifact of the prior choice.

## Posterior concentration with increasing $n$

As the sample size grows, the posterior of $\Delta$ concentrates around the
true effect size. This plot shows how precision improves:

```python
import numpy as np
import matplotlib.pyplot as plt
from bayesprop.resources.bayes_nonpaired import beta_diff_pdf

z_grid = np.linspace(-0.6, 0.8, 400)
fig, ax = plt.subplots(figsize=(9, 5))

for n, col in zip([10, 30, 100, 500], ["#E91E63", "#FF9800", "#4CAF50", "#2196F3"]):
    a_A = 1 + int(0.78 * n)
    b_A = 1 + n - int(0.78 * n)
    a_B = 1 + int(0.55 * n)
    b_B = 1 + n - int(0.55 * n)
    density = np.array([beta_diff_pdf(z, a_A, b_A, a_B, b_B) for z in z_grid])
    ax.plot(z_grid, density, color=col, linewidth=2, label=f"n = {n}")

ax.axvline(0.23, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="True Δ = 0.23")
ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Δ = θ_A − θ_B")
ax.set_ylabel("Density")
ax.set_title("Posterior Concentration with Increasing n")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

![Posterior concentration with increasing n](../images/non-paired/posterior_concentration_sample_size.png)

## Null scenario (equal rates)

Under the null ($\theta_A = \theta_B$), the model should correctly find
$BF_{01} > 1$ (evidence *for* $H_0$):

```python
rng_null = np.random.default_rng(99)
y_A_null = rng_null.binomial(1, 0.65, size=150).astype(float)
y_B_null = rng_null.binomial(1, 0.65, size=150).astype(float)

model_null = NonPairedBayesPropTest(seed=99, n_samples=50_000).fit(y_A_null, y_B_null)
model_null.print_summary()
model_null.plot_savage_dickey(title="Savage-Dickey — Null Scenario (equal rates)")
```

![Savage-Dickey null scenario](../images/non-paired/savage_dickey_null_scenario.png)

## BFDA sample-size planning

Use Bayes Factor Design Analysis to determine how many observations you need
for a given effect size. See the [BFDA guide](bfda.md) for details.

```python
from bayesprop.utils.utils import bfda_power_curve, plot_bfda_power

theta_A_hat = y_A.mean()
theta_B_hat = y_B.mean()

sample_sizes = [20, 30, 50, 75, 100, 150, 200, 300, 500]

power_curve = bfda_power_curve(
    theta_A_true=theta_A_hat,
    theta_B_true=theta_B_hat,
    sample_sizes=sample_sizes,
    design="nonpaired",
    decision_rule="bayes_factor",
    bf_threshold=3.0,
    n_sim=1000,
    seed=42,
)

plot_bfda_power(power_curve, theta_A_hat, theta_B_hat)
```

![BFDA power curve](../images/non-paired/bfda_power_curve.png)

![BFDA sensitivity to BF thresholds](../images/non-paired/bfda_sensitivity_thresholds.png)

## API

See [API Reference — Non-Paired Model](../api/bayes_nonpaired.md) for full method documentation.
