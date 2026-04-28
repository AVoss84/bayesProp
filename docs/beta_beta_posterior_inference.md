# Beta-Beta Posterior Inference for LLM Judge Score Uncertainty

## 1. Problem Setting

An LLM judge assigns a continuous score $s_i \in (0,1)$ to each evaluation item $i$. We model $s_i$ as a noisy observation of a latent true quality $\mu_i \in (0,1)$ and seek the posterior $p(\mu_i \mid s_i)$.

---

## 2. Notation

| Symbol | Domain | Description |
|--------|--------|-------------|
| $\mu_i$ | $(0,1)$ | Latent true quality of item $i$ |
| $s_i$ | $(0,1)$ | Observed judge score for item $i$ |
| $\kappa > 0$ | $\mathbb{R}^+$ | Judge concentration (precision) parameter |
| $a, b > 0$ | $\mathbb{R}^+$ | Prior hyperparameters (default: $a = b = 1$) |

---

## 3. Generative Model

### 3.1 Prior

$$
\mu_i \sim \mathrm{Beta}(a, b)
$$

$$
p(\mu_i) = \frac{\mu_i^{a-1}(1 - \mu_i)^{b-1}}{B(a, b)}
$$

With $a = b = 1$ this reduces to $\mathrm{Uniform}(0,1)$.

### 3.2 Likelihood

The observed score $s_i$ is drawn from a Beta distribution whose mean is $\mu_i$ and whose precision is $\kappa$:

$$
s_i \mid \mu_i \sim \mathrm{Beta}\!\left(\mu_i \kappa,\; (1 - \mu_i)\kappa\right)
$$

Defining $\alpha_\ell = \mu_i \kappa$ and $\beta_\ell = (1 - \mu_i)\kappa$, the density is:

$$
p(s_i \mid \mu_i) = \frac{\Gamma(\kappa)}{\Gamma(\mu_i \kappa)\,\Gamma((1 - \mu_i)\kappa)}\; s_i^{\,\mu_i \kappa - 1}\,(1 - s_i)^{(1-\mu_i)\kappa - 1}
$$

**Key property**: The mean of the likelihood is $\mathbb{E}[s_i \mid \mu_i] = \mu_i$ and the variance is:

$$
\mathrm{Var}(s_i \mid \mu_i) = \frac{\mu_i(1 - \mu_i)}{\kappa + 1}
$$

Large $\kappa$ $\Rightarrow$ small variance $\Rightarrow$ precise judge. Small $\kappa$ $\Rightarrow$ noisy judge.

---

## 4. Exact Posterior (Grid-Based)

### 4.1 Bayes' Rule

$$
p(\mu_i \mid s_i) = \frac{p(s_i \mid \mu_i)\, p(\mu_i)}{\int_0^1 p(s_i \mid \mu)\, p(\mu)\, d\mu}
$$

### 4.2 Unnormalized Log-Posterior

Taking the log of the numerator:

$$
\log p(\mu_i \mid s_i) \propto \underbrace{\log \Gamma(\kappa) - \log \Gamma(\mu_i \kappa) - \log \Gamma\!\big((1-\mu_i)\kappa\big)}_{\text{log-normalizing constant of likelihood}} \\[6pt]
+ \underbrace{(\mu_i \kappa - 1)\log s_i + \big((1-\mu_i)\kappa - 1\big)\log(1-s_i)}_{\text{log-kernel of likelihood}} \\[6pt]
+ \underbrace{(a-1)\log \mu_i + (b-1)\log(1-\mu_i)}_{\text{log-prior}}
$$

### 4.3 Why This Is Not Conjugate

In a standard Beta-Binomial model, the parameter $\mu$ appears only in the kernel $\mu^{\alpha-1}(1-\mu)^{\beta-1}$, yielding a Beta posterior. Here, $\mu_i$ appears **inside the Gamma functions** of the likelihood normalizing constant:

$$
\log \Gamma(\mu_i \kappa) + \log \Gamma\!\big((1 - \mu_i)\kappa\big)
$$

These terms are nonlinear in $\mu_i$ and cannot be absorbed into a Beta kernel. Therefore:

> **The posterior $p(\mu_i \mid s_i)$ is not a member of any standard parametric family.**

### 4.4 Numerical Evaluation

We evaluate the unnormalized log-posterior on a fine grid $\{\mu^{(j)}\}_{j=1}^G$ over $(0,1)$:

1. Compute $\ell^{(j)} = \log p(\mu^{(j)} \mid s_i)$ (unnormalized) for each grid point
2. Subtract the maximum for numerical stability: $\tilde{\ell}^{(j)} = \ell^{(j)} - \max_j \ell^{(j)}$
3. Exponentiate: $\tilde{p}^{(j)} = \exp(\tilde{\ell}^{(j)})$
4. Normalize via trapezoidal integration:

$$
p(\mu^{(j)} \mid s_i) = \frac{\tilde{p}^{(j)}}{\sum_{j=1}^{G-1} \frac{1}{2}\!\left(\tilde{p}^{(j)} + \tilde{p}^{(j+1)}\right)\!\left(\mu^{(j+1)} - \mu^{(j)}\right)}
$$

This gives an arbitrarily accurate representation of the exact posterior with $G = 1000$ grid points.

---

## 5. Beta Approximation (Moment-Matched Surrogate)

### 5.1 Derivation via Pseudo-Count Analogy

In the Beta-Binomial conjugate model with $n$ Bernoulli trials and $k$ successes:

$$
\mu \sim \mathrm{Beta}(a, b) \quad \Rightarrow \quad \mu \mid k, n \sim \mathrm{Beta}(a + k,\; b + n - k)
$$

We draw an analogy: the single continuous observation $s_i$ from a $\mathrm{Beta}(\mu_i\kappa, (1-\mu_i)\kappa)$ distribution carries approximately $\kappa$ "pseudo-observations" with empirical mean $s_i$. This motivates:

- Pseudo-successes: $k_{\text{pseudo}} = s_i \cdot \kappa$
- Pseudo-failures: $(n - k)_{\text{pseudo}} = (1 - s_i) \cdot \kappa$

### 5.2 Approximate Posterior

$$
\boxed{\mu_i \mid s_i \;\approx\; \mathrm{Beta}\!\left(s_i \kappa + a,\; (1 - s_i)\kappa + b\right)}
$$

with density:

$$
\tilde{p}(\mu_i \mid s_i) = \frac{\mu_i^{\,s_i\kappa + a - 1}\,(1-\mu_i)^{(1-s_i)\kappa + b - 1}}{B\!\left(s_i\kappa + a,\; (1-s_i)\kappa + b\right)}
$$

### 5.3 Moments of the Approximate Posterior

Defining $\tilde{\alpha} = s_i\kappa + a$ and $\tilde{\beta} = (1-s_i)\kappa + b$:

**Posterior mean:**

$$
\mathbb{E}[\mu_i \mid s_i] = \frac{\tilde{\alpha}}{\tilde{\alpha} + \tilde{\beta}} = \frac{s_i\kappa + a}{\kappa + a + b}
$$

With $a = b = 1$:

$$
\mathbb{E}[\mu_i \mid s_i] = \frac{s_i\kappa + 1}{\kappa + 2}
$$

This is a **shrinkage estimator** — the posterior mean is pulled from $s_i$ toward the prior mean $\frac{a}{a+b} = 0.5$, with strength controlled by $\kappa$:

- $\kappa \to \infty$: $\mathbb{E}[\mu_i \mid s_i] \to s_i$ (data dominates)
- $\kappa \to 0$: $\mathbb{E}[\mu_i \mid s_i] \to 0.5$ (prior dominates)

**Posterior variance:**

$$
\mathrm{Var}(\mu_i \mid s_i) = \frac{\tilde{\alpha}\,\tilde{\beta}}{(\tilde{\alpha} + \tilde{\beta})^2(\tilde{\alpha} + \tilde{\beta} + 1)}
$$

### 5.4 Credible Intervals

The $100(1-\alpha)\%$ credible interval is computed via the inverse CDF (quantile function) of the posterior Beta:

$$
\mathrm{CI}_{1-\alpha} = \left[F^{-1}\!\left(\tfrac{\alpha}{2};\; \tilde{\alpha},\, \tilde{\beta}\right),\;\; F^{-1}\!\left(1 - \tfrac{\alpha}{2};\; \tilde{\alpha},\, \tilde{\beta}\right)\right]
$$

where $F^{-1}$ is the Beta quantile function (regularized incomplete beta inverse).

---

## 6. Approximation Quality

### 6.1 When the Approximation Is Good

The Beta surrogate closely matches the exact posterior when:

- $\kappa$ is moderate ($5 \lesssim \kappa \lesssim 50$)
- $s_i$ is not too close to 0 or 1
- The prior is weak ($a, b \approx 1$)

### 6.2 When It Breaks Down

The approximation degrades when:

- $\kappa$ is very large ($\kappa > 100$): the exact posterior becomes highly peaked and the Gamma-function nonlinearity matters more
- $s_i$ is near boundaries (0 or 1): the exact posterior becomes skewed in ways the Beta cannot capture
- Strong informative priors are used

### 6.3 Error Metric

The approximation quality can be assessed via the total variation distance:

$$
\mathrm{TV} = \frac{1}{2}\int_0^1 \left|p_{\text{exact}}(\mu \mid s) - \tilde{p}(\mu \mid s)\right| d\mu
$$

or the KL divergence:

$$
D_{\mathrm{KL}}\!\left(p_{\text{exact}} \,\|\, \tilde{p}\right) = \int_0^1 p_{\text{exact}}(\mu \mid s)\,\log\frac{p_{\text{exact}}(\mu \mid s)}{\tilde{p}(\mu \mid s)}\, d\mu
$$

---

## 7. Empirical Bayes Estimation of $\kappa$

When $\kappa$ is unknown, it can be estimated from the observed scores $\{s_1, \dots, s_N\}$ using method of moments.

### 7.1 Marginal Moments

Under the generative model $\mu_i \sim \mathrm{Beta}(a, b)$ and $s_i \mid \mu_i \sim \mathrm{Beta}(\mu_i\kappa, (1-\mu_i)\kappa)$, the marginal mean and variance of $s_i$ depend on $\kappa$. As a simple approximation (treating $\mu_i$ as roughly fixed at the population mean):

$$
\mathbb{E}[s_i] \approx \bar{\mu}, \qquad \mathrm{Var}(s_i) \approx \frac{\bar{\mu}(1-\bar{\mu})}{\kappa + 1}
$$

### 7.2 Method-of-Moments Estimator

Solving for $\kappa$:

$$
\hat{\kappa} = \frac{\bar{s}(1-\bar{s})}{\hat{\sigma}^2_s} - 1
$$

where $\bar{s} = \frac{1}{N}\sum_i s_i$ and $\hat{\sigma}^2_s = \frac{1}{N}\sum_i(s_i - \bar{s})^2$.

**Interpretation:**

- High variance in scores $\Rightarrow$ small $\hat{\kappa}$ $\Rightarrow$ noisy judge $\Rightarrow$ wide posteriors
- Low variance in scores $\Rightarrow$ large $\hat{\kappa}$ $\Rightarrow$ precise judge $\Rightarrow$ narrow posteriors

---

## 8. Hierarchical Bootstrap for Aggregate Uncertainty

### 8.1 Goal

Estimate a credible interval for the population-level expected score:

$$
\theta = \mathbb{E}[\mu_i] = \frac{1}{N}\sum_{i=1}^N \mu_i
$$

### 8.2 Two-Layer Bootstrap

For $b = 1, \dots, B$ bootstrap iterations:

**Outer layer** (finite-sample uncertainty):

$$
I^{(b)} = \left(i_1^{(b)}, \dots, i_N^{(b)}\right) \sim \mathrm{Multinomial}(N;\; \tfrac{1}{N}, \dots, \tfrac{1}{N})
$$

Resample $N$ item indices with replacement. This captures uncertainty about which items are in the evaluation set.

**Inner layer** (judge noise):

For each resampled item $i_k^{(b)}$, draw a latent quality from the approximate posterior:

$$
\mu_{i_k}^{(b)} \sim \mathrm{Beta}\!\left(s_{i_k^{(b)}}\hat{\kappa} + a,\; (1 - s_{i_k^{(b)}})\hat{\kappa} + b\right)
$$

**Aggregate:**

$$
\hat{\theta}^{(b)} = \frac{1}{N}\sum_{k=1}^N \mu_{i_k}^{(b)}
$$

### 8.3 Credible Interval

The empirical distribution $\{\hat{\theta}^{(1)}, \dots, \hat{\theta}^{(B)}\}$ yields:

$$
\mathrm{CI}_{95\text{%}} = \left[\hat{\theta}^{(0.025)},\;\; \hat{\theta}^{(0.975)}\right]
$$

where $\hat{\theta}^{(q)}$ denotes the $q$-th quantile of the bootstrap distribution.

### 8.4 Variance Decomposition

The total variance of $\hat{\theta}$ decomposes as:

$$
\mathrm{Var}(\hat{\theta}) = \underbrace{\mathrm{Var}_{\text{outer}}(\hat{\theta})}_{\text{finite-sample}} + \underbrace{\mathbb{E}_{\text{outer}}\!\left[\mathrm{Var}_{\text{inner}}(\hat{\theta})\right]}_{\text{judge noise}}
$$

---

## 9. Scope and Limitations

### 9.1 What the CI Captures

$$
\mathrm{CI} \text{ covers } \mathbb{E}\!\left[\mu_i \mid \text{these specific Q/A pairs}\right]
$$

It accounts for:
- **Finite-sample noise**: which of the $N$ items enter the average
- **Judge measurement noise**: uncertainty in $\mu_i$ given a single score $s_i$

### 9.2 What the CI Does NOT Capture

$$
\text{Synthesizer generative uncertainty:} \quad (Q_i, Y_i) \sim p_{\text{synth}}(\cdot \mid X_i)
$$

Re-running the LLM synthesizer on the same contexts $X_i$ would produce different Q/A pairs, potentially yielding very different evaluation scores. This source of variance is **not** captured because the bootstrap only resamples from the fixed observed set.

To capture synthesizer uncertainty, one would need $K > 1$ independent synthesizer draws per context, forming $K$ parallel evaluation sets.

### 9.3 Assumptions

1. **Exchangeability**: Items are exchangeable draws from a common population
2. **Shared $\kappa$**: All items have the same judge precision (can be relaxed to per-metric $\kappa$)
3. **Single observation**: One judge score per item (no repeated judging)
4. **Beta approximation**: The moment-matched Beta surrogate is used in place of the exact posterior
