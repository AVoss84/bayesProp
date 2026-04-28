# Hierarchical Beta-Regression with Item-Specific Precision for Paired RAG Evaluation

## 1. Problem Setting

We compare two RAG generators — **Gemini 2.5 Flash** (Model A) and **Gemini 2.0 Flash** (Model B) — on the same set of $N$ evaluation items using a shared TFIDF retriever ($k=5$). Each item $i$ produces a pair of DeepEval metric scores $s_{A,i}, s_{B,i} \in (0,1)$ for two generator-dependent metrics: **Answer Relevancy** and **Faithfulness**.

The paired design (same retriever, same items, same LLM judge) isolates the generator effect. The statistical question is:

> **Does Model A systematically differ from Model B?**
>
> Formally: $H_0: \delta_A = 0$ vs $H_1: \delta_A \neq 0$

---

## 2. Why Not the Logistic-Normal Model?

Our first attempt used a **hierarchical logistic-normal model** (documented in [`hierarchical_logistic_normal_model.md`](hierarchical_logistic_normal_model.md)):

$$
z_{m,i} = \operatorname{logit}(s_{m,i}), \qquad z_{m,i} \sim \mathcal{N}(\theta_i + \delta_m, \sigma^2)
$$

This model assumes scores are Gaussian on the logit scale. However, **posterior predictive checks (PPC)** revealed a critical misfit:

- **Observed data**: strongly J-shaped — most scores cluster near 1.0, with a few near 0.
- **Replicated data (logistic-normal)**: symmetric, bell-shaped on the logit scale, producing a U-shape on $(0,1)$.

The logistic-normal model cannot reproduce the asymmetric, boundary-concentrated structure of DeepEval scores. The logit transform maps near-1 scores to large positive values and near-0 scores to large negative values, but the Gaussian noise assumption generates symmetric tails, creating unrealistic mass near 0 that doesn't exist in the data.

**Decision**: Replace the Gaussian likelihood on the logit scale with a **Beta likelihood** on the raw $(0,1)$ scale.

---

## 3. Why Not a Global Precision Parameter?

The next attempt used a Beta likelihood with a **single global precision** $\phi$:

$$
s_{m,i} \sim \operatorname{Beta}(\mu_{m,i} \cdot \phi, \; (1-\mu_{m,i}) \cdot \phi)
$$

PPC again showed misfit: the model predicted moderate spread for all items, but the data contained a **mixture** of:

- Items where both models score $\approx 1.0$ with very tight precision (easy questions)
- Items where scores are scattered across $(0,1)$ (hard or ambiguous questions)

A single $\phi$ cannot simultaneously accommodate both regimes. The easy items demand $\phi \gg 1$ while the hard items demand $\phi \approx 1$.

**Decision**: Introduce **item-specific precision** $\phi_i$.

---

## 4. Final Model: Hierarchical Beta-Regression with LogNormal Precision

### 4.1 Notation

| Symbol | Domain | Description |
|--------|--------|-------------|
| $N$ | $\mathbb{N}$ | Number of evaluation items |
| $s_{A,i}, s_{B,i}$ | $(0,1)$ | Observed scores for Model A and B on item $i$ |
| $\theta_i$ | $\mathbb{R}$ | Latent item difficulty (logit scale, shared across models) |
| $\delta_A$ | $\mathbb{R}$ | Model A advantage on logit scale (key parameter) |
| $\sigma_\theta$ | $\mathbb{R}^+$ | SD of item difficulties |
| $\phi_i$ | $\mathbb{R}^+$ | Item-specific precision |
| $\mu_\phi$ | $\mathbb{R}$ | Population mean of $\log \phi_i$ |
| $\sigma_\phi$ | $\mathbb{R}^+$ | Heterogeneity in log-precision across items |

### 4.2 Full Generative Model

**Hyperpriors:**

$$
\sigma_\theta \sim \operatorname{HalfNormal}(1)
$$

$$
\mu_\phi \sim \mathcal{N}(0, 2^2)
$$

$$
\sigma_\phi \sim \operatorname{HalfNormal}(1)
$$

**Item-level parameters:**

$$
\theta_i \sim \mathcal{N}(0, \sigma_\theta^2) \qquad \text{(item difficulty)}
$$

$$
\phi_i \sim \operatorname{LogNormal}(\mu_\phi, \sigma_\phi^2) \qquad \text{(item-specific precision)}
$$

**Model effect (fixed prior width):**

$$
\delta_A \sim \mathcal{N}(0, \sigma_s^2) \qquad \text{(slab width } \sigma_s = 1.0 \text{, fixed)}
$$

**Mean scores via logit link:**

$$
\mu_{A,i} = \operatorname{sigmoid}(\theta_i + \delta_A)
$$

$$
\mu_{B,i} = \operatorname{sigmoid}(\theta_i)
$$

**Beta likelihood:**

$$
s_{A,i} \sim \operatorname{Beta}(\mu_{A,i} \cdot \phi_i, \; (1 - \mu_{A,i}) \cdot \phi_i)
$$

$$
s_{B,i} \sim \operatorname{Beta}(\mu_{B,i} \cdot \phi_i, \; (1 - \mu_{B,i}) \cdot \phi_i)
$$

### 4.3 Key Design Choices

1. **LogNormal precision**: $\phi_i \sim \operatorname{LogNormal}(\mu_\phi, \sigma_\phi^2)$ ensures $\phi_i > 0$ and allows heavy-tailed variation. Easy items (near 1.0) learn large $\phi_i$ (tight concentration), while hard items learn small $\phi_i$ (diffuse spread).

2. **Non-centered parameterization**: Both $\theta_i$ and $\phi_i$ use non-centered parameterization to avoid the funnel geometry that degrades NUTS sampling when population SDs are small:

$$
\theta_i^{\text{raw}} \sim \mathcal{N}(0,1), \quad \theta_i = \sigma_\theta \cdot \theta_i^{\text{raw}}
$$

$$
\phi_i^{\text{raw}} \sim \mathcal{N}(0,1), \quad \log \phi_i = \mu_\phi + \sigma_\phi \cdot \phi_i^{\text{raw}}
$$

3. **Fixed prior width on $\delta_A$**: The prior $\delta_A \sim \mathcal{N}(0, \sigma_s^2)$ uses a fixed $\sigma_s$ (not learned). This is essential for the Savage-Dickey Bayes Factor — see §6.

4. **Shared $\phi_i$ across models**: Both $s_{A,i}$ and $s_{B,i}$ share the same precision $\phi_i$ for item $i$. This assumes item difficulty drives precision variability more than model identity, which is reasonable when both generators answer the same question with the same retrieved context.

### 4.4 Graphical Model

```
σ_θ        σ_s       μ_φ   σ_φ
 │          │         │      │
 ▼          ▼         ▼      ▼
θ_i ──→ μ_{A,i}    φ_i ─────┤
 │       ▲   │       │      │
 │       │   ▼       ▼      │
 │    δ_A   s_{A,i} ←──── φ_i
 │                           │
 └────→ μ_{B,i}             │
          │                  │
          ▼                  │
        s_{B,i} ←────────── φ_i

        (plate over i = 1, ..., N)
```

---

## 5. Posterior Inference

### 5.1 MCMC Configuration

| Parameter | Value |
|-----------|-------|
| Sampler | NUTS (No-U-Turn Sampler) via PyMC |
| Chains | 4 |
| Tune (warmup) | 2,000 |
| Draws per chain | 2,000 |
| Total posterior samples | $S = 8{,}000$ |
| Target accept rate | 0.99 |

The high target accept rate (0.99) is chosen because the Beta likelihood with near-boundary data creates sharp curvature in the posterior geometry.

### 5.2 Primary Posterior Quantities

**On the logit scale** — the marginal posterior $p(\delta_A \mid \mathbf{s}_A, \mathbf{s}_B)$:

- Posterior mean $\hat{\delta}_A$
- 95% credible interval $[\delta_A^{(0.025)}, \delta_A^{(0.975)}]$

**On the probability scale** — for each posterior draw $s$:

$$
\mu_{A,i}^{(s)} = \operatorname{sigmoid}\!\left(\theta_i^{(s)} + \delta_A^{(s)}\right), \qquad
\mu_{B,i}^{(s)} = \operatorname{sigmoid}\!\left(\theta_i^{(s)}\right)
$$

$$
\Delta^{(s)} = \frac{1}{N} \sum_{i=1}^{N} \left(\mu_{A,i}^{(s)} - \mu_{B,i}^{(s)}\right)
$$

This gives the **population-averaged probability difference** — the expected score improvement from using Model A.

### 5.3 Decision Quantities

| Quantity | Formula | Interpretation |
|----------|---------|----------------|
| Mean $\Delta$ | $\bar{\Delta} = \frac{1}{S}\sum_s \Delta^{(s)}$ | Expected score advantage of A over B |
| 95% CI | $[\Delta^{(0.025)}, \Delta^{(0.975)}]$ | Posterior uncertainty interval |
| $P(A > B)$ | $\frac{1}{S}\sum_s \mathbb{1}[\Delta^{(s)} > 0]$ | Posterior probability that A is better |

---

## 6. Simulation Validation

Before applying the model to real data, we validate parameter recovery using simulated data from a known data-generating process (DGP):

**DGP**: $N=200$ items, $\delta_A^{\text{true}} = 0.5$, $\sigma_\theta = 1.0$, $\sigma_{\text{obs}} = 0.7$.

**Recovery results**:

- $\hat{\delta}_A = 0.473$ (true: 0.5) — good recovery
- All $\hat{r} = 1.0$ — chains converged
- 95% CI covers the true value

This confirms the model and inference pipeline are correctly implemented before applying to real data.

---

## 7. Posterior Predictive Checks (PPC)

PPC is the primary model diagnostic. For each posterior draw, we generate replicated datasets:

$$
s_{m,i}^{\text{rep}} \sim \operatorname{Beta}\!\left(\mu_{m,i}^{(s)} \cdot \phi_i^{(s)},\; (1 - \mu_{m,i}^{(s)}) \cdot \phi_i^{(s)}\right)
$$

We compare the replicated data against the observed data on three dimensions:

1. **Marginal distribution of $s_A$** — does the model reproduce the J-shape?
2. **Marginal distribution of $s_B$** — same check for the baseline model
3. **Paired difference $s_A - s_B$** — does the model capture the dependence structure?

**Visualization**: KDE-based PPC plots with individual replicated KDEs (thin, transparent) overlaid on the observed KDE (bold red) and a pooled replicated KDE (dashed).

### PPC p-values

For each summary statistic $T$ (mean, std, mean difference, std of difference), we compute the two-sided posterior predictive p-value:

$$
p = 2 \cdot \min\!\left(P\!\left(T^{\text{rep}} \geq T^{\text{obs}}\right),\; P\!\left(T^{\text{rep}} \leq T^{\text{obs}}\right)\right)
$$

$p > 0.05$: the model adequately reproduces this aspect of the data.  
$p < 0.05$: potential misfit for this statistic.

### Model evolution motivated by PPC

| Model | PPC Result | Issue |
|-------|-----------|-------|
| Logistic-Normal | **FAIL** | U-shaped replicated data vs J-shaped observed |
| Beta (global $\phi$) | **FAIL** | Cannot capture mix of tight and diffuse items |
| Beta (item-specific $\phi_i$) | **PASS** | Replicated data matches J-shape and spread |

---

## 8. Bayesian Hypothesis Test: Savage-Dickey Bayes Factor

### 8.1 The Measure-Theoretic Subtlety

Under a continuous prior on $\delta_A$, the singleton $\{\delta_A = 0\}$ has **Lebesgue measure zero** — it can never receive positive posterior probability regardless of the data. To make $H_0: \delta_A = 0$ a proper hypothesis, we use a **spike-and-slab** mixed prior:

$$
\pi(\delta_A) = \pi_0 \cdot \delta_{\{0\}}(\delta_A) + (1 - \pi_0) \cdot g(\delta_A)
$$

- **Spike**: discrete mass $\pi_0 = P(H_0)$ at $\delta_A = 0$
- **Slab**: continuous density $g(\delta_A) = \mathcal{N}(0, \sigma_s^2)$

### 8.2 The Savage-Dickey Density Ratio

Because $H_0$ is nested inside $H_1$ (point restriction of a parameter in the full model), the Bayes Factor is:

$$
BF_{01} = \frac{p(\delta_A = 0 \mid D, H_1)}{g(0)}
$$

where:

- **Numerator**: posterior density at 0 under the unrestricted model, estimated via KDE on the 8,000 MCMC draws
- **Denominator**: slab prior density at 0, available analytically: $g(0) = \frac{1}{\sigma_s \sqrt{2\pi}}$

The reciprocal $BF_{10} = 1/BF_{01}$ measures evidence **against** $H_0$.

### 8.3 Why Savage-Dickey?

1. **MCMC reuse**: We already have posterior draws from the unrestricted model. No additional model fitting needed (unlike bridge sampling, RJMCMC, or the product-space method).

2. **Analytical denominator**: The slab density $g(0)$ is in closed form — the only approximation is the KDE in the numerator, which is reliable with $S = 8{,}000$ draws.

3. **Prior consistency requirement**: The fitted model uses a **fixed** prior width $\delta_A \sim \mathcal{N}(0, \sigma_s^2)$ — no hyperprior on $\sigma_s$. This ensures numerator and denominator both refer to the same model $H_1$, making the density ratio exactly consistent.

   *Why not learn $\sigma_s$?* A hierarchical prior $\sigma_s \sim \operatorname{HalfNormal}(1)$ makes the marginal slab density at 0 diverge:

   $$
   g(0) = \int_0^\infty \frac{1}{\sigma_s \sqrt{2\pi}} \cdot \operatorname{HalfNormal}(\sigma_s) \, d\sigma_s \to \infty
   $$

   due to a logarithmic singularity at $\sigma_s \to 0$.

### 8.4 Interpretation Scale (Jeffreys)

| $BF_{10}$ | Evidence against $H_0$ |
|-----------|----------------------|
| $< 1$ | Supports $H_0$ |
| $1 – 3$ | Anecdotal |
| $3 – 10$ | Moderate |
| $10 – 30$ | Strong |
| $30 – 100$ | Very strong |
| $> 100$ | Decisive |

### 8.5 Display Convention for Extreme $BF_{10}$

With strong effects, $BF_{10}$ can reach $10^{100+}$, making raw values unreadable. We use:

- For $BF_{10} > 10^4$: display as $10^k$ where $k = \lfloor\log_{10} BF_{10}\rfloor$
- Summary tables show $\log_{10} BF_{10}$ directly
- Posterior density at 0 displayed in scientific notation

---

## 9. From Bayes Factor to Posterior Model Probability

Under the spike-and-slab prior, Bayes' rule at the model level gives:

$$
\frac{P(H_0 \mid D)}{P(H_1 \mid D)} = BF_{01} \cdot \frac{\pi_0}{1 - \pi_0}
$$

Solving for $P(H_0 \mid D)$:

$$
P(H_0 \mid D) = \frac{\pi_0 \cdot BF_{01}}{\pi_0 \cdot BF_{01} + (1 - \pi_0)}
$$

With the default agnostic prior $\pi_0 = 0.5$:

$$
P(H_0 \mid D) = \frac{BF_{01}}{1 + BF_{01}}
$$

This is now a well-defined probability statement — the spike mass $\pi_0$ ensures $\{\delta_A = 0\}$ has positive prior (and hence posterior) measure.

---

## 10. Sensitivity Analysis

### 10.1 Sensitivity to Prior $P(H_0)$

We sweep $\pi_0$ from 0.01 to 0.99 and plot $P(H_0 \mid D)$ as a function of $\pi_0$. When evidence is overwhelming ($\log_{10} BF_{10} > 100$), $P(H_0 \mid D) \approx 0$ for all reasonable priors — the conclusion is robust to prior beliefs about $H_0$.

### 10.2 Sensitivity to Slab Width $\sigma_s$ (Jeffreys-Lindley)

The slab width $\sigma_s$ controls how spread out the alternative hypothesis is. The **Jeffreys-Lindley paradox** states that for fixed data, $BF_{01} \to \infty$ as $\sigma_s \to \infty$ — a very diffuse alternative always loses to the point null.

We sweep $\sigma_s \in [0.25, 5.0]$ and plot:

1. $BF_{10}$ vs $\sigma_s$ (left panel, log scale)
2. $P(H_0 \mid D)$ vs $\sigma_s$ at $\pi_0 = 0.5$ (right panel)

When both curves remain decisively against $H_0$ across the entire range, the conclusion is robust to the slab width choice.

---

## 11. Complete Analysis Pipeline

The notebook implements the following sequence:

```
1. Data preparation
   ├── Load PDF, chunk documents
   ├── Load annotated Q/A pairs
   └── Build shared TFIDF retriever (k=5)

2. Model evaluation (DeepEval)
   ├── Model A (gemini-2.5-flash): generate answers → score
   └── Model B (gemini-2.0-flash): generate answers → score

3. Simulation validation
   ├── Simulate paired scores with known δ_A = 0.5
   ├── Fit hierarchical Beta model
   └── Verify parameter recovery (δ̂_A ≈ 0.5, r̂ = 1.0)

4. Real data comparison
   ├── Extract paired scores per metric
   ├── Clip to (ε, 1-ε) for Beta likelihood
   ├── Fit hierarchical Beta model per metric
   └── Report: Mean Δ, 95% CI, P(A > B)

5. Diagnostics
   ├── MCMC trace plots (mixing, stationarity)
   ├── ArviZ summary (r̂, ESS)
   └── Posterior predictive checks (KDE-based)
       ├── s_A marginal
       ├── s_B marginal
       └── s_A − s_B paired difference
   └── PPC p-values (mean, std, paired diff)

6. Hypothesis testing
   ├── Savage-Dickey Bayes Factor (BF₁₀)
   ├── Savage-Dickey plot (posterior vs prior at δ=0)
   ├── P(H₀|D) computation
   └── Decision: Reject H₀ if BF₁₀ > 3

7. Sensitivity analysis
   ├── P(H₀|D) vs prior P(H₀) — prior robustness
   └── BF₁₀ and P(H₀|D) vs slab width σ_s — Jeffreys-Lindley
```

---

## 12. Comparison with the Logistic-Normal Model

| Aspect | Logistic-Normal | Beta (item-specific $\phi_i$) |
|--------|----------------|-------------------------------|
| Likelihood | $\mathcal{N}$ on logit scale | $\operatorname{Beta}$ on raw $(0,1)$ |
| Precision | Global $\sigma$ | Item-specific $\phi_i \sim \operatorname{LogNormal}$ |
| Boundary scores | Problematic (logit diverges) | Natural (Beta concentrates at 0 or 1) |
| PPC fit | **Fails** — U-shaped vs J-shaped | **Passes** — matches observed J-shape |
| Parameters | $\delta_A, \sigma_\theta, \sigma$ | $\delta_A, \sigma_\theta, \mu_\phi, \sigma_\phi$ |
| Hypothesis test | Same (Savage-Dickey) | Same (Savage-Dickey) |

---

## 13. Connection to Item Response Theory (IRT)

The model is a continuous-response extension of the **Rasch / 2PL IRT model**:

| IRT Concept | Our Model |
|-------------|-----------|
| Person ability | Model effect $\delta_A$ |
| Item difficulty | $\theta_i$ |
| Response | Continuous $s_{m,i} \in (0,1)$ |
| Link function | Logistic sigmoid |
| Discrimination | Fixed at 1 (Rasch-like) |
| Response distribution | Beta (vs Bernoulli in IRT) |

The LogNormal precision $\phi_i$ acts as an item-specific **reliability** parameter — analogous to how IRT discrimination parameters control the informativeness of each item.

---

## 14. Assumptions and Limitations

1. **Beta likelihood**: Assumes scores in the strict open interval $(0,1)$. Exact 0 or 1 scores are clipped to $\varepsilon = 10^{-6}$ from the boundary.

2. **Shared $\phi_i$ across models**: Both $s_{A,i}$ and $s_{B,i}$ use the same precision for item $i$. This could be relaxed to model-specific precision if evidence suggests different models have different noise levels.

3. **Single $\delta_A$**: One scalar captures the entire model difference. Item-by-model interactions are not modeled.

4. **Exchangeable items**: Items are assumed exchangeable. Topic-clustered items would need a hierarchical extension.

5. **Fixed slab width**: $\sigma_s$ is not learned from the data but checked via sensitivity analysis.

6. **KDE approximation**: The posterior density at 0 is estimated via Gaussian KDE with 8,000 draws. For extreme BF values ($>10^{50}$), the KDE estimate at 0 is effectively 0, making $BF_{10}$ astronomically large. The precise numerical value is less informative than the order of magnitude ($\log_{10} BF_{10}$).

---

## 15. Reproducibility

| Component | Version / Setting |
|-----------|-------------------|
| PyMC | $\geq 5.0$ |
| ArviZ | $\geq 0.15$ |
| Sampler | NUTS, 4 chains × 2000 draws, `target_accept=0.99` |
| Random seed | 42 (both simulation and real data) |
| DeepEval metrics | Answer Relevancy, Faithfulness |
| Retriever | TFIDF, $k=5$, shared across models |

---

## 16. Discussion: Why Not McNemar or Other Frequentist Alternatives?

### McNemar's Test

McNemar's test is a standard paired non-parametric test for $2 \times 2$ tables and might seem natural for comparing two models on the same items. However, it requires **binarizing** the scores (e.g., $s > 0.5 \to$ pass), which discards the continuous information in $(0,1)$. A score of 0.99 and 0.51 both become "pass", losing the magnitude signal that the Beta-regression model captures.

| Aspect | McNemar | Hierarchical Beta-regression |
|--------|---------|-------------------------------|
| Data used | Binary (thresholded) | Full continuous $(0,1)$ |
| Item heterogeneity | Ignored | Modeled ($\theta_i$) |
| Precision heterogeneity | Ignored | Modeled ($\phi_i \sim \operatorname{LogNormal}$) |
| Effect size | None (only a p-value) | $\Delta$, 95% CI, $P(A>B)$ |
| Evidence quantification | p-value (against $H_0$ only) | $BF_{10}$ (evidence for *or* against $H_0$) |
| Evidence for $H_0$ | Cannot provide | $BF_{01}$, $P(H_0 \mid D)$ |
| Prior sensitivity | N/A | Full Jeffreys-Lindley sweep |

McNemar provides no effect size, cannot distinguish "no evidence" from "evidence for equivalence", and requires an arbitrary binarization threshold that the analyst must justify.

### Paired Wilcoxon Signed-Rank Test

A more appropriate frequentist alternative would be the **paired Wilcoxon signed-rank test** on the raw continuous scores, since it does not require binarization. However, it still cannot model item-level heterogeneity, provide calibrated posterior probabilities, or quantify evidence in favour of the null.

### When a Frequentist Sanity Check Adds Value

If reviewers question the Bayesian machinery, a one-line Wilcoxon or even McNemar test can serve as a **robustness footnote** — confirming that the conclusion holds under a simpler framework. With $\log_{10} BF_{10} > 100$, any reasonable test will agree. The frequentist test merely confirms; the Bayesian model informs.

### Summary

The hierarchical Beta-regression subsumes what McNemar and Wilcoxon offer while additionally providing:

1. Full use of continuous score information (no binarization)
2. Item-specific precision modeling ($\phi_i$) validated by PPC
3. Calibrated posterior probabilities $P(H_0 \mid D)$ and $P(A > B)$
4. Sensitivity analysis over prior choices (slab width, prior $P(H_0)$)
5. The ability to conclude *in favour of* $H_0$ when models are equivalent (via $BF_{01}$), not just fail to reject
