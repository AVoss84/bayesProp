# Hierarchical Logistic-Normal Model for Paired RAG Evaluation

## 1. Motivation

When comparing two LLM generators (Model A vs Model B) on the same set of evaluation items, we need a statistical framework that:

- Accounts for **item-level difficulty** (some questions are harder than others for both models)
- Estimates a **shared model effect** $\delta_A$ across all items
- Provides **full posterior uncertainty** over the comparison, not just point estimates
- Handles scores in $(0,1)$ via a natural link function

The **hierarchical logistic-normal model** achieves this by working on the logit scale and using a paired design with shared item effects.

---

## 2. Setup and Notation

| Symbol | Description |
|--------|-------------|
| $N$ | Number of evaluation items (Q/A pairs) |
| $i = 1, \dots, N$ | Item index |
| $s_{A,i}, s_{B,i} \in (0,1)$ | Observed DeepEval scores for Model A and B on item $i$ |
| $z_{A,i}, z_{B,i} \in \mathbb{R}$ | Logit-transformed scores: $z = \log\!\bigl(\tfrac{s}{1-s}\bigr)$ |
| $\theta_i$ | Latent item difficulty (shared across models) |
| $\delta_A$ | Model A advantage on the logit scale (key parameter of interest) |
| $\sigma$ | Observation noise on the logit scale |
| $\sigma_\theta$ | Standard deviation of item difficulties |
| $\sigma_\delta$ | Prior scale for the model effect |

---

## 3. Data Transformation

Raw DeepEval scores $s \in [0,1]$ are first clipped to $(0,1)$ and then logit-transformed:

$$
z_{m,i} = \mathrm{logit}(s_{m,i}) = \log\!\left(\frac{s_{m,i}}{1 - s_{m,i}}\right), \quad m \in \{A, B\}
$$

This maps bounded scores to the real line, where Gaussian assumptions are more appropriate. Scores at exact 0 or 1 are clipped to $[\varepsilon, 1-\varepsilon]$ with $\varepsilon = 10^{-6}$.

---

## 4. Generative Model (Full Specification)

### 4.1 Hyperpriors

$$
\sigma_\theta \sim \text{HalfNormal}(1)
$$
$$
\sigma_\delta \sim \text{HalfNormal}(1)
$$
$$
\sigma \sim \text{HalfNormal}(1)
$$

### 4.2 Item Effects

Each item has a latent difficulty $\theta_i$ that is shared across both models:

$$
\theta_i \sim \mathcal{N}(0, \sigma_\theta^2), \quad i = 1, \dots, N
$$

### 4.3 Model Effect

Model B serves as the **baseline** ($\delta_B = 0$). Model A's advantage is:

$$
\delta_A \sim \mathcal{N}(0, \sigma_\delta^2)
$$

If $\delta_A > 0$, Model A tends to score higher; if $\delta_A < 0$, Model B is better.

### 4.4 Linear Predictors

$$
\eta_{A,i} = \theta_i + \delta_A
$$
$$
\eta_{B,i} = \theta_i
$$

### 4.5 Likelihood

$$
z_{A,i} \sim \mathcal{N}(\eta_{A,i},\; \sigma^2) = \mathcal{N}(\theta_i + \delta_A,\; \sigma^2)
$$
$$
z_{B,i} \sim \mathcal{N}(\eta_{B,i},\; \sigma^2) = \mathcal{N}(\theta_i,\; \sigma^2)
$$

---

## 5. Graphical Model

```
σ_θ       σ_δ       σ
 │         │        │
 ▼         ▼        │
θ_i ──→ η_{A,i} ──→ z_{A,i}   (observed)
 │         ▲        ▲
 │         │        │
 │      δ_A ────────┘
 │                   │
 └────→ η_{B,i} ──→ z_{B,i}   (observed)
                     ▲
                     │
                     σ

(plate over i = 1, ..., N)
```

---

## 6. Non-Centered Parameterization

For efficient MCMC sampling, the item effects use **non-centered parameterization**:

$$
\theta_i^{\text{raw}} \sim \mathcal{N}(0, 1)
$$
$$
\theta_i = \sigma_\theta \cdot \theta_i^{\text{raw}}
$$

This avoids the funnel geometry that arises when $\sigma_\theta$ is small, improving NUTS sampler performance.

---

## 7. Posterior Inference

### 7.1 MCMC Sampling

The model is fit using the **NUTS** (No-U-Turn Sampler) via PyMC:

| Parameter | Default |
|-----------|---------|
| Chains | 4 |
| Tune (warmup) | 2,000 |
| Draws (per chain) | 2,000 |
| Total posterior samples | $S = 4 \times 2{,}000 = 8{,}000$ |
| Target accept | 0.99 |

### 7.2 Posterior of $\delta_A$ (Logit Scale)

The primary output is the marginal posterior $p(\delta_A \mid \mathbf{z}_A, \mathbf{z}_B)$, summarized by:

- **Posterior mean**: $\hat{\delta}_A = \mathbb{E}[\delta_A \mid \text{data}]$
- **95% Credible Interval**: $[\delta_A^{(0.025)},\; \delta_A^{(0.975)}]$

---

## 8. Posterior Quantities on the Probability Scale

### 8.1 Item-Level Probability Difference

For each posterior draw $s = 1, \dots, S$:

$$
\mu_{A,i}^{(s)} = \sigma\!\left(\theta_i^{(s)} + \delta_A^{(s)}\right)
$$
$$
\mu_{B,i}^{(s)} = \sigma\!\left(\theta_i^{(s)}\right)
$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.

### 8.2 Average Probability Difference

The **population-averaged** effect on the probability scale:

$$
\Delta^{(s)} = \frac{1}{N} \sum_{i=1}^{N} \left(\mu_{A,i}^{(s)} - \mu_{B,i}^{(s)}\right)
$$

This gives a distribution over $\Delta$ — the expected score improvement from using Model A over Model B.

### 8.3 Decision Quantities

| Quantity | Formula | Interpretation |
|----------|---------|----------------|
| Mean $\Delta$ | $\bar{\Delta} = \frac{1}{S}\sum_s \Delta^{(s)}$ | Expected score advantage of A over B |
| 95% CI | $[\Delta^{(0.025)}, \Delta^{(0.975)}]$ | Posterior uncertainty interval |
| $P(A > B)$ | $\frac{1}{S}\sum_s \mathbb{1}[\Delta^{(s)} > 0]$ | Posterior probability that A is better |

**Decision rule**: If $P(A > B) > 0.95$ and the 95% CI excludes 0, we conclude Model A is significantly better.

---

## 9. Interpretation of Parameters

| Parameter | Meaning |
|-----------|---------|
| $\delta_A > 0$ | Model A scores higher on average (logit scale) |
| $\delta_A \approx 0$ | No difference between models |
| $\sigma_\theta$ large | Items vary widely in difficulty |
| $\sigma_\theta$ small | Items are similar in difficulty |
| $\sigma$ large | High observation noise / judge disagreement |
| $\sigma$ small | Scores are precise |

### Logit-to-Probability Conversion (approximate)

For scores near 0.5 ($\theta_i \approx 0$), a logit-scale effect $\delta_A$ corresponds to roughly:

$$
\Delta_{\text{prob}} \approx \frac{\delta_A}{4}
$$

For example, $\delta_A = 0.4$ on the logit scale $\approx 0.1$ improvement on the probability scale near the center.

---

## 10. Assumptions and Limitations

1. **Logit-normal assumption**: Scores are Gaussian on the logit scale. This breaks down when scores pile up at 0 or 1 (boundary effects cause divergences). A Beta likelihood would be more appropriate for heavily boundary-concentrated data.

2. **Shared $\sigma$**: Both models share the same observation noise $\sigma$. This assumes similar score variability across models.

3. **Exchangeable items**: The model assumes items are exchangeable draws from a common difficulty distribution. Structured item dependencies (e.g., topic clusters) are not captured.

4. **Single $\delta_A$**: One scalar captures the entire model difference. If Model A excels on some item types but not others, this is averaged out.

5. **Paired design only**: The framework requires the same items evaluated by both models. Unpaired comparisons require a different model structure.

---

## 11. Connection to Item Response Theory (IRT)

This model is closely related to the **2-parameter logistic (2PL) IRT model** used in psychometrics:

| IRT Concept | Our Model |
|-------------|-----------|
| Person ability | Model effect $\delta_A$ |
| Item difficulty | $\theta_i$ |
| Response | Logit-score $z_{m,i}$ |
| Discrimination | Fixed at 1 (Rasch-like) |

The key difference: IRT typically models binary responses, while we use continuous logit-scores with Gaussian noise.

---

## 12. Diagnostics

After fitting, check:

- **Divergences** = 0 (NUTS sampling issues)
- **$\hat{R} < 1.01$** for all parameters (chain convergence)
- **ESS > 400** per chain (effective sample size)
- **Posterior density of $\delta_A$** is unimodal and smooth
