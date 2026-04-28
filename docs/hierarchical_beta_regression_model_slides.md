---
marp: true
theme: default
paginate: true
math: katex
style: |
  /* ============================================================
     CUSTOM MARP THEME — Overview
     ============================================================
     This style block customises the default Marp theme with:
     - CSS variables (--color-*) for easy global color changes
     - A gradient accent bar on the left edge of every slide
     - A company logo watermark via layered background-image
     - Subtle radial gradients for visual depth
     - A dark "lead" class for title/section divider slides
     - Beamer-style formula blocks (.block, .block-green, etc.)
     ============================================================ */

  /* Google Fonts — Inter for a clean, modern look */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  /* ----------------------------------------------------------
     CSS Variables — change these to re-theme the entire deck
     ---------------------------------------------------------- */
  :root {
    --color-bg: #FAFBFC;
    --color-fg: #1A1A2E;
    --color-accent: #2563EB;
    --color-accent-light: #DBEAFE;
    --color-muted: #64748B;
    --color-border: #E2E8F0;
    --color-heading: #0F172A;
  }

  /* ----------------------------------------------------------
     Base slide layout
     - background-image stacks 3 layers (comma-separated):
       1. Blue radial glow (top-right corner)
       2. Purple radial glow (bottom-left corner)
       3. Company logo PNG (bottom-left, 140px wide)
     - The background-color provides the base beneath all layers
     ---------------------------------------------------------- */
  section {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 24px;
    color: var(--color-fg);
    background-color: #F0F4F8;
    background-image:
      radial-gradient(circle at 95% 5%, rgba(37,99,235,0.06) 0%, transparent 50%),
      radial-gradient(circle at 5% 95%, rgba(124,58,237,0.05) 0%, transparent 50%),
      url('NemetschekGroup_Black+Grey_Logo.png');
    background-repeat: no-repeat, no-repeat, no-repeat;
    background-position: top right, bottom left, bottom 24px left 32px;
    background-size: auto, auto, 140px auto;
    padding: 48px 56px 72px 56px;
    line-height: 1.6;
    letter-spacing: -0.01em;
  }

  /* Accent bar — a thin gradient stripe on the left edge.
     Uses ::before pseudo-element with absolute positioning. */
  section::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 5px;
    background: linear-gradient(180deg, var(--color-accent) 0%, #7C3AED 100%);
  }

  h1 {
    font-size: 38px;
    font-weight: 700;
    color: var(--color-heading);
    border-bottom: none;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
  }

  h2 {
    font-size: 28px;
    font-weight: 600;
    color: var(--color-accent);
    border-bottom: 2px solid var(--color-accent-light);
    padding-bottom: 8px;
    margin-bottom: 20px;
    letter-spacing: -0.01em;
  }

  h3 {
    font-size: 22px;
    font-weight: 600;
    color: var(--color-muted);
    margin-bottom: 4px;
  }

  strong {
    color: var(--color-heading);
    font-weight: 600;
  }

  /* Blockquotes — used for callouts / key questions.
     Styled as a card with blue left border (like Beamer alertblock). */
  blockquote {
    border-left: 4px solid var(--color-accent);
    background: var(--color-accent-light);
    padding: 12px 20px;
    margin: 16px 0;
    border-radius: 0 8px 8px 0;
    font-style: normal;
    color: var(--color-fg);
  }

  code {
    background: #F1F5F9;
    color: #1E293B;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
  }

  pre {
    background: #1E293B;
    color: #E2E8F0;
    border-radius: 8px;
    padding: 20px;
    font-size: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }

  /* Tables — styled with blue header row, alternating stripes,
     rounded corners and a subtle drop shadow. */
  table {
    font-size: 20px;
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }

  thead th {
    background: var(--color-accent);
    color: white;
    font-weight: 600;
    padding: 10px 16px;
    text-align: left;
    border: none;
  }

  tbody td {
    padding: 8px 16px;
    border-bottom: 1px solid var(--color-border);
  }

  tbody tr:nth-child(even) {
    background: #F8FAFC;
  }

  tbody tr:last-child td {
    border-bottom: none;
  }

  ul, ol {
    margin-left: 0;
    padding-left: 1.4em;
  }

  li {
    margin-bottom: 6px;
  }

  li::marker {
    color: var(--color-accent);
    font-weight: 700;
  }

  /* Images — rounded corners + shadow to match block styling */
  img {
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  }

  /* Page number — Marp renders page numbers via ::after.
     This styles the auto-generated paginator. */
  section::after {
    font-size: 13px;
    color: var(--color-muted);
    font-weight: 400;
  }

  /* ----------------------------------------------------------
     Title / section divider slide
     Activate with:  <!-- _class: lead -->
     Uses a dark gradient background, overrides text colors.
     ---------------------------------------------------------- */
  section.lead {
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: left;
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #334155 100%);
    color: white;
  }

  section.lead::before {
    background: linear-gradient(180deg, var(--color-accent) 0%, #7C3AED 50%, #EC4899 100%);
    width: 6px;
  }

  section.lead h1 {
    color: white;
    font-size: 42px;
    border-bottom: none;
  }

  section.lead h3 {
    color: #94A3B8;
    font-weight: 400;
    font-size: 24px;
  }

  section.lead strong {
    color: #60A5FA;
  }

  /* Math styling */
  .katex { font-size: 1.05em; }

  /* ----------------------------------------------------------
     Beamer-style blocks — LaTeX-inspired formula containers.

     Usage (in slide content):
       <div class="block">                    ← blue (default)
       <div class="block block-green">        ← green (results)
       <div class="block block-red">          ← red (warnings)
       <div class="block block-grey">         ← grey (definitions)
         <div class="block-title">Title</div>
         <div class="block-body">
           $$ ... formula ... $$
         </div>
       </div>

     Equivalent to LaTeX Beamer:
       \begin{block}{Title} ...          (blue)
       \begin{exampleblock}{Title} ...   (green)
       \begin{alertblock}{Title} ...     (red)
     ---------------------------------------------------------- */
  .block {
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    margin: 14px 0;
    overflow: hidden;
  }
  .block-title {
    background: var(--color-accent);
    color: white;
    font-weight: 600;
    font-size: 18px;
    padding: 6px 16px;
  }
  .block-body {
    padding: 12px 20px;
  }

  /* Variant: green (example / result) */
  .block-green .block-title {
    background: #16A34A;
  }

  /* Variant: red (alert / warning) */
  .block-red .block-title {
    background: #DC2626;
  }

  /* Variant: grey (definition / remark) */
  .block-grey .block-title {
    background: #475569;
  }

---

<!-- _class: lead — activates the dark title slide style defined above -->
<!-- _class: lead -->

# Hierarchical Beta-Regression with Item-Specific Precision

<br>

### Bayesian testing of two proportions

Comparing **Gemini 2.5 Flash** vs **Gemini 2.0 Flash**

<br>

Alexander Vosseler · 27 April 2026

---

## Problem Setting

- Compare two RAG generators on the same $N$ evaluation items
- Shared **TFIDF retriever** ($k=5$), same LLM judge
- Each item $i$ produces paired scores $s_{A,i}, s_{B,i} \in (0,1)$
- Metrics: **Answer Relevancy** and **Faithfulness** (DeepEval)

<!-- Block usage: "block" (blue header) for standard formulas -->
<div class="block">
<div class="block-title">Hypothesis Test</div>
<div class="block-body">

$$H_0: \delta_A = 0 \quad \text{vs} \quad H_1: \delta_A \neq 0$$

where $\delta_A$ is the difference in average metric scores (e.g. Faithfulness) between Model A and Model B on the logit scale

</div>
</div>

> Does Model A systematically differ from Model B?

---

## Why Not Logistic-Normal?

<!-- Block usage: "block-grey" for rejected / superseded models -->
<div class="block block-grey">
<div class="block-title">Logistic-Normal Model</div>
<div class="block-body">

$$z_{m,i} = \operatorname{logit}(s_{m,i}), \quad z_{m,i} \sim \mathcal{N}(\theta_i + \delta_m, \sigma^2)$$

</div>
</div>

**Posterior predictive checks revealed critical misfit:**

- **Observed data**: J-shaped — scores cluster near 1.0
- **Replicated data**: symmetric U-shape on $(0,1)$

The Gaussian noise on the logit scale creates unrealistic mass near 0.

---

## Why Not Global Precision?

<div class="block block-grey">
<div class="block-title">Beta with Global Precision</div>
<div class="block-body">

$$s_{m,i} \sim \operatorname{Beta}(\mu_{m,i} \cdot \phi, \; (1-\mu_{m,i}) \cdot \phi)$$

</div>
</div>

The data contains a **mixture** of:

- **Easy items**: both models score $\approx 1.0$, need $\phi \gg 1$
- **Hard items**: scores scattered across $(0,1)$, need $\phi \approx 1$

A single $\phi$ cannot accommodate both regimes.

**Decision**: Item-specific precision $\phi_i$.

---

## Model Evolution (PPC-Driven)

| Model | PPC Result | Issue |
|-------|-----------|-------|
| Logistic-Normal | **FAIL** | U-shaped vs J-shaped observed |
| Beta (global $\phi$) | **FAIL** | Can't capture tight + diffuse items |
| Beta (item $\phi_i$) | **PASS** | Matches observed J-shape and spread |

Each iteration was rejected by **posterior predictive checks** until the data-generating process was adequately captured.

---

## Notation

| Symbol | Domain | Description |
|--------|--------|-------------|
| $\theta_i$ | $\mathbb{R}$ | Item difficulty (logit scale) |
| $\delta_A$ | $\mathbb{R}$ | Model A advantage (**key parameter**) |
| $\sigma_\theta$ | $\mathbb{R}^+$ | SD of item difficulties |
| $\phi_i$ | $\mathbb{R}^+$ | Item-specific precision |
| $\mu_\phi, \sigma_\phi$ | $\mathbb{R}, \mathbb{R}^+$ | Population parameters of $\log \phi_i$ |

---

## Generative Model — Hyperpriors

<div class="block">
<div class="block-title">Hyperpriors</div>
<div class="block-body">

$$\sigma_\theta \sim \operatorname{HalfNormal}(1) \qquad \mu_\phi \sim \mathcal{N}(0, 2^2) \qquad \sigma_\phi \sim \operatorname{HalfNormal}(1)$$

$$\delta_A \sim \mathcal{N}(0, \sigma_s^2) \qquad \text{(fixed } \sigma_s = 1.0 \text{)}$$

</div>
</div>

---

## Generative Model — Item Level

<div class="block">
<div class="block-title">Item Parameters</div>
<div class="block-body">

$$\theta_i \sim \mathcal{N}(0, \sigma_\theta^2) \qquad \phi_i \sim \operatorname{LogNormal}(\mu_\phi, \sigma_\phi^2)$$

</div>
</div>

<div class="block">
<div class="block-title">Logit Link</div>
<div class="block-body">

$$\mu_{A,i} = \operatorname{sigmoid}(\theta_i + \delta_A) \qquad \mu_{B,i} = \operatorname{sigmoid}(\theta_i)$$

</div>
</div>

---

## Generative Model — Likelihood

<div class="block block-green">
<div class="block-title">Beta Likelihood (Item-Specific Precision)</div>
<div class="block-body">

$$s_{A,i} \sim \operatorname{Beta}\!\left(\mu_{A,i} \cdot \phi_i, \; (1 - \mu_{A,i}) \cdot \phi_i\right)$$

$$s_{B,i} \sim \operatorname{Beta}\!\left(\mu_{B,i} \cdot \phi_i, \; (1 - \mu_{B,i}) \cdot \phi_i\right)$$

</div>
</div>

Both models share the same $\phi_i$ per item — item difficulty drives precision variability more than model identity.

---

## Key Design Choices

1. **LogNormal precision** — ensures $\phi_i > 0$, allows heavy-tailed variation
   - Easy items → large $\phi_i$ (tight)
   - Hard items → small $\phi_i$ (diffuse)

2. **Non-centered parameterization** — avoids funnel geometry in NUTS

3. **Fixed slab width $\sigma_s$** — essential for Savage-Dickey BF

4. **Shared $\phi_i$** — same retriever, same context per item

---

## Non-Centered Parameterization

Avoids the funnel geometry that degrades NUTS sampling:

<div class="block">
<div class="block-title">Non-Centered Transform</div>
<div class="block-body">

$$\theta_i^{\text{raw}} \sim \mathcal{N}(0,1), \quad \theta_i = \sigma_\theta \cdot \theta_i^{\text{raw}}$$

$$\phi_i^{\text{raw}} \sim \mathcal{N}(0,1), \quad \log \phi_i = \mu_\phi + \sigma_\phi \cdot \phi_i^{\text{raw}}$$

</div>
</div>

Critical when population SDs ($\sigma_\theta$, $\sigma_\phi$) are small.

---

## MCMC Configuration

| Parameter | Value |
|-----------|-------|
| Sampler | NUTS (PyMC) |
| Chains | 4 |
| Warmup | 2,000 |
| Draws / chain | 2,000 |
| Total draws | 8,000 |
| Target accept | 0.99 |

High accept rate needed — Beta likelihood with near-boundary data creates sharp posterior curvature.

---

## MCMC Diagnostics: Trace Plots

![w:1100 h:520 center](trace_plot.png)

---

## Posterior Quantities

**Logit scale:**
- Marginal posterior $p(\delta_A \mid \mathbf{s}_A, \mathbf{s}_B)$
- 95% credible interval

**Probability scale** — population-averaged difference:

<div class="block">
<div class="block-title">Score Difference</div>
<div class="block-body">

$$\Delta^{(s)} = \frac{1}{N} \sum_{i=1}^{N} \left(\mu_{A,i}^{(s)} - \mu_{B,i}^{(s)}\right)$$

</div>
</div>

---

## Decision Quantities

| Quantity | Interpretation |
|----------|----------------|
| Mean $\Delta$ | Expected score advantage of A over B |
| 95% CI of $\Delta$ | Posterior uncertainty interval |
| $P(A > B)$ | Posterior probability that A is better |

<div class="block block-green">
<div class="block-title">Posterior Probability of Superiority</div>
<div class="block-body">

$$P(A > B) = \frac{1}{S}\sum_s \mathbb{1}[\Delta^{(s)} > 0]$$

</div>
</div>

---

## Results: Forest Plot & P(A > B)

![Forest plot and posterior probability of superiority](forest_plot_comparison.png)

---

## Results: Posterior of $\delta_A$

![Posterior distributions of delta_A](posterior_delta_A.png)

---

## Simulation Validation

Validate parameter recovery before real data:

- **DGP**: $N=200$, $\delta_A^{\text{true}} = 0.5$, $\sigma_\theta = 1.0$
- **Recovery**: $\hat{\delta}_A = 0.473$ — good recovery
- All $\hat{r} = 1.0$ — chains converged
- 95% CI covers the true value

Confirms model and inference pipeline are correctly implemented.

---

## Posterior Predictive Checks

For each posterior draw, generate replicated data:

<div class="block">
<div class="block-title">Replicated Data</div>
<div class="block-body">

$$s_{m,i}^{\text{rep}} \sim \operatorname{Beta}\!\left(\mu_{m,i}^{(s)} \cdot \phi_i^{(s)},\; (1 - \mu_{m,i}^{(s)}) \cdot \phi_i^{(s)}\right)$$

</div>
</div>

Compare against observed data on:
1. Marginal distribution of $s_A$
2. Marginal distribution of $s_B$
3. Paired difference $s_A - s_B$

---

## PPC Results

![w:1150 h:450 center](ppc_beta_model.png)

---

## PPC p-values

Two-sided posterior predictive p-value:

<div class="block">
<div class="block-title">PPC p-value</div>
<div class="block-body">

$$p = 2 \cdot \min\!\left(P(T^{\text{rep}} \geq T^{\text{obs}}),\; P(T^{\text{rep}} \leq T^{\text{obs}})\right)$$

</div>
</div>

- $p > 0.05$: model adequately reproduces this statistic
- $p < 0.05$: potential misfit

Summary statistics $T$: mean, std, mean difference, std of difference.

---

## Savage-Dickey Bayes Factor

### The Measure-Theoretic Problem

Under a continuous prior, $\{\delta_A = 0\}$ has **Lebesgue measure zero** — it can never get positive posterior probability.

<div class="block block-red">
<div class="block-title">Spike-and-Slab Prior</div>
<div class="block-body">

$$\pi(\delta_A) = \pi_0 \cdot \delta_{\{0\}}(\delta_A) + (1 - \pi_0) \cdot g(\delta_A)$$

</div>
</div>

---

## The Savage-Dickey Density Ratio

Since $H_0$ is nested in $H_1$:

<div class="block">
<div class="block-title">Savage-Dickey Density Ratio</div>
<div class="block-body">

$$BF_{01} = \frac{p(\delta_A = 0 \mid D, H_1)}{g(0)} \qquad \text{where } g(0) = \frac{1}{\sigma_s \sqrt{2\pi}}$$

</div>
</div>

- **Numerator**: posterior density at 0, estimated via KDE on 8,000 draws
- **Denominator**: slab prior density at 0 (analytical)

---

## Savage-Dickey: Results

![Savage-Dickey Bayes Factor plot](savage_dickey_bf.png)

---

## Why Savage-Dickey?

1. **MCMC reuse** — no additional model fitting needed
2. **Analytical denominator** — only KDE approximation in numerator
3. **Prior consistency** — fixed $\sigma_s$ ensures numerator and denominator refer to the same $H_1$

A hierarchical $\sigma_s$ would make $g(0)$ diverge:

<div class="block block-red">
<div class="block-title">Divergent Marginal Prior</div>
<div class="block-body">

$$g(0) = \int_0^\infty \frac{1}{\sigma_s \sqrt{2\pi}} \cdot \operatorname{HalfNormal}(\sigma_s) \, d\sigma_s \to \infty$$

</div>
</div>

---

## Jeffreys Interpretation Scale

| $BF_{10}$ | Evidence against $H_0$ |
|-----------|----------------------|
| $< 1$ | Supports $H_0$ |
| $1 – 3$ | Anecdotal |
| $3 – 10$ | Moderate |
| $10 – 30$ | Strong |
| $30 – 100$ | Very strong |
| $> 100$ | Decisive |

For extreme values: display as $\log_{10} BF_{10}$.

---

## From BF to Posterior Model Probability

<div class="block">
<div class="block-title">Posterior Model Probability</div>
<div class="block-body">

$$P(H_0 \mid D) = \frac{\pi_0 \cdot BF_{01}}{\pi_0 \cdot BF_{01} + (1 - \pi_0)}$$

</div>
</div>

With agnostic prior $\pi_0 = 0.5$:

<div class="block block-green">
<div class="block-title">Simplified (Equal Prior Odds)</div>
<div class="block-body">

$$P(H_0 \mid D) = \frac{BF_{01}}{1 + BF_{01}}$$

</div>
</div>

A well-defined probability statement — the spike mass ensures $\{\delta_A = 0\}$ has positive prior and posterior measure.

---

## Sensitivity Analysis

### Prior $P(H_0)$

Sweep $\pi_0$ from 0.01 to 0.99 — does the conclusion hold?

![Sensitivity to prior P(H0)](sensitivity_prior_h0.png)

---

## Sensitivity: Slab Width (Jeffreys-Lindley)

Sweep $\sigma_s \in [0.25, 5.0]$ — is the BF robust?

![Sensitivity to slab width](sensitivity_slab_width.png)

---

## Connection to Item Response Theory

| IRT Concept | Our Model |
|-------------|-----------|
| Person ability | Model effect $\delta_A$ |
| Item difficulty | $\theta_i$ |
| Response | Continuous $s_{m,i} \in (0,1)$ |
| Link function | Logistic sigmoid |
| Discrimination | Fixed at 1 (Rasch-like) |
| Response distribution | Beta (vs Bernoulli in IRT) |

$\phi_i$ acts as item-specific **reliability** — analogous to IRT discrimination.

---

## Why Not McNemar or Wilcoxon?

| Aspect | McNemar / Wilcoxon | Our Model |
|--------|-------------------|-----------|
| Data used | Binary or rank-based | Full continuous $(0,1)$ |
| Item heterogeneity | Ignored | $\theta_i$ |
| Precision heterogeneity | Ignored | $\phi_i$ |
| Effect size | None / limited | $\Delta$, 95% CI, $P(A>B)$ |
| Evidence for $H_0$ | Cannot provide | $BF_{01}$, $P(H_0 \mid D)$ |

Frequentist tests can serve as a **robustness footnote** — confirming under a simpler framework.

---

## Assumptions & Limitations

1. Scores clipped to $(\varepsilon, 1-\varepsilon)$ for Beta support
2. Shared $\phi_i$ — could relax to model-specific precision
3. Single scalar $\delta_A$ — no item × model interactions
4. Exchangeable items — no topic clustering
5. Fixed slab width — validated via sensitivity sweep
6. KDE approximation — reliable with 8,000 draws

---

## Complete Pipeline

<div class="block" style="background:#1E293B; border-radius:8px;">
<div class="block-body" style="color:#E2E8F0; padding:20px 28px; font-size:22px; line-height:1.9;">

1. **Data prep** — Chunk docs, load Q/A, build TFIDF retriever
2. **Evaluation** — Score both models with DeepEval
3. **Simulation** — Validate parameter recovery ($\delta_A = 0.5$)
4. **Inference** — Fit hierarchical Beta model per metric
5. **Diagnostics** — Trace plots, $\hat{r}$, ESS, PPC
6. **Hypothesis test** — Savage-Dickey $BF_{10}$, $P(H_0 \mid D)$
7. **Sensitivity** — Sweep $\pi_0$ and $\sigma_s$

</div>
</div>

---

## Summary

- **Beta likelihood** with **item-specific LogNormal precision** — validated by PPC
- **Paired design** isolates the generator effect
- **Savage-Dickey BF** — reuses MCMC draws, no extra model fitting
- **Full posterior inference**: $\Delta$, 95% CI, $P(A > B)$, $P(H_0 \mid D)$
- **Sensitivity analysis** confirms robustness to prior choices

---

<!-- _class: lead -->

# Thank you for your $\;\;\operatorname{softmax}\!\left(\frac{Q \cdot K^\top}{\sqrt{d_k}}\right) \cdot V$
