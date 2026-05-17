# Paired Model — Laplace Approximation

Paired logistic model with MAP + Hessian (Laplace) posterior inference.

Supports two modes:

- **Fixed priors (default)** — $\mu \sim \mathcal{N}(0, \sigma_\mu)$,
  $\delta_A \sim \mathcal{N}(0, \sigma_\delta)$ with user-chosen scales.
  2-D Newton solver.
- **Hierarchical (learned scales)** — places Inverse-Gamma hyperpriors on
  $\sigma_\mu^2$ and $\sigma_\delta^2$ so the prior widths are learned from
  data. 4-D Newton solver; the marginal prior on $\delta_A$ becomes
  Student-$t$, making the Savage–Dickey Bayes factor robust to prior
  misspecification. Activated by passing `hyperprior_mu` and
  `hyperprior_delta` to the constructor.

The sequential variant `SequentialPairedBayesPropTest` is documented on
the [Sequential designs](sequential.md) page.

::: bayesprop.resources.bayes_paired_laplace
    options:
      filters:
        - "!^Sequential"
