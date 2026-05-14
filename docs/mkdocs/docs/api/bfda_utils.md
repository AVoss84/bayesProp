# Utilities

General-purpose helpers used across the package. Grouped by purpose:

- **Data-generating processes** — `simulate_nonpaired_scores`,
  `simulate_paired_scores`: generate paired and non-paired binary
  outcomes from configurable true effects. Used by the BFDA harness
  and the operating-characteristics notebooks.
- **Frequentist baseline tests** — `fisher_exact_nonpaired_test`,
  `mcnemar_paired_test`: classical tests that serve as the frequentist
  baseline in the operating-characteristics analyses
  ([non-paired](../guide/frequentist_evaluation.md),
  [paired](../guide/frequentist_evaluation_paired.md)).
- **Bayes Factor Design Analysis (BFDA)** — `bfda_simulate`,
  `bfda_power_curve`, `find_n_for_power`, `plot_bfda_power`,
  `plot_bfda_sensitivity`: Monte-Carlo sample-size planning under a
  Bayes-factor decision rule. See the
  [sample-size planning guide](../guide/bfda.md).
- **Decision helpers** — `bf10_to_ph0`: convert a Bayes factor
  ``BF₁₀`` into the posterior probability of the null under a given
  prior ``π_H₀``.

::: bayesprop.utils.utils
