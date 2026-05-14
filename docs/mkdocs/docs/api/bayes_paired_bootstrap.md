# Paired Model — Bayesian Bootstrap

Nonparametric paired A/B test using Rubin's Bayesian bootstrap: a
Dirichlet(1, …, 1) "prior" over the empirical distribution of paired
differences yields a full posterior on `Δ = p_A − p_B` without any
parametric likelihood or prior elicitation. Decision making is routed
through the ROPE / posterior-mass framework only — Savage–Dickey BFs do
not apply because there is no parametric prior on `Δ` to evaluate at
the null.

::: bayesprop.resources.bayes_paired_bootstrap
