"""Backward-compatible re-exports for paired A/B model comparison.

Prefer importing directly from the specific modules:

- :mod:`ai_eval.resources.bayes_paired_laplace` -- Laplace approximation
- :mod:`ai_eval.resources.bayes_paired_pg` -- Polya-Gamma Gibbs sampling
"""

from bayesAB.resources.bayes_paired_laplace import (  # noqa: F401
    PairedBayesPropTest,
    _format_bf,
    sigmoid,
)
