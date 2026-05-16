# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.1.1] - 2026-05-16
### Added
- Abstract base class `BaseBayesPropTest` in `bayesprop.resources.base`
  defining the shared public API (`fit`, `decide`, `rope_test`,
  `plot_posteriors`, `plot_posterior_delta`, `print_summary`) that all
  four model classes now inherit from.
- `plot_posteriors()` method on all four models — single-panel overlay of
  the θ_A and θ_B posterior densities.
- `plot_posterior_delta()` method on all four models — single-panel KDE of
  Δ = θ_A − θ_B on the **probability scale** with 95 % CI shading.
- `PairedBayesPropTestBB`: new `theta_A_samples` / `theta_B_samples`
  attributes stored during `fit()`, plus `plot_posteriors()`,
  `plot_posterior_delta()`, and `print_summary()`.

### Changed
- `NonPairedBayesPropTest.plot_posteriors()` refactored from a two-panel
  layout to a single-panel θ_A / θ_B overlay.
- `PairedBayesPropTest.plot_posterior_delta()` (Laplace) and
  `PairedBayesPropTestPG.plot_posterior_delta()` now plot Δ = θ_A − θ_B
  on the probability scale instead of δ_A on the logit scale.
- All four model classes (`NonPairedBayesPropTest`,
  `PairedBayesPropTest`, `PairedBayesPropTestPG`,
  `PairedBayesPropTestBB`) now inherit from `BaseBayesPropTest`.
- Updated README quick-start examples to use the new `plot_posteriors()`
  and `plot_posterior_delta()` API.
- Updated notebooks (`deepeval_bayesprop_example`,
  `bayesian_AB_model_comparison_paired_laplace`,
  `bayesian_AB_model_comparison_paired_gibbs`) to use the new plot API.

### Deprecated
- `PairedBayesPropTest.plot_laplace_posterior()` — kept for backward
  compatibility; use `plot_posteriors()` + `plot_posterior_delta()` instead.
- `PairedBayesPropTestBB.plot_posterior()` — kept for backward
  compatibility; use `plot_posteriors()` + `plot_posterior_delta()` instead.


## [0.1.0.7] - 2026-05-14
### Added
- Operating characteristics module for the non-paired model
  (`bayesprop.utils.operation_characteristics`) with frequentist evaluation
  utilities (power, Type-I error, expected sample size).
- Operating characteristics module for the paired model
  (`bayesprop.utils.operation_characteristics_paired`).
- Bayesian bootstrap example for the paired design
  (`bayesprop.resources.bayes_paired_bootstrap`).
- New user-guide pages and API reference pages for the operating
  characteristics workflows.
- Unit tests covering the operating-characteristics utilities
  (`tests/test_operation_characteristics.py`).
- Shared `bayesprop.utils.utils.binarize_if_needed` helper plus a
  matching `threshold` (default `0.5`) and `verbose` argument on the
  paired classes `PairedBayesPropTest`, `PairedBayesPropTestPG`,
  `PairedBayesPropTestBB`, and `SequentialPairedBayesPropTest`. Continuous
  scores in `[0, 1]` are now auto-binarised at the configured threshold
  (mirroring the non-paired API), and out-of-range or `NaN` inputs raise
  a clear `ValueError` instead of being silently truncated. The
  Pydantic schemas `PairedLaplaceConfig` and `PairedPGConfig` gained a
  matching `threshold` field.

### Changed
- Lowered default MCMC settings for `PairedBayesPropTestPG` from
  `n_iter=2000, burn_in=500, n_chains=4` to `n_iter=1000, burn_in=200,
  n_chains=2`. The Pólya–Gamma Gibbs sampler is block-conjugate for the
  paired Bernoulli model and reaches R-hat ≈ 1.00 within ~50 iterations;
  empirically the new defaults yield ESS ≳ 1300 per chain on both small
  (`n = 10`) and realistic (`n ≥ 500`) data, while running ~3× faster.
  The matching defaults in `PairedPGConfig` (Pydantic schema) were
  lowered to keep the two surfaces consistent. Increase to
  `n_iter ≥ 2000` if you need stable Savage–Dickey BF estimates for
  very strong effects (BF tail behaviour is KDE-sensitive, not
  chain-sensitive).


## [0.1.0.6] - 2026-05-12
### Added
- Project logo and trimmed logo variant for the README and PyPI page.

### Changed
- Refreshed README with updated badges, logo, and feature overview.
- Granted additional permissions to the GitHub Actions release workflow
  to allow publishing artifacts.

### Fixed
- PyPI logo rendering on the project page.


## [0.1.0.5] - 2026-05-11
### Changed
- Documentation polish across the user guide and API reference.


## [0.1.0.4] - 2026-05-10
### Added
- Sequential update design for the **non-paired** Bayesian proportion model.
- Sequential update design for the **paired Laplace** model.
- New documentation pages describing the sequential designs and their
  decision rules.

### Changed
- README updated to highlight the new sequential-analysis capabilities.


## [0.1.0.3] - 2026-05-07
### Changed
- Code structure refactored across modules for better readability and
  maintainability (`chore: update code structure ...`).
- Improved function docstrings with detailed parameter and return-value
  descriptions; consistent formatting across the codebase.


## [0.1.0.2] - 2026-05-06
### Added
- File services and utility functions for data handling and simulation
  (`bayesprop.services.file`, `bayesprop.utils.utils`).

### Changed
- Bumped version to `0.1.0.2`.
- Enhanced explanation of the non-paired density in the documentation.
- Refined descriptions in README, Getting Started, and User Guide
  sections for clarity and consistency.
- Updated coverage badge.


## [0.1.0.1] - 2026-05-05
### Changed
- Switched PyPI badge to TestPyPI while the package was in pre-release.
- Refactor pass over notebooks and tests for readability and consistency.
- Upgraded GitHub Actions used by the PyPI publishing workflow and
  streamlined its steps.

### Removed
- Codecov configuration file (coverage is now reported via the CI badge).


## [0.1.0] - 2026-05-05
### Added
- Initial public package layout under `src/bayesprop/` with
  `resources/`, `services/`, `utils/`, and `config/` subpackages.
- Bayesian models for proportions:
  - Non-paired model (`bayes_nonpaired`)
  - Paired model with Laplace approximation (`bayes_paired_laplace`)
  - Paired model with Pólya-Gamma augmentation (`bayes_paired_pg`)
- Bayes Factor Design Analysis (BFDA) utilities and documentation.
- Pydantic data schemas (`resources.data_schemas`).
- MkDocs (Material) documentation site with user guide and
  API reference generated via `mkdocstrings`.
- GitHub Actions CI workflow for testing and coverage reporting.
- Unit tests for the core `bayesAB`/`bayesprop` modules.
- GitHub Copilot instructions for the repository.

### Fixed
- Repository URLs in the `mkdocs.yml` configuration.
