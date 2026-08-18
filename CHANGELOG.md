# Changelog

## 0.4.0 — 2026-08-18

### InterveneSim-CF

- Added a deployment-realistic factual log schema with one observed treatment outcome per row.
- Added randomized, uncertainty-selective, and online adaptive-value logging with known
  propensities and positivity controls.
- Added S-, T-, IPW, cross-fitted doubly robust, reversibility-proxy, and paired-supervision
  estimators.
- Added exact hidden-counterfactual metrics including PEHE, average-effect error, helpful and
  harmful ranking, and sign accuracy.
- Added exact score-ranked intervention budgets and a separate training-threshold calibration
  stress test.
- Added five logging seeds, sample-efficiency curves, explicit intervention costs, leave-one-task
  and leave-one-disturbance audits, and exact assistance-latency profiles.
- Added 15 public factual logs plus a machine-readable audit proving that no potential-outcome
  field was serialized.
- Added a 10-page technical report, study protocol, related-work boundary, dataset card, reviewer
  guide, and frozen candidate-, seed-, and group-level results.
- Increased the automated test suite from 37 to 45 tests.

### Result discipline

- Corrected an initial unfair comparison in which a training-calibrated DR threshold targeted 25%
  intervention but triggered on about half of unseen-policy episodes.
- Made exact equal-budget ranking primary and retained the mismatch as a calibration result.
- Preserved the negative finding that better treatment-effect estimation does not consistently
  outperform a reversibility proxy in downstream task success.

## 0.3.0 — 2026-08-13

- Added exact autonomous/assisted simulator forks and intervention-value learning.
- Added held-out policy evaluation, a five-fold exploratory seed audit, visual transfer probe,
  genuine keyboard takeover interface, report, and matched fork video.

## 0.2.0

- Added multi-seed comparison of corrective recovery labels against equal-count clean labels.

## 0.1.0

- Added the initial simulation benchmark and reproducible behavior-cloning pipeline.
