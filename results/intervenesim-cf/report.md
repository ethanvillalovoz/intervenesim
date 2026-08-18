# InterveneSim-CF research report

## Research question

Can an agent learn the individual value of requesting assistance when each training episode reveals only the outcome that actually occurred?

The learner receives one randomly timed candidate and one observed treatment arm per episode. Exact simulator forks are hidden from training and used only as an evaluation oracle.

Source data: **1,505** training candidates from **240 episodes** and **1,517** evaluation candidates from **240 unseen-policy episodes**. Results use **5 logging seeds**.

## Exact individual-effect estimation

| method               |   pehe_mean |   pehe_std |   helpful_auroc_mean |   helpful_auroc_std |   ate_error_mean |   ate_error_std |
|:---------------------|------------:|-----------:|---------------------:|--------------------:|-----------------:|----------------:|
| dr_learner           |       0.617 |      0.031 |                0.694 |               0.032 |            0.057 |           0.055 |
| ensemble_uncertainty |       0.691 |      0     |                0.703 |               0     |            0.395 |           0     |
| failure_risk         |       0.614 |      0     |                0.706 |               0     |            0.221 |           0     |
| ipw_learner          |       0.831 |      0.031 |                0.594 |               0.028 |            0.115 |           0.072 |
| paired_label_oracle  |       0.488 |      0     |                0.791 |               0     |            0.017 |           0     |
| reversibility_proxy  |       0.55  |      0.01  |                0.725 |               0.013 |            0.143 |           0.034 |
| s_learner            |       0.561 |      0.016 |                0.69  |               0.016 |            0.074 |           0.039 |
| t_learner            |       0.524 |      0.014 |                0.732 |               0.022 |            0.054 |           0.027 |

PEHE is root mean squared error against the hidden signed individual effect. The paired-label oracle is privileged and is included only as an upper-bound comparator.

## Budget-aware deployment on unseen policies

| method              |   target_budget | success_mean   | intervention_mean   | precision_mean   |
|:--------------------|----------------:|:---------------|:--------------------|:-----------------|
| paired_label_oracle |            0.25 | 66.2%          | 25.0%               | 85.0%            |
| paired_label_oracle |            0.5  | 81.7%          | 50.0%               | 73.3%            |
| failure_risk        |            0.25 | 61.3%          | 25.0%               | 65.0%            |
| failure_risk        |            0.5  | 76.2%          | 50.0%               | 62.5%            |
| t_learner           |            0.25 | 61.7%          | 25.0%               | 66.7%            |
| t_learner           |            0.5  | 78.2%          | 50.0%               | 66.7%            |
| dr_learner          |            0.25 | 62.4%          | 25.0%               | 69.7%            |
| dr_learner          |            0.5  | 78.1%          | 50.0%               | 66.3%            |
| reversibility_proxy |            0.25 | 62.0%          | 25.0%               | 68.0%            |
| reversibility_proxy |            0.5  | 79.8%          | 50.0%               | 69.7%            |

## Paired logging-seed comparisons

| logging_scheme        |   target_budget | challenger   | reference           | mean_delta   |   positive_seeds |
|:----------------------|----------------:|:-------------|:--------------------|:-------------|-----------------:|
| adaptive_value        |            0.25 | dr_learner   | reversibility_proxy | -1.3 pp      |                1 |
| adaptive_value        |            0.25 | dr_learner   | failure_risk        | -0.5 pp      |                2 |
| adaptive_value        |            0.25 | dr_learner   | paired_label_oracle | -5.5 pp      |                0 |
| adaptive_value        |            0.5  | dr_learner   | reversibility_proxy | -1.9 pp      |                1 |
| adaptive_value        |            0.5  | dr_learner   | failure_risk        | -0.4 pp      |                2 |
| adaptive_value        |            0.5  | dr_learner   | paired_label_oracle | -5.8 pp      |                0 |
| randomized            |            0.25 | dr_learner   | reversibility_proxy | +0.4 pp      |                4 |
| randomized            |            0.25 | dr_learner   | failure_risk        | +1.2 pp      |                4 |
| randomized            |            0.25 | dr_learner   | paired_label_oracle | -3.8 pp      |                0 |
| randomized            |            0.5  | dr_learner   | reversibility_proxy | -1.7 pp      |                1 |
| randomized            |            0.5  | dr_learner   | failure_risk        | +1.8 pp      |                5 |
| randomized            |            0.5  | dr_learner   | paired_label_oracle | -3.6 pp      |                0 |
| uncertainty_selective |            0.25 | dr_learner   | reversibility_proxy | -0.5 pp      |                1 |
| uncertainty_selective |            0.25 | dr_learner   | failure_risk        | +1.3 pp      |                3 |
| uncertainty_selective |            0.25 | dr_learner   | paired_label_oracle | -3.7 pp      |                0 |
| uncertainty_selective |            0.5  | dr_learner   | reversibility_proxy | -2.9 pp      |                2 |
| uncertainty_selective |            0.5  | dr_learner   | failure_risk        | +1.3 pp      |                3 |
| uncertainty_selective |            0.5  | dr_learner   | paired_label_oracle | -4.2 pp      |                1 |

## Open-world audit

Every task and disturbance type is held out in turn. Training logs contain no row from the held-out group.

| held_out_axis   | method              | success         |
|:----------------|:--------------------|:----------------|
| disturbance     | dr_learner          | 59.2% +/- 28.8% |
| disturbance     | reversibility_proxy | 55.8% +/- 31.3% |
| task            | dr_learner          | 59.2% +/- 9.9%  |
| task            | reversibility_proxy | 61.7% +/- 8.8%  |

## Assistance-latency shift

The same learned gate is evaluated when the scripted assistant arrives at the current candidate, one candidate later, or two candidates later along the exact autonomous trajectory.

|   delay_candidate_hops | method              | success        |
|-----------------------:|:--------------------|:---------------|
|                      0 | paired_label_oracle | 66.2% +/- 0.0% |
|                      0 | t_learner           | 61.7% +/- 2.4% |
|                      0 | dr_learner          | 62.4% +/- 0.9% |
|                      0 | reversibility_proxy | 62.0% +/- 0.9% |
|                      1 | paired_label_oracle | 62.5% +/- 0.0% |
|                      1 | t_learner           | 58.2% +/- 2.3% |
|                      1 | dr_learner          | 59.1% +/- 0.3% |
|                      1 | reversibility_proxy | 57.1% +/- 1.0% |
|                      2 | paired_label_oracle | 60.0% +/- 0.0% |
|                      2 | t_learner           | 55.7% +/- 2.2% |
|                      2 | dr_learner          | 55.6% +/- 1.1% |
|                      2 | reversibility_proxy | 52.6% +/- 1.1% |

## Interpretation boundaries

This benchmark validates estimators against exact simulator potential outcomes. It does not identify human-specific intervention effects, prove real-robot safety, or establish sim-to-real transfer. The reversibility proxy predicts autonomous failure under the assumption that help succeeds; it is PAINT-inspired, not a reproduction of PAINT's online algorithm.
