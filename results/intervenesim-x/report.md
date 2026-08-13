# InterveneSim-X research report

Primary results use **5 independent training seeds**, **4 object domains**, four disturbances, and **4,800** additional action labels per augmented condition.

## Autonomous disturbed success

Values are mean ± standard deviation across training seeds followed by a hierarchical 95% bootstrap interval in brackets.

| condition            | success                   |
|:---------------------|:--------------------------|
| baseline             | 39.4% ± 5.8% [33.9, 45.5] |
| contrastive_recovery | 39.4% ± 7.6% [32.5, 46.2] |
| more_demos           | 38.6% ± 2.3% [34.5, 42.8] |
| recovery_bc          | 61.4% ± 3.6% [56.6, 65.9] |

## Paired training-seed comparisons

| challenger           | reference   |   training_seeds | mean_delta   | std_delta   |   positive_seeds |   sign_permutation_p |
|:---------------------|:------------|-----------------:|:-------------|:------------|-----------------:|---------------------:|
| recovery_bc          | more_demos  |                5 | +22.8 pp     | 4.3 pp      |                5 |               0.0625 |
| contrastive_recovery | recovery_bc |                5 | -22.0 pp     | 9.4 pp      |                0 |               0.0625 |
| contrastive_recovery | more_demos  |                5 | +0.8 pp      | 5.7 pp      |                2 |               0.812  |

With 5 training seeds, the exact sign-permutation test has limited resolution; effect consistency and interval width are emphasized over a binary significance label.

## Learned help seeking

Held-out risk AUROC: **0.919**; average precision: **0.527**; recall: **0.889**.

| help_mode    | success_rate   | intervention_rate   |
|:-------------|:---------------|:--------------------|
| always_help  | 97.7%          | 100.0%              |
| learned_help | 98.4%          | 100.0%              |
| no_help      | 39.1%          | 0.0%                |
| oracle_help  | 99.2%          | 100.0%              |

The validation-selected high-recall threshold is not selective in deployment. The following post-audit threshold sweep is exploratory and is reported in full:

|   threshold | disturbed_success_rate   | disturbed_intervention_rate   | nominal_success_rate   | nominal_intervention_rate   |
|------------:|:-------------------------|:------------------------------|:-----------------------|:----------------------------|
|        0.9  | 89.8%                    | 87.5%                         | 100.0%                 | 28.1%                       |
|        0.95 | 79.7%                    | 75.0%                         | 96.9%                  | 15.6%                       |
|        0.97 | 75.8%                    | 64.8%                         | 96.9%                  | 3.1%                        |
|        0.99 | 54.7%                    | 22.7%                         | 96.9%                  | 0.0%                        |

## Boundaries

These results concern state-based simulation with scripted corrections. They do not establish real-world transfer, human intervention quality, visual robustness, or robot safety. See `docs/intervenesim-x.md` for the predeclared protocol.
