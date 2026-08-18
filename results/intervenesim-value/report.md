# InterveneSim-Value research report

Training counterfactuals: **1,505** candidate states across **3 policy seeds**. Evaluation counterfactuals: **1,517** states from **2 held-out policy seeds**.

## Helpfulness discrimination

| method           |   AUROC |   average_precision |
|:-----------------|--------:|--------------------:|
| value_gate       |   0.817 |               0.753 |
| risk_gate        |   0.706 |               0.617 |
| uncertainty_gate |   0.703 |               0.643 |

## Budget-aware deployment

| method           |   target_budget | success_rate   | intervention_rate   | request_precision   | regret_to_matched_oracle   |
|:-----------------|----------------:|:---------------|:--------------------|:--------------------|:---------------------------|
| never_help       |            0    | 45.0%          | 0.0%                | -                   | 0.0%                       |
| always_help      |            1    | 98.3%          | 100.0%              | 53.8%               | 1.3%                       |
| oracle_value     |          nan    | 99.6%          | 54.6%               | 100.0%              | 0.0%                       |
| value_gate       |            0.1  | 48.8%          | 4.6%                | 81.8%               | 0.8%                       |
| value_gate       |            0.25 | 66.2%          | 23.3%               | 91.1%               | 2.1%                       |
| value_gate       |            0.5  | 89.6%          | 54.6%               | 81.7%               | 10.0%                      |
| value_gate       |            0.75 | 95.0%          | 87.1%               | 57.9%               | 4.6%                       |
| risk_gate        |            0.1  | 52.1%          | 10.4%               | 68.0%               | 3.3%                       |
| risk_gate        |            0.25 | 60.4%          | 24.6%               | 62.7%               | 9.2%                       |
| risk_gate        |            0.5  | 73.8%          | 45.8%               | 62.7%               | 17.1%                      |
| risk_gate        |            0.75 | 86.7%          | 75.4%               | 55.2%               | 12.9%                      |
| uncertainty_gate |            0.1  | 49.2%          | 4.6%                | 90.9%               | 0.4%                       |
| uncertainty_gate |            0.25 | 58.3%          | 19.2%               | 69.6%               | 5.8%                       |
| uncertainty_gate |            0.5  | 81.7%          | 52.1%               | 70.4%               | 15.4%                      |
| uncertainty_gate |            0.75 | 90.0%          | 73.8%               | 62.7%               | 9.6%                       |

Thresholds were selected only from training-policy episodes. Oracle regret is computed at each method's realized number of evaluation interventions.

## Secondary five-fold policy-seed cross-validation

This analysis was added after the primary held-out-policy audit and is therefore exploratory. Each fold trains on four policy seeds and evaluates the fifth.

| method           |   target_budget | success        | realized_intervention_rate   |
|:-----------------|----------------:|:---------------|:-----------------------------|
| risk_gate        |            0.1  | 56.6% +/- 6.3% | 10.2%                        |
| risk_gate        |            0.25 | 64.3% +/- 5.0% | 25.1%                        |
| risk_gate        |            0.5  | 78.0% +/- 4.2% | 51.4%                        |
| risk_gate        |            0.75 | 89.1% +/- 4.0% | 74.3%                        |
| uncertainty_gate |            0.1  | 59.4% +/- 8.1% | 10.9%                        |
| uncertainty_gate |            0.25 | 68.4% +/- 8.1% | 25.6%                        |
| uncertainty_gate |            0.5  | 84.7% +/- 2.3% | 49.9%                        |
| uncertainty_gate |            0.75 | 92.0% +/- 2.0% | 74.8%                        |
| value_gate       |            0.1  | 59.2% +/- 5.1% | 10.0%                        |
| value_gate       |            0.25 | 72.4% +/- 6.1% | 25.5%                        |
| value_gate       |            0.5  | 92.7% +/- 4.0% | 53.9%                        |
| value_gate       |            0.75 | 96.4% +/- 3.0% | 81.8%                        |

|   budget | challenger   | reference        |   policy_seeds | mean_delta   | std_delta   |   positive_seeds |
|---------:|:-------------|:-----------------|---------------:|:-------------|:------------|-----------------:|
|     0.25 | value_gate   | risk_gate        |              5 | +8.1 pp      | 4.2 pp      |                5 |
|     0.25 | value_gate   | uncertainty_gate |              5 | +4.0 pp      | 4.3 pp      |                4 |
|     0.5  | value_gate   | risk_gate        |              5 | +14.7 pp     | 3.1 pp      |                5 |
|     0.5  | value_gate   | uncertainty_gate |              5 | +7.9 pp      | 2.8 pp      |                5 |

## Boundaries

The counterfactual is exact for this simulator and scripted expert only. It does not establish human-intervention value, real-robot safety, or sim-to-real transfer.
