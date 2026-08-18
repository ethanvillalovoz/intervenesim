# Reviewer guide

This page is the shortest path for auditing InterveneSim-CF without trusting the README.

## Ten-minute audit

1. Read the [v0.4 paper](../output/pdf/intervenesim-cf-report.pdf), especially Sections 2, 4,
   5, and 7.
2. Inspect [`LoggedBanditData`](../src/intervenesim/selective.py) and verify that its schema has no
   potential-outcome fields.
3. Open [`leakage_audit.json`](../results/intervenesim-cf/leakage_audit.json); all 15 public logs
   must pass.
4. Compare equal-budget results in
   [`gate_aggregate.csv`](../results/intervenesim-cf/gate_aggregate.csv) with training-threshold
   results in
   [`predeployment_gate_seed_summary.csv`](../results/intervenesim-cf/predeployment_gate_seed_summary.csv).
5. Inspect the negative group results in [`open_world.csv`](../results/intervenesim-cf/open_world.csv)
   and delayed-treatment results in
   [`latency_aggregate.csv`](../results/intervenesim-cf/latency_aggregate.csv).
6. Run `uv run pytest` and the selective smoke command from the README.

## Claim-to-artifact map

| Claim | Evidence |
|---|---|
| T-learning improves exact effect estimation over failure risk | `counterfactual_metrics.csv` |
| Equal-budget task-success improvements are small | `gate_aggregate.csv` |
| Training-threshold calibration does not transfer for DR | `predeployment_gate_seed_summary.csv` |
| Raw IPW is unstable under selective logging | `counterfactual_metrics.csv` by logging scheme |
| More factual episodes generally help | `learning_curves.csv` |
| Entire-task and disturbance shift remain difficult | `open_world.csv` |
| Intervention value changes with assistant arrival time | `latency_aggregate.csv` |
| Factual logs exclude both potential outcomes | factual `.npz` files and `leakage_audit.json` |

## Questions the release does not answer

- Does the result transfer to a physical robot?
- Does it hold for a particular human operator?
- Can a faithful PAINT implementation outperform the controlled reversibility proxy?
- Can a temporal visual model retain calibration under camera and task shift?
- Can interventions update the control policy online and reduce future assistance?
- How does performance scale with newly collected, independent simulator episodes?

These are research questions, not implied claims.
