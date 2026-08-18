# InterveneSim-CF v0.4 study protocol

Status: **timestamped internal protocol, not an external preregistration**.

This protocol was written during implementation and before inspection of the final full-scale
v0.4 tables. Because the v0.3 data and pilot runs were already known, every v0.4 result remains
exploratory. A future confirmatory study must register its hypotheses and analysis before
collecting new simulator episodes.

## Primary estimand

For candidate context `x`, treatment `a` is immediate scripted assistance (`1`) or autonomous
continuation (`0`). Terminal success is `Y`. The individual treatment effect is:

```text
tau(x) = Y(1) - Y(0)
```

The learner observes `(x, a, Y(a), propensity(a|x))`; the evaluator retains both `Y(1)` and
`Y(0)`.

## Prospective hypotheses

1. Doubly robust learning will have lower mean PEHE than uncorrected IPW under
   uncertainty-selective logging.
2. At target intervention rates of 25% and 50%, a treatment-effect gate will improve success over
   the autonomous-failure/reversibility proxy when compared over the same logging seeds.
3. Performance will degrade when an entire task or disturbance is excluded from training, with
   greater degradation for the treatment-effect learner than the reversibility proxy because
   treatment-effect supervision is sparser.
4. Increasing factual logged episodes from 60 to 240 will improve mean helpful-intervention AUROC
   and reduce PEHE, though individual logging seeds need not be monotonic.

## Fixed analysis choices

- Five primary logging seeds: 101, 211, 307, 401, 503.
- Known target treatment probability: 0.5.
- Positivity floor under selective logging: 0.1.
- Three cross-fitting folds grouped by episode.
- Primary deployment budgets: 25% and 50%; 10% and 75% are secondary. The primary analysis
  score-ranks episodes to allocate the exact same budget without using outcomes. Thresholds frozen
  on training contexts are a separate calibration stress test.
- Primary units of uncertainty: logging seeds, not candidate states.
- Exact held-out-policy candidates remain unchanged from v0.3.
- All negative and null comparisons are retained.

## Outcomes

Primary outcomes are PEHE and exact counterfactual task success near the target intervention
budget. Secondary outcomes are helpful AUROC, request precision, average-effect error, harmful
AUROC, cost-adjusted utility, and open-world results.

The repository reports bootstrap intervals across logging seeds for descriptive uncertainty. It
does not treat candidate states as independent samples for a significance test.
