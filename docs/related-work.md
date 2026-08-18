# Related work and novelty boundary

InterveneSim sits at the intersection of interactive imitation learning, selective assistance,
and heterogeneous treatment-effect estimation. This document identifies the closest work and
states what this repository does—and does not—claim.

## Selective robot assistance

[PAINT](https://arxiv.org/abs/2210.10765) learns a reversibility classifier and proactively asks
for help before entering irreversible states. It is the closest conceptual predecessor.
InterveneSim-CF studies a different estimand: the signed outcome difference caused by a specific
assistant. A high-risk state need not have positive intervention value, and an intervention can
be ineffective or harmful. The `reversibility_proxy` baseline approximates the core risk
assumption for a controlled comparison; it is not a reproduction of PAINT's online reinforcement
learning algorithm.

[ThriftyDAgger](https://proceedings.mlr.press/v164/hoque22a.html) gates expert queries using both
novelty and risk under a query budget. [HG-DAgger](https://arxiv.org/abs/1810.02890) lets a human
expert take control when necessary, while [DAgger](https://proceedings.mlr.press/v15/ross11a.html)
establishes the dataset-aggregation framework that underlies much interactive imitation
learning. These methods primarily decide which states need labels or control. InterveneSim-CF
instead tests whether the learner can estimate the causal benefit of assistance from selectively
observed terminal outcomes.

## Causal estimation

The S- and T-learner names follow the meta-learner taxonomy described by
[Künzel et al.](https://www.pnas.org/doi/10.1073/pnas.1804597116). The doubly robust learner uses
cross-fitted outcome nuisance models and an augmented inverse-propensity pseudo-outcome, drawing
on the orthogonalization principle formalized for heterogeneous effects by
[Kennedy](https://arxiv.org/abs/2004.14497).

Exact individual treatment effects are normally unobservable. The simulator's deterministic
snapshot and restore operation makes both potential outcomes measurable. InterveneSim-CF uses
that capability only for evaluation, creating a test bed for asking how well realistic
single-world logs recover intervention value.

## Claim discipline

The current contribution is a benchmark and empirical study, not a claim that the estimators are
new. A defensible novelty claim is:

> Exact simulator forks can audit intervention-effect estimators trained from one-outcome robot
> assistance logs, including under selective logging and held-out task or disturbance shift.

Before any conference submission, this boundary should be reviewed against newly published work
and the closest algorithms should be reproduced faithfully where feasible.
