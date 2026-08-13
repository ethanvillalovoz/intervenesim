# InterveneSim-X: What Should a Robot Learn From a Correction?

Ethan Villalovoz  
Independent research project, August 2026

## Abstract

Interactive imitation learning is motivated by a simple asymmetry: an expert correction
at a failure state may be more useful than another successful demonstration. InterveneSim-X
turns that intuition into a reproducible, equal-label-budget simulation study that runs on
Apple Silicon without CUDA, cloud compute, paid APIs, or robot hardware. We evaluate
behavior cloning across four Panda pick-and-place object domains, four deployment
disturbances, five independently trained policy seeds, and four additional-label budgets.
At the largest matched budget, ordinary behavior cloning on corrective recovery actions
achieves 61.4% disturbed success (95% hierarchical bootstrap interval 56.6-65.9), compared
with 38.6% (34.5-42.8) for the same number of additional clean labels. The paired gain is
22.8 percentage points and is positive for all five training seeds. A contrastive objective
that repels the rejected robot action does not improve performance, demonstrating that
more structured supervision is not automatically better. An exploratory temporal risk
gate exposes a second distinction: high offline AUROC does not guarantee selective online
intervention. A complete threshold sweep finds an operating point with 75.8% assisted
disturbed success, 64.8% disturbed interventions, and a 3.1% nominal false-alarm rate.
The repository includes the simulator, data schemas, crash-resumable runner, tests, raw
episode-level outputs, uncertainty analysis, figures, and matched rollout video.

## 1. Motivation

Behavior cloning trains on states visited by an expert but deploys on states induced by
its own imperfect actions. This distribution mismatch can compound over time. Interactive
imitation learning addresses the mismatch by querying or accepting expert control on the
learner's state distribution. Yet a practical question often gets lost in algorithmic
complexity: if labeling effort is fixed, are corrections actually more valuable than more
clean demonstrations?

InterveneSim-X isolates that comparison. The unit of budget is an action label, not a
trajectory, and the clean and recovery conditions use exactly the same number of added
labels. The goal is not to claim a new state-of-the-art robotics algorithm. It is to build
a small experiment whose causal comparison, negative results, and limitations are visible.

## 2. Experimental design

The environment is robosuite PickPlace with a Panda arm and four object variants: can,
milk, bread, and cereal. A scripted expert generates base demonstrations. A baseline MLP
policy is then rolled out under object shifts, Gaussian action noise, multi-step action
delay, and gripper slip. A privileged supervisor takes over after a disturbance or stalled
progress, producing corrective actions at learner-visited states.

Four policy conditions are compared:

- **baseline:** base clean demonstrations only;
- **more_demos:** baseline data plus an action-budgeted sample of additional clean data;
- **recovery_bc:** baseline data plus the same number of corrective recovery actions;
- **contrastive_recovery:** the recovery data plus a margin loss that pushes predictions
  away from the robot action rejected at the intervention state.

The policy uses a shared MLP trunk with task- and phase-routed action heads. The
35-dimensional state includes end-effector and object geometry, gripper state, goal
geometry, control progress, derived task predicates, and a task one-hot vector. This
augmentation resolves phase ambiguity for tall objects while retaining backward
compatibility with the released 24-dimensional single-task benchmark.

The primary endpoint is mean success over disturbed rollouts at 4,800 added labels.
Training seed is the unit of method-level uncertainty. Five independent training seeds are
evaluated on matched episode seeds. A hierarchical bootstrap resamples training seeds and
episodes. Exact sign permutations are reported as a small-sample diagnostic; with five
seeds, the minimum attainable two-sided value for a same-direction effect is 0.0625.

## 3. Autonomous-policy results

At the largest matched budget, recovery behavior cloning reaches 61.4% disturbed success,
versus 38.6% for additional clean demonstrations and 39.4% for the unaugmented baseline.
The recovery-versus-clean difference is +22.8 percentage points with a 4.3-point standard
deviation across seeds, and recovery wins for every training seed.

The improvement is disturbance-dependent. Recovery behavior cloning reaches 95.6% under
action noise, 60.0% under gripper slip, 60.6% under object shift, and 29.4% under action
delay. The matched clean-data condition reaches 76.2%, 9.4%, 50.6%, and 18.1%,
respectively. Gripper slip is the clearest example of targeted value: clean labels rarely
show how to reacquire a dropped object, while correction data concentrates precisely on
that state distribution.

The reference-seed budget curve improves from 48.4% disturbed success at 500 recovery
labels to 57.0% at 1,500 and 64.1% at 3,000, then measures 60.9% at 4,800. The last point
is not monotonic, which cautions against claiming that every additional correction helps.

## 4. Negative result: rejected-action contrast

The dataset records both the expert correction and the baseline action rejected at each
intervention state. This enables a contrastive loss that attracts the policy to the expert
action and repels it from a sufficiently different rejected action. Despite the additional
structure, the contrastive condition obtains only 39.4% disturbed success at 4,800 labels,
22.0 points below ordinary recovery behavior cloning. It loses to ordinary recovery
training for all five seeds and also reduces nominal performance.

The likely explanation is optimization interference: the rejected action is not always
globally wrong, only wrong in context, and a fixed action-space margin can over-penalize
nearby controls that remain useful during other phases. The result argues for learned
advantages, state-conditional preferences, or ranking objectives with calibrated weights
rather than naive geometric repulsion.

## 5. Learned help seeking

A static risk classifier was initially trained to predict supervisor intervention within
16 steps. Its offline ranking appeared acceptable, but it requested help at reset on every
deployment episode. This is a causal failure: before a randomized disturbance occurs, its
state can be identical to a nominal state, so no state-only classifier can infer the future
random event.

The corrected exploratory detector combines the current observation with its causal
one-step change and predicts intervention within three steps. On episode-disjoint
validation data it reaches AUROC 0.919 and average precision 0.527. The validation-selected
high-recall threshold remains non-selective online, so a complete post-audit sweep over
0.90, 0.95, 0.97, and 0.99 is reported. Threshold 0.97 yields 75.8% assisted disturbed
success while requesting help on 64.8% of disturbed episodes and 3.1% of nominal episodes.
This operating point is exploratory, not confirmatory.

The gate experiment demonstrates why offline detector metrics are insufficient. The
scientifically relevant object is the deployed success-intervention frontier, including
false alarms on nominal trajectories.

## 6. Reproducibility and boundaries

The complete study uses 13,996 base actions, 5,699 additional clean actions, 7,887
corrective actions with paired rejected actions, and 4,233 risk examples. All experiments
run locally with PyTorch MPS or CPU. The runner caches data, checkpoints, and per-condition
episode results, so interrupted multi-hour runs resume without recomputing completed work.
The public bundle includes resolved configuration, seed-level results, paired comparisons,
threshold sweep, figures, and matched video.

The study is state-based simulation with scripted corrections. It does not establish
visual robustness, real-world transfer, human correction quality, safety, or equivalence
to large vision-language-action models. The recovery supervisor bypasses synthetic actuator
corruption after takeover; persistent scene changes remain. These choices define the scope
of the recovery claim.

## 7. Conclusion

Under a controlled action-label budget, corrections gathered on the learner's failure
distribution are substantially more useful than additional clean demonstrations in this
benchmark. The result replicates across five independently trained policies and four object
domains. Two failures are equally important: naive rejected-action contrast hurts, and
high offline risk ranking does not by itself produce selective online assistance. Together,
these findings make InterveneSim-X a compact platform for studying what to label, how to
learn from corrections, and when a deployed robot should ask for help.

## References

1. Ross, S., Gordon, G., and Bagnell, D. *A Reduction of Imitation Learning and
   Structured Prediction to No-Regret Online Learning.* AISTATS, 2011.
   https://proceedings.mlr.press/v15/ross11a.html
2. Kelly, M., Sidrane, C., Driggs-Campbell, K., and Kochenderfer, M. J. *HG-DAgger:
   Interactive Imitation Learning with Human Experts.* ICRA, 2019.
   https://arxiv.org/abs/1810.02890
3. Hoque, R. et al. *ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive
   Imitation Learning.* CoRL, 2021. https://arxiv.org/abs/2109.08273
4. Zhu, Y. et al. *robosuite: A Modular Simulation Framework and Benchmark for Robot
   Learning.* 2020. https://robosuite.ai/
