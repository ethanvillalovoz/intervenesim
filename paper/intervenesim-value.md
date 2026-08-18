# InterveneSim-Value: Will Help Change the Outcome?

Ethan Villalovoz
Independent research project, August 2026

## Abstract

Robots that can ask for expert help still need to decide when that help is useful. Failure
risk is an incomplete target: a policy can be likely to fail when assistance cannot recover
it, or likely to succeed when an unnecessary takeover makes the outcome worse.
InterveneSim-Value constructs a causal target by saving the complete simulator and
controller state at fixed decision steps and replaying two futures: autonomous continuation
and scripted expert takeover. A takeover is helpful only when the assisted branch succeeds
and the matched autonomous branch fails. We train a value gate on 1,505 candidates from
three policy seeds and evaluate 1,517 candidates from two unseen policy seeds. The value
gate reaches 0.817 helpful-intervention AUROC, compared with 0.706 for a temporal failure-
risk gate and 0.703 for ensemble uncertainty. Near a 25% intervention budget, value gating
achieves 66.2% task success at 23.3% realized intervention, versus 60.4% at 24.6% for risk.
An exploratory five-fold policy-seed analysis finds value above risk on all five held-out
seeds, with mean advantages of 8.1 points near 25% intervention and 14.7 points near 50%.
A secondary frozen-encoder visual probe retains ranking signal across cameras but fails to
transfer calibration. The project also contributes deterministic snapshot/restore, a
genuine keyboard-takeover data interface, frozen candidate-level results, and a matched
counterfactual video. Claims are limited to the simulated system and scripted expert.

## 1. Research question

Interactive imitation learning can reduce distribution shift by collecting expert actions
on states visited by the learner [1, 2]. Budget-aware approaches commonly use novelty,
uncertainty, or failure risk to decide when to defer [3]. Those scores answer useful
questions, but not the causal deployment question: if the expert takes over now, will the
eventual task outcome change?

The distinction matters in both directions. A disturbance may make failure likely after the
remaining recovery window has already closed, producing high risk but zero intervention
value. Conversely, an unfamiliar-looking state may still be recoverable by the autonomous
policy, making uncertainty an expensive false alarm. A takeover can even be harmful if the
autonomous branch would have succeeded and the assisted branch fails.

This work asks whether a model trained directly on matched intervention outcomes can produce
a better success-intervention frontier than failure risk or ensemble uncertainty. It builds
on InterveneSim-X, which found that corrective recovery labels improved disturbed success by
22.8 percentage points over equal-count clean labels across five policy seeds. That earlier
study learned from corrections; this study learns when to request them.

## 2. Exact counterfactual construction

At candidate steps 10, 30, 50, 70, 90, 110, and 130, the environment stores a complete
deterministic snapshot. The snapshot contains flattened MuJoCo state, controls, mocap state,
environment clocks, termination state, wrapper random-generator state, controller targets,
observable caches, and the last raw observation. A restore test verifies equality of the
public state and the subsequent transition under a shared action.

Each saved state is replayed into two branches. In the autonomous branch, the learned policy
continues and the original disturbance process remains active. In the assisted branch, the
scripted expert takes control and synthetic actuator corruption is bypassed, matching the
released recovery protocol. Persistent scene state remains. Both terminal outcomes are
saved. The binary target is `assisted_success and not autonomous_success`; signed value is
`assisted_success - autonomous_success`.

This is an exact counterfactual inside the implemented simulator, not a claim of causal
identification in the physical world. The expert, takeover semantics, decision grid, and
disturbance process are part of the estimand.

## 3. Experimental protocol

The benchmark uses robosuite [4] Panda PickPlace tasks with can, milk, bread, and cereal
objects. Deployment conditions are nominal operation, object shift, Gaussian action noise,
multi-step action delay, and gripper slip. Candidate features concatenate the current
35-dimensional task-conditioned state, a causal one-step observation delta, and the
autonomous action. The one-step delta resets at episode boundaries.

The primary value model is a two-layer MLP trained with class-weighted binary cross entropy.
Training candidates come from policies initialized with seeds 27, 127, and 227. Evaluation
candidates come from unseen policies initialized with seeds 327 and 427. Environment reset
seeds are also disjoint, and all candidates from an episode remain in the same split.

Three learned gates share the same candidate opportunities. `value_gate` predicts positive
matched outcome value. `risk_gate` uses the previously released temporal model that predicts
imminent privileged-supervisor takeover. `uncertainty_gate` uses action variance across three
independently trained policies. `never_help`, `always_help`, and an earliest-positive-value
oracle provide controls.

For target budgets of 10%, 25%, 50%, and 75%, score thresholds are computed only from the
maximum candidate score in each training episode. A deployed gate requests help at the first
candidate above its threshold. Primary metrics are task success, realized intervention rate,
useful-request precision, nominal false alarms, regret to a matched-rate causal oracle,
AUROC, and average precision. The full frontier is reported.

## 4. Primary held-out-policy results

The training split contains 1,505 candidate states from 240 episodes; 41.5% have positive
intervention value and 4.3% have negative value. The evaluation split contains 1,517 states
from 240 episodes; 45.4% are helpful and 4.4% are harmful. Positive value is concentrated
in gripper slip (84.2% of evaluation candidates) and action delay (63.8%), but is not absent
from nominal episodes (9.7%). This is why disturbance identity alone is not a sufficient
decision rule.

The value gate ranks helpful candidates with AUROC 0.817 and average precision 0.753.
Failure risk obtains 0.706 and 0.617; ensemble uncertainty obtains 0.703 and 0.643. At the
25% target budget, value reaches 66.2% success while intervening on 23.3% of episodes. Risk
reaches 60.4% at 24.6%, and uncertainty reaches 58.3% at 19.2%. Value requests have 91.1%
positive counterfactual precision, compared with 62.7% for risk.

At the 50% target budget, value reaches 89.6% success at 54.6% intervention, compared with
73.8% at 45.8% for risk and 81.7% at 52.1% for uncertainty. These realized rates are not
identical, so the figure and tables preserve both axes rather than describing the success
gap as a matched-budget treatment effect.

Never asking succeeds on 45.0% of evaluation episodes. Always asking reaches 98.3% but uses
the expert on every episode. The causal oracle reaches 99.6% while requesting help on 54.6%.
At its 54.6% realized intervention point, the learned value gate has 10.0 points of regret
to the matched-rate oracle, leaving meaningful headroom.

## 5. Exploratory cross-seed robustness

The primary split was fixed before result inspection. After the primary audit, a secondary
five-fold leave-one-policy-seed-out analysis was added. Each fold trains value on four
policy seeds, evaluates the fifth, retains episode grouping, and derives all thresholds from
the four training seeds. Because the analysis was added afterward, it is explicitly
exploratory.

Near the 25% target budget, value averages 72.4% success at 25.5% intervention; risk averages
64.3% at 25.1%, and uncertainty averages 68.4% at 25.6%. The paired value-minus-risk
difference is +8.1 points with a 4.2-point standard deviation and is positive on all five
held-out policy seeds. Value exceeds uncertainty on four of five seeds by 4.0 points on
average.

Near the 50% target, value averages 92.7% success at 53.9% intervention. Risk averages 78.0%
at 51.4% and uncertainty averages 84.7% at 49.9%. Value beats both on every held-out seed:
+14.7 +/- 3.1 points over risk and +7.9 +/- 2.8 over uncertainty. The small number of policy
seeds supports consistency and effect-size reporting, not asymptotic significance claims.

## 6. Secondary visual and human interfaces

The primary result uses simulator state to isolate the causal question. A secondary visual
pilot tests whether the label can support perception research. A frozen ImageNet ResNet-18
[5] encodes 160 x 120 RGB observations. A small learned head combines its 512 features with
robot state, control progress, and task identity while withholding object and goal
coordinates. Training uses 249 front-view images; evaluation uses 512 images from an unseen
policy seed across front-view and agent-view cameras.

Front-view AUROC is 0.712 with 80.6% recall at threshold 0.5. Zero-shot agent-view AUROC is
0.696, indicating that some ranking signal transfers. Calibration does not: only 1.2% of
agent-view candidates score above 0.5 and recall falls to 1.9%. This is a useful negative
result and not evidence of a deployable visual gate. Lighting and textures were not varied.

The release also includes `capture-human-corrections`. A learned policy controls the
simulator until a person presses `T`, after which keyboard Cartesian and gripper commands
are recorded with the exact takeover step and a per-action human mask. Human archives use
a distinct schema and are never silently mixed with scripted data. No participants were
recruited and no human-performance result is claimed.

## 7. Threats to validity

The largest limitation is external validity. MuJoCo state restore makes paired outcomes
exact for the implemented dynamics, but physical systems cannot generally be rewound. The
scripted expert has privileged geometry and deterministic logic. Assisted branches bypass
synthetic actuator corruption, which represents a reliable expert control channel rather
than a shared hardware failure. Different experts or takeover channels define different
values.

Candidate decisions occur on a fixed grid rather than continuously. Episode-level budgets
are targeted through training score quantiles, so realized evaluation rates can differ.
Primary uncertainty is also limited by five total policy initializations. The post-audit
cross-validation reuses all seeds across different folds and is therefore not independent
replication.

The visual dataset is small and uses a pretrained encoder. It probes representation and
calibration, not end-to-end visual control. No result establishes real-robot safety,
sim-to-real transfer, human intervention quality, or robustness to lighting and texture.

## 8. Reproducibility

The entire pipeline runs locally on an Apple M4 Pro with 64 GB unified memory using PyTorch
MPS; CPU is supported and CUDA is not required. All software is free, dependencies are
locked, and no cloud service, paid API, or physical robot is used. Collection is sharded by
policy seed and task. Datasets, model checkpoints, and completed folds are cached so an
interrupted run resumes without discarding completed work.

The public artifact freezes candidate-level branch outcomes, gate episode records, resolved
configurations, thresholds, metrics, cross-seed tables, plots, a matched counterfactual
video, model and dataset cards, and tests that assert the reported primary and exploratory
claims. Large observation tensors and checkpoints are regenerated locally.

## 9. Conclusion

Failure risk and uncertainty are not the same quantity as intervention value. In this
benchmark, learning directly from matched autonomous and assisted futures produces a better
held-out-policy success-intervention tradeoff, more precise requests, and consistent
advantages across five exploratory leave-one-policy-seed-out folds. Exact simulation forks
make the causal target inspectable; the visual calibration failure and strict claim
boundaries keep the result honest. InterveneSim-Value is a compact platform for studying
not only how robots recover from correction, but when they should ask for it.

## References

1. Ross, S., Gordon, G., and Bagnell, D. A Reduction of Imitation Learning and Structured
   Prediction to No-Regret Online Learning. AISTATS, 2011.
   https://proceedings.mlr.press/v15/ross11a.html
2. Kelly, M., Sidrane, C., Driggs-Campbell, K., and Kochenderfer, M. J. HG-DAgger:
   Interactive Imitation Learning with Human Experts. ICRA, 2019.
   https://arxiv.org/abs/1810.02890
3. Hoque, R. et al. ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive
   Imitation Learning. CoRL, 2021. https://arxiv.org/abs/2109.08273
4. Zhu, Y. et al. robosuite: A Modular Simulation Framework and Benchmark for Robot
   Learning. 2020. https://robosuite.ai/
5. He, K., Zhang, X., Ren, S., and Sun, J. Deep Residual Learning for Image Recognition.
   CVPR, 2016. https://arxiv.org/abs/1512.03385
