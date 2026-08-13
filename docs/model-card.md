# InterveneSim-X model card

## Policy

The expanded policy is a compact PyTorch MLP with a shared trunk and task- and phase-routed
action heads. A phase classifier supplies the route at deployment. Training uses mean
squared behavior-cloning loss; the contrastive ablation adds a fixed-margin penalty away
from paired rejected actions.

The policy receives 35-dimensional state observations and emits four continuous commands:
Cartesian end-effector translation and gripper control. It does not consume camera pixels,
language, privileged simulator state, disturbance identity, or an intervention flag.

## Risk gate

The exploratory risk network is a two-layer MLP over the current observation concatenated
with its causal one-step delta. It predicts privileged supervisor takeover within three
steps. Evaluation must call `reset()` between episodes so the temporal delta cannot cross
episode boundaries.

Episode-disjoint validation metrics are AUROC 0.919, average precision 0.527, Brier score
0.083, and recall 0.889 at the validation-selected threshold. That threshold is not
selective online. The full deployment threshold sweep is published because online
success-intervention tradeoffs—not offline AUROC alone—determine usefulness.

## Evaluation

The primary policy result uses five independent policy-training seeds, four object domains,
four disturbances, and matched evaluation episode seeds. At 4,800 added labels, recovery
behavior cloning reaches 61.4% disturbed success, compared with 38.6% for equal-count clean
labels. Contrastive recovery reaches 39.4% and is a negative result.

## Limitations

These models are research artifacts for state-based simulation. They are not safe control
systems, real-robot policies, visual policies, or general manipulation models. The scripted
expert and privileged supervisor encode benchmark-specific knowledge. Deployment outside
the exact simulator protocol is unsupported.
