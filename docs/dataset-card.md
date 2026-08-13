# InterveneSim-X dataset card

## Summary

InterveneSim-X generates state-action data in robosuite's four single-object Panda
PickPlace variants. The public repository contains aggregate and episode-level experiment
records; the larger trajectory archives are regenerated locally because they depend on
simulator execution and are excluded from Git.

## Generated training data

| Split | Samples | Purpose |
|---|---:|---|
| Base clean demonstrations | 13,996 | Shared initial behavior-cloning data |
| Additional clean pool | 5,699 | Equal-budget clean control |
| Corrective recovery pool | 7,887 | Learner-state expert corrections |
| Paired rejected actions | 7,887 | Robot actions rejected at correction states |
| Risk examples | 4,233 | Pre-intervention and successful nominal states |
| Positive 16-step risk labels | 950 | Supervisor takeover within the original horizon |
| Positive 3-step detector labels | 231 | Imminent takeover for the temporal gate |

Every recovery action includes task, phase, disturbance, episode, intervention flag, and a
paired rejected action. Sample-budget selection is stratified across task and disturbance.

## Observation and action schema

The expanded 35-dimensional observation includes robot end-effector position, gripper
state, object and goal position, relative geometry, joint velocity, control progress,
lifted/proximity predicates, object geometry, and task one-hot identity. The policy predicts
three Cartesian delta commands plus gripper control.

The original 24-dimensional single-task schema remains loadable. Dataset loaders are
backward-compatible with the released v1 archives.

## Collection

A scripted expert supplies successful clean trajectories. The baseline policy is deployed
under object shift, action noise, action delay, and gripper slip. A privileged supervisor
triggers on a fired disturbance, stalled progress, or a hard rollout horizon. Once takeover
begins, the collector records the expert correction and the baseline action it rejected.

## Intended use

The data supports controlled studies of behavior cloning, recovery learning, intervention
timing, and rejected-action objectives. It should not be treated as human teleoperation
data, real-robot data, a safety validation set, or evidence of sim-to-real transfer.

## Reproduction

Run `uv run intervenesim research --config configs/research.yaml`. Generated trajectory
archives appear under `artifacts/runs/research-v1/datasets`; checkpoints appear beside them.
The run is resumable and the resolved configuration is preserved with the outputs.
