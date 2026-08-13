# Benchmark protocol

## Research question

Under an equal additional-sample budget, does targeted recovery data improve a
behavior-cloned manipulation policy's robustness more than additional clean expert
demonstrations?

## Hypothesis

A policy trained on successful demonstrations alone encounters covariate shift after a
mistake. Expert corrections beginning in those off-distribution states should improve
disturbed-task success and autonomous recovery more efficiently than clean demonstrations
that revisit nominal states.

## Task

- Simulator: MuJoCo through robosuite 1.5
- Robot: Franka Emika Panda
- Environment: `PickPlaceCan`
- Control: normalized operational-space pose deltas plus a binary gripper command
- Observation: compact task state (end-effector pose, gripper state, can pose, target pose,
  and relative geometry)
- Success: robosuite's binary task success criterion

The first release deliberately uses state observations. This isolates the recovery-data
hypothesis from visual representation learning and makes the benchmark practical on
CPU-only and Apple Silicon systems. Pixel observations are a documented extension, not a
hidden requirement.

## Conditions

All learned conditions begin from the same baseline checkpoint and base dataset.

1. **Baseline**: clean successful demonstrations only.
2. **More demos**: baseline data plus additional clean expert samples.
3. **Recovery data**: baseline data plus expert recovery samples collected after a
   baseline rollout reaches a risky or failed state.

Conditions 2 and 3 receive the same number of additional action-labeled samples. Recovery
samples are truncated deterministically when necessary to enforce this budget.

## Failure families

- `object_shift`: move the can during the approach phase.
- `action_noise`: add bounded Gaussian noise to Cartesian policy actions.
- `action_delay`: execute actions after a fixed control delay.
- `gripper_slip`: release and relocate the can after it has been lifted.
- `mixed`: sample one of the preceding failures per episode.

Each disturbance is deterministic for a given episode seed. The benchmark records its
type, parameters, trigger step, and whether it actually fired.

## Intervention collection

The learned baseline begins each disturbed rollout. A simulator-state supervisor monitors
task progress. It triggers when a configured disturbance fires and the rollout enters an
off-nominal state, or when task progress stalls beyond a fixed horizon. At that point, a
scripted expert takes control from the current physical state and attempts to recover.
The supervisor's corrective command bypasses synthetic action corruption after takeover;
one-shot physical-state disturbances remain in the scene. This isolates the value of
labeling recovery states from robust control under a continuously corrupted actuator
channel.

Only the expert-controlled recovery segment is added to the intervention dataset. The
supervisor and expert may use privileged simulator state during data collection; learned
policies do not receive the intervention flag or expert phase.

## Primary metrics

- Nominal success rate
- Disturbed success rate by failure family
- Recovery success rate after a disturbance fires
- Mean completion steps among successful episodes
- Autonomous success per 1,000 additional labeled actions
- Wilson 95% confidence intervals for per-condition success rates
- Exact paired McNemar test over seed-matched disturbed episodes

## Reproducibility

- Python and dependency versions are locked with `uv.lock`.
- Each command accepts a seed and stores its resolved configuration.
- Dataset files include provenance and schema metadata.
- Checkpoints include normalization statistics and training configuration.
- Evaluation writes one row per episode before generating aggregate tables.
- Fast smoke settings exercise the full pipeline in continuous integration.

## Interpretation boundaries

This benchmark evaluates simulated, state-based manipulation. It does not claim real-world
transfer, visual robustness, safety certification, or equivalence between scripted and
human interventions. Its purpose is to test a focused data-efficiency claim in a setting
that is inexpensive and reproducible.
