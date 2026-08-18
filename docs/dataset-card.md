# InterveneSim-Value dataset card

## Summary

InterveneSim-Value generates matched outcome pairs at fixed intervention candidates in
robosuite's four Panda PickPlace object domains. The complete simulator and controller state
is saved once, then replayed into autonomous-continuation and scripted-expert branches.

The repository publishes aggregate and candidate-level tables. Compressed observations,
images, and checkpoints are regenerated locally and excluded from Git because of size.

## Primary counterfactual data

| Split | Policy seeds | Episodes | Candidate states | Helpful | Harmful |
|---|---|---:|---:|---:|---:|
| Train | 27, 127, 227 | 240 | 1,505 | 624 (41.5%) | 65 (4.3%) |
| Evaluation | 327, 427 | 240 | 1,517 | 688 (45.4%) | 66 (4.4%) |

`helpful` means assisted success and autonomous failure. `harmful` means autonomous success
and assisted failure. Both branch outcomes and signed value are retained, preventing the
binary label from hiding unsuccessful takeovers.

Each row includes episode and policy seed, task, disturbance, candidate step, autonomous
and assisted success, the helpful label, signed value, and ensemble uncertainty. Local NPZ
archives additionally contain the state observation, causal one-step delta, and autonomous
action used by the value gate.

## Visual pilot data

The secondary dataset contains 498 training images and 512 held-out-policy images across
front-view and agent-view cameras. The head trains on 249 front-view images. Each image is
paired with robot proprioception that excludes object and goal coordinates and inherits the
same counterfactual outcome semantics.

This is a small representation probe. It is not a large-scale vision dataset, and it does
not vary lighting or textures.

## Human correction schema

`capture-human-corrections` produces a separate NPZ archive containing observation, chosen
action, episode, step, a boolean human-control mask, takeover timing, and episode metadata.
Human and scripted data are never silently pooled. No human dataset is distributed with
v0.3 and no human-study result is claimed.

## Intended use and limitations

The data supports controlled studies of intervention value, optimal stopping, failure risk,
uncertainty, and simulated corrective control. It should not be treated as real-robot data,
human-preference data, a safety validation set, or evidence of sim-to-real transfer.

## Reproduction

Run `uv run intervenesim research --config configs/research.yaml` to produce prerequisite
policies, followed by `uv run intervenesim value-research --config configs/value.yaml`.
Frozen public tables and resolved configurations live under `results/intervenesim-value`.
