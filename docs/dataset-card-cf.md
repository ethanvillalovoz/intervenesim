# InterveneSim-CF factual-log dataset card

## Purpose

These logs support evaluation of intervention-effect estimators under the information available
in deployment: one context, one treatment decision, and one realized terminal outcome.

## Schema

| Field | Meaning |
|---|---|
| `contexts` | Observation, one-step observation change, and autonomous action |
| `treatments` | Whether assistance was requested |
| `outcomes` | Terminal success for the selected treatment only |
| `propensities` | Known probability that the logger selected assistance |
| `episode_ids` | Source episode identifier |
| `policy_seeds` | Autonomous-policy training seed |
| `tasks` | Manipulated object/task |
| `disturbances` | Deployment condition |
| `steps` | Candidate decision step |

The files intentionally exclude autonomous and assisted potential-outcome pairs. A machine-read
leakage audit is published with every benchmark run.

## Construction

One candidate is sampled per source episode. The treatment is drawn from either a randomized or
uncertainty-selective logging policy, and only the realized arm is copied from the exact simulator
fork archive. The full protocol uses five logging seeds for each policy.

## Appropriate uses

- Testing causal meta-learners and off-policy corrections.
- Measuring sensitivity to logging-policy selection bias.
- Auditing estimated effects against a separately held exact-fork evaluator.
- Studying help-request policies under fixed budgets and explicit intervention costs.

## Inappropriate uses

- Claiming effects for a human operator.
- Treating rows as real-robot trials.
- Training on the paired archive and describing the result as single-world learning.
- Treating candidate rows or repeated logging seeds as independently collected physical trials.

## Ethical and safety notes

No human participants, physical robots, or safety-critical deployment are involved. The scripted
assistant and simulated disturbances are abstractions. Results must not be interpreted as a
certification of autonomous-system safety.
