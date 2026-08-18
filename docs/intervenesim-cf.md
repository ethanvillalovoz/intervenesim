# InterveneSim-CF protocol

## Question

Can a robot learn when assistance will change its outcome when a deployment log contains only
the outcome of the action that was actually taken?

InterveneSim-Value v0.3 answered an easier question: if both simulator futures are available at
every training state, can a value model rank useful takeovers? InterveneSim-CF v0.4 removes that
privileged supervision. Exact forks become a hidden evaluator rather than a source of training
labels.

## Separation between learner and evaluator

The two data schemas form a deliberate information boundary:

- `CounterfactualData` contains autonomous and assisted potential outcomes. Only the logger and
  evaluator may read it.
- `LoggedBanditData` contains context, chosen treatment, realized outcome, and known logging
  propensity. It cannot serialize either potential-outcome field.

The public leakage audit opens every frozen log and rejects the fields `autonomous_success`,
`assisted_success`, `helpful`, and `signed_value`.

## Logged-decision construction

Each source episode contributes one uniformly sampled intervention candidate. This avoids
pretending that all seven candidate states could independently reach a terminal outcome in one
deployed trajectory. A logging policy then chooses autonomous continuation or immediate scripted
assistance.

Two policies are evaluated:

1. **Randomized:** assistance probability is 0.5 for every context.
2. **Uncertainty-selective:** probability depends on policy-ensemble uncertainty and is clipped
   to `[0.1, 0.9]` to preserve overlap.
3. **Adaptive-value:** begins with a randomized warm-up, refits an S-learner every 60 episodes
   using only past factual outcomes, and uses an epsilon-soft treatment policy based on predicted
   benefit. Every online propensity is retained for later correction.

Only the selected arm is copied into the factual log. Five independently sampled logging seeds
measure sensitivity to which decision times and outcomes happened to be observed.

## Estimators

- **S-learner:** one outcome model with treatment as an input.
- **T-learner:** separate autonomous and assisted outcome models with inverse-propensity weights.
- **IPW learner:** regresses an inverse-propensity pseudo-outcome.
- **DR learner:** cross-fits the two outcome models, constructs an augmented inverse-propensity
  pseudo-outcome, and regresses the individual treatment effect.
- **Reversibility proxy:** predicts autonomous failure and assumes that requesting help repairs
  it. This is a PAINT-inspired risk/reversibility baseline, not a reproduction of PAINT.
- **Paired-label oracle:** trains on both exact futures. It is privileged and reported only as an
  upper-bound comparator.

The neural heads are deliberately small and run on CPU or Apple Metal. No CUDA or paid service is
required.

## Evaluation

The evaluator reveals exact held-out potential outcomes only after fitting. It reports:

- precision in estimation of heterogeneous effect (PEHE);
- error in average treatment effect;
- AUROC and average precision for helpful interventions;
- harmful-intervention ranking;
- task success, realized intervention rate, and useful-request precision;
- cost-adjusted utility;
- paired differences over logging seeds;
- logged-data learning curves;
- leave-one-task-out and leave-one-disturbance-out performance.
- exact assistance-latency shifts at zero, one, and two candidate intervals.

The primary comparison score-ranks evaluation episodes and allocates exactly the same intervention
budget to every method without reading outcomes. A secondary calibration stress test freezes
thresholds from unlabeled training contexts and reports the resulting deployment rate separately.
Policy evaluation uses policy seeds 327 and 427, which are absent from all training logs.

## Commands

```bash
uv run intervenesim selective-research --config configs/selective.yaml
```

The quick pipeline check is:

```bash
uv run intervenesim selective-research \
  --config configs/selective-smoke.yaml \
  --output artifacts/runs/selective-smoke
```

## Boundaries

The benchmark measures estimator error against exact simulator potential outcomes. It does not
identify a human operator's individual effect, establish real-robot safety, or demonstrate
sim-to-real transfer. The scripted assistant is one fixed treatment, and assistance begins
immediately in the training estimand. The latency audit changes only arrival time along the exact
autonomous trajectory; it does not model different human skills. Assistant identity, partial
takeovers, temporal vision, and online policy improvement remain follow-on experiments rather
than being silently folded into the primary claim.
