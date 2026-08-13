# InterveneSim-Value research protocol

## Primary question

Can a learned gate predict the **causal value of expert takeover** more efficiently than a
gate that predicts failure risk?

InterveneSim-X showed that corrective data improves autonomous policies, but its risk gate
also showed that offline classification quality does not imply selective online help. This
study changes the target. Instead of asking whether failure is likely, it asks whether
expert takeover changes the outcome from failure to success from the same physical state.

## Counterfactual label

At fixed candidate steps on an autonomous rollout, the complete simulator and controller
state is snapshotted and replayed into two matched futures:

1. **autonomous branch:** the policy continues under the original disturbance;
2. **assisted branch:** the scripted expert takes over and synthetic actuator corruption is
   bypassed, matching the released recovery protocol.

The binary helpful-intervention label is

```text
helpful = assisted_success and not autonomous_success
```

The signed outcome value is `assisted_success - autonomous_success`, so harmful takeover
remains representable. Both raw outcomes are retained; they are never reconstructed from a
single label.

## Splits and leakage controls

- Entire episodes, including every candidate state derived from them, remain in one split.
- Training and evaluation use disjoint environment reset seeds.
- The primary generalization test trains the value model on counterfactual branches from
  policy seeds 27, 127, and 227 and evaluates it on unseen policy seeds 327 and 427.
- Candidate decisions occur on a fixed 20-step grid beginning at step 10 and ending at
  step 130. All gates see the same opportunities.
- A one-step observation delta is causal and resets at episode boundaries.
- Thresholds are selected from the training split to meet predeclared assistance budgets;
  evaluation thresholds are never tuned against evaluation outcomes.

## Methods

- `never_help`: complete the autonomous branch.
- `always_help`: request expert takeover at the first candidate.
- `risk_gate`: the released temporal failure-risk score.
- `uncertainty_gate`: disagreement among independently trained policies.
- `value_gate`: predicted probability that takeover changes failure into success.
- `oracle_value`: earliest candidate with a positive matched counterfactual label.

The primary comparison is `value_gate` versus `risk_gate` at matched **realized evaluation
intervention rates**. Results are presented as a complete success-intervention frontier,
not as one favorable threshold.

## Primary metrics

1. assisted success at target training-score budgets of 10%, 25%, 50%, and 75%;
2. regret to the counterfactual oracle at comparable intervention rate;
3. precision of help requests: fraction of requested episodes where takeover is helpful;
4. nominal false-alarm rate;
5. helpful-intervention AUROC and average precision on episode-disjoint evaluation data.

Training policy seed is the unit of uncertainty for final method comparisons. Episode
bootstrap intervals are secondary because decision candidates within an episode are
correlated.

## Secondary extensions

### Vision

A frozen compact visual encoder is evaluated using RGB plus proprioception. Visual results
are secondary and must report held-out camera, lighting, texture, and object variations.
State-based counterfactual value remains the primary causal experiment.

### Human correction capture

A keyboard/gamepad interface records actual takeover timing and actions in simulation.
Human data is kept distinct from scripted labels and is not silently pooled into the
primary result.

## Boundaries

The counterfactual is exact only for the simulated system and the scripted expert. It does
not identify the value of a particular human intervention, real-robot safety, or sim-to-real
transfer. The candidate grid is a benchmark choice rather than a claim about continuous
optimal stopping.
