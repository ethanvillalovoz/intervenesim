# InterveneSim-X research protocol

## Research questions

InterveneSim-X extends the released single-seed benchmark around three predeclared
questions:

1. **Correction value:** under a fixed number of additional action labels, do recovery
   corrections improve disturbed autonomous success more than clean demonstrations across
   independently trained policies?
2. **Correction structure:** does learning from the robot action rejected at an
   intervention improve over treating the correction as ordinary behavior-cloning data?
3. **Help seeking:** can a learned risk model request expert control before failure while
   using fewer interventions than always-on assistance?

The primary autonomous endpoint is mean disturbed success across training seeds. The
primary help-seeking endpoint is assisted success as a function of intervention rate.

## Data and method conditions

Every policy seed uses the same base demonstrations and augmentation pools. The clean and
recovery conditions are subsampled to identical added-action budgets.

- `baseline`: base clean demonstrations only.
- `more_demos`: baseline plus additional clean actions.
- `recovery_bc`: baseline plus corrective actions, trained with ordinary MSE behavior
  cloning.
- `contrastive_recovery`: the same samples as `recovery_bc`, with a margin loss that also
  moves the prediction away from the baseline action rejected at the corrected state.

The contrastive loss is applied only when the rejected and corrective actions differ by a
minimum distance. This prevents artificial repulsion when the supervisor intervenes even
though the baseline action already agrees with the expert.

## Learned intervention gate (exploratory correction)

The recovery collector preserves pre-intervention rollout states from both successful and
unsuccessful expert-takeover attempts. The predeclared static, 16-step predictor ranked its
episode-disjoint validation examples but requested help at reset on every deployment
episode. That is a causal error: before a randomized disturbance occurs, the state may be
identical to a nominal state, so the model cannot infer the future disturbance.

The corrected exploratory detector combines each current observation with its causal
one-step change and predicts an imminent intervention within three steps. Earlier states
and nominal successful rollouts are negative examples. Training uses episode-disjoint
validation data, class-balanced binary cross entropy, and a validation-selected threshold
targeting high recall. The failed static gate is retained as an explicit negative finding;
the correction is not presented as predeclared evidence.

After observing the high-recall gate's nominal false alarms, thresholds 0.90, 0.95, 0.97,
and 0.99 are evaluated as a complete exploratory operating-point sweep. These thresholds
were selected after diagnosis and are therefore descriptive rather than confirmatory.

The gate is evaluated without privileged state access. Once it requests help, the scripted
expert takes over and synthetic actuator corruption is bypassed, matching the correction
collection protocol. Reported metrics include success, intervention rate, nominal false
alarm rate, request timing, risk AUROC, average precision, Brier score, and calibration.
Help requests are disabled during the first five control steps because the benchmark's
privileged supervisor cannot yet trigger under any disturbance. This causal warm-up also
prevents a reset-distribution classifier transient from being mistaken for a selective
request.

## Generalization

The multi-domain benchmark uses the four single-object robosuite `PickPlace` variants:
can, milk, bread, and cereal. These share the Panda embodiment but vary object geometry,
initial placement distribution, and target bin. Each observation includes object geometry
and a task identifier. Generalization is evaluated over held-out disturbance strengths and
unseen combinations; object-domain leave-one-out experiments are secondary because the
task identifier itself cannot provide semantic information for an unseen category.

## Statistics

- At least five independent policy-training seeds for final claims.
- Evaluation episodes are seed-matched across methods within each training seed.
- Per-seed success is the unit used for method-level uncertainty.
- Report mean, standard deviation, paired seed-level differences, and a hierarchical
  bootstrap interval that resamples training seeds and episodes.
- Per-episode McNemar tests are retained only as conditional diagnostics for a fixed pair
  of trained policies, not as the primary evidence across training randomness.

## Compute and interpretation boundaries

All reference experiments must run on Apple Silicon or CPU using free software. The
project does not claim real-world transfer, safety, equivalence between scripted and human
corrections, or evidence about large vision-language-action models. Visual experiments,
when included, are reported separately from the state-based primary result.

The multi-domain observation augments the original 24-D benchmark state with control
progress, derived lifted/proximity predicates, object geometry, and a task one-hot vector.
This makes phase-dependent commands identifiable for tall objects while leaving the
released single-task benchmark and checkpoints backward compatible.
