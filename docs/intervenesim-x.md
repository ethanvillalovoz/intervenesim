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

## Learned intervention gate

The recovery collector preserves pre-intervention rollout states. A state receives a
positive risk label if the privileged supervisor will intervene within a fixed prediction
horizon. Earlier states and nominal successful rollouts are negative examples. Training
uses episode-disjoint validation data, class-balanced binary cross entropy, and a
validation-selected threshold targeting high recall.

The gate is evaluated without privileged state access. Once it requests help, the scripted
expert takes over and synthetic actuator corruption is bypassed, matching the correction
collection protocol. Reported metrics include success, intervention rate, nominal false
alarm rate, request timing, risk AUROC, average precision, Brier score, and calibration.

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
