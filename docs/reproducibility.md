# Reproducing InterveneSim-CF

## Reference environment

The published experiment was run locally on a Mac mini with an Apple M4 Pro and 64 GB of
unified memory using PyTorch MPS. CUDA is not required. CPU is supported but slower. All
dependencies are free and locked in `uv.lock`.

## Setup

```bash
git clone https://github.com/ethanvillalovoz/intervenesim.git
cd intervenesim
uv sync --locked --extra dev
uv run intervenesim doctor
uv run pytest
```

`doctor` constructs the simulator, reports the selected PyTorch accelerator, and verifies
that the scripted expert completes PickPlaceCan.

## Fast end-to-end validation

```bash
uv run intervenesim smoke
uv run intervenesim research \
  --config configs/research-smoke.yaml \
  --output artifacts/runs/research-smoke
```

## Full experiment

```bash
uv run intervenesim research --config configs/research.yaml
uv run intervenesim value-research --config configs/value.yaml
uv run intervenesim selective-research --config configs/selective.yaml
```

The first command builds the prerequisite policies under `artifacts/runs/research-v1`.
The second collects matched counterfactual branches and stores the v0.3 experiment under
`artifacts/runs/value-v1`. The third hides one branch per logged decision, trains causal
estimators, and audits them against the held-out exact forks. It writes the frozen v0.4 result to
`results/intervenesim-cf`.
Existing complete artifacts are loaded on restart, so a failed or interrupted run resumes
at the next unfinished condition. Do not delete the run directory between resumptions.

## Primary v0.3 configuration

The frozen resolved configuration is
[`results/intervenesim-value/config.resolved.yaml`](../results/intervenesim-value/config.resolved.yaml).
It defines three training-policy seeds, two unseen evaluation-policy seeds, four objects,
five deployment conditions, seven fixed candidate steps, and four target intervention
budgets. Thresholds are derived only from training-policy episodes.

The five-fold policy-seed analysis in the frozen results is explicitly exploratory because
it was added after inspection of the primary three-seed-train/two-seed-test audit.

## v0.4 single-world configuration

The resolved protocol is
[`results/intervenesim-cf/config.resolved.yaml`](../results/intervenesim-cf/config.resolved.yaml).
It produces randomized, uncertainty-selective, and online adaptive-value logs for five logging
seeds. Each log exposes one candidate and one observed outcome per source episode. The leakage
audit rejects any serialized potential-outcome field. Equal-budget evaluation is primary;
training-threshold results are a separate calibration stress test.

## Optional vision and human capture

```bash
uv sync --locked --extra vision
uv run intervenesim visual-research --config configs/visual.yaml
uv run intervenesim capture-human-corrections \
  --checkpoint artifacts/runs/research-v1/checkpoints/seed-27/baseline.pt \
  --output artifacts/human/corrections.npz
```

The visual command downloads free ImageNet ResNet-18 weights on first use. The human command
opens a local simulator window: press `T` to take over and `Esc` to end the episode. Human
archives use a separate schema and are not part of the published scripted result.

## Earlier v0.2 configuration

The frozen resolved configuration is
[`results/intervenesim-x/config.resolved.yaml`](../results/intervenesim-x/config.resolved.yaml).
It defines:

- training seeds `27, 127, 227, 327, 427`;
- label budgets `500, 1500, 3000, 4800`;
- object tasks `can, milk, bread, cereal`;
- eight matched evaluation episodes per task and disturbance;
- 80 baseline epochs, 35 fine-tuning epochs, and 50 risk epochs;
- MLP widths `256, 256, 128` and risk widths `128, 128`.

## Rebuild the papers and videos

After running the benchmark, install the paper-only dependencies and rebuild:

```bash
uv sync --locked --extra paper
uv run python scripts/build_paper.py
uv run python scripts/build_value_paper.py
uv run python scripts/build_cf_paper.py
uv run intervenesim record-value-counterfactual \
  --run artifacts/runs/value-v1 \
  --output artifacts/videos/intervenesim-value-fork.mp4
uv run intervenesim record-research-comparison \
  --run artifacts/runs/research-v1 \
  --output artifacts/videos/intervenesim-x.mp4 \
  --task cereal \
  --disturbance gripper_slip
```

The PDF builder uses ReportLab. Visual verification uses Poppler rendering, `pdfplumber`,
and `pypdf`; those paper-only tools are not required for the simulator or benchmark.

## Interpretation checklist

- Treat every v0.4 hypothesis as exploratory; the protocol is timestamped internally but was not
  externally preregistered.
- Use score-ranked exact budgets for method comparisons and the predeployment table for threshold
  calibration claims.
- Treat logging seed—not candidate row—as the primary unit of v0.4 method uncertainty.
- Do not call the reversibility proxy a PAINT reproduction.
- Treat adaptive logging as selective data collection, not online control-policy improvement.
- Treat the v0.3 three-seed-train/two-seed-test split as primary.
- Treat five-fold value-gate comparisons as exploratory.
- Compare gates using both realized intervention rate and task success.
- Do not equate failure probability with positive intervention value.
- Treat cross-camera visual ranking and cross-camera calibration as separate findings.
- Treat training seed—not episode—as the primary unit of method uncertainty.
- Do not call the budget curve monotonic; the reference seed declines at 4,800 labels.
- Treat the rejected-action contrastive objective as a negative result.
- Treat the threshold sweep as exploratory because it followed the help-gate audit.
- Do not infer real-world transfer, human-label quality, or safety.
