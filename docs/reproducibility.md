# Reproducing InterveneSim-X

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
```

The full run stores data, checkpoints, and results under `artifacts/runs/research-v1`.
Existing complete artifacts are loaded on restart, so a failed or interrupted run resumes
at the next unfinished condition. Do not delete the run directory between resumptions.

## Published configuration

The frozen resolved configuration is
[`results/intervenesim-x/config.resolved.yaml`](../results/intervenesim-x/config.resolved.yaml).
It defines:

- training seeds `27, 127, 227, 327, 427`;
- label budgets `500, 1500, 3000, 4800`;
- object tasks `can, milk, bread, cereal`;
- eight matched evaluation episodes per task and disturbance;
- 80 baseline epochs, 35 fine-tuning epochs, and 50 risk epochs;
- MLP widths `256, 256, 128` and risk widths `128, 128`.

## Rebuild the paper and video

After running the benchmark, install the paper-only dependencies and rebuild:

```bash
uv sync --locked --extra paper
uv run python scripts/build_paper.py
uv run intervenesim record-research-comparison \
  --run artifacts/runs/research-v1 \
  --output artifacts/videos/intervenesim-x.mp4 \
  --task cereal \
  --disturbance gripper_slip
```

The PDF builder uses ReportLab. Visual verification uses Poppler rendering, `pdfplumber`,
and `pypdf`; those paper-only tools are not required for the simulator or benchmark.

## Interpretation checklist

- Treat training seed—not episode—as the primary unit of method uncertainty.
- Do not call the budget curve monotonic; the reference seed declines at 4,800 labels.
- Treat the rejected-action contrastive objective as a negative result.
- Treat the threshold sweep as exploratory because it followed the help-gate audit.
- Do not infer visual robustness, real-world transfer, human-label quality, or safety.
