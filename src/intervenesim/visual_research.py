from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from intervenesim.benchmark import system_manifest
from intervenesim.config import VisualConfig
from intervenesim.visual import (
    VisualData,
    collect_visual_counterfactuals,
    run_visual_probe,
)

ProgressCallback = Callable[[str], None]


def run_visual_research(
    config: VisualConfig,
    output_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    if output_dir is not None:
        config = replace(config, output_dir=str(output_dir))
    output = Path(config.output_dir)
    datasets = output / "datasets"
    results = output / "results"
    datasets.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    started = time.time()
    split_parts: dict[str, list[VisualData]] = {"train": [], "eval": []}
    for split, policy_seed in (
        ("train", config.train_policy_seed),
        ("eval", config.eval_policy_seed),
    ):
        checkpoint = (
            Path(config.research_run) / "checkpoints" / f"seed-{policy_seed}" / "baseline.pt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing prerequisite policy checkpoint: {checkpoint}")
        for task_index, task in enumerate(config.tasks):
            path = datasets / f"{split}-{task}.npz"
            if path.exists():
                data = VisualData.load(path)
            else:
                _emit(progress, f"collecting {split} RGB counterfactuals / {task}")
                data = collect_visual_counterfactuals(
                    checkpoint,
                    task,
                    config.disturbances,
                    config.episodes_per_task,
                    config.seed + (0 if split == "train" else 1_000_000) + task_index * 100_000,
                    config.candidate_steps,
                    config.max_steps,
                    config.cameras,
                    device=config.device,
                )
                data.save(path)
            split_parts[split].append(data)
    train = VisualData.concatenate(split_parts["train"])
    evaluation = VisualData.concatenate(split_parts["eval"])
    _emit(progress, "extracting frozen visual features and training value probe")
    probe = run_visual_probe(
        train,
        evaluation,
        results,
        epochs=config.epochs,
        seed=config.seed,
        device=config.device,
    )
    write_visual_montage(evaluation, results / "visual_camera_montage.png")
    report = _report(train, evaluation, probe)
    (results / "report.md").write_text(report, encoding="utf-8")
    manifest: dict[str, object] = {
        "elapsed_seconds": time.time() - started,
        "config": config.to_dict(),
        "system": system_manifest(),
        "train_samples": train.sample_count,
        "eval_samples": evaluation.sample_count,
        "probe": probe,
        "report": str(results / "report.md"),
        "montage": str(results / "visual_camera_montage.png"),
    }
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _emit(progress, f"visual value pilot complete: {results / 'report.md'}")
    return manifest


def _report(train: VisualData, evaluation: VisualData, probe: dict[str, object]) -> str:
    metrics = probe["metrics"]
    rows = [
        {
            "camera": camera,
            "AUROC": values["auroc"],
            "average_precision": values["average_precision"],
        }
        for camera, values in metrics.items()
    ]
    import pandas as pd

    table = pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")
    return "\n".join(
        [
            "# InterveneSim-Value visual pilot",
            "",
            f"A frozen ImageNet ResNet-18 visual encoder plus a learned value head uses "
            f"**{train.sample_count:,}** training images. Evaluation contains "
            f"**{evaluation.sample_count:,}** images from a held-out policy seed.",
            "",
            "The head is trained only on `frontview`. `agentview` is a zero-shot camera-transfer "
            "test. Proprioception includes robot state, control time, and task identity but "
            "excludes "
            "object and goal position.",
            "",
            table,
            "",
            "This is a secondary failure-detection probe, not an end-to-end visual control policy.",
            "",
        ]
    )


def write_visual_montage(data: VisualData, path: str | Path) -> Path:
    """Write paired front/agent camera observations from matched causal states."""
    rows: list[tuple[int, int]] = []
    for task in ("can", "milk", "bread", "cereal"):
        front_indices = np.flatnonzero(
            (data.tasks == task) & (data.cameras == "frontview") & data.helpful
        )
        if not len(front_indices):
            front_indices = np.flatnonzero((data.tasks == task) & (data.cameras == "frontview"))
        if not len(front_indices):
            continue
        front = int(front_indices[0])
        matches = np.flatnonzero(
            (data.tasks == task)
            & (data.cameras == "agentview")
            & (data.episode_ids == data.episode_ids[front])
            & (data.steps == data.steps[front])
        )
        if len(matches):
            rows.append((front, int(matches[0])))
    if not rows:
        raise ValueError("visual dataset has no matched camera pairs")
    scale = 1.5
    panel_width, panel_height = int(160 * scale), int(120 * scale)
    header_height, label_height = 52, 26
    row_count = (len(rows) + 1) // 2
    canvas = Image.new(
        "RGB",
        (panel_width * 4, header_height + row_count * (panel_height + label_height)),
        "#111827",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((14, 12), "Held-out visual observations", fill="white", font=font)
    draw.text(
        (14, 30),
        "same causal state / two cameras / object and goal coordinates withheld",
        fill="#9ca3af",
        font=font,
    )
    for row_index, (front, agent) in enumerate(rows):
        top = header_height + (row_index // 2) * (panel_height + label_height)
        pair_left = (row_index % 2) * panel_width * 2
        for column, index in enumerate((front, agent)):
            image = Image.fromarray(data.images[index]).resize(
                (panel_width, panel_height), Image.Resampling.LANCZOS
            )
            left = pair_left + column * panel_width
            canvas.paste(image, (left, top))
            label = (
                f"{data.tasks[index]} · {data.cameras[index]} · step {data.steps[index]} · "
                f"helpful={bool(data.helpful[index])}"
            )
            draw.text(
                (left + 8, top + panel_height + 7),
                label,
                fill="white",
                font=font,
            )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress:
        progress(message)
