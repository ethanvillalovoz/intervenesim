from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from intervenesim.collection import (
    collect_clean_demonstrations,
    collect_recovery_demonstrations,
)
from intervenesim.config import BenchmarkConfig, TrainConfig
from intervenesim.dataset import TrajectoryData
from intervenesim.evaluation import evaluate_policies
from intervenesim.policy import PolicyAgent, train_policy
from intervenesim.reporting import write_report


def run_benchmark(
    config: BenchmarkConfig,
    output_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    if output_dir is not None:
        config = replace(config, output_dir=str(output_dir))
    output = Path(config.output_dir)
    datasets_dir = output / "datasets"
    checkpoints_dir = output / "checkpoints"
    results_dir = output / "results"
    for directory in (datasets_dir, checkpoints_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    started = time.time()
    _emit(progress, "collecting baseline expert demonstrations")
    base, base_summary = collect_clean_demonstrations(
        config.base_episodes,
        config.seed,
        config.max_steps,
        config.expert_attempt_multiplier,
        progress,
    )
    base_path = base.save(datasets_dir / "base_clean.npz")
    baseline_checkpoint = checkpoints_dir / "baseline.pt"
    _emit(progress, "training baseline behavior-cloning policy")
    baseline_train = train_policy(
        base,
        baseline_checkpoint,
        TrainConfig(
            epochs=config.train_epochs,
            hidden_dims=config.hidden_dims,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            device=config.device,
            seed=config.seed,
        ),
        condition="baseline",
    )
    baseline_policy = PolicyAgent.load(baseline_checkpoint, device=config.device)
    _emit(progress, "collecting equal-budget clean augmentation data")
    extra, extra_summary = collect_clean_demonstrations(
        config.extra_episodes,
        config.seed + 100_000,
        config.max_steps,
        config.expert_attempt_multiplier,
        progress,
    )
    _emit(progress, "collecting post-failure recovery interventions")
    recovery, recovery_summary = collect_recovery_demonstrations(
        baseline_policy,
        config.recovery_episodes,
        config.seed + 200_000,
        config.max_steps,
        max(config.expert_attempt_multiplier, 4),
        progress=progress,
    )
    sample_budget = min(extra.sample_count, recovery.sample_count)
    if sample_budget == 0:
        raise RuntimeError("augmentation datasets were empty")
    extra_budget = extra.sample_budget(sample_budget, config.seed + 300_000)
    recovery_budget = recovery.sample_budget(sample_budget, config.seed + 300_000)
    extra_path = extra_budget.save(datasets_dir / "extra_clean_equal_budget.npz")
    recovery_path = recovery_budget.save(datasets_dir / "recovery_equal_budget.npz")
    more_demos_data = TrajectoryData.concatenate(
        [base, extra_budget],
        metadata={"condition": "more_demos", "additional_sample_budget": sample_budget},
    )
    recovery_data = TrajectoryData.concatenate(
        [base, recovery_budget],
        metadata={"condition": "recovery_data", "additional_sample_budget": sample_budget},
    )
    more_demos_checkpoint = checkpoints_dir / "more_demos.pt"
    recovery_checkpoint = checkpoints_dir / "recovery_data.pt"
    fine_tune_common = dict(
        epochs=config.fine_tune_epochs,
        hidden_dims=config.hidden_dims,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate * 0.5,
        weight_decay=config.weight_decay,
        device=config.device,
        fine_tune_from=str(baseline_checkpoint),
    )
    _emit(progress, "fine-tuning equal-budget additional-demonstration condition")
    more_demos_train = train_policy(
        more_demos_data,
        more_demos_checkpoint,
        TrainConfig(seed=config.seed + 1, **fine_tune_common),
        condition="more_demos",
    )
    _emit(progress, "fine-tuning equal-budget recovery-data condition")
    recovery_train = train_policy(
        recovery_data,
        recovery_checkpoint,
        TrainConfig(seed=config.seed + 1, **fine_tune_common),
        condition="recovery_data",
    )
    _emit(progress, "evaluating all policies on matched episode seeds")
    episodes = evaluate_policies(
        checkpoints={
            "baseline": baseline_checkpoint,
            "more_demos": more_demos_checkpoint,
            "recovery_data": recovery_checkpoint,
        },
        disturbances=config.disturbances,
        episodes=config.eval_episodes,
        seed=config.seed + 400_000,
        max_steps=config.max_steps,
        device=config.device,
        progress=progress,
    )
    manifest: dict[str, object] = {
        "elapsed_seconds": time.time() - started,
        "config": config.to_dict(),
        "sample_budget": sample_budget,
        "collection": {
            "base": base_summary.to_dict(),
            "extra": extra_summary.to_dict(),
            "recovery": recovery_summary.to_dict(),
        },
        "training": {
            "baseline": baseline_train,
            "more_demos": more_demos_train,
            "recovery_data": recovery_train,
        },
        "datasets": {
            "base": str(base_path),
            "extra_equal_budget": str(extra_path),
            "recovery_equal_budget": str(recovery_path),
        },
        "system": system_manifest(),
    }
    artifacts = write_report(episodes, results_dir, sample_budget, manifest)
    manifest["artifacts"] = artifacts
    (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _emit(progress, f"benchmark complete: {artifacts['report']}")
    return manifest


def system_manifest() -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "numpy": np.__version__,
        "git_commit": commit,
    }


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress:
        progress(message)
