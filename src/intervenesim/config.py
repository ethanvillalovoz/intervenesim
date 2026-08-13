from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 27
    base_episodes: int = 120
    extra_episodes: int = 40
    recovery_episodes: int = 60
    train_epochs: int = 80
    fine_tune_epochs: int = 35
    eval_episodes: int = 50
    hidden_dims: tuple[int, ...] = (256, 256, 128)
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    device: str = "auto"
    max_steps: int = 260
    disturbances: tuple[str, ...] = (
        "nominal",
        "object_shift",
        "action_noise",
        "action_delay",
        "gripper_slip",
    )
    output_dir: str = "artifacts/runs/latest"
    expert_attempt_multiplier: int = 3

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        with Path(path).open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        if "hidden_dims" in raw:
            raw["hidden_dims"] = tuple(raw["hidden_dims"])
        if "disturbances" in raw:
            raw["disturbances"] = tuple(raw["disturbances"])
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        data["disturbances"] = list(self.disturbances)
        return data


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    hidden_dims: tuple[int, ...]
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: str
    seed: int
    fine_tune_from: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchConfig:
    seed: int = 27
    training_seeds: tuple[int, ...] = (27, 127, 227, 327, 427)
    budgets: tuple[int, ...] = (500, 1500, 3000, 4800)
    tasks: tuple[str, ...] = ("can", "milk", "bread", "cereal")
    base_episodes_per_task: int = 30
    extra_episodes_per_task: int = 12
    recovery_episodes_per_task: int = 18
    risk_nominal_episodes_per_task: int = 6
    risk_horizon: int = 16
    risk_detection_horizon: int = 3
    minimum_help_step: int = 5
    help_thresholds: tuple[float, ...] = (0.90, 0.95, 0.97, 0.99)
    train_epochs: int = 80
    fine_tune_epochs: int = 35
    risk_epochs: int = 50
    eval_episodes: int = 8
    hidden_dims: tuple[int, ...] = (256, 256, 128)
    risk_hidden_dims: tuple[int, ...] = (128, 128)
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    contrastive_weight: float = 0.25
    contrastive_margin: float = 0.20
    contrastive_min_distance: float = 0.08
    device: str = "auto"
    max_steps: int = 260
    disturbances: tuple[str, ...] = (
        "nominal",
        "object_shift",
        "action_noise",
        "action_delay",
        "gripper_slip",
    )
    output_dir: str = "artifacts/runs/research-v1"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ResearchConfig:
        with Path(path).open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        for key in (
            "training_seeds",
            "budgets",
            "tasks",
            "hidden_dims",
            "risk_hidden_dims",
            "help_thresholds",
            "disturbances",
        ):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "training_seeds",
            "budgets",
            "tasks",
            "hidden_dims",
            "risk_hidden_dims",
            "help_thresholds",
            "disturbances",
        ):
            data[key] = list(data[key])
        return data


@dataclass(frozen=True)
class ValueConfig:
    seed: int = 27
    train_policy_seeds: tuple[int, ...] = (27, 127, 227)
    eval_policy_seeds: tuple[int, ...] = (327, 427)
    tasks: tuple[str, ...] = ("can", "milk", "bread", "cereal")
    disturbances: tuple[str, ...] = (
        "nominal",
        "object_shift",
        "action_noise",
        "action_delay",
        "gripper_slip",
    )
    candidate_steps: tuple[int, ...] = (10, 30, 50, 70, 90, 110, 130)
    target_budgets: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
    train_episodes: int = 4
    eval_episodes: int = 6
    value_epochs: int = 60
    value_hidden_dims: tuple[int, ...] = (128, 128)
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    max_steps: int = 260
    device: str = "auto"
    research_run: str = "artifacts/runs/research-v1"
    output_dir: str = "artifacts/runs/value-v1"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ValueConfig:
        with Path(path).open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        for key in (
            "train_policy_seeds",
            "eval_policy_seeds",
            "tasks",
            "disturbances",
            "candidate_steps",
            "target_budgets",
            "value_hidden_dims",
        ):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "train_policy_seeds",
            "eval_policy_seeds",
            "tasks",
            "disturbances",
            "candidate_steps",
            "target_budgets",
            "value_hidden_dims",
        ):
            data[key] = list(data[key])
        return data


@dataclass(frozen=True)
class VisualConfig:
    seed: int = 117
    train_policy_seed: int = 27
    eval_policy_seed: int = 327
    tasks: tuple[str, ...] = ("can", "milk", "bread", "cereal")
    disturbances: tuple[str, ...] = (
        "nominal",
        "object_shift",
        "action_noise",
        "action_delay",
        "gripper_slip",
    )
    candidate_steps: tuple[int, ...] = (10, 30, 50, 70, 90, 110, 130)
    cameras: tuple[str, ...] = ("frontview", "agentview")
    episodes_per_task: int = 2
    epochs: int = 30
    max_steps: int = 260
    device: str = "auto"
    research_run: str = "artifacts/runs/research-v1"
    output_dir: str = "artifacts/runs/visual-v1"

    @classmethod
    def from_yaml(cls, path: str | Path) -> VisualConfig:
        with Path(path).open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        for key in ("tasks", "disturbances", "candidate_steps", "cameras"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("tasks", "disturbances", "candidate_steps", "cameras"):
            data[key] = list(data[key])
        return data
