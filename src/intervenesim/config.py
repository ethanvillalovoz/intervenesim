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
    base_episodes: int = 120
    extra_episodes: int = 48
    recovery_episodes: int = 72
    risk_nominal_episodes: int = 24
    risk_horizon: int = 16
    train_epochs: int = 80
    fine_tune_epochs: int = 35
    risk_epochs: int = 50
    eval_episodes: int = 12
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
        for key in ("training_seeds", "budgets", "hidden_dims", "risk_hidden_dims", "disturbances"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("training_seeds", "budgets", "hidden_dims", "risk_hidden_dims", "disturbances"):
            data[key] = list(data[key])
        return data
