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
