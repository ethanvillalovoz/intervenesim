from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset


@dataclass
class TrajectoryData:
    observations: np.ndarray
    actions: np.ndarray
    episode_ids: np.ndarray
    sources: np.ndarray
    disturbances: np.ndarray
    intervention: np.ndarray
    phases: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    rejected_actions: np.ndarray | None = None
    rejection_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.episode_ids = np.asarray(self.episode_ids, dtype=np.int32)
        self.sources = np.asarray(self.sources, dtype="U16")
        self.disturbances = np.asarray(self.disturbances, dtype="U24")
        self.intervention = np.asarray(self.intervention, dtype=np.bool_)
        self.phases = np.asarray(self.phases, dtype=np.int8)
        if self.rejected_actions is None:
            self.rejected_actions = np.zeros_like(self.actions, dtype=np.float32)
        else:
            self.rejected_actions = np.asarray(self.rejected_actions, dtype=np.float32)
        if self.rejection_mask is None:
            self.rejection_mask = np.zeros(len(self.actions), dtype=np.bool_)
        else:
            self.rejection_mask = np.asarray(self.rejection_mask, dtype=np.bool_)
        self.validate()

    @classmethod
    def empty(cls, observation_dim: int, action_dim: int) -> TrajectoryData:
        return cls(
            observations=np.empty((0, observation_dim), dtype=np.float32),
            actions=np.empty((0, action_dim), dtype=np.float32),
            episode_ids=np.empty(0, dtype=np.int32),
            sources=np.empty(0, dtype="U16"),
            disturbances=np.empty(0, dtype="U24"),
            intervention=np.empty(0, dtype=np.bool_),
            phases=np.empty(0, dtype=np.int8),
            rejected_actions=np.empty((0, action_dim), dtype=np.float32),
            rejection_mask=np.empty(0, dtype=np.bool_),
        )

    @property
    def sample_count(self) -> int:
        return int(len(self.observations))

    @property
    def episode_count(self) -> int:
        return int(len(np.unique(self.episode_ids))) if self.sample_count else 0

    @property
    def observation_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def validate(self) -> None:
        if self.observations.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("observations and actions must be rank-2 arrays")
        lengths = {
            len(self.observations),
            len(self.actions),
            len(self.episode_ids),
            len(self.sources),
            len(self.disturbances),
            len(self.intervention),
            len(self.phases),
            len(self.rejected_actions),
            len(self.rejection_mask),
        }
        if len(lengths) != 1:
            raise ValueError(f"dataset arrays have inconsistent lengths: {sorted(lengths)}")
        if self.rejected_actions.shape != self.actions.shape:
            raise ValueError("rejected_actions must have the same shape as actions")
        if (
            not np.isfinite(self.observations).all()
            or not np.isfinite(self.actions).all()
            or not np.isfinite(self.rejected_actions).all()
        ):
            raise ValueError("dataset contains non-finite observations or actions")

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **self.metadata,
            "schema_version": 2,
            "sample_count": self.sample_count,
            "episode_count": self.episode_count,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
        }
        np.savez_compressed(
            output,
            observations=self.observations,
            actions=self.actions,
            episode_ids=self.episode_ids,
            sources=self.sources,
            disturbances=self.disturbances,
            intervention=self.intervention,
            phases=self.phases,
            rejected_actions=self.rejected_actions,
            rejection_mask=self.rejection_mask,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> TrajectoryData:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            rejected_actions = (
                archive["rejected_actions"]
                if "rejected_actions" in archive.files
                else np.zeros_like(archive["actions"])
            )
            rejection_mask = (
                archive["rejection_mask"]
                if "rejection_mask" in archive.files
                else np.zeros(len(archive["actions"]), dtype=np.bool_)
            )
            return cls(
                observations=archive["observations"],
                actions=archive["actions"],
                episode_ids=archive["episode_ids"],
                sources=archive["sources"],
                disturbances=archive["disturbances"],
                intervention=archive["intervention"],
                phases=archive["phases"],
                metadata=metadata,
                rejected_actions=rejected_actions,
                rejection_mask=rejection_mask,
            )

    def sample_budget(self, count: int, seed: int) -> TrajectoryData:
        if count >= self.sample_count:
            return self
        if count < 0:
            raise ValueError("sample budget must be non-negative")
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(self.sample_count, size=count, replace=False))
        return self.take(indices, metadata={**self.metadata, "sample_budget": count})

    def take(self, indices: np.ndarray, metadata: dict[str, Any] | None = None) -> TrajectoryData:
        return TrajectoryData(
            observations=self.observations[indices],
            actions=self.actions[indices],
            episode_ids=self.episode_ids[indices],
            sources=self.sources[indices],
            disturbances=self.disturbances[indices],
            intervention=self.intervention[indices],
            phases=self.phases[indices],
            metadata=metadata or dict(self.metadata),
            rejected_actions=self.rejected_actions[indices],
            rejection_mask=self.rejection_mask[indices],
        )

    @classmethod
    def concatenate(
        cls,
        datasets: Iterable[TrajectoryData],
        metadata: dict[str, Any] | None = None,
    ) -> TrajectoryData:
        parts = [part for part in datasets if part.sample_count]
        if not parts:
            raise ValueError("cannot concatenate an empty dataset collection")
        obs_dim = parts[0].observation_dim
        action_dim = parts[0].action_dim
        if any(p.observation_dim != obs_dim or p.action_dim != action_dim for p in parts):
            raise ValueError("all datasets must share observation and action dimensions")
        episode_ids: list[np.ndarray] = []
        offset = 0
        for part in parts:
            local = part.episode_ids - part.episode_ids.min()
            episode_ids.append(local + offset)
            offset += int(local.max()) + 1
        return cls(
            observations=np.concatenate([p.observations for p in parts]),
            actions=np.concatenate([p.actions for p in parts]),
            episode_ids=np.concatenate(episode_ids),
            sources=np.concatenate([p.sources for p in parts]),
            disturbances=np.concatenate([p.disturbances for p in parts]),
            intervention=np.concatenate([p.intervention for p in parts]),
            phases=np.concatenate([p.phases for p in parts]),
            metadata=metadata or {},
            rejected_actions=np.concatenate([p.rejected_actions for p in parts]),
            rejection_mask=np.concatenate([p.rejection_mask for p in parts]),
        )


class BehaviorCloningDataset(Dataset[tuple[np.ndarray, np.ndarray, np.ndarray, np.bool_]]):
    def __init__(self, data: TrajectoryData, indices: np.ndarray | None = None) -> None:
        self.data = data
        self.indices = np.arange(data.sample_count) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.bool_]:
        row = self.indices[index]
        return (
            self.data.observations[row],
            self.data.actions[row],
            self.data.rejected_actions[row],
            self.data.rejection_mask[row],
        )


class TrajectoryBuilder:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.episode_ids: list[int] = []
        self.sources: list[str] = []
        self.disturbances: list[str] = []
        self.intervention: list[bool] = []
        self.phases: list[int] = []
        self.rejected_actions: list[np.ndarray] = []
        self.rejection_mask: list[bool] = []

    def append(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        episode_id: int,
        source: str,
        disturbance: str,
        intervention: bool,
        phase: int,
        rejected_action: np.ndarray | None = None,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.episode_ids.append(episode_id)
        self.sources.append(source)
        self.disturbances.append(disturbance)
        self.intervention.append(intervention)
        self.phases.append(phase)
        self.rejected_actions.append(
            np.zeros_like(action, dtype=np.float32)
            if rejected_action is None
            else np.asarray(rejected_action, dtype=np.float32).copy()
        )
        self.rejection_mask.append(rejected_action is not None)

    def extend(self, other: TrajectoryBuilder) -> None:
        self.observations.extend(other.observations)
        self.actions.extend(other.actions)
        self.episode_ids.extend(other.episode_ids)
        self.sources.extend(other.sources)
        self.disturbances.extend(other.disturbances)
        self.intervention.extend(other.intervention)
        self.phases.extend(other.phases)
        self.rejected_actions.extend(other.rejected_actions)
        self.rejection_mask.extend(other.rejection_mask)

    def build(
        self,
        observation_dim: int,
        action_dim: int,
        metadata: dict[str, Any],
    ) -> TrajectoryData:
        if not self.observations:
            return TrajectoryData.empty(observation_dim, action_dim)
        return TrajectoryData(
            observations=np.stack(self.observations),
            actions=np.stack(self.actions),
            episode_ids=np.asarray(self.episode_ids),
            sources=np.asarray(self.sources),
            disturbances=np.asarray(self.disturbances),
            intervention=np.asarray(self.intervention),
            phases=np.asarray(self.phases),
            metadata=metadata,
            rejected_actions=np.stack(self.rejected_actions),
            rejection_mask=np.asarray(self.rejection_mask),
        )
