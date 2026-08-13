from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from intervenesim.policy import Normalizer, resolve_device


@dataclass
class RiskData:
    observations: np.ndarray
    targets: np.ndarray
    episode_ids: np.ndarray
    disturbances: np.ndarray
    steps_to_intervention: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.targets = np.asarray(self.targets, dtype=np.float32)
        self.episode_ids = np.asarray(self.episode_ids, dtype=np.int32)
        self.disturbances = np.asarray(self.disturbances, dtype="U24")
        self.steps_to_intervention = np.asarray(self.steps_to_intervention, dtype=np.int16)
        lengths = {
            len(self.observations),
            len(self.targets),
            len(self.episode_ids),
            len(self.disturbances),
            len(self.steps_to_intervention),
        }
        if len(lengths) != 1:
            raise ValueError("risk dataset arrays must have equal lengths")
        if self.observations.ndim != 2:
            raise ValueError("risk observations must be rank-2")
        if not np.isin(self.targets, [0.0, 1.0]).all():
            raise ValueError("risk targets must be binary")

    @property
    def sample_count(self) -> int:
        return int(len(self.targets))

    @property
    def positive_count(self) -> int:
        return int(self.targets.sum())

    @property
    def observation_dim(self) -> int:
        return int(self.observations.shape[1])

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **self.metadata,
            "schema_version": 1,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
        }
        np.savez_compressed(
            output,
            observations=self.observations,
            targets=self.targets,
            episode_ids=self.episode_ids,
            disturbances=self.disturbances,
            steps_to_intervention=self.steps_to_intervention,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> RiskData:
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                observations=archive["observations"],
                targets=archive["targets"],
                episode_ids=archive["episode_ids"],
                disturbances=archive["disturbances"],
                steps_to_intervention=archive["steps_to_intervention"],
                metadata=json.loads(str(archive["metadata"].item())),
            )

    @classmethod
    def concatenate(cls, parts: list[RiskData]) -> RiskData:
        selected = [part for part in parts if part.sample_count]
        if not selected:
            raise ValueError("cannot concatenate empty risk datasets")
        episode_ids: list[np.ndarray] = []
        offset = 0
        for part in selected:
            local = part.episode_ids - part.episode_ids.min()
            episode_ids.append(local + offset)
            offset += int(local.max()) + 1
        return cls(
            observations=np.concatenate([part.observations for part in selected]),
            targets=np.concatenate([part.targets for part in selected]),
            episode_ids=np.concatenate(episode_ids),
            disturbances=np.concatenate([part.disturbances for part in selected]),
            steps_to_intervention=np.concatenate([part.steps_to_intervention for part in selected]),
        )


class RiskBuilder:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.targets: list[float] = []
        self.episode_ids: list[int] = []
        self.disturbances: list[str] = []
        self.steps_to_intervention: list[int] = []

    def append_episode(
        self,
        observations: list[np.ndarray],
        intervention_index: int | None,
        horizon: int,
        episode_id: int,
        disturbance: str,
    ) -> None:
        for index, observation in enumerate(observations):
            steps = -1 if intervention_index is None else intervention_index - index
            target = float(intervention_index is not None and 0 <= steps < horizon)
            self.observations.append(np.asarray(observation, dtype=np.float32).copy())
            self.targets.append(target)
            self.episode_ids.append(episode_id)
            self.disturbances.append(disturbance)
            self.steps_to_intervention.append(steps)

    def build(self, observation_dim: int, metadata: dict[str, Any]) -> RiskData:
        observations = (
            np.stack(self.observations)
            if self.observations
            else np.empty((0, observation_dim), dtype=np.float32)
        )
        return RiskData(
            observations=observations,
            targets=np.asarray(self.targets),
            episode_ids=np.asarray(self.episode_ids),
            disturbances=np.asarray(self.disturbances),
            steps_to_intervention=np.asarray(self.steps_to_intervention),
            metadata=metadata,
        )


class RiskNetwork(nn.Module):
    def __init__(self, observation_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = observation_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)


@dataclass(frozen=True)
class RiskTrainConfig:
    epochs: int
    hidden_dims: tuple[int, ...]
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: str
    seed: int
    target_recall: float = 0.9


class RiskAgent:
    def __init__(
        self,
        model: RiskNetwork,
        normalizer: Normalizer,
        threshold: float,
        device: torch.device,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.threshold = threshold
        self.device = device
        self.metadata = metadata or {}

    @torch.inference_mode()
    def score(self, observation: np.ndarray) -> float:
        normalized = self.normalizer.transform(np.asarray(observation, dtype=np.float32))
        tensor = torch.from_numpy(normalized).to(self.device).unsqueeze(0)
        return float(torch.sigmoid(self.model(tensor)).cpu().item())

    def requests_help(self, observation: np.ndarray) -> bool:
        return self.score(observation) >= self.threshold

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> RiskAgent:
        resolved = resolve_device(device)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = RiskNetwork(checkpoint["observation_dim"], tuple(checkpoint["hidden_dims"]))
        model.load_state_dict(checkpoint["model"])
        return cls(
            model,
            Normalizer(checkpoint["normalizer_mean"], checkpoint["normalizer_std"]),
            float(checkpoint["threshold"]),
            resolved,
            checkpoint.get("metadata", {}),
        )


def train_risk_model(
    data: RiskData,
    output_path: str | Path,
    config: RiskTrainConfig,
) -> dict[str, Any]:
    _seed_everything(config.seed)
    device = resolve_device(config.device)
    rng = np.random.default_rng(config.seed)
    validation_episodes = _stratified_validation_episodes(data, rng)
    validation_mask = np.asarray([ep in validation_episodes for ep in data.episode_ids])
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    normalizer = Normalizer.fit(data.observations[train_indices])
    observations = normalizer.transform(data.observations).astype(np.float32)
    train_targets = data.targets[train_indices]
    positives = max(1.0, float(train_targets.sum()))
    negatives = max(1.0, float(len(train_targets) - positives))
    positive_weight = torch.tensor(negatives / positives, device=device)
    dataset = TensorDataset(
        torch.from_numpy(observations[train_indices]),
        torch.from_numpy(train_targets),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = RiskNetwork(data.observation_dim, config.hidden_dims).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    for epoch in range(config.epochs):
        model.train()
        train_losses: list[float] = []
        for batch_observations, batch_targets in loader:
            batch_observations = batch_observations.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(batch_observations), batch_targets, pos_weight=positive_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        probabilities = _predict(model, observations[validation_indices], device)
        validation_loss = _binary_log_loss(data.targets[validation_indices], probabilities)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    probabilities = _predict(model, observations[validation_indices], device)
    targets = data.targets[validation_indices]
    threshold = select_threshold(targets, probabilities, config.target_recall)
    metrics = binary_metrics(targets, probabilities, threshold)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "train_config": asdict(config),
        "sample_count": data.sample_count,
        "positive_count": data.positive_count,
        "validation_episodes": sorted(validation_episodes),
        "validation_metrics": metrics,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "observation_dim": data.observation_dim,
            "hidden_dims": list(config.hidden_dims),
            "normalizer_mean": normalizer.mean,
            "normalizer_std": normalizer.std,
            "threshold": threshold,
            "metadata": metadata,
        },
        output,
    )
    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    return {"checkpoint": str(output), "threshold": threshold, **metadata}


def select_threshold(targets: np.ndarray, probabilities: np.ndarray, target_recall: float) -> float:
    positives = probabilities[targets.astype(bool)]
    if not len(positives):
        return 0.5
    quantile = max(0.0, min(1.0, 1.0 - target_recall))
    return float(np.quantile(positives, quantile))


def _stratified_validation_episodes(data: RiskData, rng: np.random.Generator) -> set[int]:
    positive_episodes: list[int] = []
    negative_episodes: list[int] = []
    for episode_id in np.unique(data.episode_ids):
        targets = data.targets[data.episode_ids == episode_id]
        destination = positive_episodes if targets.any() else negative_episodes
        destination.append(int(episode_id))
    selected: set[int] = set()
    for group in (positive_episodes, negative_episodes):
        rng.shuffle(group)
        if len(group) > 1:
            count = max(1, int(round(0.2 * len(group))))
            selected.update(group[:count])
    if not selected:
        all_episodes = np.unique(data.episode_ids)
        if len(all_episodes) > 1:
            selected.add(int(all_episodes[0]))
    return selected


def binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities >= threshold
    true = targets.astype(bool)
    tp = int((predictions & true).sum())
    fp = int((predictions & ~true).sum())
    fn = int((~predictions & true).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "auroc": _auroc(targets, probabilities),
        "average_precision": _average_precision(targets, probabilities),
        "brier": float(np.mean((probabilities - targets) ** 2)),
        "precision": precision,
        "recall": recall,
        "positive_rate": float(predictions.mean()),
    }


def _predict(model: nn.Module, observations: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(observations).to(device))
        return torch.sigmoid(logits).cpu().numpy()


def _binary_log_loss(targets: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)))


def _auroc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    positive = probabilities[targets == 1]
    negative = probabilities[targets == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    wins = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def _average_precision(targets: np.ndarray, probabilities: np.ndarray) -> float:
    positive_count = int(targets.sum())
    if not positive_count:
        return float("nan")
    order = np.argsort(-probabilities, kind="stable")
    ordered = targets[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float((precision * ordered).sum() / positive_count)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
