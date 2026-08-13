from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from intervenesim.config import TrainConfig
from intervenesim.dataset import BehaviorCloningDataset, TrajectoryData


class MLPPolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = observation_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            width = hidden
        layers.extend([nn.Linear(width, action_dim), nn.Tanh()])
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, observations: np.ndarray) -> Normalizer:
        mean = observations.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = observations.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std < 1e-5, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, observations: np.ndarray) -> np.ndarray:
        return (observations - self.mean) / self.std


class PolicyAgent:
    def __init__(
        self,
        model: MLPPolicy,
        normalizer: Normalizer,
        device: torch.device,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.device = device
        self.metadata = metadata or {}

    @torch.inference_mode()
    def action(self, observation: np.ndarray) -> np.ndarray:
        normalized = self.normalizer.transform(np.asarray(observation, dtype=np.float32))
        tensor = torch.from_numpy(normalized).to(self.device).unsqueeze(0)
        action = self.model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
        # The benchmark's expert controls translation and gripper state. Rotation remains
        # fixed, so projected zeroes prevent tiny regression residuals from accumulating
        # into an out-of-distribution wrist orientation during closed-loop rollout.
        action[3:6] = 0.0
        return action

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> PolicyAgent:
        resolved = resolve_device(device)
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = MLPPolicy(
            checkpoint["observation_dim"],
            checkpoint["action_dim"],
            tuple(checkpoint["hidden_dims"]),
        )
        model.load_state_dict(checkpoint["model"])
        normalizer = Normalizer(
            mean=np.asarray(checkpoint["normalizer_mean"], dtype=np.float32),
            std=np.asarray(checkpoint["normalizer_std"], dtype=np.float32),
        )
        return cls(model, normalizer, resolved, checkpoint.get("metadata", {}))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def train_policy(
    data: TrajectoryData,
    output_path: str | Path,
    config: TrainConfig,
    condition: str,
) -> dict[str, Any]:
    _seed_everything(config.seed)
    device = resolve_device(config.device)
    rng = np.random.default_rng(config.seed)
    episode_ids = np.unique(data.episode_ids)
    rng.shuffle(episode_ids)
    validation_count = max(1, int(round(0.1 * len(episode_ids)))) if len(episode_ids) > 1 else 0
    validation_episodes = set(episode_ids[:validation_count].tolist())
    validation_mask = np.asarray([ep in validation_episodes for ep in data.episode_ids])
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if not len(train_indices):
        train_indices = np.arange(data.sample_count)
    if config.fine_tune_from:
        initial = torch.load(config.fine_tune_from, map_location="cpu", weights_only=False)
        normalizer = Normalizer(
            mean=np.asarray(initial["normalizer_mean"], dtype=np.float32),
            std=np.asarray(initial["normalizer_std"], dtype=np.float32),
        )
    else:
        initial = None
        normalizer = Normalizer.fit(data.observations[train_indices])
    normalized_data = TrajectoryData(
        observations=normalizer.transform(data.observations).astype(np.float32),
        actions=data.actions,
        episode_ids=data.episode_ids,
        sources=data.sources,
        disturbances=data.disturbances,
        intervention=data.intervention,
        phases=data.phases,
        metadata=data.metadata,
        rejected_actions=data.rejected_actions,
        rejection_mask=data.rejection_mask,
    )
    model = MLPPolicy(data.observation_dim, data.action_dim, config.hidden_dims).to(device)
    if initial is not None:
        model.load_state_dict(initial["model"])
    train_loader = DataLoader(
        BehaviorCloningDataset(normalized_data, train_indices),
        batch_size=min(config.batch_size, len(train_indices)),
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    contrastive_weight = float(config.extra.get("contrastive_weight", 0.0))
    contrastive_margin = float(config.extra.get("contrastive_margin", 0.2))
    contrastive_min_distance = float(config.extra.get("contrastive_min_distance", 0.08))
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    for epoch in range(config.epochs):
        model.train()
        losses: list[float] = []
        contrastive_losses: list[float] = []
        for observations, actions, rejected_actions, rejection_mask in train_loader:
            observations = observations.to(device=device, dtype=torch.float32)
            actions = actions.to(device=device, dtype=torch.float32)
            rejected_actions = rejected_actions.to(device=device, dtype=torch.float32)
            rejection_mask = rejection_mask.to(device=device, dtype=torch.bool)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(observations)
            imitation_loss = criterion(predictions, actions)
            correction_loss = contrastive_correction_loss(
                predictions,
                actions,
                rejected_actions,
                rejection_mask,
                margin=contrastive_margin,
                min_distance=contrastive_min_distance,
            )
            loss = imitation_loss + contrastive_weight * correction_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            contrastive_losses.append(float(correction_loss.detach().cpu()))
        validation = _validation_loss(model, normalized_data, validation_indices, device)
        train_loss = float(np.mean(losses))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "contrastive_loss": float(np.mean(contrastive_losses)),
                "validation_loss": validation,
            }
        )
        score = validation if np.isfinite(validation) else train_loss
        if score < best_validation:
            best_validation = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "condition": condition,
        "sample_count": data.sample_count,
        "episode_count": data.episode_count,
        "device": str(device),
        "train_config": asdict(config),
        "sources": {
            str(key): int(value)
            for key, value in zip(*np.unique(data.sources, return_counts=True), strict=True)
        },
        "rejected_sample_count": int(data.rejection_mask.sum()),
        "contrastive_weight": contrastive_weight,
        "contrastive_margin": contrastive_margin,
        "contrastive_min_distance": contrastive_min_distance,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "observation_dim": data.observation_dim,
            "action_dim": data.action_dim,
            "hidden_dims": list(config.hidden_dims),
            "normalizer_mean": normalizer.mean,
            "normalizer_std": normalizer.std,
            "metadata": metadata,
        },
        output,
    )
    history_path = output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "checkpoint": str(output),
        "history": str(history_path),
        "best_validation_loss": best_validation,
        **metadata,
    }


def contrastive_correction_loss(
    predictions: torch.Tensor,
    corrections: torch.Tensor,
    rejected_actions: torch.Tensor,
    rejection_mask: torch.Tensor,
    margin: float,
    min_distance: float,
) -> torch.Tensor:
    """Margin loss that ranks a correction closer than the rejected robot action."""
    correction_distance = torch.mean((predictions - corrections) ** 2, dim=-1)
    rejected_distance = torch.mean((predictions - rejected_actions) ** 2, dim=-1)
    pair_distance = torch.sqrt(torch.sum((corrections - rejected_actions) ** 2, dim=-1))
    eligible = rejection_mask & (pair_distance >= min_distance)
    if not torch.any(eligible):
        return predictions.sum() * 0.0
    ranking = torch.relu(margin + correction_distance - rejected_distance)
    return ranking[eligible].mean()


@torch.inference_mode()
def _validation_loss(
    model: nn.Module,
    data: TrajectoryData,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    if not len(indices):
        return float("nan")
    model.eval()
    observations = torch.from_numpy(data.observations[indices]).to(device)
    actions = torch.from_numpy(data.actions[indices]).to(device)
    return float(nn.functional.mse_loss(model(observations), actions).cpu())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
