from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from intervenesim.counterfactual import CounterfactualData
from intervenesim.policy import Normalizer, resolve_device
from intervenesim.risk import RiskAgent, binary_metrics


class ValueNetwork(nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = feature_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class ValueTrainConfig:
    epochs: int
    hidden_dims: tuple[int, ...]
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: str
    seed: int


class ValueAgent:
    def __init__(
        self,
        model: ValueNetwork,
        normalizer: Normalizer,
        device: torch.device,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.device = device
        self.metadata = metadata or {}

    @torch.inference_mode()
    def score(
        self, observation: np.ndarray, observation_delta: np.ndarray, action: np.ndarray
    ) -> float:
        features = np.concatenate([observation, observation_delta, action]).astype(np.float32)
        normalized = self.normalizer.transform(features)
        tensor = torch.from_numpy(normalized).to(self.device).unsqueeze(0)
        return float(torch.sigmoid(self.model(tensor)).cpu().item())

    @torch.inference_mode()
    def scores(self, data: CounterfactualData) -> np.ndarray:
        features = self.normalizer.transform(data.features).astype(np.float32)
        tensor = torch.from_numpy(features).to(self.device)
        return torch.sigmoid(self.model(tensor)).cpu().numpy()

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> ValueAgent:
        resolved = resolve_device(device)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = ValueNetwork(checkpoint["feature_dim"], tuple(checkpoint["hidden_dims"]))
        model.load_state_dict(checkpoint["model"])
        return cls(
            model,
            Normalizer(checkpoint["normalizer_mean"], checkpoint["normalizer_std"]),
            resolved,
            checkpoint.get("metadata", {}),
        )


def train_value_model(
    data: CounterfactualData,
    output_path: str | Path,
    config: ValueTrainConfig,
) -> dict[str, Any]:
    _seed_everything(config.seed)
    device = resolve_device(config.device)
    train_indices, validation_indices = episode_split(data.episode_ids, config.seed)
    features = data.features
    targets = data.helpful.astype(np.float32)
    normalizer = Normalizer.fit(features[train_indices])
    normalized = normalizer.transform(features).astype(np.float32)
    train_targets = targets[train_indices]
    positives = max(1.0, float(train_targets.sum()))
    negatives = max(1.0, float(len(train_targets) - positives))
    positive_weight = torch.tensor(negatives / positives, device=device)
    dataset = TensorDataset(
        torch.from_numpy(normalized[train_indices]),
        torch.from_numpy(train_targets),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = ValueNetwork(normalized.shape[1], config.hidden_dims).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(config.epochs):
        model.train()
        losses: list[float] = []
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(batch_features), batch_targets, pos_weight=positive_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probabilities = _predict(model, normalized[validation_indices], device)
        validation_loss = _log_loss(targets[validation_indices], probabilities)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
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
    probabilities = _predict(model, normalized[validation_indices], device)
    metrics = binary_metrics(targets[validation_indices], probabilities, 0.5)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "train_config": asdict(config),
        "sample_count": data.sample_count,
        "helpful_count": int(data.helpful.sum()),
        "validation_episodes": sorted(set(data.episode_ids[validation_indices].tolist())),
        "validation_metrics": metrics,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "feature_dim": int(normalized.shape[1]),
            "hidden_dims": list(config.hidden_dims),
            "normalizer_mean": normalizer.mean,
            "normalizer_std": normalizer.std,
            "metadata": metadata,
        },
        output,
    )
    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    return {"checkpoint": str(output), **metadata}


def risk_scores(data: CounterfactualData, risk_checkpoint: str | Path) -> np.ndarray:
    agent = RiskAgent.load(risk_checkpoint, device="cpu")
    features = np.concatenate([data.observations, data.observation_deltas], axis=1)
    normalized = agent.normalizer.transform(features).astype(np.float32)
    with torch.inference_mode():
        logits = agent.model(torch.from_numpy(normalized).to(agent.device))
        return torch.sigmoid(logits).cpu().numpy()


def budget_thresholds(
    scores: np.ndarray,
    episode_ids: np.ndarray,
    budgets: tuple[float, ...],
) -> dict[float, float]:
    maxima = np.asarray(
        [scores[episode_ids == episode].max() for episode in np.unique(episode_ids)]
    )
    return {
        budget: float(np.quantile(maxima, max(0.0, min(1.0, 1.0 - budget)))) for budget in budgets
    }


def evaluate_counterfactual_gates(
    data: CounterfactualData,
    score_sets: dict[str, np.ndarray],
    thresholds: dict[str, dict[float, float]],
    budgets: tuple[float, ...],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    frame = data.to_frame()
    for name, values in score_sets.items():
        frame[f"score_{name}"] = values
    for episode_id, group in frame.groupby("episode_id", sort=False):
        group = group.sort_values("step")
        autonomous_success = bool(group["autonomous_success"].iloc[0])
        common = {
            "episode_id": int(episode_id),
            "policy_seed": int(group["policy_seed"].iloc[0]),
            "task": str(group["task"].iloc[0]),
            "disturbance": str(group["disturbance"].iloc[0]),
        }
        records.append(
            {
                **common,
                "method": "never_help",
                "target_budget": 0.0,
                "success": autonomous_success,
                "help_requested": False,
                "helpful_request": False,
                "request_step": np.nan,
                "request_score": np.nan,
            }
        )
        earliest = group.iloc[0]
        records.append(_request_record(common, "always_help", 1.0, earliest, np.nan))
        helpful = group.loc[group["helpful"]]
        if len(helpful):
            records.append(_request_record(common, "oracle_value", np.nan, helpful.iloc[0], 1.0))
        else:
            records.append(
                {
                    **common,
                    "method": "oracle_value",
                    "target_budget": np.nan,
                    "success": autonomous_success,
                    "help_requested": False,
                    "helpful_request": False,
                    "request_step": np.nan,
                    "request_score": np.nan,
                }
            )
        for method, _ in score_sets.items():
            for budget in budgets:
                threshold = thresholds[method][budget]
                selected = group.loc[group[f"score_{method}"] >= threshold]
                if len(selected):
                    records.append(
                        _request_record(
                            common,
                            method,
                            budget,
                            selected.iloc[0],
                            float(selected.iloc[0][f"score_{method}"]),
                        )
                    )
                else:
                    records.append(
                        {
                            **common,
                            "method": method,
                            "target_budget": budget,
                            "success": autonomous_success,
                            "help_requested": False,
                            "helpful_request": False,
                            "request_step": np.nan,
                            "request_score": np.nan,
                        }
                    )
    return pd.DataFrame(records)


def summarize_gate_evaluation(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, budget), group in episodes.groupby(
        ["method", "target_budget"], dropna=False, sort=False
    ):
        requests = group["help_requested"].astype(bool)
        requested = group.loc[requests]
        episode_ids = group["episode_id"].to_numpy()
        all_episode_rows = episodes.loc[episodes["method"] == "never_help"]
        baseline = all_episode_rows.loc[all_episode_rows["episode_id"].isin(episode_ids)]
        base_successes = int(baseline["success"].sum())
        helpful_episodes = int(
            episodes.loc[
                (episodes["method"] == "oracle_value") & episodes["episode_id"].isin(episode_ids),
                "help_requested",
            ].sum()
        )
        oracle_success = (base_successes + min(int(requests.sum()), helpful_episodes)) / len(group)
        observed_success = float(group["success"].mean())
        rows.append(
            {
                "method": method,
                "target_budget": budget,
                "episodes": len(group),
                "success_rate": observed_success,
                "intervention_rate": float(requests.mean()),
                "matched_oracle_success_rate": oracle_success,
                "regret_to_matched_oracle": oracle_success - observed_success,
                "request_precision": (
                    float(requested["helpful_request"].mean()) if len(requested) else np.nan
                ),
                "nominal_false_alarm_rate": float(
                    group.loc[group["disturbance"] == "nominal", "help_requested"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def episode_split(episode_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode_ids).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    validation_count = max(1, int(round(0.2 * len(episodes))))
    if validation_count >= len(episodes):
        validation_count = 1
    validation = set(episodes[:validation_count].tolist())
    validation_mask = np.asarray([episode in validation for episode in episode_ids])
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def _request_record(
    common: dict[str, Any], method: str, budget: float, row: pd.Series, score: float
) -> dict[str, Any]:
    return {
        **common,
        "method": method,
        "target_budget": budget,
        "success": bool(row["assisted_success"]),
        "help_requested": True,
        "helpful_request": bool(row["helpful"]),
        "request_step": int(row["step"]),
        "request_score": score,
    }


def _predict(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(torch.from_numpy(features).to(device))).cpu().numpy()


def _log_loss(targets: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
