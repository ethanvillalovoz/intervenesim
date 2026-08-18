from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from intervenesim.counterfactual import _assisted_branch
from intervenesim.disturbances import Disturbance
from intervenesim.environment import PickPlaceEnv
from intervenesim.policy import Normalizer, PolicyAgent, resolve_device
from intervenesim.risk import binary_metrics


@dataclass
class VisualData:
    images: np.ndarray
    proprioception: np.ndarray
    autonomous_success: np.ndarray
    assisted_success: np.ndarray
    episode_ids: np.ndarray
    steps: np.ndarray
    cameras: np.ndarray
    tasks: np.ndarray
    disturbances: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.images = np.asarray(self.images, dtype=np.uint8)
        self.proprioception = np.asarray(self.proprioception, dtype=np.float32)
        self.autonomous_success = np.asarray(self.autonomous_success, dtype=bool)
        self.assisted_success = np.asarray(self.assisted_success, dtype=bool)
        self.episode_ids = np.asarray(self.episode_ids, dtype=np.int32)
        self.steps = np.asarray(self.steps, dtype=np.int16)
        self.cameras = np.asarray(self.cameras, dtype="U24")
        self.tasks = np.asarray(self.tasks, dtype="U16")
        self.disturbances = np.asarray(self.disturbances, dtype="U24")

    @property
    def helpful(self) -> np.ndarray:
        return self.assisted_success & ~self.autonomous_success

    @property
    def sample_count(self) -> int:
        return len(self.images)

    def subset(self, mask: np.ndarray) -> VisualData:
        return VisualData(
            self.images[mask],
            self.proprioception[mask],
            self.autonomous_success[mask],
            self.assisted_success[mask],
            self.episode_ids[mask],
            self.steps[mask],
            self.cameras[mask],
            self.tasks[mask],
            self.disturbances[mask],
            self.metadata,
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            images=self.images,
            proprioception=self.proprioception,
            autonomous_success=self.autonomous_success,
            assisted_success=self.assisted_success,
            episode_ids=self.episode_ids,
            steps=self.steps,
            cameras=self.cameras,
            tasks=self.tasks,
            disturbances=self.disturbances,
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> VisualData:
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                archive["images"],
                archive["proprioception"],
                archive["autonomous_success"],
                archive["assisted_success"],
                archive["episode_ids"],
                archive["steps"],
                archive["cameras"],
                archive["tasks"],
                archive["disturbances"],
                json.loads(str(archive["metadata"].item())),
            )

    @classmethod
    def concatenate(cls, parts: list[VisualData]) -> VisualData:
        ids: list[np.ndarray] = []
        offset = 0
        for part in parts:
            local = part.episode_ids - part.episode_ids.min(initial=0)
            ids.append(local + offset)
            offset += int(local.max(initial=-1)) + 1
        return cls(
            np.concatenate([part.images for part in parts]),
            np.concatenate([part.proprioception for part in parts]),
            np.concatenate([part.autonomous_success for part in parts]),
            np.concatenate([part.assisted_success for part in parts]),
            np.concatenate(ids),
            np.concatenate([part.steps for part in parts]),
            np.concatenate([part.cameras for part in parts]),
            np.concatenate([part.tasks for part in parts]),
            np.concatenate([part.disturbances for part in parts]),
            {"parts": [part.metadata for part in parts]},
        )


def collect_visual_counterfactuals(
    policy_checkpoint: str | Path,
    task: str,
    disturbances: tuple[str, ...],
    episodes: int,
    seed: int,
    candidate_steps: tuple[int, ...],
    max_steps: int,
    cameras: tuple[str, ...] = ("frontview", "agentview"),
    width: int = 160,
    height: int = 120,
    device: str = "auto",
) -> VisualData:
    policy = PolicyAgent.load(policy_checkpoint, device=device)
    rows: list[dict[str, Any]] = []
    episode_id = 0
    with PickPlaceEnv(
        max_steps=max_steps, offscreen=True, task=task, task_conditioning=True
    ) as env:
        for disturbance_index, disturbance_name in enumerate(disturbances):
            for episode in range(episodes):
                episode_seed = seed + disturbance_index * 10_000 + episode
                disturbance = Disturbance(disturbance_name, episode_seed + 50_000)
                disturbance.reset(env.action_dim)
                state = env.reset(episode_seed)
                candidates: list[dict[str, Any]] = []
                info: dict[str, Any] = {"success": False}
                for step in range(max_steps):
                    state = disturbance.before_step(env, state, step)
                    if step in candidate_steps:
                        candidates.append(
                            {
                                "snapshot": env.snapshot(),
                                "disturbance": deepcopy(disturbance),
                                "images": {
                                    camera: env.capture_frame(width, height, camera)
                                    for camera in cameras
                                },
                                "proprioception": proprioception(state.observation),
                                "step": step,
                            }
                        )
                    action = disturbance.transform_action(policy.action(state.observation), step)
                    state, _, done, info = env.step(action)
                    if done:
                        break
                autonomous_success = bool(info["success"])
                for candidate in candidates:
                    assisted_success = _assisted_branch(env, candidate["snapshot"], max_steps)
                    for camera, image in candidate["images"].items():
                        rows.append(
                            {
                                "image": image,
                                "proprioception": candidate["proprioception"],
                                "autonomous_success": autonomous_success,
                                "assisted_success": assisted_success,
                                "episode_id": episode_id,
                                "step": candidate["step"],
                                "camera": camera,
                                "disturbance": candidate["disturbance"].name,
                            }
                        )
                episode_id += 1
    return VisualData(
        images=np.stack([row["image"] for row in rows]),
        proprioception=np.stack([row["proprioception"] for row in rows]),
        autonomous_success=np.asarray([row["autonomous_success"] for row in rows]),
        assisted_success=np.asarray([row["assisted_success"] for row in rows]),
        episode_ids=np.asarray([row["episode_id"] for row in rows]),
        steps=np.asarray([row["step"] for row in rows]),
        cameras=np.asarray([row["camera"] for row in rows]),
        tasks=np.full(len(rows), task),
        disturbances=np.asarray([row["disturbance"] for row in rows]),
        metadata={
            "policy_checkpoint": str(policy_checkpoint),
            "task": task,
            "seed": seed,
            "cameras": list(cameras),
            "image_size": [height, width],
        },
    )


def proprioception(observation: np.ndarray) -> np.ndarray:
    """Select robot/time/task inputs while excluding object and goal state."""
    return np.concatenate([observation[0:5], observation[17:25], observation[31:35]]).astype(
        np.float32
    )


class VisualValueHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def run_visual_probe(
    train: VisualData,
    evaluation: VisualData,
    output_dir: str | Path,
    epochs: int = 30,
    seed: int = 27,
    device: str = "auto",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved = resolve_device(device)
    train_front = train.subset(train.cameras == "frontview")
    train_features = _frozen_resnet_features(train_front, resolved)
    eval_features = _frozen_resnet_features(evaluation, resolved)
    normalizer = Normalizer.fit(train_front.proprioception)
    train_proprio = normalizer.transform(train_front.proprioception)
    eval_proprio = normalizer.transform(evaluation.proprioception)
    train_inputs = np.concatenate([train_features, train_proprio], axis=1).astype(np.float32)
    eval_inputs = np.concatenate([eval_features, eval_proprio], axis=1).astype(np.float32)
    targets = train_front.helpful.astype(np.float32)
    _seed_everything(seed)
    head = VisualValueHead(train_inputs.shape[1]).to(resolved)
    positives = max(1.0, float(targets.sum()))
    negatives = max(1.0, float(len(targets) - positives))
    positive_weight = torch.tensor(negatives / positives, device=resolved)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_inputs), torch.from_numpy(targets)),
        batch_size=min(128, len(targets)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-6)
    for _ in range(epochs):
        for features, batch_targets in loader:
            features = features.to(resolved)
            batch_targets = batch_targets.to(resolved)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                head(features), batch_targets, pos_weight=positive_weight
            )
            loss.backward()
            optimizer.step()
    head.eval()
    with torch.inference_mode():
        probabilities = (
            torch.sigmoid(head(torch.from_numpy(eval_inputs).to(resolved))).cpu().numpy()
        )
    metrics: dict[str, dict[str, float]] = {}
    for camera in np.unique(evaluation.cameras):
        mask = evaluation.cameras == camera
        metrics[str(camera)] = binary_metrics(evaluation.helpful[mask], probabilities[mask], 0.5)
    torch.save(
        {
            "head": head.state_dict(),
            "feature_dim": train_inputs.shape[1],
            "proprio_mean": normalizer.mean,
            "proprio_std": normalizer.std,
            "metadata": {
                "encoder": "torchvision/resnet18/IMAGENET1K_V1",
                "encoder_frozen": True,
                "train_camera": "frontview",
                "metrics": metrics,
            },
        },
        output / "visual_value_probe.pt",
    )
    (output / "visual_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return {"checkpoint": str(output / "visual_value_probe.pt"), "metrics": metrics}


def _frozen_resnet_features(data: VisualData, device: torch.device) -> np.ndarray:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as error:
        raise RuntimeError("install visual dependencies with `uv sync --extra vision`") from error
    weights = ResNet18_Weights.IMAGENET1K_V1
    encoder = resnet18(weights=weights)
    encoder.fc = nn.Identity()
    encoder = encoder.to(device).eval()
    transform = weights.transforms()
    features: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, data.sample_count, 64):
            images = torch.from_numpy(data.images[start : start + 64]).permute(0, 3, 1, 2)
            batch = transform(images).to(device)
            features.append(encoder(batch).cpu().numpy())
    return np.concatenate(features).astype(np.float32)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
