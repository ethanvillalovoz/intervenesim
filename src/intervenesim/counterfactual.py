from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intervenesim.disturbances import Disturbance
from intervenesim.environment import EnvSnapshot, PickPlaceEnv
from intervenesim.expert import ScriptedExpert
from intervenesim.policy import PolicyAgent


@dataclass
class CounterfactualData:
    observations: np.ndarray
    observation_deltas: np.ndarray
    autonomous_actions: np.ndarray
    ensemble_uncertainty: np.ndarray
    autonomous_success: np.ndarray
    assisted_success: np.ndarray
    steps: np.ndarray
    episode_ids: np.ndarray
    policy_seeds: np.ndarray
    tasks: np.ndarray
    disturbances: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.observation_deltas = np.asarray(self.observation_deltas, dtype=np.float32)
        self.autonomous_actions = np.asarray(self.autonomous_actions, dtype=np.float32)
        self.ensemble_uncertainty = np.asarray(self.ensemble_uncertainty, dtype=np.float32)
        self.autonomous_success = np.asarray(self.autonomous_success, dtype=bool)
        self.assisted_success = np.asarray(self.assisted_success, dtype=bool)
        self.steps = np.asarray(self.steps, dtype=np.int16)
        self.episode_ids = np.asarray(self.episode_ids, dtype=np.int32)
        self.policy_seeds = np.asarray(self.policy_seeds, dtype=np.int32)
        self.tasks = np.asarray(self.tasks, dtype="U16")
        self.disturbances = np.asarray(self.disturbances, dtype="U24")
        lengths = {
            len(self.observations),
            len(self.observation_deltas),
            len(self.autonomous_actions),
            len(self.ensemble_uncertainty),
            len(self.autonomous_success),
            len(self.assisted_success),
            len(self.steps),
            len(self.episode_ids),
            len(self.policy_seeds),
            len(self.tasks),
            len(self.disturbances),
        }
        if len(lengths) != 1:
            raise ValueError("counterfactual arrays must have equal lengths")

    @property
    def sample_count(self) -> int:
        return len(self.observations)

    @property
    def helpful(self) -> np.ndarray:
        return self.assisted_success & ~self.autonomous_success

    @property
    def signed_value(self) -> np.ndarray:
        return self.assisted_success.astype(np.int8) - self.autonomous_success.astype(np.int8)

    @property
    def features(self) -> np.ndarray:
        return np.concatenate(
            [self.observations, self.observation_deltas, self.autonomous_actions], axis=1
        ).astype(np.float32)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            observations=self.observations,
            observation_deltas=self.observation_deltas,
            autonomous_actions=self.autonomous_actions,
            ensemble_uncertainty=self.ensemble_uncertainty,
            autonomous_success=self.autonomous_success,
            assisted_success=self.assisted_success,
            steps=self.steps,
            episode_ids=self.episode_ids,
            policy_seeds=self.policy_seeds,
            tasks=self.tasks,
            disturbances=self.disturbances,
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> CounterfactualData:
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                observations=archive["observations"],
                observation_deltas=archive["observation_deltas"],
                autonomous_actions=archive["autonomous_actions"],
                ensemble_uncertainty=archive["ensemble_uncertainty"],
                autonomous_success=archive["autonomous_success"],
                assisted_success=archive["assisted_success"],
                steps=archive["steps"],
                episode_ids=archive["episode_ids"],
                policy_seeds=archive["policy_seeds"],
                tasks=archive["tasks"],
                disturbances=archive["disturbances"],
                metadata=json.loads(str(archive["metadata"].item())),
            )

    @classmethod
    def concatenate(cls, parts: list[CounterfactualData]) -> CounterfactualData:
        if not parts:
            raise ValueError("cannot concatenate empty counterfactual data")
        ids: list[np.ndarray] = []
        offset = 0
        for part in parts:
            local = part.episode_ids - part.episode_ids.min(initial=0)
            ids.append(local + offset)
            offset += int(local.max(initial=-1)) + 1
        return cls(
            observations=np.concatenate([part.observations for part in parts]),
            observation_deltas=np.concatenate([part.observation_deltas for part in parts]),
            autonomous_actions=np.concatenate([part.autonomous_actions for part in parts]),
            ensemble_uncertainty=np.concatenate([part.ensemble_uncertainty for part in parts]),
            autonomous_success=np.concatenate([part.autonomous_success for part in parts]),
            assisted_success=np.concatenate([part.assisted_success for part in parts]),
            steps=np.concatenate([part.steps for part in parts]),
            episode_ids=np.concatenate(ids),
            policy_seeds=np.concatenate([part.policy_seeds for part in parts]),
            tasks=np.concatenate([part.tasks for part in parts]),
            disturbances=np.concatenate([part.disturbances for part in parts]),
            metadata={"parts": [part.metadata for part in parts]},
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "episode_id": self.episode_ids,
                "policy_seed": self.policy_seeds,
                "task": self.tasks,
                "disturbance": self.disturbances,
                "step": self.steps,
                "autonomous_success": self.autonomous_success,
                "assisted_success": self.assisted_success,
                "helpful": self.helpful,
                "signed_value": self.signed_value,
                "ensemble_uncertainty": self.ensemble_uncertainty,
            }
        )

    def subset(self, mask: np.ndarray) -> CounterfactualData:
        return CounterfactualData(
            self.observations[mask],
            self.observation_deltas[mask],
            self.autonomous_actions[mask],
            self.ensemble_uncertainty[mask],
            self.autonomous_success[mask],
            self.assisted_success[mask],
            self.steps[mask],
            self.episode_ids[mask],
            self.policy_seeds[mask],
            self.tasks[mask],
            self.disturbances[mask],
            self.metadata,
        )


def collect_counterfactual_data(
    policy_checkpoint: str | Path,
    ensemble_checkpoints: tuple[str | Path, ...],
    task: str,
    disturbances: tuple[str, ...],
    episodes: int,
    seed: int,
    candidate_steps: tuple[int, ...],
    max_steps: int,
    policy_seed: int,
    device: str = "auto",
) -> CounterfactualData:
    policy = PolicyAgent.load(policy_checkpoint, device=device)
    ensemble = [PolicyAgent.load(path, device=device) for path in ensemble_checkpoints]
    rows: list[dict[str, Any]] = []
    episode_id = 0
    with PickPlaceEnv(max_steps=max_steps, task=task, task_conditioning=True) as env:
        for disturbance_index, disturbance_name in enumerate(disturbances):
            for episode in range(episodes):
                episode_seed = seed + disturbance_index * 10_000 + episode
                disturbance_seed = episode_seed + 50_000
                disturbance = Disturbance(disturbance_name, disturbance_seed)
                disturbance.reset(env.action_dim)
                state = env.reset(episode_seed)
                previous = state.observation.copy()
                candidates: list[dict[str, Any]] = []
                info: dict[str, Any] = {"success": False}
                for step in range(max_steps):
                    state = disturbance.before_step(env, state, step)
                    if step in candidate_steps:
                        action = policy.action(state.observation)
                        actions = np.stack(
                            [member.action(state.observation) for member in ensemble]
                        )
                        candidates.append(
                            {
                                "snapshot": env.snapshot(),
                                "disturbance": deepcopy(disturbance),
                                "observation": state.observation.copy(),
                                "observation_delta": state.observation - previous,
                                "action": action,
                                "uncertainty": float(actions.var(axis=0).mean()),
                                "step": step,
                            }
                        )
                    previous = state.observation.copy()
                    action = disturbance.transform_action(policy.action(state.observation), step)
                    state, _, done, info = env.step(action)
                    if done:
                        break
                autonomous_success = bool(info["success"])
                for candidate in candidates:
                    assisted_success = _assisted_branch(
                        env,
                        candidate["snapshot"],
                        max_steps,
                    )
                    rows.append(
                        {
                            **candidate,
                            "autonomous_success": autonomous_success,
                            "assisted_success": assisted_success,
                            "episode_id": episode_id,
                        }
                    )
                episode_id += 1
    if not rows:
        observation_dim = policy.normalizer.mean.shape[0]
        action_dim = policy.model.action_dim
        return CounterfactualData(
            np.empty((0, observation_dim), np.float32),
            np.empty((0, observation_dim), np.float32),
            np.empty((0, action_dim), np.float32),
            np.empty(0, np.float32),
            np.empty(0, bool),
            np.empty(0, bool),
            np.empty(0, np.int16),
            np.empty(0, np.int32),
            np.empty(0, np.int32),
            np.empty(0, "U16"),
            np.empty(0, "U24"),
            {},
        )
    return CounterfactualData(
        observations=np.stack([row["observation"] for row in rows]),
        observation_deltas=np.stack([row["observation_delta"] for row in rows]),
        autonomous_actions=np.stack([row["action"] for row in rows]),
        ensemble_uncertainty=np.asarray([row["uncertainty"] for row in rows]),
        autonomous_success=np.asarray([row["autonomous_success"] for row in rows]),
        assisted_success=np.asarray([row["assisted_success"] for row in rows]),
        steps=np.asarray([row["step"] for row in rows]),
        episode_ids=np.asarray([row["episode_id"] for row in rows]),
        policy_seeds=np.full(len(rows), policy_seed),
        tasks=np.full(len(rows), task),
        disturbances=np.asarray([row["disturbance"].name for row in rows]),
        metadata={
            "policy_checkpoint": str(policy_checkpoint),
            "task": task,
            "seed": seed,
            "episodes": episodes,
            "candidate_steps": list(candidate_steps),
            "branch_semantics": asdict(
                BranchSemantics(
                    autonomous_disturbance_continues=True,
                    assisted_corruption_bypassed=True,
                )
            ),
        },
    )


@dataclass(frozen=True)
class BranchSemantics:
    autonomous_disturbance_continues: bool
    assisted_corruption_bypassed: bool


def _assisted_branch(env: PickPlaceEnv, snapshot: EnvSnapshot, max_steps: int) -> bool:
    state = env.restore(snapshot)
    expert = ScriptedExpert()
    expert.reset(state, recovering=True)
    info: dict[str, Any] = {"success": False}
    for _ in range(snapshot.step, max_steps):
        state, _, done, info = env.step(expert.action(state))
        if done:
            break
    return bool(info["success"])
