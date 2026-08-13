from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from intervenesim.dataset import TrajectoryBuilder, TrajectoryData
from intervenesim.disturbances import DISTURBANCE_NAMES, Disturbance
from intervenesim.environment import PickPlaceEnv, TaskState
from intervenesim.expert import ScriptedExpert
from intervenesim.policy import PolicyAgent

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class CollectionSummary:
    requested_episodes: int
    successful_episodes: int
    attempted_episodes: int
    samples: int
    intervention_samples: int
    disturbance_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_episodes": self.requested_episodes,
            "successful_episodes": self.successful_episodes,
            "attempted_episodes": self.attempted_episodes,
            "samples": self.samples,
            "intervention_samples": self.intervention_samples,
            "disturbance_counts": self.disturbance_counts,
        }


def collect_clean_demonstrations(
    episodes: int,
    seed: int,
    max_steps: int,
    attempt_multiplier: int = 3,
    progress: ProgressCallback | None = None,
) -> tuple[TrajectoryData, CollectionSummary]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    builder = TrajectoryBuilder()
    successful = 0
    attempts = 0
    with PickPlaceEnv(max_steps=max_steps) as env:
        while successful < episodes and attempts < episodes * attempt_multiplier:
            episode_seed = seed + attempts
            state = env.reset(episode_seed)
            expert = ScriptedExpert()
            expert.reset(state)
            episode = TrajectoryBuilder()
            for _ in range(max_steps):
                phase = int(expert.phase)
                action = expert.action(state)
                episode.append(
                    state.observation,
                    action,
                    episode_id=successful,
                    source="clean",
                    disturbance="nominal",
                    intervention=False,
                    phase=phase,
                )
                state, _, done, info = env.step(action)
                if done:
                    break
            attempts += 1
            if info["success"]:
                builder.extend(episode)
                successful += 1
                if progress:
                    progress(f"clean demonstration {successful}/{episodes}")
    if successful < episodes:
        raise RuntimeError(f"expert produced only {successful}/{episodes} successful episodes")
    summary = CollectionSummary(
        requested_episodes=episodes,
        successful_episodes=successful,
        attempted_episodes=attempts,
        samples=len(builder.observations),
        intervention_samples=0,
        disturbance_counts={"nominal": successful},
    )
    data = builder.build(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        metadata={"kind": "clean", "seed": seed, "collection": summary.to_dict()},
    )
    return data, summary


def collect_recovery_demonstrations(
    policy: PolicyAgent,
    episodes: int,
    seed: int,
    max_steps: int,
    attempt_multiplier: int = 4,
    disturbance_names: tuple[str, ...] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[TrajectoryData, CollectionSummary]:
    names = disturbance_names or tuple(name for name in DISTURBANCE_NAMES if name != "nominal")
    if not names or any(name == "nominal" for name in names):
        raise ValueError("recovery collection requires one or more non-nominal disturbances")
    builder = TrajectoryBuilder()
    successful = 0
    attempts = 0
    counts = {name: 0 for name in names}
    with PickPlaceEnv(max_steps=max_steps) as env:
        while successful < episodes and attempts < episodes * attempt_multiplier:
            disturbance_name = names[attempts % len(names)]
            episode_seed = seed + attempts
            disturbance = Disturbance(disturbance_name, episode_seed + 10_000)
            disturbance.reset(env.action_dim)
            state = env.reset(episode_seed)
            expert = ScriptedExpert()
            intervening = False
            episode = TrajectoryBuilder()
            last_progress = _progress_score(state)
            stalled_steps = 0
            for step in range(max_steps):
                state = disturbance.before_step(env, state, step)
                progress_score = _progress_score(state)
                if progress_score <= last_progress + 1e-4:
                    stalled_steps += 1
                else:
                    stalled_steps = 0
                    last_progress = progress_score
                disturbance_ready = (
                    disturbance.fired and step >= (disturbance.trigger_step or 0) + 2
                )
                if not intervening and (disturbance_ready or stalled_steps >= 45 or step >= 130):
                    intervening = True
                    expert.reset(state, recovering=True)
                if intervening:
                    phase = int(expert.phase)
                    action = expert.action(state)
                    episode.append(
                        state.observation,
                        action,
                        episode_id=successful,
                        source="recovery",
                        disturbance=disturbance_name,
                        intervention=True,
                        phase=phase,
                    )
                else:
                    action = policy.action(state.observation)
                executed = disturbance.transform_action(action, step)
                state, _, done, info = env.step(executed)
                if done:
                    break
            attempts += 1
            if info["success"] and episode.observations:
                builder.extend(episode)
                successful += 1
                counts[disturbance_name] += 1
                if progress:
                    progress(f"recovery demonstration {successful}/{episodes} ({disturbance_name})")
    if successful < episodes:
        raise RuntimeError(f"expert recovered only {successful}/{episodes} requested episodes")
    summary = CollectionSummary(
        requested_episodes=episodes,
        successful_episodes=successful,
        attempted_episodes=attempts,
        samples=len(builder.observations),
        intervention_samples=len(builder.observations),
        disturbance_counts=counts,
    )
    data = builder.build(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        metadata={"kind": "recovery", "seed": seed, "collection": summary.to_dict()},
    )
    return data, summary


def _progress_score(state: TaskState) -> float:
    eef_to_can = float(np.linalg.norm(state.eef_pos - state.can_pos))
    can_to_goal = float(np.linalg.norm(state.can_pos[:2] - state.goal_pos[:2]))
    score = -0.2 * eef_to_can
    if state.can_lifted:
        score += 1.0 - can_to_goal
    if state.near_goal:
        score += 1.0
    return score
