from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from intervenesim.dataset import TrajectoryBuilder, TrajectoryData
from intervenesim.disturbances import DISTURBANCE_NAMES, Disturbance
from intervenesim.environment import PickPlaceEnv
from intervenesim.expert import ScriptedExpert
from intervenesim.policy import PolicyAgent
from intervenesim.risk import RiskBuilder, RiskData
from intervenesim.supervision import InterventionSupervisor

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


@dataclass(frozen=True)
class RecoveryCollection:
    demonstrations: TrajectoryData
    risk_data: RiskData
    summary: CollectionSummary


def collect_clean_demonstrations(
    episodes: int,
    seed: int,
    max_steps: int,
    attempt_multiplier: int = 3,
    progress: ProgressCallback | None = None,
    task: str = "can",
    task_conditioning: bool = False,
) -> tuple[TrajectoryData, CollectionSummary]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    builder = TrajectoryBuilder()
    successful = 0
    attempts = 0
    with PickPlaceEnv(max_steps=max_steps, task=task, task_conditioning=task_conditioning) as env:
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
        metadata={
            "kind": "clean",
            "seed": seed,
            "task": task,
            "task_conditioning": task_conditioning,
            "collection": summary.to_dict(),
        },
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
    task: str = "can",
    task_conditioning: bool = False,
) -> tuple[TrajectoryData, CollectionSummary]:
    bundle = collect_recovery_bundle(
        policy,
        episodes,
        seed,
        max_steps,
        attempt_multiplier,
        disturbance_names,
        risk_horizon=16,
        nominal_risk_episodes=0,
        progress=progress,
        task=task,
        task_conditioning=task_conditioning,
    )
    return bundle.demonstrations, bundle.summary


def collect_recovery_bundle(
    policy: PolicyAgent,
    episodes: int,
    seed: int,
    max_steps: int,
    attempt_multiplier: int = 4,
    disturbance_names: tuple[str, ...] | None = None,
    risk_horizon: int = 16,
    nominal_risk_episodes: int = 0,
    progress: ProgressCallback | None = None,
    task: str = "can",
    task_conditioning: bool = False,
) -> RecoveryCollection:
    names = disturbance_names or tuple(name for name in DISTURBANCE_NAMES if name != "nominal")
    if not names or any(name == "nominal" for name in names):
        raise ValueError("recovery collection requires one or more non-nominal disturbances")
    builder = TrajectoryBuilder()
    risk_builder = RiskBuilder()
    successful = 0
    attempts = 0
    counts = {name: 0 for name in names}
    with PickPlaceEnv(max_steps=max_steps, task=task, task_conditioning=task_conditioning) as env:
        while successful < episodes and attempts < episodes * attempt_multiplier:
            disturbance_name = names[attempts % len(names)]
            episode_seed = seed + attempts
            disturbance = Disturbance(disturbance_name, episode_seed + 10_000)
            disturbance.reset(env.action_dim)
            state = env.reset(episode_seed)
            expert = ScriptedExpert()
            supervisor = InterventionSupervisor()
            supervisor.reset(state)
            intervening = False
            intervention_index: int | None = None
            episode = TrajectoryBuilder()
            risk_observations: list[np.ndarray] = []
            for step in range(max_steps):
                state = disturbance.before_step(env, state, step)
                if not intervening:
                    risk_observations.append(state.observation.copy())
                if not intervening and supervisor.should_intervene(state, disturbance, step):
                    intervening = True
                    intervention_index = len(risk_observations) - 1
                    expert.reset(state, recovering=True)
                if intervening:
                    phase = int(expert.phase)
                    rejected_action = policy.action(state.observation)
                    action = expert.action(state)
                    episode.append(
                        state.observation,
                        action,
                        episode_id=successful,
                        source="recovery",
                        disturbance=disturbance_name,
                        intervention=True,
                        phase=phase,
                        rejected_action=rejected_action,
                    )
                else:
                    action = policy.action(state.observation)
                # The disturbance creates the off-nominal state. Once the privileged
                # supervisor takes control, execute its correction directly so the
                # collected label represents recovery from that state rather than the
                # expert's ability to fight a permanently corrupted control channel.
                executed = action if intervening else disturbance.transform_action(action, step)
                state, _, done, info = env.step(executed)
                if done:
                    break
            attempts += 1
            if info["success"] and episode.observations:
                builder.extend(episode)
                risk_builder.append_episode(
                    risk_observations,
                    intervention_index,
                    risk_horizon,
                    successful,
                    disturbance_name,
                )
                successful += 1
                counts[disturbance_name] += 1
                if progress:
                    progress(f"recovery demonstration {successful}/{episodes} ({disturbance_name})")
        nominal_successful = 0
        nominal_attempts = 0
        while (
            nominal_successful < nominal_risk_episodes
            and nominal_attempts < nominal_risk_episodes * attempt_multiplier
        ):
            state = env.reset(seed + 500_000 + nominal_attempts)
            observations: list[np.ndarray] = []
            for _ in range(max_steps):
                observations.append(state.observation.copy())
                state, _, done, info = env.step(policy.action(state.observation))
                if done:
                    break
            nominal_attempts += 1
            if info["success"]:
                risk_builder.append_episode(
                    observations,
                    intervention_index=None,
                    horizon=risk_horizon,
                    episode_id=episodes + nominal_successful,
                    disturbance="nominal",
                )
                nominal_successful += 1
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
        metadata={
            "kind": "recovery",
            "seed": seed,
            "task": task,
            "task_conditioning": task_conditioning,
            "collection": summary.to_dict(),
        },
    )
    risk_data = risk_builder.build(
        observation_dim=env.observation_dim,
        metadata={
            "kind": "intervention_risk",
            "seed": seed,
            "horizon": risk_horizon,
            "nominal_episodes": nominal_risk_episodes,
            "task": task,
            "task_conditioning": task_conditioning,
        },
    )
    return RecoveryCollection(data, risk_data, summary)
