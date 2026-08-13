from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from intervenesim.disturbances import Disturbance
from intervenesim.environment import PickPlaceEnv
from intervenesim.policy import PolicyAgent


def evaluate_policies(
    checkpoints: dict[str, str | Path],
    disturbances: tuple[str, ...],
    episodes: int,
    seed: int,
    max_steps: int,
    device: str = "auto",
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    with PickPlaceEnv(max_steps=max_steps) as env:
        for condition, checkpoint in checkpoints.items():
            policy = PolicyAgent.load(checkpoint, device=device)
            for disturbance_name in disturbances:
                for episode in range(episodes):
                    episode_seed = seed + episode
                    disturbance_seed = seed + 50_000 + episode
                    disturbance = Disturbance(disturbance_name, disturbance_seed)
                    disturbance.reset(env.action_dim)
                    state = env.reset(episode_seed)
                    total_reward = 0.0
                    for step in range(max_steps):
                        state = disturbance.before_step(env, state, step)
                        action = policy.action(state.observation)
                        action = disturbance.transform_action(action, step)
                        state, reward, done, info = env.step(action)
                        total_reward += reward
                        if done:
                            break
                    records.append(
                        {
                            "condition": condition,
                            "disturbance": disturbance_name,
                            "episode": episode,
                            "seed": episode_seed,
                            "success": bool(info["success"]),
                            "disturbance_fired": disturbance.fired,
                            "recovered": bool(info["success"] and disturbance.fired),
                            "steps": step + 1,
                            "total_reward": total_reward,
                            "trigger_step": disturbance.trigger_step,
                            "final_can_x": float(state.can_pos[0]),
                            "final_can_y": float(state.can_pos[1]),
                            "final_can_z": float(state.can_pos[2]),
                        }
                    )
                if progress:
                    progress(f"evaluated {condition} / {disturbance_name}")
    return pd.DataFrame.from_records(records)
