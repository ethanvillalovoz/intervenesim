from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from intervenesim.disturbances import Disturbance
from intervenesim.environment import PickPlaceEnv
from intervenesim.policy import PolicyAgent


@dataclass
class HumanCorrectionData:
    observations: np.ndarray
    actions: np.ndarray
    episode_ids: np.ndarray
    steps: np.ndarray
    human_control: np.ndarray
    metadata: dict[str, Any]

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            observations=np.asarray(self.observations, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.float32),
            episode_ids=np.asarray(self.episode_ids, dtype=np.int32),
            steps=np.asarray(self.steps, dtype=np.int16),
            human_control=np.asarray(self.human_control, dtype=bool),
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )
        return output


def capture_human_corrections(
    policy_checkpoint: str | Path,
    output: str | Path,
    task: str = "can",
    disturbance_name: str = "gripper_slip",
    episodes: int = 5,
    seed: int = 27,
    max_steps: int = 260,
    device: str = "auto",
) -> HumanCorrectionData:
    """Run policy-first rollouts and record keyboard takeover actions after the T key."""
    from pynput.keyboard import Key
    from robosuite.devices import Keyboard

    class TakeoverKeyboard(Keyboard):
        takeover = False

        def on_release(self, key):  # noqa: ANN001
            try:
                if key.char == "t":
                    self.takeover = True
                    return
            except AttributeError:
                if key == Key.esc:
                    self._reset_state = 1
                    self._enabled = False
                    return
            super().on_release(key)

    policy = PolicyAgent.load(policy_checkpoint, device=device)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_ids: list[int] = []
    steps: list[int] = []
    human_control: list[bool] = []
    episode_summaries: list[dict[str, Any]] = []
    print("Policy is active. Press T to take over; Esc ends the current episode.")
    with PickPlaceEnv(max_steps=max_steps, render=True, task=task, task_conditioning=True) as env:
        keyboard = TakeoverKeyboard(env=env._env)
        for episode in range(episodes):
            episode_seed = seed + episode
            disturbance = Disturbance(disturbance_name, episode_seed + 50_000)
            disturbance.reset(env.action_dim)
            state = env.reset(episode_seed)
            keyboard.takeover = False
            keyboard.start_control()
            env.render()
            takeover_step: int | None = None
            info: dict[str, Any] = {"success": False}
            for step in range(max_steps):
                started = time.time()
                state = disturbance.before_step(env, state, step)
                if keyboard.takeover:
                    if takeover_step is None:
                        takeover_step = step
                    action_dict = keyboard.input2action()
                    if action_dict is None:
                        break
                    action = _device_action(env, action_dict)
                    is_human = True
                else:
                    action = disturbance.transform_action(policy.action(state.observation), step)
                    is_human = False
                observations.append(state.observation.copy())
                actions.append(action.copy())
                episode_ids.append(episode)
                steps.append(step)
                human_control.append(is_human)
                state, _, done, info = env.step(action)
                env.render()
                remaining = 1 / 20 - (time.time() - started)
                if remaining > 0:
                    time.sleep(remaining)
                if done:
                    break
            episode_summaries.append(
                {
                    "episode": episode,
                    "seed": episode_seed,
                    "success": bool(info["success"]),
                    "takeover_step": takeover_step,
                    "disturbance_fired": disturbance.fired,
                    "disturbance_trigger_step": disturbance.trigger_step,
                }
            )
            print(
                f"episode {episode + 1}/{episodes}: success={bool(info['success'])}, "
                f"takeover_step={takeover_step}"
            )
    data = HumanCorrectionData(
        observations=np.stack(observations),
        actions=np.stack(actions),
        episode_ids=np.asarray(episode_ids),
        steps=np.asarray(steps),
        human_control=np.asarray(human_control),
        metadata={
            "schema_version": 1,
            "policy_checkpoint": str(policy_checkpoint),
            "task": task,
            "disturbance": disturbance_name,
            "seed": seed,
            "episodes": episode_summaries,
            "controls": {
                "takeover": "t",
                "reset_episode": "escape",
                "translation": "arrow keys, period, semicolon",
                "gripper": "space",
                "rotation": "e/r, y/h, o/p",
            },
        },
    )
    data.save(output)
    return data


def _device_action(env: PickPlaceEnv, input_actions: dict[str, np.ndarray]) -> np.ndarray:
    robot = env._env.robots[0]
    action_dict = deepcopy(input_actions)
    for arm in robot.arms:
        controller = robot.part_controllers[arm]
        suffix = "delta" if controller.input_type == "delta" else "abs"
        action_dict[arm] = input_actions[f"{arm}_{suffix}"]
    return np.asarray(robot.create_action_vector(action_dict), dtype=np.float32)
