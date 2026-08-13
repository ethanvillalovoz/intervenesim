from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from intervenesim.environment import PickPlaceEnv, TaskState

DISTURBANCE_NAMES = ("nominal", "object_shift", "action_noise", "action_delay", "gripper_slip")


@dataclass
class Disturbance:
    name: str
    seed: int
    fired: bool = False
    trigger_step: int | None = None
    parameters: dict[str, float | int | list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in DISTURBANCE_NAMES:
            raise ValueError(f"Unknown disturbance {self.name!r}; choose from {DISTURBANCE_NAMES}")
        self._rng = np.random.default_rng(self.seed)
        self._delay: deque[np.ndarray] = deque()
        self._configured = False

    def reset(self, action_dim: int) -> None:
        self.fired = False
        self.trigger_step = None
        self.parameters = {}
        self._delay.clear()
        self._configured = True
        if self.name == "object_shift":
            angle = float(self._rng.uniform(0, 2 * np.pi))
            radius = float(self._rng.uniform(0.07, 0.11))
            self.parameters = {"dx": radius * np.cos(angle), "dy": radius * np.sin(angle)}
        elif self.name == "action_noise":
            self.parameters = {"sigma": 0.18}
        elif self.name == "action_delay":
            delay = int(self._rng.integers(3, 6))
            self.parameters = {"delay_steps": delay}
            for _ in range(delay):
                self._delay.append(np.zeros(action_dim, dtype=np.float32))
        elif self.name == "gripper_slip":
            angle = float(self._rng.uniform(0, 2 * np.pi))
            radius = float(self._rng.uniform(0.045, 0.085))
            self.parameters = {"dx": radius * np.cos(angle), "dy": radius * np.sin(angle)}

    def transform_action(self, action: np.ndarray, step: int) -> np.ndarray:
        if not self._configured:
            self.reset(len(action))
        transformed = np.asarray(action, dtype=np.float32).copy()
        if self.name == "action_noise" and step >= 8:
            sigma = float(self.parameters["sigma"])
            transformed[:3] += self._rng.normal(0.0, sigma, size=3).astype(np.float32)
            if not self.fired:
                self._fire(step)
        elif self.name == "action_delay":
            self._delay.append(transformed)
            transformed = self._delay.popleft()
            if step >= int(self.parameters["delay_steps"]) and not self.fired:
                self._fire(step)
        return np.clip(transformed, -1.0, 1.0)

    def before_step(self, env: PickPlaceEnv, state: TaskState, step: int) -> TaskState:
        if (
            self.name == "object_shift"
            and not self.fired
            and 14 <= step <= 28
            and not state.can_lifted
        ):
            position = state.can_pos.copy()
            position[0] = np.clip(position[0] + float(self.parameters["dx"]), -0.18, 0.24)
            position[1] = np.clip(position[1] + float(self.parameters["dy"]), -0.43, -0.10)
            position[2] = env.object_rest_z
            env.teleport_can(position)
            self._fire(step)
            return env.task_state()
        if self.name == "gripper_slip" and not self.fired and state.can_lifted:
            position = state.can_pos.copy()
            position[0] = np.clip(position[0] + float(self.parameters["dx"]), -0.18, 0.24)
            position[1] = np.clip(position[1] + float(self.parameters["dy"]), -0.43, 0.18)
            position[2] = env.object_rest_z + 0.02
            env.teleport_can(position)
            self._fire(step)
            return env.task_state()
        return state

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seed": self.seed,
            "fired": self.fired,
            "trigger_step": self.trigger_step,
            "parameters": self.parameters,
        }

    def _fire(self, step: int) -> None:
        self.fired = True
        self.trigger_step = step
