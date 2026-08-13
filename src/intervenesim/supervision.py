from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from intervenesim.disturbances import Disturbance
from intervenesim.environment import TaskState


@dataclass
class InterventionSupervisor:
    """Privileged reference supervisor used to create intervention labels."""

    stall_horizon: int = 45
    hard_step: int = 130
    post_disturbance_delay: int = 2
    stalled_steps: int = 0
    last_progress: float = float("-inf")

    def reset(self, state: TaskState) -> None:
        self.stalled_steps = 0
        self.last_progress = progress_score(state)

    def should_intervene(self, state: TaskState, disturbance: Disturbance, step: int) -> bool:
        score = progress_score(state)
        if score <= self.last_progress + 1e-4:
            self.stalled_steps += 1
        else:
            self.stalled_steps = 0
            self.last_progress = score
        disturbance_ready = (
            disturbance.fired
            and step >= (disturbance.trigger_step or 0) + self.post_disturbance_delay
        )
        return bool(
            disturbance_ready or self.stalled_steps >= self.stall_horizon or step >= self.hard_step
        )


def progress_score(state: TaskState) -> float:
    eef_to_object = float(np.linalg.norm(state.eef_pos - state.can_pos))
    object_to_goal = float(np.linalg.norm(state.can_pos[:2] - state.goal_pos[:2]))
    score = -0.2 * eef_to_object
    if state.can_lifted:
        score += 1.0 - object_to_goal
    if state.near_goal:
        score += 1.0
    return score
