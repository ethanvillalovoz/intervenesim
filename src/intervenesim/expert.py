from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from intervenesim.environment import TaskState


class Phase(IntEnum):
    APPROACH = 0
    DESCEND = 1
    CLOSE = 2
    LIFT = 3
    MOVE = 4
    LOWER = 5
    OPEN = 6
    RETREAT = 7


@dataclass
class ScriptedExpert:
    """Closed-loop operational-space expert that can start from recovery states."""

    position_scale: float = 0.05
    phase: Phase = Phase.APPROACH
    phase_steps: int = 0

    def reset(self, state: TaskState, recovering: bool = False) -> None:
        self.phase_steps = 0
        if state.can_lifted:
            self.phase = Phase.MOVE
        elif recovering and state.near_can and self._gripper_closed(state):
            self.phase = Phase.CLOSE
        else:
            self.phase = Phase.APPROACH

    def action(self, state: TaskState) -> np.ndarray:
        self._advance(state)
        target, gripper = self._target(state)
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip((target - state.eef_pos) / self.position_scale, -1.0, 1.0)
        action[6] = gripper
        self.phase_steps += 1
        return action

    def _advance(self, state: TaskState) -> None:
        can = state.can_pos
        eef = state.eef_pos
        if self.phase is Phase.APPROACH:
            target = can + np.array([0.0, 0.0, state.grasp_offset_z + 0.105], dtype=np.float32)
            if np.linalg.norm(target - eef) < 0.02 or self.phase_steps > 70:
                self._set_phase(Phase.DESCEND)
        elif self.phase is Phase.DESCEND:
            target = can + np.array([0.0, 0.0, state.grasp_offset_z], dtype=np.float32)
            if np.linalg.norm(target - eef) < 0.014 or self.phase_steps > 55:
                self._set_phase(Phase.CLOSE)
        elif self.phase is Phase.CLOSE:
            if state.can_lifted:
                self._set_phase(Phase.MOVE)
            elif self.phase_steps > 20:
                self._set_phase(Phase.LIFT)
        elif self.phase is Phase.LIFT:
            if state.can_lifted and eef[2] > 0.995:
                self._set_phase(Phase.MOVE)
            elif self.phase_steps > 55 and not state.can_lifted:
                self._set_phase(Phase.APPROACH)
        elif self.phase is Phase.MOVE:
            target = self._goal_eef(state, z_offset=0.23)
            if np.linalg.norm(target - eef) < 0.028:
                self._set_phase(Phase.LOWER)
            elif not state.can_lifted and self.phase_steps > 8:
                self._set_phase(Phase.APPROACH)
        elif self.phase is Phase.LOWER:
            target = self._goal_eef(state, z_offset=0.10)
            if np.linalg.norm(target - eef) < 0.022 or self.phase_steps > 65:
                self._set_phase(Phase.OPEN)
        elif self.phase is Phase.OPEN:
            if self.phase_steps > 18:
                self._set_phase(Phase.RETREAT)

    def _target(self, state: TaskState) -> tuple[np.ndarray, float]:
        can = state.can_pos
        eef = state.eef_pos
        if self.phase is Phase.APPROACH:
            return (
                can + np.array([0.0, 0.0, state.grasp_offset_z + 0.105], dtype=np.float32),
                -1.0,
            )
        if self.phase is Phase.DESCEND:
            return can + np.array([0.0, 0.0, state.grasp_offset_z], dtype=np.float32), -1.0
        if self.phase is Phase.CLOSE:
            return eef.copy(), 1.0
        if self.phase is Phase.LIFT:
            return np.array([eef[0], eef[1], 1.08], dtype=np.float32), 1.0
        if self.phase is Phase.MOVE:
            return self._goal_eef(state, z_offset=0.23), 1.0
        if self.phase is Phase.LOWER:
            return self._goal_eef(state, z_offset=0.10), 1.0
        if self.phase is Phase.OPEN:
            return eef.copy(), -1.0
        return np.array([eef[0] - 0.08, eef[1], 1.04], dtype=np.float32), -1.0

    @staticmethod
    def _goal_eef(state: TaskState, z_offset: float) -> np.ndarray:
        # The object is held slightly to the positive-x side of the Panda tool center.
        transport_clearance = 0.13 if z_offset > 0.15 else 0.0
        return np.array(
            [
                state.goal_pos[0] - 0.022,
                state.goal_pos[1] - 0.005,
                state.place_eef_z + transport_clearance,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _gripper_closed(state: TaskState) -> bool:
        return bool(np.mean(np.abs(state.gripper_qpos)) < 0.012)

    def _set_phase(self, phase: Phase) -> None:
        self.phase = phase
        self.phase_steps = 0
