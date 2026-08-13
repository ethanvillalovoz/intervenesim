from __future__ import annotations

import contextlib
import io
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")


@dataclass(frozen=True)
class TaskState:
    observation: np.ndarray
    eef_pos: np.ndarray
    can_pos: np.ndarray
    goal_pos: np.ndarray
    gripper_qpos: np.ndarray
    can_to_eef: np.ndarray
    can_to_goal: np.ndarray
    can_lifted: bool
    near_can: bool
    near_goal: bool


class PickPlaceEnv:
    """Thin, deterministic wrapper around robosuite's Panda PickPlaceCan task."""

    observation_names = (
        "eef_x",
        "eef_y",
        "eef_z",
        "gripper_left",
        "gripper_right",
        "can_x",
        "can_y",
        "can_z",
        "goal_x",
        "goal_y",
        "goal_z",
        "can_to_eef_x",
        "can_to_eef_y",
        "can_to_eef_z",
        "can_to_goal_x",
        "can_to_goal_y",
        "can_to_goal_z",
        "joint_vel_0",
        "joint_vel_1",
        "joint_vel_2",
        "joint_vel_3",
        "joint_vel_4",
        "joint_vel_5",
        "joint_vel_6",
    )

    def __init__(
        self,
        max_steps: int = 260,
        render: bool = False,
        offscreen: bool = False,
    ) -> None:
        # Delay imports so dataset/model unit tests do not require a simulator context.
        with contextlib.redirect_stderr(io.StringIO()):
            import robosuite as suite
            from robosuite.controllers import load_composite_controller_config
            from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

        logging.getLogger("robosuite").setLevel(logging.ERROR)
        ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)
        for handler in ROBOSUITE_DEFAULT_LOGGER.handlers:
            handler.setLevel(logging.ERROR)
        controller = load_composite_controller_config(controller="BASIC")
        self._env = suite.make(
            env_name="PickPlaceCan",
            robots="Panda",
            controller_configs=controller,
            has_renderer=render,
            has_offscreen_renderer=offscreen,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=True,
            control_freq=20,
            horizon=max_steps,
            initialization_noise=None,
            ignore_done=False,
        )
        self.max_steps = max_steps
        self._offscreen = offscreen
        self._step = 0
        self._last_raw: dict[str, Any] | None = None
        self._rng = np.random.default_rng(0)

    @property
    def action_dim(self) -> int:
        return int(self._env.action_dim)

    @property
    def observation_dim(self) -> int:
        return len(self.observation_names)

    @property
    def step_count(self) -> int:
        return self._step

    def reset(self, seed: int) -> TaskState:
        self._rng = np.random.default_rng(seed)
        # robosuite 1.5 placement samplers use NumPy's legacy global generator.
        legacy_state = np.random.get_state()
        np.random.seed(seed)
        try:
            raw = self._env.reset()
        finally:
            np.random.set_state(legacy_state)
        self._step = 0
        self._last_raw = raw
        return self.task_state(raw)

    def step(self, action: np.ndarray) -> tuple[TaskState, float, bool, dict[str, Any]]:
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        raw, reward, done, info = self._env.step(clipped)
        self._step += 1
        self._last_raw = raw
        success = self.success
        done = bool(done or success or self._step >= self.max_steps)
        result_info = dict(info)
        result_info.update(success=success, step=self._step)
        return self.task_state(raw), float(reward), done, result_info

    @property
    def success(self) -> bool:
        return bool(self._env._check_success())

    @property
    def target_position(self) -> np.ndarray:
        # Can is the fourth object / fourth target bin in PickPlaceCan.
        return np.asarray(self._env.target_bin_placements[3], dtype=np.float32).copy()

    def task_state(self, raw: dict[str, Any] | None = None) -> TaskState:
        if raw is None:
            if self._last_raw is None:
                raise RuntimeError("reset() must be called before reading task state")
            raw = self._last_raw
        eef = np.asarray(raw["robot0_eef_pos"], dtype=np.float32)
        can = np.asarray(raw["Can_pos"], dtype=np.float32)
        goal = self.target_position
        gripper = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32)
        joint_vel = np.asarray(raw["robot0_joint_vel"], dtype=np.float32)
        can_to_eef = eef - can
        can_to_goal = goal - can
        observation = np.concatenate(
            [eef, gripper, can, goal, can_to_eef, can_to_goal, joint_vel]
        ).astype(np.float32)
        return TaskState(
            observation=observation,
            eef_pos=eef,
            can_pos=can,
            goal_pos=goal,
            gripper_qpos=gripper,
            can_to_eef=can_to_eef,
            can_to_goal=can_to_goal,
            can_lifted=bool(can[2] > 0.925),
            near_can=bool(np.linalg.norm(can_to_eef) < 0.075),
            near_goal=bool(np.linalg.norm(can[:2] - goal[:2]) < 0.075),
        )

    def teleport_can(self, position: np.ndarray) -> None:
        qpos = np.asarray(self._env.sim.data.get_joint_qpos("Can_joint0")).copy()
        qpos[:3] = np.asarray(position, dtype=np.float64)
        self._env.sim.data.set_joint_qpos("Can_joint0", qpos)
        self._env.sim.data.set_joint_qvel("Can_joint0", np.zeros(6, dtype=np.float64))
        self._env.sim.forward()
        self._refresh_observation()

    def _refresh_observation(self) -> None:
        # The observable cache is updated inside _get_observations.
        self._last_raw = self._env._get_observations(force_update=True)

    def render(self) -> None:
        with contextlib.suppress(Exception):
            self._env.render()

    def capture_frame(
        self,
        width: int = 320,
        height: int = 240,
        camera: str = "frontview",
    ) -> np.ndarray:
        """Render an RGB frame without requiring an on-screen window."""
        if not self._offscreen:
            raise RuntimeError("create PickPlaceEnv with offscreen=True to capture frames")
        frame = np.asarray(
            self._env.sim.render(camera_name=camera, width=width, height=height),
            dtype=np.uint8,
        )
        # MuJoCo's offscreen framebuffer has an OpenGL bottom-left origin.
        return np.flipud(frame).copy()

    def close(self) -> None:
        self._env.close()

    def __enter__(self) -> PickPlaceEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
