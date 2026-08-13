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
    task_name: str
    object_geometry: np.ndarray
    place_eef_z: float
    grasp_offset_z: float


class PickPlaceEnv:
    """Thin, deterministic wrapper around robosuite's Panda PickPlaceCan task."""

    base_observation_names = (
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

    task_specs = {
        "can": ("PickPlaceCan", "Can", 3),
        "milk": ("PickPlaceMilk", "Milk", 0),
        "bread": ("PickPlaceBread", "Bread", 1),
        "cereal": ("PickPlaceCereal", "Cereal", 2),
    }
    grasp_offsets = {"can": 0.025, "milk": 0.025, "bread": 0.01, "cereal": 0.05}

    def __init__(
        self,
        max_steps: int = 260,
        render: bool = False,
        offscreen: bool = False,
        task: str = "can",
        task_conditioning: bool = False,
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
        if task not in self.task_specs:
            raise ValueError(f"unknown task {task!r}; choose from {tuple(self.task_specs)}")
        env_name, object_name, target_index = self.task_specs[task]
        controller = load_composite_controller_config(controller="BASIC")
        self._env = suite.make(
            env_name=env_name,
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
            z_rotation=0.0,
        )
        self.max_steps = max_steps
        self._offscreen = offscreen
        self.task_name = task
        self._object_name = object_name
        self._target_index = target_index
        self._task_conditioning = task_conditioning
        self._grasp_offset_z = self.grasp_offsets[task]
        task_names = tuple(self.task_specs)
        self._task_one_hot = np.asarray([name == task for name in task_names], dtype=np.float32)
        object_model = self._env.objects[self._target_index]
        self._object_geometry = np.asarray(
            [
                float(object_model.horizontal_radius),
                float(object_model.top_offset[2]),
                float(-object_model.bottom_offset[2]),
            ],
            dtype=np.float32,
        )
        self.observation_names = self.base_observation_names
        if task_conditioning:
            self.observation_names += (
                "object_radius",
                "object_top_extent",
                "object_bottom_extent",
                "task_can",
                "task_milk",
                "task_bread",
                "task_cereal",
            )
        self._step = 0
        self._last_raw: dict[str, Any] | None = None
        self._rng = np.random.default_rng(0)
        self._rest_object_z = 0.86

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
        self._rest_object_z = float(raw[f"{self._object_name}_pos"][2])
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
        return np.asarray(
            self._env.target_bin_placements[self._target_index], dtype=np.float32
        ).copy()

    @property
    def object_rest_z(self) -> float:
        return self._rest_object_z

    def task_state(self, raw: dict[str, Any] | None = None) -> TaskState:
        if raw is None:
            if self._last_raw is None:
                raise RuntimeError("reset() must be called before reading task state")
            raw = self._last_raw
        eef = np.asarray(raw["robot0_eef_pos"], dtype=np.float32)
        can = np.asarray(raw[f"{self._object_name}_pos"], dtype=np.float32)
        goal = self.target_position
        gripper = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32)
        joint_vel = np.asarray(raw["robot0_joint_vel"], dtype=np.float32)
        can_to_eef = eef - can
        can_to_goal = goal - can
        observation_parts = [eef, gripper, can, goal, can_to_eef, can_to_goal, joint_vel]
        if self._task_conditioning:
            observation_parts.extend([self._object_geometry, self._task_one_hot])
        observation = np.concatenate(observation_parts).astype(np.float32)
        place_eef_z = float(goal[2] + self._object_geometry[2] + self._grasp_offset_z)
        return TaskState(
            observation=observation,
            eef_pos=eef,
            can_pos=can,
            goal_pos=goal,
            gripper_qpos=gripper,
            can_to_eef=can_to_eef,
            can_to_goal=can_to_goal,
            can_lifted=bool(can[2] > self._rest_object_z + 0.055),
            near_can=bool(np.linalg.norm(can_to_eef) < 0.075),
            near_goal=bool(np.linalg.norm(can[:2] - goal[:2]) < 0.075),
            task_name=self.task_name,
            object_geometry=self._object_geometry.copy(),
            place_eef_z=place_eef_z,
            grasp_offset_z=self._grasp_offset_z,
        )

    def teleport_can(self, position: np.ndarray) -> None:
        joint_name = f"{self._object_name}_joint0"
        qpos = np.asarray(self._env.sim.data.get_joint_qpos(joint_name)).copy()
        qpos[:3] = np.asarray(position, dtype=np.float64)
        self._env.sim.data.set_joint_qpos(joint_name, qpos)
        self._env.sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
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
