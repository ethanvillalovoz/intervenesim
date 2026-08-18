from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

from intervenesim.counterfactual import CounterfactualData
from intervenesim.disturbances import Disturbance
from intervenesim.environment import PickPlaceEnv
from intervenesim.expert import ScriptedExpert
from intervenesim.policy import PolicyAgent


@dataclass(frozen=True)
class VideoRollout:
    frames: list[np.ndarray]
    success: bool
    steps: int
    trigger_step: int | None


def record_benchmark_comparison(
    run_dir: str | Path,
    output: str | Path,
    disturbance_name: str = "gripper_slip",
    fps: int = 20,
) -> dict[str, object]:
    """Record a matched episode where recovery data succeeds and baseline BC fails."""
    run = Path(run_dir)
    episodes = pd.read_csv(run / "results" / "episodes.csv")
    seed = select_comparison_seed(episodes, disturbance_name)
    disturbance_seed = seed + 50_000
    baseline = _record_rollout(
        run / "checkpoints" / "baseline.pt",
        disturbance_name,
        seed,
        disturbance_seed,
    )
    recovery = _record_rollout(
        run / "checkpoints" / "recovery_data.pt",
        disturbance_name,
        seed,
        disturbance_seed,
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_side_by_side(baseline, recovery, destination, disturbance_name, seed, fps)
    return {
        "output": str(destination),
        "disturbance": disturbance_name,
        "seed": seed,
        "baseline_success": baseline.success,
        "recovery_success": recovery.success,
        "baseline_steps": baseline.steps,
        "recovery_steps": recovery.steps,
    }


def record_research_comparison(
    run_dir: str | Path,
    output: str | Path,
    task: str = "cereal",
    disturbance_name: str = "gripper_slip",
    training_seed: int = 27,
    budget: int = 4800,
    fps: int = 20,
) -> dict[str, object]:
    """Record a matched InterveneSim-X baseline failure and recovery-policy success."""
    run = Path(run_dir)
    episodes = pd.read_csv(run / "results" / "autonomous_episodes.csv")
    subset = episodes.loc[
        (episodes["training_seed"] == training_seed)
        & (episodes["task"] == task)
        & (episodes["disturbance"] == disturbance_name)
        & (episodes["budget"].isin([0, budget]))
    ]
    paired = subset.pivot_table(
        index="seed", columns="condition", values="success", aggfunc="first"
    )
    required = {"baseline", "recovery_bc"}
    if not required.issubset(paired.columns):
        raise ValueError("research results must contain baseline and recovery_bc conditions")
    candidates = paired.index[(~paired["baseline"].astype(bool)) & paired["recovery_bc"]]
    if not len(candidates):
        raise ValueError("no matched research episode has a baseline failure and recovery success")
    seed = int(candidates[0])
    disturbance_seed = seed + 50_000
    checkpoint_dir = run / "checkpoints" / f"seed-{training_seed}"
    baseline = _record_rollout(
        checkpoint_dir / "baseline.pt",
        disturbance_name,
        seed,
        disturbance_seed,
        task=task,
        task_conditioning=True,
    )
    recovery = _record_rollout(
        checkpoint_dir / f"recovery-bc-{budget}.pt",
        disturbance_name,
        seed,
        disturbance_seed,
        task=task,
        task_conditioning=True,
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_side_by_side(
        baseline,
        recovery,
        destination,
        f"{task} / {disturbance_name}",
        seed,
        fps,
    )
    return {
        "output": str(destination),
        "task": task,
        "disturbance": disturbance_name,
        "seed": seed,
        "training_seed": training_seed,
        "budget": budget,
        "baseline_success": baseline.success,
        "recovery_success": recovery.success,
    }


def record_value_counterfactual(
    value_run_dir: str | Path,
    output: str | Path,
    fps: int = 20,
) -> dict[str, object]:
    """Record two futures forked from one helpful held-out simulator state."""
    value_run = Path(value_run_dir)
    part, row = _select_helpful_value_candidate(value_run / "datasets")
    data = CounterfactualData.load(part)
    metadata = data.metadata
    task = str(row["task"])
    disturbance_name = str(row["disturbance"])
    candidate_step = int(row["step"])
    episodes = int(metadata["episodes"])
    local_episode_id = int(row["episode_id"])
    disturbance_names = tuple(
        yaml.safe_load((value_run / "config.resolved.yaml").read_text())["disturbances"]
    )
    disturbance_index = disturbance_names.index(disturbance_name)
    episode_index = local_episode_id - disturbance_index * episodes
    episode_seed = int(metadata["seed"]) + disturbance_index * 10_000 + episode_index
    policy = PolicyAgent.load(metadata["policy_checkpoint"])
    autonomous, assisted = _record_counterfactual_fork(
        policy,
        task,
        disturbance_name,
        episode_seed,
        candidate_step,
    )
    if autonomous.success != bool(row["autonomous_success"]):
        raise RuntimeError("replayed autonomous outcome does not match the frozen dataset")
    if assisted.success != bool(row["assisted_success"]):
        raise RuntimeError("replayed assisted outcome does not match the frozen dataset")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_counterfactual_fork(
        autonomous,
        assisted,
        destination,
        task,
        disturbance_name,
        candidate_step,
        fps,
    )
    return {
        "output": str(destination),
        "source_dataset": str(part),
        "task": task,
        "disturbance": disturbance_name,
        "episode_seed": episode_seed,
        "candidate_step": candidate_step,
        "autonomous_success": autonomous.success,
        "assisted_success": assisted.success,
    }


def _select_helpful_value_candidate(dataset_dir: Path) -> tuple[Path, pd.Series]:
    preferences = (("gripper_slip", 70), ("object_shift", 30))
    parts = sorted(dataset_dir.glob("eval-seed-*.npz"))
    for disturbance, minimum_step in preferences:
        for part in parts:
            frame = CounterfactualData.load(part).to_frame()
            candidates = frame.loc[
                frame["helpful"]
                & (frame["disturbance"] == disturbance)
                & (frame["step"] >= minimum_step)
            ]
            if not candidates.empty:
                return part, candidates.iloc[0]
    raise ValueError("no helpful held-out counterfactual candidate found")


def _record_counterfactual_fork(
    policy: PolicyAgent,
    task: str,
    disturbance_name: str,
    episode_seed: int,
    candidate_step: int,
    max_steps: int = 260,
) -> tuple[VideoRollout, VideoRollout]:
    disturbance = Disturbance(disturbance_name, episode_seed + 50_000)
    with PickPlaceEnv(
        max_steps=max_steps,
        offscreen=True,
        task=task,
        task_conditioning=True,
    ) as env:
        disturbance.reset(env.action_dim)
        state = env.reset(episode_seed)
        for step in range(candidate_step + 1):
            state = disturbance.before_step(env, state, step)
            if step == candidate_step:
                snapshot = env.snapshot()
                branch_disturbance = deepcopy(disturbance)
                break
            action = disturbance.transform_action(policy.action(state.observation), step)
            state, _, done, _ = env.step(action)
            if done:
                raise RuntimeError("episode ended before the selected counterfactual state")

        state = env.restore(snapshot)
        autonomous_frames = [env.capture_frame()]
        autonomous_info: dict[str, object] = {"success": False}
        for step in range(candidate_step, max_steps):
            state = branch_disturbance.before_step(env, state, step)
            action = branch_disturbance.transform_action(policy.action(state.observation), step)
            state, _, done, autonomous_info = env.step(action)
            autonomous_frames.append(env.capture_frame())
            if done:
                break

        state = env.restore(snapshot)
        expert = ScriptedExpert()
        expert.reset(state, recovering=True)
        assisted_frames = [env.capture_frame()]
        assisted_info: dict[str, object] = {"success": False}
        for _ in range(candidate_step, max_steps):
            state, _, done, assisted_info = env.step(expert.action(state))
            assisted_frames.append(env.capture_frame())
            if done:
                break

    return (
        VideoRollout(
            autonomous_frames,
            bool(autonomous_info["success"]),
            len(autonomous_frames) - 1,
            0,
        ),
        VideoRollout(
            assisted_frames,
            bool(assisted_info["success"]),
            len(assisted_frames) - 1,
            0,
        ),
    )


def _write_counterfactual_fork(
    autonomous: VideoRollout,
    assisted: VideoRollout,
    output: Path,
    task: str,
    disturbance_name: str,
    candidate_step: int,
    fps: int,
) -> None:
    frame_count = max(len(autonomous.frames), len(assisted.frames))
    hold_frames = fps * 2
    with imageio.get_writer(
        output,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for index in range(frame_count + hold_frames):
            left = autonomous.frames[min(index, len(autonomous.frames) - 1)]
            right = assisted.frames[min(index, len(assisted.frames) - 1)]
            left_panel = _annotate(left, "Autonomous policy", _status(autonomous, index), index)
            right_panel = _annotate(right, "Expert takeover", _status(assisted, index), index)
            combined = np.concatenate([left_panel, right_panel], axis=1)
            image = Image.fromarray(combined)
            draw = ImageDraw.Draw(image)
            text = (
                f"Exact simulator fork | {task} / {disturbance_name.replace('_', ' ')} "
                f"| decision step {candidate_step}"
            )
            draw.rounded_rectangle((128, 38, 512, 60), radius=6, fill=(17, 24, 39, 220))
            draw.text((142, 43), text, fill="white", font=ImageFont.load_default())
            writer.append_data(np.asarray(image))


def select_comparison_seed(episodes: pd.DataFrame, disturbance_name: str) -> int:
    subset = episodes.loc[episodes["disturbance"] == disturbance_name]
    if subset.empty:
        raise ValueError(f"no evaluation episodes found for {disturbance_name!r}")
    paired = subset.pivot(index="seed", columns="condition", values="success")
    required = {"baseline", "recovery_data"}
    if not required.issubset(paired.columns):
        raise ValueError("results must contain baseline and recovery_data conditions")
    candidates = paired.index[(~paired["baseline"].astype(bool)) & paired["recovery_data"]]
    if len(candidates) == 0:
        raise ValueError(
            f"no matched {disturbance_name!r} episode has a baseline failure and recovery success"
        )
    return int(candidates[0])


def _record_rollout(
    checkpoint: Path,
    disturbance_name: str,
    seed: int,
    disturbance_seed: int,
    max_steps: int = 260,
    task: str = "can",
    task_conditioning: bool = False,
) -> VideoRollout:
    policy = PolicyAgent.load(checkpoint)
    disturbance = Disturbance(disturbance_name, disturbance_seed)
    frames: list[np.ndarray] = []
    with PickPlaceEnv(
        max_steps=max_steps,
        offscreen=True,
        task=task,
        task_conditioning=task_conditioning,
    ) as env:
        disturbance.reset(env.action_dim)
        state = env.reset(seed)
        frames.append(env.capture_frame())
        info: dict[str, object] = {"success": False}
        for step in range(max_steps):
            state = disturbance.before_step(env, state, step)
            action = disturbance.transform_action(policy.action(state.observation), step)
            state, _, done, info = env.step(action)
            frames.append(env.capture_frame())
            if done:
                break
    return VideoRollout(
        frames=frames,
        success=bool(info["success"]),
        steps=step + 1,
        trigger_step=disturbance.trigger_step,
    )


def _write_side_by_side(
    baseline: VideoRollout,
    recovery: VideoRollout,
    output: Path,
    disturbance_name: str,
    seed: int,
    fps: int,
) -> None:
    frame_count = max(len(baseline.frames), len(recovery.frames))
    hold_frames = fps * 2
    with imageio.get_writer(
        output,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for index in range(frame_count + hold_frames):
            left = baseline.frames[min(index, len(baseline.frames) - 1)]
            right = recovery.frames[min(index, len(recovery.frames) - 1)]
            left_status = _status(baseline, index)
            right_status = _status(recovery, index)
            left_panel = _annotate(left, "Baseline BC", left_status, index)
            right_panel = _annotate(right, "Recovery-data BC", right_status, index)
            combined = np.concatenate([left_panel, right_panel], axis=1)
            header = Image.fromarray(combined)
            draw = ImageDraw.Draw(header)
            text = f"Matched {disturbance_name.replace('_', ' ')} | seed {seed}"
            draw.rounded_rectangle((185, 38, 455, 60), radius=6, fill=(17, 24, 39, 220))
            draw.text((198, 43), text, fill="white", font=ImageFont.load_default())
            writer.append_data(np.asarray(header))


def _status(rollout: VideoRollout, frame_index: int) -> str:
    if frame_index >= len(rollout.frames) - 1:
        return "SUCCESS" if rollout.success else "FAILURE"
    if rollout.trigger_step is not None and frame_index >= rollout.trigger_step:
        return "DISTURBANCE FIRED"
    return "RUNNING"


def _annotate(frame: np.ndarray, label: str, status: str, step: int) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0, 150))
    draw.text((10, 7), label, fill="white", font=font)
    status_color = {
        "SUCCESS": (16, 185, 129, 255),
        "FAILURE": (239, 68, 68, 255),
        "DISTURBANCE FIRED": (245, 158, 11, 255),
    }.get(status, (203, 213, 225, 255))
    draw.text((190, 7), status, fill=status_color, font=font)
    draw.text((10, image.height - 18), f"step {step}", fill="white", font=font)
    return np.asarray(image)
