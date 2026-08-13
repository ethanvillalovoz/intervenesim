from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from intervenesim.disturbances import Disturbance
from intervenesim.environment import PickPlaceEnv
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
