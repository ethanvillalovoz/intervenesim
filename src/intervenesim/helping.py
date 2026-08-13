from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from intervenesim.disturbances import Disturbance
from intervenesim.environment import PickPlaceEnv
from intervenesim.expert import ScriptedExpert
from intervenesim.policy import PolicyAgent
from intervenesim.risk import RiskAgent
from intervenesim.supervision import InterventionSupervisor

HELP_MODES = ("no_help", "learned_help", "oracle_help", "always_help")


def evaluate_help_seeking(
    policy_checkpoint: str | Path,
    risk_checkpoint: str | Path,
    disturbances: tuple[str, ...],
    episodes: int,
    seed: int,
    max_steps: int,
    device: str = "auto",
    progress: Callable[[str], None] | None = None,
    task: str = "can",
    task_conditioning: bool = False,
) -> pd.DataFrame:
    policy = PolicyAgent.load(policy_checkpoint, device=device)
    risk = RiskAgent.load(risk_checkpoint, device=device)
    records: list[dict[str, object]] = []
    with PickPlaceEnv(max_steps=max_steps, task=task, task_conditioning=task_conditioning) as env:
        for mode in HELP_MODES:
            for disturbance_name in disturbances:
                for episode in range(episodes):
                    episode_seed = seed + episode
                    disturbance = Disturbance(disturbance_name, seed + 50_000 + episode)
                    disturbance.reset(env.action_dim)
                    state = env.reset(episode_seed)
                    expert = ScriptedExpert()
                    supervisor = InterventionSupervisor()
                    supervisor.reset(state)
                    help_requested = False
                    help_step: int | None = None
                    request_score: float | None = None
                    max_risk_score = 0.0
                    for step in range(max_steps):
                        state = disturbance.before_step(env, state, step)
                        oracle_request = supervisor.should_intervene(state, disturbance, step)
                        risk_score = risk.score(state.observation)
                        max_risk_score = max(max_risk_score, risk_score)
                        request = (
                            (mode == "always_help" and step == 0)
                            or (mode == "oracle_help" and oracle_request)
                            or (mode == "learned_help" and risk_score >= risk.threshold)
                        )
                        if not help_requested and request:
                            help_requested = True
                            help_step = step
                            request_score = risk_score
                            expert.reset(state, recovering=True)
                        if help_requested:
                            action = expert.action(state)
                        else:
                            action = policy.action(state.observation)
                            action = disturbance.transform_action(action, step)
                        state, _, done, info = env.step(action)
                        if done:
                            break
                    records.append(
                        {
                            "help_mode": mode,
                            "task": task,
                            "disturbance": disturbance_name,
                            "episode": episode,
                            "seed": episode_seed,
                            "success": bool(info["success"]),
                            "help_requested": help_requested,
                            "help_step": help_step,
                            "request_score": request_score,
                            "max_risk_score": max_risk_score,
                            "risk_threshold": risk.threshold,
                            "disturbance_fired": disturbance.fired,
                            "disturbance_trigger_step": disturbance.trigger_step,
                            "steps": step + 1,
                        }
                    )
                if progress:
                    progress(f"evaluated {mode} / {disturbance_name}")
    return pd.DataFrame.from_records(records)


def summarize_help_seeking(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (mode, disturbance), group in episodes.groupby(["help_mode", "disturbance"]):
        requests = group["help_requested"].astype(bool)
        request_steps = group.loc[requests, "help_step"]
        rows.append(
            {
                "help_mode": mode,
                "disturbance": disturbance,
                "episodes": int(len(group)),
                "success_rate": float(group["success"].mean()),
                "intervention_rate": float(requests.mean()),
                "mean_request_step": (
                    float(request_steps.mean()) if len(request_steps) else float("nan")
                ),
                "mean_steps": float(group["steps"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["disturbance", "help_mode"]).reset_index(drop=True)
