from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from intervenesim.counterfactual import CounterfactualData
from intervenesim.policy import Normalizer, resolve_device
from intervenesim.risk import binary_metrics

LoggingScheme = Literal["randomized", "uncertainty_selective", "adaptive_value"]
EstimatorName = Literal[
    "s_learner",
    "t_learner",
    "ipw_learner",
    "dr_learner",
    "reversibility_proxy",
    "paired_label_oracle",
]


@dataclass
class LoggedBanditData:
    """Deployment-realistic intervention log containing exactly one observed outcome per row.

    Potential outcomes are deliberately absent from this schema. Exact paired outcomes remain in
    :class:`CounterfactualData` and may only be passed to evaluation code.
    """

    contexts: np.ndarray
    treatments: np.ndarray
    outcomes: np.ndarray
    propensities: np.ndarray
    episode_ids: np.ndarray
    policy_seeds: np.ndarray
    tasks: np.ndarray
    disturbances: np.ndarray
    steps: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.contexts = np.asarray(self.contexts, dtype=np.float32)
        self.treatments = np.asarray(self.treatments, dtype=bool)
        self.outcomes = np.asarray(self.outcomes, dtype=np.float32)
        self.propensities = np.asarray(self.propensities, dtype=np.float32)
        self.episode_ids = np.asarray(self.episode_ids, dtype=np.int32)
        self.policy_seeds = np.asarray(self.policy_seeds, dtype=np.int32)
        self.tasks = np.asarray(self.tasks, dtype="U16")
        self.disturbances = np.asarray(self.disturbances, dtype="U24")
        self.steps = np.asarray(self.steps, dtype=np.int16)
        lengths = {
            len(self.contexts),
            len(self.treatments),
            len(self.outcomes),
            len(self.propensities),
            len(self.episode_ids),
            len(self.policy_seeds),
            len(self.tasks),
            len(self.disturbances),
            len(self.steps),
        }
        if len(lengths) != 1:
            raise ValueError("logged-bandit arrays must have equal lengths")
        if self.contexts.ndim != 2:
            raise ValueError("contexts must be rank-2")
        if not np.isin(self.outcomes, [0.0, 1.0]).all():
            raise ValueError("observed outcomes must be binary")
        if not ((self.propensities > 0.0) & (self.propensities < 1.0)).all():
            raise ValueError("propensities must lie strictly between zero and one")

    @property
    def sample_count(self) -> int:
        return int(len(self.outcomes))

    @property
    def assist_rate(self) -> float:
        return float(self.treatments.mean())

    def subset(self, mask: np.ndarray) -> LoggedBanditData:
        mask = np.asarray(mask, dtype=bool)
        return LoggedBanditData(
            contexts=self.contexts[mask],
            treatments=self.treatments[mask],
            outcomes=self.outcomes[mask],
            propensities=self.propensities[mask],
            episode_ids=self.episode_ids[mask],
            policy_seeds=self.policy_seeds[mask],
            tasks=self.tasks[mask],
            disturbances=self.disturbances[mask],
            steps=self.steps[mask],
            metadata=self.metadata,
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **self.metadata,
            "schema_version": 1,
            "sample_count": self.sample_count,
            "assist_rate": self.assist_rate,
            "contains_potential_outcomes": False,
        }
        np.savez_compressed(
            output,
            contexts=self.contexts,
            treatments=self.treatments,
            outcomes=self.outcomes,
            propensities=self.propensities,
            episode_ids=self.episode_ids,
            policy_seeds=self.policy_seeds,
            tasks=self.tasks,
            disturbances=self.disturbances,
            steps=self.steps,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> LoggedBanditData:
        with np.load(path, allow_pickle=False) as archive:
            forbidden = {"autonomous_success", "assisted_success", "signed_value", "helpful"}
            leaked = forbidden.intersection(archive.files)
            if leaked:
                raise ValueError(
                    f"single-world log contains forbidden potential outcomes: {leaked}"
                )
            return cls(
                contexts=archive["contexts"],
                treatments=archive["treatments"],
                outcomes=archive["outcomes"],
                propensities=archive["propensities"],
                episode_ids=archive["episode_ids"],
                policy_seeds=archive["policy_seeds"],
                tasks=archive["tasks"],
                disturbances=archive["disturbances"],
                steps=archive["steps"],
                metadata=json.loads(str(archive["metadata"].item())),
            )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "episode_id": self.episode_ids,
                "policy_seed": self.policy_seeds,
                "task": self.tasks,
                "disturbance": self.disturbances,
                "step": self.steps,
                "assistance_observed": self.treatments,
                "outcome_observed": self.outcomes,
                "logging_propensity": self.propensities,
            }
        )


def simulate_single_world_log(
    oracle: CounterfactualData,
    seed: int,
    scheme: LoggingScheme = "randomized",
    assist_rate: float = 0.5,
    positivity_floor: float = 0.1,
) -> LoggedBanditData:
    """Reveal one randomly timed, one-treatment outcome per source episode.

    The simulator's paired outcomes are used only inside this logging boundary to reveal the
    realized arm. Neither unobserved arm is retained in the returned object or its serialized
    representation.
    """

    if not 0.0 < assist_rate < 1.0:
        raise ValueError("assist_rate must lie strictly between zero and one")
    if not 0.0 < positivity_floor < 0.5:
        raise ValueError("positivity_floor must lie between zero and 0.5")
    rng = np.random.default_rng(seed)
    indices = _sample_candidate_indices(oracle, rng)
    contexts = oracle.features[indices]
    if scheme == "randomized":
        propensities = np.full(len(indices), assist_rate, dtype=np.float32)
    elif scheme == "uncertainty_selective":
        uncertainty = oracle.ensemble_uncertainty[indices]
        standardized = (uncertainty - uncertainty.mean()) / max(float(uncertainty.std()), 1e-6)
        intercept = _calibrate_logistic_intercept(standardized, assist_rate)
        propensities = _sigmoid(intercept + standardized)
        propensities = np.clip(propensities, positivity_floor, 1.0 - positivity_floor).astype(
            np.float32
        )
    elif scheme == "adaptive_value":
        raise ValueError("adaptive_value requires simulate_adaptive_single_world_log")
    else:
        raise ValueError(f"unknown logging scheme: {scheme}")
    treatments = rng.random(len(indices)) < propensities
    outcomes = np.where(
        treatments,
        oracle.assisted_success[indices],
        oracle.autonomous_success[indices],
    ).astype(np.float32)
    return LoggedBanditData(
        contexts=contexts,
        treatments=treatments,
        outcomes=outcomes,
        propensities=propensities,
        episode_ids=oracle.episode_ids[indices],
        policy_seeds=oracle.policy_seeds[indices],
        tasks=oracle.tasks[indices],
        disturbances=oracle.disturbances[indices],
        steps=oracle.steps[indices],
        metadata={
            "source": "exact-pair oracle with one arm hidden",
            "logging_scheme": scheme,
            "seed": seed,
            "target_assist_rate": assist_rate,
            "positivity_floor": positivity_floor,
            "decision_time": "one uniformly sampled candidate per episode",
        },
    )


def simulate_adaptive_single_world_log(
    oracle: CounterfactualData,
    seed: int,
    train_config: CATETrainConfig,
    *,
    assist_rate: float = 0.5,
    positivity_floor: float = 0.1,
    update_interval: int = 60,
    temperature: float = 0.15,
) -> LoggedBanditData:
    """Run an online epsilon-soft logger updated only from previously observed outcomes."""

    if update_interval < 4:
        raise ValueError("update_interval must be at least four")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    rng = np.random.default_rng(seed)
    indices = _sample_candidate_indices(oracle, rng)
    rng.shuffle(indices)
    contexts = oracle.features[indices]
    treatments = np.zeros(len(indices), dtype=bool)
    outcomes = np.zeros(len(indices), dtype=np.float32)
    propensities = np.zeros(len(indices), dtype=np.float32)
    for start in range(0, len(indices), update_interval):
        stop = min(len(indices), start + update_interval)
        if start < update_interval:
            probabilities = np.full(stop - start, assist_rate, dtype=np.float32)
        else:
            history = LoggedBanditData(
                contexts=contexts[:start],
                treatments=treatments[:start],
                outcomes=outcomes[:start],
                propensities=propensities[:start],
                episode_ids=oracle.episode_ids[indices[:start]],
                policy_seeds=oracle.policy_seeds[indices[:start]],
                tasks=oracle.tasks[indices[:start]],
                disturbances=oracle.disturbances[indices[:start]],
                steps=oracle.steps[indices[:start]],
                metadata={"online_prefix": start},
            )
            estimator = fit_cate_estimator(
                history,
                "s_learner",
                CATETrainConfig(
                    epochs=train_config.epochs,
                    hidden_dims=train_config.hidden_dims,
                    batch_size=train_config.batch_size,
                    learning_rate=train_config.learning_rate,
                    weight_decay=train_config.weight_decay,
                    device=train_config.device,
                    seed=train_config.seed + start,
                    crossfit_folds=train_config.crossfit_folds,
                ),
            )
            effect = estimator.effects(contexts[start:stop])
            soft_choice = _sigmoid(effect / temperature)
            probabilities = (
                positivity_floor + (1.0 - 2.0 * positivity_floor) * soft_choice
            ).astype(np.float32)
        propensities[start:stop] = probabilities
        treatments[start:stop] = rng.random(stop - start) < probabilities
        outcomes[start:stop] = np.where(
            treatments[start:stop],
            oracle.assisted_success[indices[start:stop]],
            oracle.autonomous_success[indices[start:stop]],
        ).astype(np.float32)
    return LoggedBanditData(
        contexts=contexts,
        treatments=treatments,
        outcomes=outcomes,
        propensities=propensities,
        episode_ids=oracle.episode_ids[indices],
        policy_seeds=oracle.policy_seeds[indices],
        tasks=oracle.tasks[indices],
        disturbances=oracle.disturbances[indices],
        steps=oracle.steps[indices],
        metadata={
            "source": "exact-pair oracle with one arm hidden",
            "logging_scheme": "adaptive_value",
            "seed": seed,
            "warmup_assist_rate": assist_rate,
            "positivity_floor": positivity_floor,
            "update_interval": update_interval,
            "temperature": temperature,
            "decision_time": "one uniformly sampled candidate per episode",
            "online_information": "past factual outcomes only",
        },
    )


@dataclass(frozen=True)
class CATETrainConfig:
    epochs: int = 60
    hidden_dims: tuple[int, ...] = (64, 64)
    batch_size: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    device: str = "auto"
    seed: int = 27
    crossfit_folds: int = 3


class _Network(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass
class _FittedModel:
    model: _Network
    normalizer: Normalizer
    device: torch.device
    binary: bool

    @torch.inference_mode()
    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = self.normalizer.transform(np.asarray(features, dtype=np.float32))
        output = self.model(torch.from_numpy(normalized).to(self.device))
        if self.binary:
            output = torch.sigmoid(output)
        return output.cpu().numpy()


class CATEEstimator:
    def __init__(self, method: EstimatorName, models: tuple[_FittedModel, ...]) -> None:
        self.method = method
        self.models = models

    def effects(self, contexts: np.ndarray) -> np.ndarray:
        contexts = np.asarray(contexts, dtype=np.float32)
        if self.method == "s_learner":
            model = self.models[0]
            control = model.predict(_with_treatment(contexts, False))
            assisted = model.predict(_with_treatment(contexts, True))
            return assisted - control
        if self.method == "t_learner":
            control, assisted = self.models
            return assisted.predict(contexts) - control.predict(contexts)
        if self.method in {"ipw_learner", "dr_learner", "paired_label_oracle"}:
            return np.clip(self.models[0].predict(contexts), -1.0, 1.0)
        if self.method == "reversibility_proxy":
            return 1.0 - self.models[0].predict(contexts)
        raise ValueError(f"unknown estimator method: {self.method}")


def fit_cate_estimator(
    data: LoggedBanditData,
    method: EstimatorName,
    config: CATETrainConfig,
) -> CATEEstimator:
    """Fit a treatment-effect estimator without access to paired outcomes."""

    if data.sample_count < 4:
        raise ValueError("at least four logged decisions are required")
    treatment = data.treatments.astype(np.float32)
    if method == "s_learner":
        model = _fit_model(
            _with_treatment(data.contexts, treatment),
            data.outcomes,
            np.ones(data.sample_count),
            config,
            binary=True,
            seed_offset=1,
        )
        return CATEEstimator(method, (model,))
    if method in {"t_learner", "reversibility_proxy"}:
        control_mask = ~data.treatments
        assisted_mask = data.treatments
        if not control_mask.any() or not assisted_mask.any():
            raise ValueError("both treatment arms must be represented")
        control = _fit_model(
            data.contexts[control_mask],
            data.outcomes[control_mask],
            1.0 / (1.0 - data.propensities[control_mask]),
            config,
            binary=True,
            seed_offset=2,
        )
        if method == "reversibility_proxy":
            return CATEEstimator(method, (control,))
        assisted = _fit_model(
            data.contexts[assisted_mask],
            data.outcomes[assisted_mask],
            1.0 / data.propensities[assisted_mask],
            config,
            binary=True,
            seed_offset=3,
        )
        return CATEEstimator(method, (control, assisted))
    if method == "ipw_learner":
        pseudo_outcome = treatment * data.outcomes / data.propensities - (
            1.0 - treatment
        ) * data.outcomes / (1.0 - data.propensities)
    elif method == "dr_learner":
        control, assisted = _cross_fitted_outcomes(data, config)
        pseudo_outcome = (
            assisted
            - control
            + treatment * (data.outcomes - assisted) / data.propensities
            - (1.0 - treatment) * (data.outcomes - control) / (1.0 - data.propensities)
        )
    else:
        raise ValueError(f"{method} requires exact paired labels or is not implemented")
    effect_model = _fit_model(
        data.contexts,
        pseudo_outcome,
        np.ones(data.sample_count),
        config,
        binary=False,
        seed_offset=4,
    )
    return CATEEstimator(method, (effect_model,))


def fit_paired_label_oracle(
    data: CounterfactualData,
    config: CATETrainConfig,
) -> CATEEstimator:
    """Privileged upper bound trained on both simulator futures; never a deployable method."""

    model = _fit_model(
        data.features,
        data.signed_value.astype(np.float32),
        np.ones(data.sample_count),
        config,
        binary=False,
        seed_offset=17,
    )
    return CATEEstimator("paired_label_oracle", (model,))


def counterfactual_metrics(
    oracle: CounterfactualData,
    predicted_effect: np.ndarray,
) -> dict[str, float]:
    """Score estimates against hidden individual treatment effects."""

    effect = np.asarray(predicted_effect, dtype=np.float64)
    truth = oracle.signed_value.astype(np.float64)
    ranking = np.clip((effect + 1.0) / 2.0, 0.0, 1.0)
    helpful = binary_metrics(oracle.helpful, ranking, 0.5)
    harmful = binary_metrics(truth < 0.0, 1.0 - ranking, 0.5)
    return {
        "pehe": float(np.sqrt(np.mean((effect - truth) ** 2))),
        "mae": float(np.mean(np.abs(effect - truth))),
        "ate": float(effect.mean()),
        "true_ate": float(truth.mean()),
        "ate_error": float(abs(effect.mean() - truth.mean())),
        "helpful_auroc": helpful["auroc"],
        "helpful_average_precision": helpful["average_precision"],
        "harmful_auroc": harmful["auroc"],
        "sign_accuracy": float(
            np.mean(
                np.sign(effect[np.abs(effect) >= 0.05]) == np.sign(truth[np.abs(effect) >= 0.05])
            )
        )
        if np.any(np.abs(effect) >= 0.05)
        else 0.0,
    }


def delayed_assistance_data(
    oracle: CounterfactualData,
    candidate_hops: int,
) -> CounterfactualData:
    """Construct an exact delayed-help profile along the logged autonomous trajectory.

    A one-hop delay means the policy continues to the next saved candidate before the same
    scripted assistant takes over. If no later candidate exists, the request has no effect and
    the autonomous outcome is used. This changes the assistance treatment, not the context.
    """

    if candidate_hops < 0:
        raise ValueError("candidate_hops must be non-negative")
    assisted = np.empty(oracle.sample_count, dtype=bool)
    for episode_id in np.unique(oracle.episode_ids):
        indices = np.flatnonzero(oracle.episode_ids == episode_id)
        indices = indices[np.argsort(oracle.steps[indices], kind="stable")]
        for position, index in enumerate(indices):
            future = position + candidate_hops
            assisted[index] = (
                oracle.assisted_success[indices[future]]
                if future < len(indices)
                else oracle.autonomous_success[index]
            )
    return CounterfactualData(
        observations=oracle.observations.copy(),
        observation_deltas=oracle.observation_deltas.copy(),
        autonomous_actions=oracle.autonomous_actions.copy(),
        ensemble_uncertainty=oracle.ensemble_uncertainty.copy(),
        autonomous_success=oracle.autonomous_success.copy(),
        assisted_success=assisted,
        steps=oracle.steps.copy(),
        episode_ids=oracle.episode_ids.copy(),
        policy_seeds=oracle.policy_seeds.copy(),
        tasks=oracle.tasks.copy(),
        disturbances=oracle.disturbances.copy(),
        metadata={
            **oracle.metadata,
            "assistance_profile": "delayed_scripted_expert",
            "delay_candidate_hops": candidate_hops,
            "late_request_fallback": "autonomous outcome",
        },
    )


def evaluate_cost_sensitive_policy(
    oracle: CounterfactualData,
    predicted_effect: np.ndarray,
    costs: tuple[float, ...],
    method: str,
) -> pd.DataFrame:
    """Evaluate the first request whose estimated benefit exceeds its intervention cost."""

    frame = oracle.to_frame()
    frame["predicted_effect"] = predicted_effect
    records: list[dict[str, Any]] = []
    for cost in costs:
        for episode_id, group in frame.groupby("episode_id", sort=False):
            group = group.sort_values("step")
            selected = group.loc[group["predicted_effect"] > cost]
            request = len(selected) > 0
            row = selected.iloc[0] if request else group.iloc[0]
            success = bool(row["assisted_success"]) if request else bool(row["autonomous_success"])
            records.append(
                {
                    "episode_id": int(episode_id),
                    "method": method,
                    "cost": cost,
                    "success": success,
                    "help_requested": request,
                    "helpful_request": bool(row["helpful"]) if request else False,
                    "utility": float(success) - cost * float(request),
                }
            )
    return pd.DataFrame(records)


def evaluate_ranked_budget_policy(
    oracle: CounterfactualData,
    score_sets: dict[str, np.ndarray],
    budgets: tuple[float, ...],
) -> pd.DataFrame:
    """Evaluate gates at an exact episode budget using scores but no outcome labels.

    This transductive audit removes threshold-calibration differences between methods. It ranks
    each episode by its maximum candidate score, selects exactly the requested number of
    episodes, then intervenes at the first candidate above the selected-set cutoff. Stable
    episode-id tie breaking prevents a large tied score from exceeding the budget.
    """

    frame = oracle.to_frame()
    for method, scores in score_sets.items():
        if len(scores) != oracle.sample_count:
            raise ValueError(f"score length mismatch for {method}")
        frame[f"score_{method}"] = scores
    selections: dict[tuple[str, float], tuple[set[int], float]] = {}
    for method in score_sets:
        maxima = (
            frame.groupby("episode_id", sort=False)[f"score_{method}"]
            .max()
            .rename("score")
            .reset_index()
            .sort_values(["score", "episode_id"], ascending=[False, True], kind="stable")
        )
        for budget in budgets:
            count = min(len(maxima), max(0, int(round(float(budget) * len(maxima)))))
            selected = maxima.iloc[:count]
            selected_ids = set(selected["episode_id"].astype(int).tolist())
            cutoff = float(selected["score"].min()) if count else float("inf")
            selections[(method, budget)] = selected_ids, cutoff

    records: list[dict[str, Any]] = []
    for episode_id, group in frame.groupby("episode_id", sort=False):
        group = group.sort_values("step")
        common = {
            "episode_id": int(episode_id),
            "policy_seed": int(group["policy_seed"].iloc[0]),
            "task": str(group["task"].iloc[0]),
            "disturbance": str(group["disturbance"].iloc[0]),
        }
        autonomous_success = bool(group["autonomous_success"].iloc[0])
        records.append(
            {
                **common,
                "method": "never_help",
                "target_budget": 0.0,
                "success": autonomous_success,
                "help_requested": False,
                "helpful_request": False,
                "request_step": np.nan,
                "request_score": np.nan,
            }
        )
        records.append(_ranked_request_record(common, "always_help", 1.0, group.iloc[0], np.nan))
        helpful = group.loc[group["helpful"]]
        if len(helpful):
            records.append(
                _ranked_request_record(common, "oracle_value", np.nan, helpful.iloc[0], 1.0)
            )
        else:
            records.append(
                {
                    **common,
                    "method": "oracle_value",
                    "target_budget": np.nan,
                    "success": autonomous_success,
                    "help_requested": False,
                    "helpful_request": False,
                    "request_step": np.nan,
                    "request_score": np.nan,
                }
            )
        for method in score_sets:
            for budget in budgets:
                selected_ids, cutoff = selections[(method, budget)]
                if int(episode_id) in selected_ids:
                    candidates = group.loc[group[f"score_{method}"] >= cutoff]
                    row = candidates.iloc[0] if len(candidates) else group.iloc[-1]
                    records.append(
                        _ranked_request_record(
                            common,
                            method,
                            budget,
                            row,
                            float(row[f"score_{method}"]),
                        )
                    )
                else:
                    records.append(
                        {
                            **common,
                            "method": method,
                            "target_budget": budget,
                            "success": autonomous_success,
                            "help_requested": False,
                            "helpful_request": False,
                            "request_step": np.nan,
                            "request_score": np.nan,
                        }
                    )
    return pd.DataFrame(records)


def summarize_cost_sensitive_policy(episodes: pd.DataFrame) -> pd.DataFrame:
    return (
        episodes.groupby(["method", "cost"], sort=False)
        .agg(
            episodes=("episode_id", "size"),
            success_rate=("success", "mean"),
            intervention_rate=("help_requested", "mean"),
            useful_request_rate=("helpful_request", "mean"),
            mean_utility=("utility", "mean"),
        )
        .reset_index()
    )


def _ranked_request_record(
    common: dict[str, Any], method: str, budget: float, row: pd.Series, score: float
) -> dict[str, Any]:
    return {
        **common,
        "method": method,
        "target_budget": budget,
        "success": bool(row["assisted_success"]),
        "help_requested": True,
        "helpful_request": bool(row["helpful"]),
        "request_step": int(row["step"]),
        "request_score": score,
    }


def _cross_fitted_outcomes(
    data: LoggedBanditData,
    config: CATETrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    folds = _episode_folds(data.episode_ids, config.crossfit_folds, config.seed)
    control = np.zeros(data.sample_count, dtype=np.float32)
    assisted = np.zeros(data.sample_count, dtype=np.float32)
    for fold_index, validation in enumerate(folds):
        training = ~validation
        fold = data.subset(training)
        estimator = fit_cate_estimator(
            fold,
            "t_learner",
            CATETrainConfig(
                epochs=config.epochs,
                hidden_dims=config.hidden_dims,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                device=config.device,
                seed=config.seed + 101 * (fold_index + 1),
                crossfit_folds=config.crossfit_folds,
            ),
        )
        control_model, assisted_model = estimator.models
        control[validation] = control_model.predict(data.contexts[validation])
        assisted[validation] = assisted_model.predict(data.contexts[validation])
    return control, assisted


def _episode_folds(episode_ids: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    unique = np.unique(episode_ids).copy()
    if count < 2 or count > len(unique):
        raise ValueError("crossfit_folds must be between two and the number of episodes")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    groups = np.array_split(unique, count)
    return [np.isin(episode_ids, group) for group in groups]


def _fit_model(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    config: CATETrainConfig,
    *,
    binary: bool,
    seed_offset: int,
) -> _FittedModel:
    _seed_everything(config.seed + seed_offset)
    device = resolve_device(config.device)
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / max(float(weights.mean()), 1e-6)
    normalizer = Normalizer.fit(features)
    normalized = normalizer.transform(features).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(normalized), torch.from_numpy(targets), torch.from_numpy(weights)
    )
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed + seed_offset),
    )
    model = _Network(normalized.shape[1], config.hidden_dims).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    for _ in range(config.epochs):
        model.train()
        for batch_features, batch_targets, batch_weights in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            batch_weights = batch_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            if binary:
                losses = nn.functional.binary_cross_entropy_with_logits(
                    predictions, batch_targets, reduction="none"
                )
            else:
                losses = nn.functional.huber_loss(
                    predictions, batch_targets, reduction="none", delta=1.0
                )
            loss = (losses * batch_weights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return _FittedModel(model.eval(), normalizer, device, binary)


def _with_treatment(contexts: np.ndarray, treatment: np.ndarray | bool) -> np.ndarray:
    contexts = np.asarray(contexts, dtype=np.float32)
    values = np.asarray(treatment, dtype=np.float32)
    if values.ndim == 0:
        values = np.full(len(contexts), float(values), dtype=np.float32)
    return np.concatenate([contexts, values.reshape(-1, 1)], axis=1)


def _calibrate_logistic_intercept(scores: np.ndarray, target: float) -> float:
    lower, upper = -12.0, 12.0
    for _ in range(60):
        middle = (lower + upper) / 2.0
        if float(_sigmoid(middle + scores).mean()) < target:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def _sample_candidate_indices(oracle: CounterfactualData, rng: np.random.Generator) -> np.ndarray:
    selected = []
    for episode_id in np.unique(oracle.episode_ids):
        candidates = np.flatnonzero(oracle.episode_ids == episode_id)
        selected.append(int(rng.choice(candidates)))
    return np.asarray(selected, dtype=np.int64)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
