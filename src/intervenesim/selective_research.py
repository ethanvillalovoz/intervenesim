from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from intervenesim.benchmark import system_manifest
from intervenesim.config import SelectiveConfig
from intervenesim.counterfactual import CounterfactualData
from intervenesim.selective import (
    CATETrainConfig,
    counterfactual_metrics,
    delayed_assistance_data,
    evaluate_cost_sensitive_policy,
    evaluate_ranked_budget_policy,
    fit_cate_estimator,
    fit_paired_label_oracle,
    simulate_adaptive_single_world_log,
    simulate_single_world_log,
    summarize_cost_sensitive_policy,
)
from intervenesim.value import (
    budget_thresholds,
    evaluate_counterfactual_gates,
    risk_scores,
    summarize_gate_evaluation,
)

ProgressCallback = Callable[[str], None]


def run_selective_research(
    config: SelectiveConfig,
    output_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run the single-world intervention-value benchmark from frozen exact forks."""

    if output_dir is not None:
        config = replace(config, output_dir=str(output_dir))
    output = Path(config.output_dir)
    logs_dir = output / "logs"
    for directory in (output, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    started = time.time()
    source = Path(config.source_run) / "datasets"
    train_path = source / "train_counterfactuals.npz"
    eval_path = source / "eval_counterfactuals.npz"
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            "selective-research requires the frozen value-research datasets; "
            "run `intervenesim value-research` first"
        )
    train = CounterfactualData.load(train_path)
    evaluation = CounterfactualData.load(eval_path)
    train_config = _train_config(config)
    _emit(progress, "fitting privileged paired-label upper bound")
    paired_oracle = fit_paired_label_oracle(train, train_config)
    paired_train_scores = paired_oracle.effects(train.features)
    paired_eval_scores = paired_oracle.effects(evaluation.features)
    risk_checkpoint = Path(config.risk_checkpoint)
    if not risk_checkpoint.exists():
        raise FileNotFoundError(f"missing risk baseline checkpoint: {risk_checkpoint}")
    baseline_train_scores = {
        "paired_label_oracle": paired_train_scores,
        "failure_risk": risk_scores(train, risk_checkpoint),
        "ensemble_uncertainty": train.ensemble_uncertainty,
    }
    baseline_eval_scores = {
        "paired_label_oracle": paired_eval_scores,
        "failure_risk": risk_scores(evaluation, risk_checkpoint),
        "ensemble_uncertainty": evaluation.ensemble_uncertainty,
    }

    metric_rows: list[dict[str, Any]] = []
    gate_rows: list[pd.DataFrame] = []
    predeployment_gate_rows: list[pd.DataFrame] = []
    summary_rows: list[pd.DataFrame] = []
    cost_rows: list[pd.DataFrame] = []
    overlap_rows: list[pd.DataFrame] = []
    latency_rows: list[pd.DataFrame] = []
    latency_profiles = {
        hops: delayed_assistance_data(evaluation, hops) for hops in config.latency_candidate_hops
    }
    for scheme in config.logging_schemes:
        for logging_seed in config.logging_seeds:
            _emit(progress, f"single-world log / {scheme} / seed {logging_seed}")
            log = _simulate_log(train, scheme, logging_seed, config, train_config)
            log_path = logs_dir / f"{scheme}-seed-{logging_seed}.npz"
            log.save(log_path)
            log.to_frame().to_csv(log_path.with_suffix(".csv"), index=False)
            overlap_rows.append(
                pd.DataFrame(
                    {
                        "logging_scheme": scheme,
                        "logging_seed": logging_seed,
                        "propensity": log.propensities,
                        "assistance_observed": log.treatments,
                    }
                )
            )
            score_sets_train = dict(baseline_train_scores)
            score_sets_eval = dict(baseline_eval_scores)
            for method in config.estimators:
                _emit(progress, f"fitting {method} / {scheme} / seed {logging_seed}")
                estimator = fit_cate_estimator(
                    log,
                    method,  # type: ignore[arg-type]
                    replace(train_config, seed=config.seed + logging_seed),
                )
                train_score = estimator.effects(train.features)
                eval_score = estimator.effects(evaluation.features)
                score_sets_train[method] = train_score
                score_sets_eval[method] = eval_score
            thresholds = {
                method: budget_thresholds(scores, train.episode_ids, config.target_budgets)
                for method, scores in score_sets_train.items()
            }
            predeployment_episodes = evaluate_counterfactual_gates(
                evaluation, score_sets_eval, thresholds, config.target_budgets
            )
            predeployment_episodes.insert(0, "logging_seed", logging_seed)
            predeployment_episodes.insert(0, "logging_scheme", scheme)
            predeployment_gate_rows.append(predeployment_episodes)
            episodes = evaluate_ranked_budget_policy(
                evaluation, score_sets_eval, config.target_budgets
            )
            episodes.insert(0, "logging_seed", logging_seed)
            episodes.insert(0, "logging_scheme", scheme)
            gate_rows.append(episodes)
            summary = summarize_gate_evaluation(episodes)
            summary.insert(0, "logging_seed", logging_seed)
            summary.insert(0, "logging_scheme", scheme)
            summary_rows.append(summary)
            latency_methods = {
                method: scores
                for method, scores in score_sets_eval.items()
                if method
                in {
                    "t_learner",
                    "dr_learner",
                    "reversibility_proxy",
                    "paired_label_oracle",
                }
            }
            for hops, profile in latency_profiles.items():
                latency_episodes = evaluate_ranked_budget_policy(
                    profile, latency_methods, config.target_budgets
                )
                latency_summary = summarize_gate_evaluation(latency_episodes)
                latency_summary.insert(0, "delay_candidate_hops", hops)
                latency_summary.insert(0, "logging_seed", logging_seed)
                latency_summary.insert(0, "logging_scheme", scheme)
                latency_rows.append(latency_summary)
            for method, scores in score_sets_eval.items():
                metric_rows.append(
                    {
                        "logging_scheme": scheme,
                        "logging_seed": logging_seed,
                        "method": method,
                        **counterfactual_metrics(evaluation, scores),
                    }
                )
                if method in {
                    "s_learner",
                    "t_learner",
                    "ipw_learner",
                    "dr_learner",
                    "paired_label_oracle",
                }:
                    cost = evaluate_cost_sensitive_policy(
                        evaluation, scores, config.intervention_costs, method
                    )
                    cost.insert(0, "logging_seed", logging_seed)
                    cost.insert(0, "logging_scheme", scheme)
                    cost_rows.append(cost)

    metrics = pd.DataFrame(metric_rows)
    gate_episodes = pd.concat(gate_rows, ignore_index=True)
    predeployment_gate_episodes = pd.concat(predeployment_gate_rows, ignore_index=True)
    gate_summary = pd.concat(summary_rows, ignore_index=True)
    cost_episodes = pd.concat(cost_rows, ignore_index=True)
    cost_summary = _summarize_cost_replicates(cost_episodes)
    overlap = pd.concat(overlap_rows, ignore_index=True)
    latency_seed_summary = pd.concat(latency_rows, ignore_index=True)
    latency_aggregate = _aggregate_latency(latency_seed_summary)
    aggregate = _aggregate_gate_replicates(gate_summary, config.seed)
    paired = _paired_differences(gate_summary)

    _emit(progress, "running logged-data learning curves")
    learning_curves = _learning_curves(train, evaluation, config, train_config, progress=progress)
    _emit(progress, "running held-out task and disturbance audit")
    open_world = _open_world_audit(train, evaluation, config, train_config, progress=progress)

    metrics.to_csv(output / "counterfactual_metrics.csv", index=False)
    gate_episodes.to_csv(output / "gate_episodes.csv.gz", index=False, compression="gzip")
    predeployment_gate_episodes.to_csv(
        output / "predeployment_gate_episodes.csv.gz", index=False, compression="gzip"
    )
    predeployment_summary = []
    for (scheme, seed), group in predeployment_gate_episodes.groupby(
        ["logging_scheme", "logging_seed"]
    ):
        summary = summarize_gate_evaluation(group)
        summary.insert(0, "logging_seed", seed)
        summary.insert(0, "logging_scheme", scheme)
        predeployment_summary.append(summary)
    pd.concat(predeployment_summary, ignore_index=True).to_csv(
        output / "predeployment_gate_seed_summary.csv", index=False
    )
    gate_summary.to_csv(output / "gate_seed_summary.csv", index=False)
    aggregate.to_csv(output / "gate_aggregate.csv", index=False)
    paired.to_csv(output / "paired_seed_differences.csv", index=False)
    cost_episodes.to_csv(output / "cost_episodes.csv.gz", index=False, compression="gzip")
    cost_summary.to_csv(output / "cost_summary.csv", index=False)
    overlap.to_csv(output / "logging_overlap.csv", index=False)
    latency_seed_summary.to_csv(output / "latency_seed_summary.csv", index=False)
    latency_aggregate.to_csv(output / "latency_aggregate.csv", index=False)
    learning_curves.to_csv(output / "learning_curves.csv", index=False)
    open_world.to_csv(output / "open_world.csv", index=False)
    _plot_policy_frontier(aggregate, output / "single_world_frontier.png")
    _plot_estimation(metrics, output / "effect_estimation.png")
    _plot_learning_curves(learning_curves, output / "learning_curves.png")
    _plot_open_world(open_world, output / "open_world.png")
    _plot_overlap(overlap, output / "logging_overlap.png")
    _plot_latency(latency_aggregate, output / "latency_audit.png")
    report = _markdown_report(
        train, evaluation, metrics, aggregate, paired, open_world, latency_aggregate, config
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    audit = _audit_logs(logs_dir)
    (output / "leakage_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    manifest: dict[str, object] = {
        "elapsed_seconds": time.time() - started,
        "config": config.to_dict(),
        "system": system_manifest(),
        "source": {
            "train_candidates": train.sample_count,
            "train_episodes": int(len(np.unique(train.episode_ids))),
            "evaluation_candidates": evaluation.sample_count,
            "evaluation_episodes": int(len(np.unique(evaluation.episode_ids))),
        },
        "leakage_audit": audit,
        "artifacts": {
            "report": str(output / "report.md"),
            "counterfactual_metrics": str(output / "counterfactual_metrics.csv"),
            "gate_aggregate": str(output / "gate_aggregate.csv"),
            "paired_differences": str(output / "paired_seed_differences.csv"),
            "learning_curves": str(output / "learning_curves.csv"),
            "open_world": str(output / "open_world.csv"),
            "latency_audit": str(output / "latency_aggregate.csv"),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _emit(progress, f"single-world benchmark complete: {output / 'report.md'}")
    return manifest


def _learning_curves(
    train: CounterfactualData,
    evaluation: CounterfactualData,
    config: SelectiveConfig,
    train_config: CATETrainConfig,
    progress: ProgressCallback | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scheme in config.logging_schemes:
        for seed in config.logging_seeds:
            log = _simulate_log(train, scheme, seed, config, train_config)
            rng = np.random.default_rng(seed + 8_003)
            order = rng.permutation(log.sample_count)
            for count in config.learning_curve_episodes:
                if count > log.sample_count:
                    continue
                _emit(progress, f"learning curve / {scheme} / seed {seed} / {count} episodes")
                mask = np.zeros(log.sample_count, dtype=bool)
                mask[order[:count]] = True
                subset = log.subset(mask)
                estimator = fit_cate_estimator(
                    subset,
                    "dr_learner",
                    replace(train_config, seed=config.seed + seed + count),
                )
                eval_scores = estimator.effects(evaluation.features)
                metric = counterfactual_metrics(evaluation, eval_scores)
                episodes = evaluate_ranked_budget_policy(
                    evaluation,
                    {"dr_learner": eval_scores},
                    config.target_budgets,
                )
                summary = summarize_gate_evaluation(episodes)
                for row in summary.loc[summary["method"] == "dr_learner"].to_dict("records"):
                    rows.append(
                        {
                            "logging_scheme": scheme,
                            "logging_seed": seed,
                            "logged_episodes": count,
                            "target_budget": row["target_budget"],
                            "success_rate": row["success_rate"],
                            "intervention_rate": row["intervention_rate"],
                            "request_precision": row["request_precision"],
                            "pehe": metric["pehe"],
                            "helpful_auroc": metric["helpful_auroc"],
                        }
                    )
    return pd.DataFrame(rows)


def _open_world_audit(
    train: CounterfactualData,
    evaluation: CounterfactualData,
    config: SelectiveConfig,
    train_config: CATETrainConfig,
    progress: ProgressCallback | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    axes = {
        "task": (train.tasks, evaluation.tasks),
        "disturbance": (train.disturbances, evaluation.disturbances),
    }
    for axis in config.ood_axes:
        if axis not in axes:
            raise ValueError(f"unsupported OOD axis: {axis}")
        train_labels, eval_labels = axes[axis]
        for held_out in np.unique(eval_labels):
            source = train.subset(train_labels != held_out)
            target = evaluation.subset(eval_labels == held_out)
            for seed in config.ood_logging_seeds:
                log = simulate_single_world_log(
                    source,
                    seed,
                    scheme="randomized",
                    assist_rate=config.assist_rate,
                    positivity_floor=config.positivity_floor,
                )
                for method in config.ood_estimators:
                    _emit(progress, f"OOD {axis}={held_out} / {method} / seed {seed}")
                    estimator = fit_cate_estimator(
                        log,
                        method,  # type: ignore[arg-type]
                        replace(train_config, seed=config.seed + seed),
                    )
                    target_scores = estimator.effects(target.features)
                    metric = counterfactual_metrics(target, target_scores)
                    episodes = evaluate_ranked_budget_policy(
                        target,
                        {method: target_scores},
                        config.target_budgets,
                    )
                    summary = summarize_gate_evaluation(episodes)
                    for row in summary.loc[summary["method"] == method].to_dict("records"):
                        rows.append(
                            {
                                "held_out_axis": axis,
                                "held_out_value": held_out,
                                "logging_seed": seed,
                                "method": method,
                                "target_budget": row["target_budget"],
                                "success_rate": row["success_rate"],
                                "intervention_rate": row["intervention_rate"],
                                "request_precision": row["request_precision"],
                                "pehe": metric["pehe"],
                                "helpful_auroc": metric["helpful_auroc"],
                            }
                        )
    return pd.DataFrame(rows)


def _aggregate_gate_replicates(summary: pd.DataFrame, seed: int) -> pd.DataFrame:
    selected = summary.loc[~summary["method"].isin(["never_help", "always_help", "oracle_value"])]
    rows: list[dict[str, Any]] = []
    for keys, group in selected.groupby(
        ["logging_scheme", "method", "target_budget"], dropna=False, sort=False
    ):
        scheme, method, budget = keys
        success_low, success_high = _bootstrap_interval(group["success_rate"].to_numpy(), seed)
        rows.append(
            {
                "logging_scheme": scheme,
                "method": method,
                "target_budget": budget,
                "logging_seeds": int(group["logging_seed"].nunique()),
                "success_mean": float(group["success_rate"].mean()),
                "success_std": float(group["success_rate"].std(ddof=1)),
                "success_ci_low": success_low,
                "success_ci_high": success_high,
                "intervention_mean": float(group["intervention_rate"].mean()),
                "precision_mean": float(group["request_precision"].mean()),
                "regret_mean": float(group["regret_to_matched_oracle"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _paired_differences(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary.loc[
        summary["method"].isin(
            ["dr_learner", "reversibility_proxy", "failure_risk", "paired_label_oracle"]
        )
    ]
    rows: list[dict[str, Any]] = []
    for (scheme, budget), group in selected.groupby(["logging_scheme", "target_budget"]):
        pivot = group.pivot_table(
            index="logging_seed", columns="method", values="success_rate", aggfunc="first"
        )
        if "dr_learner" not in pivot:
            continue
        for reference in ("reversibility_proxy", "failure_risk", "paired_label_oracle"):
            if reference not in pivot:
                continue
            differences = (pivot["dr_learner"] - pivot[reference]).dropna()
            rows.append(
                {
                    "logging_scheme": scheme,
                    "target_budget": budget,
                    "challenger": "dr_learner",
                    "reference": reference,
                    "logging_seeds": len(differences),
                    "mean_delta": float(differences.mean()),
                    "std_delta": float(differences.std(ddof=1)),
                    "positive_seeds": int((differences > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def _summarize_cost_replicates(episodes: pd.DataFrame) -> pd.DataFrame:
    seed_summaries = []
    for (scheme, seed), group in episodes.groupby(["logging_scheme", "logging_seed"]):
        summary = summarize_cost_sensitive_policy(group)
        summary.insert(0, "logging_seed", seed)
        summary.insert(0, "logging_scheme", scheme)
        seed_summaries.append(summary)
    data = pd.concat(seed_summaries, ignore_index=True)
    return (
        data.groupby(["logging_scheme", "method", "cost"], sort=False)
        .agg(
            logging_seeds=("logging_seed", "nunique"),
            success_mean=("success_rate", "mean"),
            intervention_mean=("intervention_rate", "mean"),
            utility_mean=("mean_utility", "mean"),
            utility_std=("mean_utility", "std"),
        )
        .reset_index()
    )


def _aggregate_latency(data: pd.DataFrame) -> pd.DataFrame:
    selected = data.loc[
        data["method"].isin(
            ["t_learner", "dr_learner", "reversibility_proxy", "paired_label_oracle"]
        )
    ]
    return (
        selected.groupby(
            ["logging_scheme", "delay_candidate_hops", "method", "target_budget"],
            sort=False,
        )
        .agg(
            logging_seeds=("logging_seed", "nunique"),
            success_mean=("success_rate", "mean"),
            success_std=("success_rate", "std"),
            intervention_mean=("intervention_rate", "mean"),
            precision_mean=("request_precision", "mean"),
        )
        .reset_index()
    )


def _bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(5_000)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _audit_logs(logs_dir: Path) -> dict[str, Any]:
    forbidden = {"autonomous_success", "assisted_success", "signed_value", "helpful"}
    files = sorted(logs_dir.glob("*.npz"))
    violations: dict[str, list[str]] = {}
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            leaked = sorted(forbidden.intersection(archive.files))
        if leaked:
            violations[path.name] = leaked
    return {
        "files_checked": len(files),
        "forbidden_fields": sorted(forbidden),
        "violations": violations,
        "passed": not violations,
    }


def _plot_policy_frontier(data: pd.DataFrame, path: Path) -> None:
    selected = data.loc[
        (data["logging_scheme"] == "randomized")
        & data["method"].isin(
            [
                "dr_learner",
                "t_learner",
                "reversibility_proxy",
                "failure_risk",
                "paired_label_oracle",
            ]
        )
    ].copy()
    selected["method"] = selected["method"].map(_method_labels())
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=selected,
        x="intervention_mean",
        y="success_mean",
        hue="method",
        marker="o",
        ax=axis,
    )
    axis.set(xlim=(-0.03, 1.03), ylim=(0, 1.03))
    axis.set_xlabel("Realized intervention rate")
    axis.set_ylabel("Exact counterfactual success")
    axis.set_title("Learning help value from one observed future")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_estimation(metrics: pd.DataFrame, path: Path) -> None:
    selected = metrics.loc[
        (metrics["logging_scheme"] == "randomized")
        & metrics["method"].isin(
            ["s_learner", "t_learner", "ipw_learner", "dr_learner", "paired_label_oracle"]
        )
    ].copy()
    selected["method"] = selected["method"].map(_method_labels())
    sns.set_theme(style="whitegrid", context="talk")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    sns.barplot(data=selected, x="method", y="pehe", hue="method", legend=False, ax=axes[0])
    sns.stripplot(data=selected, x="method", y="pehe", color="#111827", ax=axes[0])
    sns.barplot(
        data=selected,
        x="method",
        y="helpful_auroc",
        hue="method",
        legend=False,
        ax=axes[1],
    )
    sns.stripplot(data=selected, x="method", y="helpful_auroc", color="#111827", ax=axes[1])
    axes[0].set_title("Individual-effect error (lower is better)")
    axes[1].set_title("Helpful-intervention ranking")
    axes[0].tick_params(axis="x", labelrotation=22)
    axes[1].tick_params(axis="x", labelrotation=22)
    axes[1].set_ylim(0.4, 1.0)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_learning_curves(data: pd.DataFrame, path: Path) -> None:
    selected = data.loc[data["target_budget"] == 0.25].copy()
    selected["logging_scheme"] = selected["logging_scheme"].map(
        {
            "randomized": "Randomized",
            "uncertainty_selective": "Uncertainty-selective",
            "adaptive_value": "Online adaptive-value",
        }
    )
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(9, 5.8))
    sns.lineplot(
        data=selected,
        x="logged_episodes",
        y="success_rate",
        hue="logging_scheme",
        marker="o",
        errorbar="sd",
        ax=axis,
    )
    axis.set_ylim(0.35, 0.9)
    axis.set_xlabel("Episodes with one observed future")
    axis.set_ylabel("Success near 25% target intervention")
    axis.set_title("Logged-data sample efficiency")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_open_world(data: pd.DataFrame, path: Path) -> None:
    selected = data.loc[data["target_budget"] == 0.25].copy()
    selected["method"] = selected["method"].map(_method_labels())
    sns.set_theme(style="whitegrid", context="talk")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for axis_plot, axis_name in zip(axes, ("task", "disturbance"), strict=True):
        subset = selected.loc[selected["held_out_axis"] == axis_name]
        sns.barplot(
            data=subset,
            x="held_out_value",
            y="success_rate",
            hue="method",
            errorbar="sd",
            ax=axis_plot,
        )
        axis_plot.set_title(f"Unseen {axis_name}")
        axis_plot.set_xlabel("")
        axis_plot.tick_params(axis="x", labelrotation=20)
    axes[0].set_ylabel("Exact counterfactual success")
    axes[1].set_ylabel("")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_overlap(data: pd.DataFrame, path: Path) -> None:
    data = data.copy()
    data["logging_scheme"] = data["logging_scheme"].map(
        {
            "randomized": "Randomized",
            "uncertainty_selective": "Uncertainty-selective",
            "adaptive_value": "Online adaptive-value",
        }
    )
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(9, 5.8))
    sns.histplot(
        data=data,
        x="propensity",
        hue="logging_scheme",
        bins=20,
        stat="density",
        common_norm=False,
        element="step",
        ax=axis,
    )
    axis.set_xlabel("Probability that assistance was logged")
    axis.set_title("Logging-policy overlap and positivity")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_latency(data: pd.DataFrame, path: Path) -> None:
    selected = data.loc[
        (data["logging_scheme"] == "randomized") & (data["target_budget"] == 0.25)
    ].copy()
    selected["method"] = selected["method"].map(_method_labels())
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(10, 5.8))
    sns.lineplot(
        data=selected,
        x="delay_candidate_hops",
        y="success_mean",
        hue="method",
        marker="o",
        ax=axis,
    )
    axis.set_xlabel("Assistance latency (candidate intervals)")
    axis.set_ylabel("Success at 25% intervention")
    axis.set_title("A help gate is conditional on who arrives when")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _markdown_report(
    train: CounterfactualData,
    evaluation: CounterfactualData,
    metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    open_world: pd.DataFrame,
    latency: pd.DataFrame,
    config: SelectiveConfig,
) -> str:
    metric_table = (
        metrics.loc[metrics["logging_scheme"] == "randomized"]
        .groupby("method")[["pehe", "helpful_auroc", "ate_error"]]
        .agg(["mean", "std"])
        .round(3)
    )
    metric_table.columns = ["_".join(column) for column in metric_table.columns]
    frontier = aggregate.loc[
        (aggregate["logging_scheme"] == "randomized")
        & aggregate["target_budget"].isin([0.25, 0.50])
        & aggregate["method"].isin(
            [
                "dr_learner",
                "t_learner",
                "reversibility_proxy",
                "failure_risk",
                "paired_label_oracle",
            ]
        )
    ].copy()
    for column in ("success_mean", "intervention_mean", "precision_mean"):
        frontier[column] = frontier[column].map(lambda value: f"{100 * value:.1f}%")
    paired_table = paired.loc[paired["target_budget"].isin([0.25, 0.50])].copy()
    paired_table["mean_delta"] = paired_table["mean_delta"].map(
        lambda value: f"{100 * value:+.1f} pp"
    )
    ood = (
        open_world.loc[open_world["target_budget"] == 0.25]
        .groupby(["held_out_axis", "method"])["success_rate"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ood["success"] = ood.apply(
        lambda row: f"{100 * row['mean']:.1f}% +/- {100 * row['std']:.1f}%", axis=1
    )
    latency_table = latency.loc[
        (latency["logging_scheme"] == "randomized") & (latency["target_budget"] == 0.25)
    ].copy()
    latency_table["success"] = latency_table.apply(
        lambda row: f"{100 * row['success_mean']:.1f}% +/- {100 * row['success_std']:.1f}%",
        axis=1,
    )
    return "\n".join(
        [
            "# InterveneSim-CF research report",
            "",
            "## Research question",
            "",
            "Can an agent learn the individual value of requesting assistance when each "
            "training episode reveals only the outcome that actually occurred?",
            "",
            "The learner receives one randomly timed candidate and one observed treatment arm "
            "per episode. Exact simulator forks are hidden from training and used only as an "
            "evaluation oracle.",
            "",
            f"Source data: **{train.sample_count:,}** training candidates from "
            f"**{len(np.unique(train.episode_ids))} episodes** and "
            f"**{evaluation.sample_count:,}** evaluation candidates from "
            f"**{len(np.unique(evaluation.episode_ids))} unseen-policy episodes**. Results use "
            f"**{len(config.logging_seeds)} logging seeds**.",
            "",
            "## Exact individual-effect estimation",
            "",
            metric_table.to_markdown(),
            "",
            "PEHE is root mean squared error against the hidden signed individual effect. The "
            "paired-label oracle is privileged and is included only as an upper-bound comparator.",
            "",
            "## Budget-aware deployment on unseen policies",
            "",
            frontier[
                [
                    "method",
                    "target_budget",
                    "success_mean",
                    "intervention_mean",
                    "precision_mean",
                ]
            ].to_markdown(index=False),
            "",
            "## Paired logging-seed comparisons",
            "",
            paired_table[
                [
                    "logging_scheme",
                    "target_budget",
                    "challenger",
                    "reference",
                    "mean_delta",
                    "positive_seeds",
                ]
            ].to_markdown(index=False),
            "",
            "## Open-world audit",
            "",
            "Every task and disturbance type is held out in turn. Training logs contain no row "
            "from the held-out group.",
            "",
            ood[["held_out_axis", "method", "success"]].to_markdown(index=False),
            "",
            "## Assistance-latency shift",
            "",
            "The same learned gate is evaluated when the scripted assistant arrives at the "
            "current candidate, one candidate later, or two candidates later along the exact "
            "autonomous trajectory.",
            "",
            latency_table[["delay_candidate_hops", "method", "success"]].to_markdown(index=False),
            "",
            "## Interpretation boundaries",
            "",
            "This benchmark validates estimators against exact simulator potential outcomes. It "
            "does not identify human-specific intervention effects, prove real-robot safety, or "
            "establish sim-to-real transfer. The reversibility proxy predicts autonomous failure "
            "under the assumption that help succeeds; it is PAINT-inspired, not a reproduction "
            "of PAINT's online algorithm.",
            "",
        ]
    )


def _train_config(config: SelectiveConfig) -> CATETrainConfig:
    return CATETrainConfig(
        epochs=config.epochs,
        hidden_dims=config.hidden_dims,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        seed=config.seed,
        crossfit_folds=config.crossfit_folds,
    )


def _method_labels() -> dict[str, str]:
    return {
        "paired_label_oracle": "Paired supervision",
        "failure_risk": "Failure risk",
        "ensemble_uncertainty": "Ensemble uncertainty",
        "s_learner": "S-learner",
        "t_learner": "T-learner",
        "ipw_learner": "IPW learner",
        "dr_learner": "DR learner",
        "reversibility_proxy": "Reversibility proxy",
    }


def _simulate_log(
    oracle: CounterfactualData,
    scheme: str,
    seed: int,
    config: SelectiveConfig,
    train_config: CATETrainConfig,
):
    if scheme == "adaptive_value":
        adaptive_config = replace(train_config, epochs=config.adaptive_epochs, seed=seed)
        return simulate_adaptive_single_world_log(
            oracle,
            seed,
            adaptive_config,
            assist_rate=config.assist_rate,
            positivity_floor=config.positivity_floor,
            update_interval=config.adaptive_update_interval,
            temperature=config.adaptive_temperature,
        )
    return simulate_single_world_log(
        oracle,
        seed,
        scheme=scheme,  # type: ignore[arg-type]
        assist_rate=config.assist_rate,
        positivity_floor=config.positivity_floor,
    )


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress:
        progress(message)
