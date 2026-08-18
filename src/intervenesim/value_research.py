from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from intervenesim.benchmark import system_manifest
from intervenesim.config import ValueConfig
from intervenesim.counterfactual import CounterfactualData, collect_counterfactual_data
from intervenesim.risk import binary_metrics
from intervenesim.value import (
    ValueAgent,
    ValueTrainConfig,
    budget_thresholds,
    evaluate_counterfactual_gates,
    risk_scores,
    summarize_gate_evaluation,
    train_value_model,
)

ProgressCallback = Callable[[str], None]


def run_value_research(
    config: ValueConfig,
    output_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    if output_dir is not None:
        config = replace(config, output_dir=str(output_dir))
    output = Path(config.output_dir)
    datasets = output / "datasets"
    checkpoints = output / "checkpoints"
    results = output / "results"
    for directory in (datasets, checkpoints, results):
        directory.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    started = time.time()
    research_run = Path(config.research_run)
    all_policy_seeds = config.train_policy_seeds + config.eval_policy_seeds
    ensemble_paths = tuple(
        research_run / "checkpoints" / f"seed-{seed}" / "baseline.pt"
        for seed in config.train_policy_seeds
    )
    missing = [str(path) for path in ensemble_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing prerequisite policy checkpoints: {missing}")

    split_parts: dict[str, list[CounterfactualData]] = {"train": [], "eval": []}
    for policy_seed in all_policy_seeds:
        split = "train" if policy_seed in config.train_policy_seeds else "eval"
        episodes = config.train_episodes if split == "train" else config.eval_episodes
        policy_path = research_run / "checkpoints" / f"seed-{policy_seed}" / "baseline.pt"
        if not policy_path.exists():
            raise FileNotFoundError(f"missing prerequisite policy checkpoint: {policy_path}")
        for task_index, task in enumerate(config.tasks):
            part_path = datasets / f"{split}-seed-{policy_seed}-{task}.npz"
            if part_path.exists():
                part = CounterfactualData.load(part_path)
            else:
                _emit(progress, f"collecting {split} counterfactuals / seed {policy_seed} / {task}")
                part = collect_counterfactual_data(
                    policy_path,
                    ensemble_paths,
                    task,
                    config.disturbances,
                    episodes,
                    config.seed
                    + (0 if split == "train" else 1_000_000)
                    + policy_seed * 1_000
                    + task_index * 100_000,
                    config.candidate_steps,
                    config.max_steps,
                    policy_seed,
                    config.device,
                )
                part.save(part_path)
            split_parts[split].append(part)
    train = CounterfactualData.concatenate(split_parts["train"])
    evaluation = CounterfactualData.concatenate(split_parts["eval"])
    train.save(datasets / "train_counterfactuals.npz")
    evaluation.save(datasets / "eval_counterfactuals.npz")
    train.to_frame().to_csv(results / "train_counterfactuals.csv", index=False)
    evaluation.to_frame().to_csv(results / "eval_counterfactuals.csv", index=False)

    value_checkpoint = checkpoints / "value_gate.pt"
    if value_checkpoint.exists():
        value_agent = ValueAgent.load(value_checkpoint, device=config.device)
        value_training: dict[str, object] = {
            "checkpoint": str(value_checkpoint),
            "cached": True,
            **value_agent.metadata,
        }
    else:
        _emit(progress, "training counterfactual value gate")
        value_training = train_value_model(
            train,
            value_checkpoint,
            ValueTrainConfig(
                epochs=config.value_epochs,
                hidden_dims=config.value_hidden_dims,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                device=config.device,
                seed=config.seed,
            ),
        )
        value_agent = ValueAgent.load(value_checkpoint, device=config.device)

    risk_checkpoint = research_run / "checkpoints" / "risk_gate.pt"
    train_scores = {
        "value_gate": value_agent.scores(train),
        "risk_gate": risk_scores(train, risk_checkpoint),
        "uncertainty_gate": train.ensemble_uncertainty,
    }
    evaluation_scores = {
        "value_gate": value_agent.scores(evaluation),
        "risk_gate": risk_scores(evaluation, risk_checkpoint),
        "uncertainty_gate": evaluation.ensemble_uncertainty,
    }
    thresholds = {
        method: budget_thresholds(scores, train.episode_ids, config.target_budgets)
        for method, scores in train_scores.items()
    }
    episodes = evaluate_counterfactual_gates(
        evaluation, evaluation_scores, thresholds, config.target_budgets
    )
    episodes.to_csv(results / "gate_episodes.csv", index=False)
    summary = summarize_gate_evaluation(episodes)
    summary.to_csv(results / "gate_summary.csv", index=False)
    metrics = {
        method: binary_metrics(evaluation.helpful, scores, 0.5)
        for method, scores in evaluation_scores.items()
    }
    (results / "gate_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (results / "thresholds.json").write_text(
        json.dumps(
            {
                method: {str(budget): threshold for budget, threshold in values.items()}
                for method, values in thresholds.items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    _emit(progress, "running secondary leave-one-policy-seed-out analysis")
    crossval = _cross_validate_policy_seeds(
        train,
        evaluation,
        risk_checkpoint,
        checkpoints / "crossval",
        config,
    )
    crossval.to_csv(results / "crossval_gate_episodes.csv", index=False)
    crossval_summary = summarize_gate_evaluation(crossval)
    crossval_summary.to_csv(results / "crossval_gate_summary.csv", index=False)
    crossval_seed_rates = (
        crossval.groupby(["policy_seed", "method", "target_budget"], dropna=False)[
            ["success", "help_requested"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"success": "success_rate", "help_requested": "intervention_rate"})
    )
    crossval_seed_rates.to_csv(results / "crossval_seed_rates.csv", index=False)
    crossval_paired = _crossval_paired(crossval_seed_rates)
    crossval_paired.to_csv(results / "crossval_paired.csv", index=False)
    _plot_frontier(summary, results / "value_frontier.png")
    _plot_precision(summary, results / "request_precision.png")
    _plot_crossval(crossval_seed_rates, results / "crossval_success.png")
    report = _markdown_report(
        train, evaluation, summary, metrics, crossval_seed_rates, crossval_paired, config
    )
    (results / "report.md").write_text(report, encoding="utf-8")
    manifest: dict[str, object] = {
        "elapsed_seconds": time.time() - started,
        "config": config.to_dict(),
        "system": system_manifest(),
        "datasets": {
            "train_samples": train.sample_count,
            "train_episodes": int(len(np.unique(train.episode_ids))),
            "train_helpful": int(train.helpful.sum()),
            "eval_samples": evaluation.sample_count,
            "eval_episodes": int(len(np.unique(evaluation.episode_ids))),
            "eval_helpful": int(evaluation.helpful.sum()),
        },
        "value_training": value_training,
        "thresholds": thresholds,
        "gate_metrics": metrics,
        "artifacts": {
            "report": str(results / "report.md"),
            "gate_summary": str(results / "gate_summary.csv"),
            "gate_episodes": str(results / "gate_episodes.csv"),
            "frontier": str(results / "value_frontier.png"),
            "precision": str(results / "request_precision.png"),
            "crossval_seed_rates": str(results / "crossval_seed_rates.csv"),
            "crossval_paired": str(results / "crossval_paired.csv"),
            "crossval_success": str(results / "crossval_success.png"),
        },
    }
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _emit(progress, f"counterfactual value benchmark complete: {results / 'report.md'}")
    return manifest


def _plot_frontier(summary: pd.DataFrame, path: Path) -> None:
    data = summary.loc[summary["method"].isin(["value_gate", "risk_gate", "uncertainty_gate"])]
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=data,
        x="intervention_rate",
        y="success_rate",
        hue="method",
        marker="o",
        ax=axis,
    )
    controls = summary.loc[summary["method"].isin(["never_help", "always_help", "oracle_value"])]
    sns.scatterplot(
        data=controls,
        x="intervention_rate",
        y="success_rate",
        hue="method",
        marker="X",
        s=180,
        legend=False,
        ax=axis,
    )
    axis.set(xlim=(-0.03, 1.03), ylim=(0, 1.03))
    axis.set_xlabel("Episodes requesting expert takeover")
    axis.set_ylabel("Counterfactually evaluated success")
    axis.set_title("Causal value of assistance")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_precision(summary: pd.DataFrame, path: Path) -> None:
    data = summary.loc[summary["method"].isin(["value_gate", "risk_gate", "uncertainty_gate"])]
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=data,
        x="intervention_rate",
        y="request_precision",
        hue="method",
        marker="o",
        ax=axis,
    )
    axis.set(xlim=(-0.03, 1.03), ylim=(0, 1.03))
    axis.set_xlabel("Episodes requesting expert takeover")
    axis.set_ylabel("Requests with positive counterfactual value")
    axis.set_title("Are help requests actually useful?")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_crossval(seed_rates: pd.DataFrame, path: Path) -> None:
    data = seed_rates.loc[
        seed_rates["method"].isin(["value_gate", "risk_gate", "uncertainty_gate"])
        & seed_rates["target_budget"].isin([0.25, 0.5])
    ].copy()
    labels = {
        "value_gate": "Value gate",
        "risk_gate": "Failure risk",
        "uncertainty_gate": "Ensemble uncertainty",
    }
    data["method"] = data["method"].map(labels)
    data["budget"] = data["target_budget"].map(
        {0.25: "25% target budget", 0.5: "50% target budget"}
    )
    sns.set_theme(style="whitegrid", context="talk")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    order = ["Value gate", "Failure risk", "Ensemble uncertainty"]
    palette = ["#2563eb", "#ef4444", "#f59e0b"]
    for axis, budget in zip(axes, ("25% target budget", "50% target budget"), strict=True):
        subset = data.loc[data["budget"] == budget]
        sns.barplot(
            data=subset,
            x="method",
            y="success_rate",
            order=order,
            hue="method",
            hue_order=order,
            errorbar="sd",
            palette=palette,
            legend=False,
            ax=axis,
        )
        sns.stripplot(
            data=subset,
            x="method",
            y="success_rate",
            order=order,
            color="#111827",
            alpha=0.65,
            jitter=0.12,
            size=5,
            ax=axis,
        )
        axis.set_title(budget)
        axis.set_xlabel("")
        axis.tick_params(axis="x", labelrotation=15)
        axis.set_ylim(0.45, 1.01)
        axis.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
    axes[0].set_ylabel("Held-out policy success")
    axes[1].set_ylabel("")
    figure.suptitle("Five-fold policy-seed robustness", y=1.01)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown_report(
    train: CounterfactualData,
    evaluation: CounterfactualData,
    summary: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
    crossval_seed_rates: pd.DataFrame,
    crossval_paired: pd.DataFrame,
    config: ValueConfig,
) -> str:
    table = summary.copy()
    for column in (
        "success_rate",
        "intervention_rate",
        "request_precision",
        "nominal_false_alarm_rate",
        "matched_oracle_success_rate",
        "regret_to_matched_oracle",
    ):
        table[column] = table[column].map(
            lambda value: "-" if pd.isna(value) else f"{100 * value:.1f}%"
        )
    selected = table[
        [
            "method",
            "target_budget",
            "success_rate",
            "intervention_rate",
            "request_precision",
            "regret_to_matched_oracle",
        ]
    ].to_markdown(index=False)
    metric_rows = pd.DataFrame(
        [
            {
                "method": method,
                "AUROC": values["auroc"],
                "average_precision": values["average_precision"],
            }
            for method, values in metrics.items()
        ]
    ).to_markdown(index=False, floatfmt=".3f")
    crossval_aggregate = (
        crossval_seed_rates.loc[
            crossval_seed_rates["method"].isin(["value_gate", "risk_gate", "uncertainty_gate"])
        ]
        .groupby(["method", "target_budget"])
        .agg(
            success_mean=("success_rate", "mean"),
            success_std=("success_rate", "std"),
            intervention_mean=("intervention_rate", "mean"),
        )
        .reset_index()
    )
    crossval_aggregate["success"] = crossval_aggregate.apply(
        lambda row: f"{100 * row['success_mean']:.1f}% +/- {100 * row['success_std']:.1f}%",
        axis=1,
    )
    crossval_aggregate["realized_intervention_rate"] = crossval_aggregate["intervention_mean"].map(
        lambda value: f"{100 * value:.1f}%"
    )
    crossval_table = crossval_aggregate[
        ["method", "target_budget", "success", "realized_intervention_rate"]
    ].to_markdown(index=False)
    paired_table = crossval_paired.copy()
    paired_table["mean_delta"] = paired_table["mean_delta"].map(
        lambda value: f"{100 * value:+.1f} pp"
    )
    paired_table["std_delta"] = paired_table["std_delta"].map(lambda value: f"{100 * value:.1f} pp")
    paired_table = paired_table.to_markdown(index=False)
    return "\n".join(
        [
            "# InterveneSim-Value research report",
            "",
            f"Training counterfactuals: **{train.sample_count:,}** candidate states across "
            f"**{len(config.train_policy_seeds)} policy seeds**. Evaluation counterfactuals: "
            f"**{evaluation.sample_count:,}** states from "
            f"**{len(config.eval_policy_seeds)} held-out policy seeds**.",
            "",
            "## Helpfulness discrimination",
            "",
            metric_rows,
            "",
            "## Budget-aware deployment",
            "",
            selected,
            "",
            "Thresholds were selected only from training-policy episodes. Oracle regret is "
            "computed at each method's realized number of evaluation interventions.",
            "",
            "## Secondary five-fold policy-seed cross-validation",
            "",
            "This analysis was added after the primary held-out-policy audit and is therefore "
            "exploratory. Each fold trains on four policy seeds and evaluates the fifth.",
            "",
            crossval_table,
            "",
            paired_table,
            "",
            "## Boundaries",
            "",
            "The counterfactual is exact for this simulator and scripted expert only. It does "
            "not establish human-intervention value, real-robot safety, or sim-to-real transfer.",
            "",
        ]
    )


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress:
        progress(message)


def _cross_validate_policy_seeds(
    train: CounterfactualData,
    evaluation: CounterfactualData,
    risk_checkpoint: Path,
    checkpoint_dir: Path,
    config: ValueConfig,
) -> pd.DataFrame:
    all_data = CounterfactualData.concatenate([train, evaluation])
    parts: list[pd.DataFrame] = []
    for held_out_seed in np.unique(all_data.policy_seeds):
        train_data = all_data.subset(all_data.policy_seeds != held_out_seed)
        eval_data = all_data.subset(all_data.policy_seeds == held_out_seed)
        checkpoint = checkpoint_dir / f"held-out-{held_out_seed}.pt"
        if not checkpoint.exists():
            train_value_model(
                train_data,
                checkpoint,
                ValueTrainConfig(
                    epochs=config.value_epochs,
                    hidden_dims=config.value_hidden_dims,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                    device=config.device,
                    seed=config.seed + int(held_out_seed),
                ),
            )
        agent = ValueAgent.load(checkpoint, device=config.device)
        train_scores = {
            "value_gate": agent.scores(train_data),
            "risk_gate": risk_scores(train_data, risk_checkpoint),
            "uncertainty_gate": train_data.ensemble_uncertainty,
        }
        eval_scores = {
            "value_gate": agent.scores(eval_data),
            "risk_gate": risk_scores(eval_data, risk_checkpoint),
            "uncertainty_gate": eval_data.ensemble_uncertainty,
        }
        thresholds = {
            method: budget_thresholds(scores, train_data.episode_ids, config.target_budgets)
            for method, scores in train_scores.items()
        }
        parts.append(
            evaluate_counterfactual_gates(eval_data, eval_scores, thresholds, config.target_budgets)
        )
    return pd.concat(parts, ignore_index=True)


def _crossval_paired(seed_rates: pd.DataFrame) -> pd.DataFrame:
    selected = seed_rates.loc[
        seed_rates["method"].isin(["value_gate", "risk_gate", "uncertainty_gate"])
    ]
    rows: list[dict[str, object]] = []
    for budget in (0.25, 0.5):
        pivot = selected.loc[selected["target_budget"] == budget].pivot(
            index="policy_seed", columns="method", values="success_rate"
        )
        for reference in ("risk_gate", "uncertainty_gate"):
            differences = (pivot["value_gate"] - pivot[reference]).dropna()
            rows.append(
                {
                    "budget": budget,
                    "challenger": "value_gate",
                    "reference": reference,
                    "policy_seeds": len(differences),
                    "mean_delta": float(differences.mean()),
                    "std_delta": float(differences.std(ddof=1)),
                    "positive_seeds": int((differences > 0).sum()),
                }
            )
    return pd.DataFrame(rows)
