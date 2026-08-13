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
from intervenesim.collection import collect_clean_demonstrations, collect_recovery_bundle
from intervenesim.config import ResearchConfig, TrainConfig
from intervenesim.dataset import TrajectoryData
from intervenesim.evaluation import evaluate_policies
from intervenesim.helping import evaluate_help_seeking, summarize_help_seeking
from intervenesim.policy import train_policy
from intervenesim.risk import RiskAgent, RiskData, RiskTrainConfig, train_risk_model

ProgressCallback = Callable[[str], None]


def run_research(
    config: ResearchConfig,
    output_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    if output_dir is not None:
        config = replace(config, output_dir=str(output_dir))
    output = Path(config.output_dir)
    datasets_dir = output / "datasets"
    checkpoints_dir = output / "checkpoints"
    results_dir = output / "results"
    for directory in (datasets_dir, checkpoints_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    started = time.time()

    base_path = datasets_dir / "base_multitask.npz"
    extra_path = datasets_dir / "extra_multitask.npz"
    if base_path.exists() and extra_path.exists():
        _emit(progress, "loading cached multi-domain clean datasets")
        base = TrajectoryData.load(base_path)
        extra = TrajectoryData.load(extra_path)
    else:
        base_parts: list[TrajectoryData] = []
        extra_parts: list[TrajectoryData] = []
        for task_index, task in enumerate(config.tasks):
            _emit(progress, f"collecting base demonstrations / {task}")
            part, _ = collect_clean_demonstrations(
                config.base_episodes_per_task,
                config.seed + task_index * 10_000,
                config.max_steps,
                progress=progress,
                task=task,
                task_conditioning=True,
            )
            base_parts.append(part)
            _emit(progress, f"collecting clean augmentation pool / {task}")
            part, _ = collect_clean_demonstrations(
                config.extra_episodes_per_task,
                config.seed + 100_000 + task_index * 10_000,
                config.max_steps,
                progress=progress,
                task=task,
                task_conditioning=True,
            )
            extra_parts.append(part)
        base = TrajectoryData.concatenate(
            base_parts, metadata={"kind": "base_multitask", "tasks": list(config.tasks)}
        )
        extra = TrajectoryData.concatenate(
            extra_parts, metadata={"kind": "extra_multitask", "tasks": list(config.tasks)}
        )
        base.save(base_path)
        extra.save(extra_path)

    reference_seed = config.training_seeds[0]
    reference_dir = checkpoints_dir / f"seed-{reference_seed}"
    reference_checkpoint = reference_dir / "baseline.pt"
    if not reference_checkpoint.exists():
        _emit(progress, f"training reference multi-domain baseline / seed {reference_seed}")
        _train_baseline(base, reference_checkpoint, config, reference_seed)

    recovery_path = datasets_dir / "recovery_multitask.npz"
    risk_path = datasets_dir / "risk_multitask.npz"
    if recovery_path.exists() and risk_path.exists():
        _emit(progress, "loading cached recovery and risk datasets")
        recovery = TrajectoryData.load(recovery_path)
        risk_data = RiskData.load(risk_path)
    else:
        from intervenesim.policy import PolicyAgent

        reference_policy = PolicyAgent.load(reference_checkpoint, device=config.device)
        recovery_parts: list[TrajectoryData] = []
        risk_parts: list[RiskData] = []
        for task_index, task in enumerate(config.tasks):
            _emit(progress, f"collecting corrective interventions / {task}")
            bundle = collect_recovery_bundle(
                reference_policy,
                config.recovery_episodes_per_task,
                config.seed + 200_000 + task_index * 10_000,
                config.max_steps,
                attempt_multiplier=5,
                risk_horizon=config.risk_horizon,
                nominal_risk_episodes=config.risk_nominal_episodes_per_task,
                progress=progress,
                task=task,
                task_conditioning=True,
            )
            recovery_parts.append(bundle.demonstrations)
            risk_parts.append(bundle.risk_data)
        recovery = TrajectoryData.concatenate(
            recovery_parts,
            metadata={"kind": "recovery_multitask", "tasks": list(config.tasks)},
        )
        risk_data = RiskData.concatenate(risk_parts)
        recovery.save(recovery_path)
        risk_data.save(risk_path)

    available_budget = min(extra.sample_count, recovery.sample_count)
    requested_budgets = tuple(budget for budget in config.budgets if budget <= available_budget)
    if not requested_budgets:
        raise RuntimeError(
            f"no configured budget fits available pool of {available_budget} samples"
        )
    max_budget = max(requested_budgets)

    risk_checkpoint = checkpoints_dir / "risk_gate.pt"
    if risk_checkpoint.exists():
        cached_risk = RiskAgent.load(risk_checkpoint, device="cpu")
        risk_train = {
            "checkpoint": str(risk_checkpoint),
            "threshold": cached_risk.threshold,
            "cached": True,
            **cached_risk.metadata,
        }
    else:
        _emit(progress, "training learned intervention gate")
        risk_train = train_risk_model(
            risk_data,
            risk_checkpoint,
            RiskTrainConfig(
                epochs=config.risk_epochs,
                hidden_dims=config.risk_hidden_dims,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                device=config.device,
                seed=config.seed,
                detection_horizon=config.risk_detection_horizon,
            ),
        )

    autonomous_parts: list[pd.DataFrame] = []
    for training_seed in config.training_seeds:
        seed_dir = checkpoints_dir / f"seed-{training_seed}"
        baseline_checkpoint = seed_dir / "baseline.pt"
        if not baseline_checkpoint.exists():
            _emit(progress, f"training baseline / seed {training_seed}")
            _train_baseline(base, baseline_checkpoint, config, training_seed)
        baseline_result = results_dir / f"autonomous-seed-{training_seed}-baseline.csv"
        if not baseline_result.exists():
            episodes = _evaluate_across_tasks(
                {"baseline": baseline_checkpoint}, config, training_seed, progress
            )
            episodes["training_seed"] = training_seed
            episodes["budget"] = 0
            episodes.to_csv(baseline_result, index=False)
        autonomous_parts.append(pd.read_csv(baseline_result))

        budgets_for_seed = requested_budgets if training_seed == reference_seed else (max_budget,)
        for budget in budgets_for_seed:
            result_path = results_dir / f"autonomous-seed-{training_seed}-budget-{budget}.csv"
            if result_path.exists():
                autonomous_parts.append(pd.read_csv(result_path))
                continue
            _emit(progress, f"training equal-budget methods / seed {training_seed} / {budget}")
            extra_budget = extra.stratified_sample_budget(
                budget, config.seed + 300_000 + budget, by=("tasks",)
            )
            recovery_budget = recovery.stratified_sample_budget(
                budget,
                config.seed + 300_000 + budget,
                by=("tasks", "disturbances"),
            )
            more_data = TrajectoryData.concatenate([base, extra_budget])
            recovery_training_data = TrajectoryData.concatenate([base, recovery_budget])
            common = dict(
                epochs=config.fine_tune_epochs,
                hidden_dims=config.hidden_dims,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate * 0.5,
                weight_decay=config.weight_decay,
                device=config.device,
                seed=training_seed + 1,
                fine_tune_from=str(baseline_checkpoint),
            )
            more_checkpoint = seed_dir / f"more-demos-{budget}.pt"
            recovery_checkpoint = seed_dir / f"recovery-bc-{budget}.pt"
            contrastive_checkpoint = seed_dir / f"contrastive-recovery-{budget}.pt"
            if not more_checkpoint.exists():
                train_policy(
                    more_data,
                    more_checkpoint,
                    TrainConfig(**common),
                    condition="more_demos",
                )
            if not recovery_checkpoint.exists():
                train_policy(
                    recovery_training_data,
                    recovery_checkpoint,
                    TrainConfig(**common),
                    condition="recovery_bc",
                )
            if not contrastive_checkpoint.exists():
                train_policy(
                    recovery_training_data,
                    contrastive_checkpoint,
                    TrainConfig(
                        **common,
                        extra={
                            "contrastive_weight": config.contrastive_weight,
                            "contrastive_margin": config.contrastive_margin,
                            "contrastive_min_distance": config.contrastive_min_distance,
                        },
                    ),
                    condition="contrastive_recovery",
                )
            episodes = _evaluate_across_tasks(
                {
                    "more_demos": more_checkpoint,
                    "recovery_bc": recovery_checkpoint,
                    "contrastive_recovery": contrastive_checkpoint,
                },
                config,
                training_seed,
                progress,
            )
            episodes["training_seed"] = training_seed
            episodes["budget"] = budget
            episodes.to_csv(result_path, index=False)
            autonomous_parts.append(episodes)
    autonomous = pd.concat(autonomous_parts, ignore_index=True)
    autonomous.to_csv(results_dir / "autonomous_episodes.csv", index=False)

    help_path = results_dir / "help_episodes.csv"
    if help_path.exists():
        help_episodes = pd.read_csv(help_path)
    else:
        help_parts: list[pd.DataFrame] = []
        for task in config.tasks:
            _emit(progress, f"evaluating learned help seeking / {task}")
            part = evaluate_help_seeking(
                reference_checkpoint,
                risk_checkpoint,
                config.disturbances,
                config.eval_episodes,
                config.seed + 600_000,
                config.max_steps,
                device=config.device,
                progress=progress,
                task=task,
                task_conditioning=True,
                minimum_help_step=config.minimum_help_step,
            )
            help_parts.append(part)
        help_episodes = pd.concat(help_parts, ignore_index=True)
        help_episodes.to_csv(help_path, index=False)

    sweep_path = results_dir / "help_threshold_sweep_episodes.csv"
    if sweep_path.exists():
        help_sweep_episodes = pd.read_csv(sweep_path)
    else:
        sweep_parts: list[pd.DataFrame] = []
        for task in config.tasks:
            for threshold in config.help_thresholds:
                _emit(progress, f"evaluating help threshold {threshold:.2f} / {task}")
                part = evaluate_help_seeking(
                    reference_checkpoint,
                    risk_checkpoint,
                    config.disturbances,
                    config.eval_episodes,
                    config.seed + 600_000,
                    config.max_steps,
                    device=config.device,
                    task=task,
                    task_conditioning=True,
                    minimum_help_step=config.minimum_help_step,
                    help_modes=("learned_help",),
                    risk_threshold=threshold,
                )
                sweep_parts.append(part)
        help_sweep_episodes = pd.concat(sweep_parts, ignore_index=True)
        help_sweep_episodes.to_csv(sweep_path, index=False)

    artifacts = write_research_report(
        autonomous,
        help_episodes,
        help_sweep_episodes,
        results_dir,
        config,
        max_budget,
        risk_train,
    )
    manifest: dict[str, object] = {
        "elapsed_seconds": time.time() - started,
        "config": config.to_dict(),
        "system": system_manifest(),
        "datasets": {
            "base_samples": base.sample_count,
            "extra_samples": extra.sample_count,
            "recovery_samples": recovery.sample_count,
            "rejected_recovery_samples": int(recovery.rejection_mask.sum()),
            "risk_samples": risk_data.sample_count,
            "risk_positive_samples": risk_data.positive_count,
        },
        "available_budget": available_budget,
        "evaluated_budgets": list(requested_budgets),
        "risk_training": risk_train,
        "artifacts": artifacts,
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    _emit(progress, f"research benchmark complete: {artifacts['report']}")
    return manifest


def _train_baseline(
    base: TrajectoryData,
    checkpoint: Path,
    config: ResearchConfig,
    seed: int,
) -> dict[str, object]:
    return train_policy(
        base,
        checkpoint,
        TrainConfig(
            epochs=config.train_epochs,
            hidden_dims=config.hidden_dims,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            device=config.device,
            seed=seed,
        ),
        condition="baseline",
    )


def _evaluate_across_tasks(
    checkpoints: dict[str, Path],
    config: ResearchConfig,
    training_seed: int,
    progress: ProgressCallback | None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for task in config.tasks:
        part = evaluate_policies(
            checkpoints,
            config.disturbances,
            config.eval_episodes,
            config.seed + 400_000,
            config.max_steps,
            device=config.device,
            progress=progress,
            task=task,
            task_conditioning=True,
        )
        part["training_seed"] = training_seed
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def write_research_report(
    autonomous: pd.DataFrame,
    help_episodes: pd.DataFrame,
    help_sweep_episodes: pd.DataFrame,
    output_dir: Path,
    config: ResearchConfig,
    max_budget: int,
    risk_training: dict[str, object],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final = autonomous.loc[
        (autonomous["budget"].isin([0, max_budget]))
        & (autonomous["training_seed"].isin(config.training_seeds))
    ].copy()
    final["disturbed"] = final["disturbance"] != "nominal"
    seed_rates = (
        final.groupby(["training_seed", "condition", "disturbed"], as_index=False)["success"]
        .mean()
        .rename(columns={"success": "success_rate"})
    )
    seed_rates.to_csv(output_dir / "autonomous_seed_rates.csv", index=False)
    aggregate = (
        seed_rates.groupby(["condition", "disturbed"])["success_rate"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    intervals: list[tuple[float, float]] = []
    for _, row in aggregate.iterrows():
        subset = final.loc[
            (final["condition"] == row["condition"]) & (final["disturbed"] == row["disturbed"])
        ]
        intervals.append(hierarchical_bootstrap_interval(subset, config.seed))
    aggregate["ci95_low"] = [interval[0] for interval in intervals]
    aggregate["ci95_high"] = [interval[1] for interval in intervals]
    aggregate.to_csv(output_dir / "autonomous_summary.csv", index=False)

    reference_seed = config.training_seeds[0]
    budget_curve = autonomous.loc[
        (autonomous["training_seed"] == reference_seed) & (autonomous["disturbance"] != "nominal")
    ]
    budget_curve = (
        budget_curve.groupby(["condition", "budget"], as_index=False)["success"]
        .mean()
        .rename(columns={"success": "success_rate"})
    )
    budget_curve.to_csv(output_dir / "budget_curve.csv", index=False)
    help_summary = summarize_help_seeking(help_episodes)
    help_summary.to_csv(output_dir / "help_summary.csv", index=False)
    help_sweep = _summarize_threshold_sweep(help_sweep_episodes)
    help_sweep.to_csv(output_dir / "help_threshold_sweep.csv", index=False)

    _plot_final(seed_rates, output_dir / "multiseed_success.png")
    _plot_budget_curve(budget_curve, output_dir / "budget_efficiency.png")
    _plot_help(help_summary, help_sweep, output_dir / "help_efficiency.png")

    paired_rows: list[dict[str, object]] = []
    disturbed_rates = seed_rates.loc[seed_rates["disturbed"]].pivot(
        index="training_seed", columns="condition", values="success_rate"
    )
    for challenger, reference in (
        ("recovery_bc", "more_demos"),
        ("contrastive_recovery", "recovery_bc"),
        ("contrastive_recovery", "more_demos"),
    ):
        paired_rows.append(paired_seed_comparison(disturbed_rates, challenger, reference))
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "paired_seed_comparisons.csv", index=False)

    report = _research_markdown(
        aggregate, paired, help_summary, help_sweep, max_budget, risk_training, config
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return {
        "report": str(report_path),
        "autonomous_summary": str(output_dir / "autonomous_summary.csv"),
        "seed_rates": str(output_dir / "autonomous_seed_rates.csv"),
        "budget_curve": str(output_dir / "budget_curve.csv"),
        "help_summary": str(output_dir / "help_summary.csv"),
        "help_threshold_sweep": str(output_dir / "help_threshold_sweep.csv"),
        "paired_comparisons": str(output_dir / "paired_seed_comparisons.csv"),
        "multiseed_figure": str(output_dir / "multiseed_success.png"),
        "budget_figure": str(output_dir / "budget_efficiency.png"),
        "help_figure": str(output_dir / "help_efficiency.png"),
    }


def hierarchical_bootstrap_interval(
    episodes: pd.DataFrame,
    seed: int,
    repetitions: int = 2000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    training_seeds = np.unique(episodes["training_seed"])
    estimates = np.empty(repetitions, dtype=np.float64)
    groups = {
        training_seed: episodes.loc[episodes["training_seed"] == training_seed, "success"].to_numpy(
            dtype=np.float64
        )
        for training_seed in training_seeds
    }
    for repetition in range(repetitions):
        sampled_seeds = rng.choice(training_seeds, size=len(training_seeds), replace=True)
        seed_means = []
        for training_seed in sampled_seeds:
            values = groups[training_seed]
            seed_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        estimates[repetition] = np.mean(seed_means)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def paired_seed_comparison(
    rates: pd.DataFrame,
    challenger: str,
    reference: str,
) -> dict[str, object]:
    differences = (rates[challenger] - rates[reference]).dropna().to_numpy(dtype=np.float64)
    observed = abs(float(differences.mean()))
    if not len(differences):
        p_value = float("nan")
    else:
        signs = np.asarray(
            [
                [1.0 if mask & (1 << index) else -1.0 for index in range(len(differences))]
                for mask in range(2 ** len(differences))
            ]
        )
        permuted = np.abs(np.mean(signs * np.abs(differences), axis=1))
        p_value = float(np.mean(permuted >= observed - 1e-12))
    return {
        "challenger": challenger,
        "reference": reference,
        "training_seeds": int(len(differences)),
        "mean_delta": float(differences.mean()) if len(differences) else float("nan"),
        "std_delta": float(differences.std(ddof=1)) if len(differences) > 1 else float("nan"),
        "positive_seeds": int((differences > 0).sum()),
        "sign_permutation_p": p_value,
    }


def _plot_final(seed_rates: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    data = seed_rates.loc[seed_rates["disturbed"]]
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(data=data, x="condition", y="success_rate", errorbar="sd", ax=axis)
    sns.stripplot(
        data=data,
        x="condition",
        y="success_rate",
        color="black",
        alpha=0.7,
        jitter=0.08,
        ax=axis,
    )
    axis.set_ylim(0, 1.0)
    axis.set_xlabel("Training method")
    axis.set_ylabel("Mean disturbed success")
    axis.set_title("InterveneSim-X: independent training seeds")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_budget_curve(data: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.lineplot(
        data=data,
        x="budget",
        y="success_rate",
        hue="condition",
        marker="o",
        ax=axis,
    )
    axis.set_ylim(0, 1.0)
    axis.set_xlabel("Additional labeled actions")
    axis.set_ylabel("Mean disturbed success")
    axis.set_title("Intervention data efficiency (reference training seed)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _summarize_threshold_sweep(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for threshold, group in episodes.groupby("risk_threshold"):
        disturbed = group.loc[group["disturbance"] != "nominal"]
        nominal = group.loc[group["disturbance"] == "nominal"]
        rows.append(
            {
                "threshold": float(threshold),
                "disturbed_episodes": int(len(disturbed)),
                "disturbed_success_rate": float(disturbed["success"].mean()),
                "disturbed_intervention_rate": float(disturbed["help_requested"].mean()),
                "nominal_success_rate": float(nominal["success"].mean()),
                "nominal_intervention_rate": float(nominal["help_requested"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def _plot_help(data: pd.DataFrame, sweep: pd.DataFrame, path: Path) -> None:
    disturbed = data.loc[data["disturbance"] != "nominal"]
    aggregate = disturbed.groupby("help_mode", as_index=False).agg(
        success_rate=("success_rate", "mean"),
        intervention_rate=("intervention_rate", "mean"),
    )
    sweep_points = sweep.rename(
        columns={
            "disturbed_success_rate": "success_rate",
            "disturbed_intervention_rate": "intervention_rate",
        }
    ).copy()
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=aggregate,
        x="intervention_rate",
        y="success_rate",
        hue="help_mode",
        s=180,
        ax=axis,
    )
    label_offsets = {
        "always_help": (-80, -16),
        "learned_help": (-82, 8),
        "oracle_help": (-76, 23),
        "no_help": (7, 6),
    }
    for row in aggregate.itertuples():
        axis.annotate(
            row.help_mode,
            (row.intervention_rate, row.success_rate),
            xytext=label_offsets.get(row.help_mode, (7, 6)),
            textcoords="offset points",
            fontsize=9,
        )
    axis.plot(
        sweep_points["intervention_rate"],
        sweep_points["success_rate"],
        color="#6f42c1",
        marker="o",
        linewidth=2,
        label="threshold sweep",
    )
    for row in sweep_points.itertuples():
        axis.annotate(
            f"{row.threshold:.2f}",
            (row.intervention_rate, row.success_rate),
            xytext=(5, -14),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(0, 1.03)
    axis.set_xlabel("Episodes requesting expert help")
    axis.set_ylabel("Assisted task success")
    axis.set_title("Success versus intervention rate")
    axis.legend().remove()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _research_markdown(
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    help_summary: pd.DataFrame,
    help_sweep: pd.DataFrame,
    max_budget: int,
    risk_training: dict[str, object],
    config: ResearchConfig,
) -> str:
    disturbed = aggregate.loc[aggregate["disturbed"]].copy()
    disturbed["success"] = disturbed.apply(
        lambda row: (
            f"{100 * row['mean']:.1f}% ± {100 * row['std']:.1f}% "
            f"[{100 * row['ci95_low']:.1f}, {100 * row['ci95_high']:.1f}]"
        ),
        axis=1,
    )
    table = disturbed[["condition", "success"]].to_markdown(index=False)
    pair_table = paired.copy()
    pair_table["mean_delta"] = pair_table["mean_delta"].map(lambda value: f"{100 * value:+.1f} pp")
    pair_table["std_delta"] = pair_table["std_delta"].map(lambda value: f"{100 * value:.1f} pp")
    pair_table["sign_permutation_p"] = pair_table["sign_permutation_p"].map(
        lambda value: f"{value:.3g}"
    )
    risk_metrics = risk_training.get("validation_metrics", {})
    if not risk_metrics and isinstance(risk_training.get("cached"), bool):
        risk_text = "Risk checkpoint loaded from the cached run; see `manifest.json`."
    else:
        risk_text = (
            f"Held-out risk AUROC: **{float(risk_metrics.get('auroc', float('nan'))):.3f}**; "
            "average precision: "
            f"**{float(risk_metrics.get('average_precision', float('nan'))):.3f}**; "
            f"recall: **{float(risk_metrics.get('recall', float('nan'))):.3f}**."
        )
    help_disturbed = help_summary.loc[help_summary["disturbance"] != "nominal"]
    help_aggregate = help_disturbed.groupby("help_mode").agg(
        success_rate=("success_rate", "mean"), intervention_rate=("intervention_rate", "mean")
    )
    help_table = help_aggregate.map(lambda value: f"{100 * value:.1f}%").to_markdown()
    sweep_table = help_sweep.copy()
    sweep_table["threshold"] = sweep_table["threshold"].map(lambda value: f"{value:.2f}")
    for column in (
        "disturbed_success_rate",
        "disturbed_intervention_rate",
        "nominal_success_rate",
        "nominal_intervention_rate",
    ):
        sweep_table[column] = sweep_table[column].map(lambda value: f"{100 * value:.1f}%")
    sweep_table = sweep_table.drop(columns=["disturbed_episodes"]).to_markdown(index=False)
    return "\n".join(
        [
            "# InterveneSim-X research report",
            "",
            f"Primary results use **{len(config.training_seeds)} independent training seeds**, "
            f"**{len(config.tasks)} object domains**, four disturbances, and "
            f"**{max_budget:,}** additional "
            "action labels per augmented condition.",
            "",
            "## Autonomous disturbed success",
            "",
            "Values are mean ± standard deviation across training seeds followed by a "
            "hierarchical 95% bootstrap interval in brackets.",
            "",
            table,
            "",
            "## Paired training-seed comparisons",
            "",
            pair_table.to_markdown(index=False),
            "",
            f"With {len(config.training_seeds)} training seeds, the exact sign-permutation test "
            "has limited resolution; "
            "effect consistency and interval width are emphasized over a binary "
            "significance label.",
            "",
            "## Learned help seeking",
            "",
            risk_text,
            "",
            help_table,
            "",
            "The validation-selected high-recall threshold is not selective in deployment. "
            "The following post-audit threshold sweep is exploratory and is reported in full:",
            "",
            sweep_table,
            "",
            "## Boundaries",
            "",
            "These results concern state-based simulation with scripted corrections. They do not "
            "establish real-world transfer, human intervention quality, visual robustness, or "
            "robot safety. See `docs/intervenesim-x.md` for the predeclared protocol.",
            "",
        ]
    )


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress:
        progress(message)
