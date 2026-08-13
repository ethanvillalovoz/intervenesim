from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def summarize_results(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (condition, disturbance), group in episodes.groupby(["condition", "disturbance"]):
        successes = int(group["success"].sum())
        count = int(len(group))
        low, high = wilson_interval(successes, count)
        successful_steps = group.loc[group["success"], "steps"]
        rows.append(
            {
                "condition": condition,
                "disturbance": disturbance,
                "episodes": count,
                "successes": successes,
                "success_rate": successes / count if count else float("nan"),
                "ci95_low": low,
                "ci95_high": high,
                "mean_success_steps": (
                    float(successful_steps.mean()) if len(successful_steps) else float("nan")
                ),
                "disturbance_fire_rate": float(group["disturbance_fired"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["disturbance", "condition"]).reset_index(drop=True)


def wilson_interval(successes: int, count: int, z: float = 1.96) -> tuple[float, float]:
    if count == 0:
        return float("nan"), float("nan")
    p = successes / count
    denominator = 1 + z**2 / count
    center = (p + z**2 / (2 * count)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * count)) / count) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def write_report(
    episodes: pd.DataFrame,
    output_dir: str | Path,
    sample_budget: int,
    manifest: dict[str, object],
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episodes_path = output / "episodes.csv"
    summary_path = output / "summary.csv"
    figure_path = output / "success_rates.png"
    report_path = output / "report.md"
    manifest_path = output / "manifest.json"
    episodes.to_csv(episodes_path, index=False)
    summary = summarize_results(episodes)
    summary.to_csv(summary_path, index=False)
    _plot_success(summary, figure_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_markdown_report(episodes, summary, sample_budget), encoding="utf-8")
    return {
        "episodes": str(episodes_path),
        "summary": str(summary_path),
        "figure": str(figure_path),
        "report": str(report_path),
        "manifest": str(manifest_path),
    }


def _plot_success(summary: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(13, 7))
    sns.barplot(
        data=summary,
        x="disturbance",
        y="success_rate",
        hue="condition",
        palette="colorblind",
        ax=axis,
    )
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Evaluation condition")
    axis.set_ylabel("Task success rate")
    axis.set_title("InterveneSim: autonomous recovery under controlled failures")
    axis.legend(title="Training condition", loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def paired_comparison(
    episodes: pd.DataFrame,
    challenger: str,
    reference: str,
    include_nominal: bool = False,
) -> dict[str, float | int | str]:
    """Compute an exact paired success comparison over matched episode seeds."""
    selected = episodes if include_nominal else episodes.loc[episodes["disturbance"] != "nominal"]
    paired = selected.pivot(index=["disturbance", "seed"], columns="condition", values="success")
    if challenger not in paired or reference not in paired:
        raise ValueError("both requested conditions must be present")
    challenger_success = paired[challenger].astype(bool)
    reference_success = paired[reference].astype(bool)
    wins = int((challenger_success & ~reference_success).sum())
    losses = int((~challenger_success & reference_success).sum())
    discordant = wins + losses
    tail = min(wins, losses)
    p_value = (
        min(1.0, 2 * sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant)
        if discordant
        else 1.0
    )
    return {
        "challenger": challenger,
        "reference": reference,
        "episodes": int(len(paired)),
        "challenger_rate": float(challenger_success.mean()),
        "reference_rate": float(reference_success.mean()),
        "delta": float(challenger_success.mean() - reference_success.mean()),
        "wins": wins,
        "losses": losses,
        "ties": int(len(paired) - discordant),
        "mcnemar_exact_p": p_value,
    }


def _markdown_report(
    episodes: pd.DataFrame,
    summary: pd.DataFrame,
    sample_budget: int,
) -> str:
    pivot = summary.pivot(index="disturbance", columns="condition", values="success_rate")
    order = ["nominal", "object_shift", "action_noise", "action_delay", "gripper_slip"]
    pivot = pivot.reindex([name for name in order if name in pivot.index])
    table = pivot.map(lambda value: f"{100 * value:.1f}%" if np.isfinite(value) else "—")
    paired = paired_comparison(episodes, "recovery_data", "more_demos")
    baseline = paired_comparison(episodes, "recovery_data", "baseline")
    nominal = pivot.loc["nominal"] if "nominal" in pivot.index else None
    nominal_sentence = ""
    if nominal is not None:
        nominal_sentence = (
            f" Nominal success was {100 * nominal['recovery_data']:.1f}% versus "
            f"{100 * nominal['more_demos']:.1f}% for additional clean demonstrations."
        )
    return "\n".join(
        [
            "# InterveneSim benchmark report",
            "",
            "The two augmentation conditions each received "
            f"**{sample_budget:,} additional labeled actions**.",
            "",
            "## Headline result",
            "",
            f"Across {paired['episodes']} disturbed, seed-matched episodes, recovery-data "
            f"training achieved **{100 * paired['challenger_rate']:.1f}%** success versus "
            f"**{100 * paired['reference_rate']:.1f}%** for additional clean demonstrations "
            f"(**{100 * paired['delta']:+.1f} percentage points**). It won "
            f"{paired['wins']} paired episodes and lost {paired['losses']} "
            f"(two-sided exact McNemar p={paired['mcnemar_exact_p']:.3g}).",
            "",
            f"Against the unaugmented baseline, recovery-data training improved disturbed "
            f"success from {100 * baseline['reference_rate']:.1f}% to "
            f"{100 * baseline['challenger_rate']:.1f}%." + nominal_sentence,
            "",
            "## Autonomous success",
            "",
            table.to_markdown(),
            "",
            "## Interpretation",
            "",
            "This report describes a state-based simulation benchmark. Results do not "
            "establish real-world transfer, visual robustness, or safety. See "
            "`docs/benchmark.md` for the predeclared protocol and interpretation boundaries. "
            "The confidence intervals and paired test quantify evaluation uncertainty for one "
            "training run; they do not capture variance across independently trained policies.",
            "",
        ]
    )
