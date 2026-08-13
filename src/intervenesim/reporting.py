from __future__ import annotations

import json
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
    report_path.write_text(_markdown_report(summary, sample_budget), encoding="utf-8")
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


def _markdown_report(summary: pd.DataFrame, sample_budget: int) -> str:
    pivot = summary.pivot(index="disturbance", columns="condition", values="success_rate")
    table = pivot.map(lambda value: f"{100 * value:.1f}%" if np.isfinite(value) else "—")
    return "\n".join(
        [
            "# InterveneSim benchmark report",
            "",
            "The two augmentation conditions each received "
            f"**{sample_budget:,} additional labeled actions**.",
            "",
            "## Autonomous success",
            "",
            table.to_markdown(),
            "",
            "## Interpretation",
            "",
            "This report describes a state-based simulation benchmark. Results do not "
            "establish real-world transfer, visual robustness, or safety. See "
            "`docs/benchmark.md` for the predeclared protocol and interpretation boundaries.",
            "",
        ]
    )
