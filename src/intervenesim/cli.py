from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console

from intervenesim.benchmark import run_benchmark
from intervenesim.config import BenchmarkConfig, ResearchConfig
from intervenesim.environment import PickPlaceEnv
from intervenesim.expert import ScriptedExpert
from intervenesim.research import run_research
from intervenesim.video import record_benchmark_comparison

app = typer.Typer(
    no_args_is_help=True,
    help="Intervention-guided recovery benchmark for simulated robot manipulation.",
)
console = Console()


@app.command()
def doctor() -> None:
    """Verify the local simulator, expert, and PyTorch accelerator."""
    result: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }
    with PickPlaceEnv(max_steps=260) as env:
        state = env.reset(27)
        expert = ScriptedExpert()
        expert.reset(state)
        steps = 0
        for _ in range(env.max_steps):
            state, _, done, info = env.step(expert.action(state))
            steps += 1
            if done:
                break
        result.update(
            simulator="robosuite/PickPlaceCan",
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            expert_success=bool(info["success"]),
            expert_steps=steps,
        )
    console.print_json(json.dumps(result))
    if not result["expert_success"]:
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "configs/benchmark.yaml"
    ),
    output: Annotated[
        Path | None, typer.Option(help="Override the configured output directory.")
    ] = None,
) -> None:
    """Run the complete equal-budget benchmark."""
    resolved = BenchmarkConfig.from_yaml(config)
    manifest = run_benchmark(resolved, output_dir=output, progress=_progress)
    console.print(f"[bold green]Complete[/bold green]: {manifest['artifacts']['report']}")


@app.command()
def smoke(
    output: Annotated[Path, typer.Option()] = Path("artifacts/runs/smoke"),
) -> None:
    """Run a tiny end-to-end pipeline for development and CI."""
    config = BenchmarkConfig.from_yaml("configs/smoke.yaml")
    manifest = run_benchmark(config, output_dir=output, progress=_progress)
    report = manifest["artifacts"]["report"]
    console.print(f"[bold green]Smoke test complete[/bold green]: {report}")


@app.command("record-comparison")
def record_comparison(
    run: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "artifacts/runs/benchmark-v1"
    ),
    output: Annotated[Path, typer.Option()] = Path("artifacts/videos/comparison.mp4"),
    disturbance: Annotated[str, typer.Option()] = "gripper_slip",
) -> None:
    """Record a matched baseline-failure / recovery-success benchmark episode."""
    result = record_benchmark_comparison(run, output, disturbance)
    console.print_json(json.dumps(result))


@app.command()
def research(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "configs/research.yaml"
    ),
    output: Annotated[
        Path | None, typer.Option(help="Override the configured output directory.")
    ] = None,
) -> None:
    """Run the multi-seed InterveneSim-X research benchmark."""
    resolved = ResearchConfig.from_yaml(config)
    manifest = run_research(resolved, output_dir=output, progress=_progress)
    console.print(f"[bold green]Complete[/bold green]: {manifest['artifacts']['report']}")


def _progress(message: str) -> None:
    console.print(f"[cyan]•[/cyan] {message}")


if __name__ == "__main__":
    app()
