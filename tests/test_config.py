from intervenesim.config import BenchmarkConfig, ResearchConfig


def test_config_loads_sequences_as_tuples() -> None:
    config = BenchmarkConfig.from_yaml("configs/smoke.yaml")
    assert config.hidden_dims == (64, 64)
    assert config.base_episodes == 3


def test_research_config_loads_tasks_and_seeds() -> None:
    config = ResearchConfig.from_yaml("configs/research-smoke.yaml")
    assert config.tasks == ("can", "milk")
    assert config.training_seeds == (71, 72)
    assert config.budgets == (100, 200)
