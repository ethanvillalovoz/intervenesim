from intervenesim.config import BenchmarkConfig


def test_config_loads_sequences_as_tuples() -> None:
    config = BenchmarkConfig.from_yaml("configs/smoke.yaml")
    assert config.hidden_dims == (64, 64)
    assert config.base_episodes == 3
