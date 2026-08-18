from intervenesim.config import BenchmarkConfig, ResearchConfig, ValueConfig, VisualConfig


def test_config_loads_sequences_as_tuples() -> None:
    config = BenchmarkConfig.from_yaml("configs/smoke.yaml")
    assert config.hidden_dims == (64, 64)
    assert config.base_episodes == 3


def test_research_config_loads_tasks_and_seeds() -> None:
    config = ResearchConfig.from_yaml("configs/research-smoke.yaml")
    assert config.tasks == ("can", "milk")
    assert config.training_seeds == (71, 72)
    assert config.budgets == (100, 200)
    assert config.minimum_help_step == 5
    assert config.risk_detection_horizon == 3
    assert config.help_thresholds == (0.9, 0.97)


def test_value_config_loads_counterfactual_protocol() -> None:
    config = ValueConfig.from_yaml("configs/value-smoke.yaml")
    assert config.train_policy_seeds == (27,)
    assert config.eval_policy_seeds == (127,)
    assert config.candidate_steps == (20, 60)
    assert config.target_budgets == (0.25, 0.5)


def test_visual_config_loads_camera_transfer_protocol() -> None:
    config = VisualConfig.from_yaml("configs/visual-smoke.yaml")
    assert config.tasks == ("can",)
    assert config.cameras == ("frontview", "agentview")
    assert config.train_policy_seed == 27
    assert config.eval_policy_seed == 127
