import numpy as np

from intervenesim.counterfactual import CounterfactualData
from intervenesim.selective import (
    CATETrainConfig,
    LoggedBanditData,
    counterfactual_metrics,
    delayed_assistance_data,
    evaluate_cost_sensitive_policy,
    evaluate_ranked_budget_policy,
    fit_cate_estimator,
    simulate_adaptive_single_world_log,
    simulate_single_world_log,
)


def synthetic_oracle(episodes: int = 40, candidates: int = 3) -> CounterfactualData:
    episode_ids = np.repeat(np.arange(episodes), candidates)
    feature = np.repeat(np.linspace(-1.0, 1.0, episodes), candidates)
    autonomous = feature > 0.2
    assisted = feature > -0.3
    rows = len(feature)
    return CounterfactualData(
        observations=feature[:, None],
        observation_deltas=np.zeros((rows, 1)),
        autonomous_actions=np.zeros((rows, 1)),
        ensemble_uncertainty=np.linspace(0.0, 1.0, rows),
        autonomous_success=autonomous,
        assisted_success=assisted,
        steps=np.tile(np.asarray([10, 30, 50]), episodes),
        episode_ids=episode_ids,
        policy_seeds=np.ones(rows),
        tasks=np.asarray(["can"] * rows),
        disturbances=np.asarray(["noise"] * rows),
        metadata={"test": True},
    )


def test_single_world_log_has_one_factual_outcome_per_episode(tmp_path) -> None:
    oracle = synthetic_oracle()
    log = simulate_single_world_log(oracle, seed=11)
    assert log.sample_count == 40
    assert len(np.unique(log.episode_ids)) == 40
    selected = {
        (int(episode), int(step)): index
        for index, (episode, step) in enumerate(zip(oracle.episode_ids, oracle.steps, strict=True))
    }
    for row in range(log.sample_count):
        oracle_row = selected[(int(log.episode_ids[row]), int(log.steps[row]))]
        expected = (
            oracle.assisted_success[oracle_row]
            if log.treatments[row]
            else oracle.autonomous_success[oracle_row]
        )
        assert log.outcomes[row] == expected
    path = log.save(tmp_path / "single-world.npz")
    with np.load(path, allow_pickle=False) as archive:
        assert "autonomous_success" not in archive.files
        assert "assisted_success" not in archive.files
    loaded = LoggedBanditData.load(path)
    np.testing.assert_array_equal(loaded.outcomes, log.outcomes)


def test_selective_logger_enforces_overlap() -> None:
    log = simulate_single_world_log(
        synthetic_oracle(),
        seed=19,
        scheme="uncertainty_selective",
        positivity_floor=0.15,
    )
    assert log.propensities.min() >= 0.15
    assert log.propensities.max() <= 0.85
    assert np.std(log.propensities) > 0.05


def test_adaptive_logger_uses_only_past_factual_history() -> None:
    oracle = synthetic_oracle(episodes=80)
    config = CATETrainConfig(
        epochs=2,
        hidden_dims=(8,),
        batch_size=32,
        learning_rate=1e-3,
        device="cpu",
        seed=3,
        crossfit_folds=2,
    )
    log = simulate_adaptive_single_world_log(
        oracle,
        seed=31,
        train_config=config,
        positivity_floor=0.15,
        update_interval=20,
    )
    assert log.sample_count == 80
    assert log.metadata["online_information"] == "past factual outcomes only"
    assert log.propensities.min() >= 0.15
    assert log.propensities.max() <= 0.85
    assert np.std(log.propensities[20:]) > 0.0


def test_all_single_world_estimators_return_finite_effects() -> None:
    oracle = synthetic_oracle(episodes=80)
    log = simulate_single_world_log(oracle, seed=23)
    config = CATETrainConfig(
        epochs=2,
        hidden_dims=(8,),
        batch_size=32,
        learning_rate=1e-3,
        device="cpu",
        seed=5,
        crossfit_folds=2,
    )
    for method in (
        "s_learner",
        "t_learner",
        "ipw_learner",
        "dr_learner",
        "reversibility_proxy",
    ):
        effects = fit_cate_estimator(log, method, config).effects(oracle.features)
        assert effects.shape == (oracle.sample_count,)
        assert np.isfinite(effects).all()
        assert ((effects >= -1.0) & (effects <= 1.0)).all()


def test_exact_counterfactual_metrics_and_cost_policy() -> None:
    oracle = synthetic_oracle()
    exact = oracle.signed_value.astype(np.float32)
    metrics = counterfactual_metrics(oracle, exact)
    assert metrics["pehe"] == 0.0
    assert metrics["ate_error"] == 0.0
    episodes = evaluate_cost_sensitive_policy(oracle, exact, (0.25,), "exact")
    assert len(episodes) == len(np.unique(oracle.episode_ids))
    assert episodes["utility"].notna().all()


def test_ranked_budget_policy_uses_exact_episode_budget() -> None:
    oracle = synthetic_oracle(episodes=40)
    scores = np.linspace(0.0, 1.0, oracle.sample_count)
    episodes = evaluate_ranked_budget_policy(oracle, {"test_gate": scores}, (0.25,))
    selected = episodes.loc[episodes["method"] == "test_gate"]
    assert len(selected) == 40
    assert selected["help_requested"].sum() == 10


def test_delayed_assistance_uses_later_exact_branch() -> None:
    oracle = synthetic_oracle(episodes=4)
    delayed = delayed_assistance_data(oracle, candidate_hops=1)
    for episode in range(4):
        indices = np.flatnonzero(oracle.episode_ids == episode)
        np.testing.assert_array_equal(
            delayed.assisted_success[indices[:-1]], oracle.assisted_success[indices[1:]]
        )
        assert delayed.assisted_success[indices[-1]] == oracle.autonomous_success[indices[-1]]
