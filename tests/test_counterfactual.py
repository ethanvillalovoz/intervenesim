import numpy as np

from intervenesim.counterfactual import CounterfactualData
from intervenesim.value import budget_thresholds, evaluate_counterfactual_gates


def synthetic_data() -> CounterfactualData:
    return CounterfactualData(
        observations=np.zeros((4, 2)),
        observation_deltas=np.zeros((4, 2)),
        autonomous_actions=np.zeros((4, 1)),
        ensemble_uncertainty=np.asarray([0.1, 0.9, 0.2, 0.3]),
        autonomous_success=np.asarray([False, False, True, True]),
        assisted_success=np.asarray([False, True, True, False]),
        steps=np.asarray([10, 30, 10, 30]),
        episode_ids=np.asarray([0, 0, 1, 1]),
        policy_seeds=np.asarray([1, 1, 2, 2]),
        tasks=np.asarray(["can"] * 4),
        disturbances=np.asarray(["slip", "slip", "nominal", "nominal"]),
        metadata={"test": True},
    )


def test_counterfactual_data_preserves_signed_outcomes(tmp_path) -> None:
    data = synthetic_data()
    np.testing.assert_array_equal(data.helpful, [False, True, False, False])
    np.testing.assert_array_equal(data.signed_value, [0, 1, 0, -1])
    loaded = CounterfactualData.load(data.save(tmp_path / "counterfactual.npz"))
    np.testing.assert_array_equal(loaded.signed_value, data.signed_value)


def test_budget_thresholds_use_episode_maxima() -> None:
    scores = np.asarray([0.1, 0.9, 0.2, 0.3])
    thresholds = budget_thresholds(scores, np.asarray([0, 0, 1, 1]), (0.5,))
    assert thresholds[0.5] == 0.6


def test_gate_uses_earliest_candidate_above_threshold() -> None:
    data = synthetic_data()
    scores = {"value_gate": np.asarray([0.1, 0.9, 0.8, 0.7])}
    thresholds = {"value_gate": {0.5: 0.75}}
    episodes = evaluate_counterfactual_gates(data, scores, thresholds, (0.5,))
    selected = episodes.loc[episodes["method"] == "value_gate"].set_index("episode_id")
    assert selected.loc[0, "request_step"] == 30
    assert selected.loc[0, "success"]
    assert selected.loc[1, "request_step"] == 10
    assert selected.loc[1, "success"]
