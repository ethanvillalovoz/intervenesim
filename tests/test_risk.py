import numpy as np

from intervenesim.risk import (
    RiskBuilder,
    RiskData,
    binary_metrics,
    select_threshold,
    temporal_features,
)


def test_risk_data_round_trip(tmp_path) -> None:
    builder = RiskBuilder()
    builder.append_episode(
        [np.full(3, step, dtype=np.float32) for step in range(6)],
        intervention_index=5,
        horizon=2,
        episode_id=0,
        disturbance="noise",
    )
    data = builder.build(3, {"test": True})
    assert data.positive_count == 2
    loaded = RiskData.load(data.save(tmp_path / "risk.npz"))
    np.testing.assert_array_equal(loaded.targets, [0, 0, 0, 0, 1, 1])


def test_risk_metrics_are_perfect_for_separable_predictions() -> None:
    targets = np.array([0, 0, 1, 1], dtype=np.float32)
    probabilities = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    threshold = select_threshold(targets, probabilities, target_recall=1.0)
    metrics = binary_metrics(targets, probabilities, threshold)
    assert metrics["auroc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_temporal_features_reset_delta_between_episodes() -> None:
    observations = np.asarray([[1, 2], [3, 5], [10, 20], [13, 25]], dtype=np.float32)
    features = temporal_features(observations, np.asarray([0, 0, 1, 1]))
    np.testing.assert_array_equal(features[:, :2], observations)
    np.testing.assert_array_equal(features[:, 2:], [[0, 0], [2, 3], [0, 0], [3, 5]])
