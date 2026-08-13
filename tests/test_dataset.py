import numpy as np

from intervenesim.dataset import TrajectoryData


def make_data(samples: int = 12, episode_offset: int = 0) -> TrajectoryData:
    return TrajectoryData(
        observations=np.arange(samples * 4, dtype=np.float32).reshape(samples, 4),
        actions=np.zeros((samples, 2), dtype=np.float32),
        episode_ids=np.repeat(np.arange(3) + episode_offset, samples // 3),
        sources=np.full(samples, "clean"),
        disturbances=np.full(samples, "nominal"),
        intervention=np.zeros(samples, dtype=bool),
        phases=np.zeros(samples, dtype=np.int8),
        metadata={"test": True},
    )


def test_round_trip(tmp_path) -> None:
    path = make_data().save(tmp_path / "dataset.npz")
    loaded = TrajectoryData.load(path)
    np.testing.assert_array_equal(loaded.observations, make_data().observations)
    assert loaded.sample_count == 12
    assert loaded.episode_count == 3
    assert loaded.metadata["schema_version"] == 2


def test_equal_budget_is_deterministic() -> None:
    data = make_data()
    first = data.sample_budget(5, seed=7)
    second = data.sample_budget(5, seed=7)
    np.testing.assert_array_equal(first.observations, second.observations)


def test_concatenate_offsets_episode_ids() -> None:
    merged = TrajectoryData.concatenate([make_data(), make_data(episode_offset=10)])
    assert merged.sample_count == 24
    assert merged.episode_count == 6


def test_rejected_actions_survive_budget_and_round_trip(tmp_path) -> None:
    data = make_data()
    data.rejected_actions[3] = np.array([0.5, -0.5], dtype=np.float32)
    data.rejection_mask[3] = True
    loaded = TrajectoryData.load(data.save(tmp_path / "pairs.npz"))
    assert loaded.rejection_mask.sum() == 1
    np.testing.assert_array_equal(loaded.rejected_actions[3], [0.5, -0.5])


def test_stratified_budget_balances_available_groups() -> None:
    data = make_data()
    data.tasks[:6] = "can"
    data.tasks[6:] = "milk"
    sampled = data.stratified_sample_budget(6, seed=2, by=("tasks",))
    names, counts = np.unique(sampled.tasks, return_counts=True)
    assert dict(zip(names, counts, strict=True)) == {"can": 3, "milk": 3}
