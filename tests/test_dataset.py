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
    assert loaded.metadata["schema_version"] == 1


def test_equal_budget_is_deterministic() -> None:
    data = make_data()
    first = data.sample_budget(5, seed=7)
    second = data.sample_budget(5, seed=7)
    np.testing.assert_array_equal(first.observations, second.observations)


def test_concatenate_offsets_episode_ids() -> None:
    merged = TrajectoryData.concatenate([make_data(), make_data(episode_offset=10)])
    assert merged.sample_count == 24
    assert merged.episode_count == 6
