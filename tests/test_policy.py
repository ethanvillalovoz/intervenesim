import numpy as np

from intervenesim.config import TrainConfig
from intervenesim.dataset import TrajectoryData
from intervenesim.policy import PolicyAgent, train_policy


def test_policy_fits_small_mapping(tmp_path) -> None:
    rng = np.random.default_rng(3)
    observations = rng.normal(size=(128, 4)).astype(np.float32)
    actions = np.tanh(observations[:, :2]).astype(np.float32)
    data = TrajectoryData(
        observations=observations,
        actions=actions,
        episode_ids=np.repeat(np.arange(16), 8),
        sources=np.full(128, "clean"),
        disturbances=np.full(128, "nominal"),
        intervention=np.zeros(128, dtype=bool),
        phases=np.zeros(128, dtype=np.int8),
    )
    checkpoint = tmp_path / "policy.pt"
    result = train_policy(
        data,
        checkpoint,
        TrainConfig(
            epochs=10,
            hidden_dims=(32, 32),
            batch_size=32,
            learning_rate=1e-2,
            weight_decay=0.0,
            device="cpu",
            seed=4,
        ),
        condition="test",
    )
    policy = PolicyAgent.load(checkpoint, device="cpu")
    prediction = policy.action(observations[0])
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()
    assert result["sample_count"] == 128


def test_robot_action_projection_holds_rotation_fixed(tmp_path) -> None:
    rng = np.random.default_rng(8)
    observations = rng.normal(size=(32, 4)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(32, 7)).astype(np.float32)
    data = TrajectoryData(
        observations=observations,
        actions=actions,
        episode_ids=np.repeat(np.arange(4), 8),
        sources=np.full(32, "clean"),
        disturbances=np.full(32, "nominal"),
        intervention=np.zeros(32, dtype=bool),
        phases=np.zeros(32, dtype=np.int8),
    )
    checkpoint = tmp_path / "robot_policy.pt"
    train_policy(
        data,
        checkpoint,
        TrainConfig(
            epochs=1,
            hidden_dims=(16,),
            batch_size=16,
            learning_rate=1e-3,
            weight_decay=0.0,
            device="cpu",
            seed=2,
        ),
        condition="projection_test",
    )
    action = PolicyAgent.load(checkpoint, device="cpu").action(observations[0])
    np.testing.assert_array_equal(action[3:6], np.zeros(3, dtype=np.float32))
