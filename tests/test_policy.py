import numpy as np
import torch

from intervenesim.config import TrainConfig
from intervenesim.dataset import TrajectoryData
from intervenesim.policy import MLPPolicy, PolicyAgent, contrastive_correction_loss, train_policy


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


def test_contrastive_loss_prefers_correction_over_rejected_action() -> None:
    correction = torch.tensor([[1.0, 0.0]])
    rejected = torch.tensor([[0.0, 1.0]])
    mask = torch.tensor([True])
    good = contrastive_correction_loss(
        correction, correction, rejected, mask, margin=0.2, min_distance=0.1
    )
    bad = contrastive_correction_loss(
        rejected, correction, rejected, mask, margin=0.2, min_distance=0.1
    )
    assert good < bad


def test_multitask_policy_routes_examples_to_separate_heads() -> None:
    model = MLPPolicy(observation_dim=6, action_dim=1, hidden_dims=(4,), task_head_count=2)
    assert model.heads is not None
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.heads[0][0].bias.fill_(-1.0)
        model.heads[1][0].bias.fill_(1.0)
    observations = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 2.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 2.0],
        ]
    )
    predictions = model(observations).squeeze(-1)
    assert predictions[0] < 0
    assert predictions[1] > 0
