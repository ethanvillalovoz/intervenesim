import numpy as np

from intervenesim.disturbances import Disturbance


def test_action_delay_is_deterministic() -> None:
    left = Disturbance("action_delay", seed=4)
    right = Disturbance("action_delay", seed=4)
    left.reset(7)
    right.reset(7)
    actions = [np.full(7, index, dtype=np.float32) for index in range(8)]
    for step, action in enumerate(actions):
        np.testing.assert_array_equal(
            left.transform_action(action, step), right.transform_action(action, step)
        )
    assert left.metadata() == right.metadata()


def test_noise_stays_in_action_bounds() -> None:
    disturbance = Disturbance("action_noise", seed=2)
    disturbance.reset(7)
    result = disturbance.transform_action(np.ones(7, dtype=np.float32), step=12)
    assert np.all(result <= 1.0)
    assert np.all(result >= -1.0)
    assert disturbance.fired
