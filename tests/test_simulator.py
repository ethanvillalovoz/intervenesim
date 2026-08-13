import numpy as np
import pytest

from intervenesim.environment import PickPlaceEnv
from intervenesim.expert import ScriptedExpert


def test_expert_completes_pick_place() -> None:
    with PickPlaceEnv(max_steps=260) as env:
        state = env.reset(27)
        expert = ScriptedExpert()
        expert.reset(state)
        for _ in range(env.max_steps):
            state, _, done, info = env.step(expert.action(state))
            if done:
                break
    assert info["success"]


@pytest.mark.parametrize("task", ["can", "milk", "bread", "cereal"])
def test_expert_completes_each_object_domain(task: str) -> None:
    with PickPlaceEnv(max_steps=260, task=task, task_conditioning=True) as env:
        state = env.reset(108)
        expert = ScriptedExpert()
        expert.reset(state)
        for _ in range(env.max_steps):
            state, _, done, info = env.step(expert.action(state))
            if done:
                break
    assert info["success"]
    assert len(state.observation) == 35


def test_snapshot_restore_replays_identical_transition() -> None:
    with PickPlaceEnv(max_steps=40, task="can", task_conditioning=True) as env:
        state = env.reset(91)
        action = np.asarray([0.2, -0.1, 0.05, 0, 0, 0, -1], dtype=np.float32)
        for _ in range(4):
            state, _, _, _ = env.step(action)
        snapshot = env.snapshot()
        first_state, first_reward, first_done, first_info = env.step(action)
        restored = env.restore(snapshot)
        np.testing.assert_allclose(restored.observation, state.observation, atol=1e-6)
        second_state, second_reward, second_done, second_info = env.step(action)
        np.testing.assert_allclose(second_state.observation, first_state.observation, atol=1e-6)
        assert second_reward == first_reward
        assert second_done == first_done
        assert second_info["success"] == first_info["success"]
