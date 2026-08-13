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
