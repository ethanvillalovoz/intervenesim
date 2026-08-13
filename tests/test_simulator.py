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
