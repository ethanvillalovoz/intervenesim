import numpy as np

from intervenesim.visual import VisualData, proprioception


def test_visual_data_round_trip_and_camera_subset(tmp_path) -> None:
    data = VisualData(
        images=np.zeros((2, 8, 10, 3), dtype=np.uint8),
        proprioception=np.zeros((2, 17)),
        autonomous_success=np.asarray([False, True]),
        assisted_success=np.asarray([True, True]),
        episode_ids=np.asarray([0, 0]),
        steps=np.asarray([10, 10]),
        cameras=np.asarray(["frontview", "agentview"]),
        tasks=np.asarray(["can", "can"]),
        disturbances=np.asarray(["slip", "slip"]),
        metadata={"test": True},
    )
    loaded = VisualData.load(data.save(tmp_path / "visual.npz"))
    front = loaded.subset(loaded.cameras == "frontview")
    assert front.sample_count == 1
    assert front.helpful[0]


def test_visual_proprioception_excludes_object_and_goal_coordinates() -> None:
    observation = np.arange(35, dtype=np.float32)
    selected = proprioception(observation)
    np.testing.assert_array_equal(
        selected, np.concatenate([observation[0:5], observation[17:25], observation[31:35]])
    )
    assert not np.isin(np.arange(5, 17), selected).any()
