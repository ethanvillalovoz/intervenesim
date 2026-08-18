import json

import numpy as np

from intervenesim.human import HumanCorrectionData


def test_human_correction_archive_preserves_takeover_mask(tmp_path) -> None:
    data = HumanCorrectionData(
        observations=np.zeros((3, 4)),
        actions=np.zeros((3, 2)),
        episode_ids=np.asarray([0, 0, 0]),
        steps=np.asarray([4, 5, 6]),
        human_control=np.asarray([False, True, True]),
        metadata={"controls": {"takeover": "t"}},
    )
    path = data.save(tmp_path / "human.npz")
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["human_control"], [False, True, True])
        metadata = json.loads(str(archive["metadata"].item()))
    assert metadata["controls"]["takeover"] == "t"
