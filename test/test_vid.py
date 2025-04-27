from pathlib import Path

import numpy as np
import pytest

from camvidlog2.vid import generate_frames_cv2, get_frame_by_no


def test_generate_frames_cv2(video_path: Path):
    for frame_no, array in generate_frames_cv2(video_path):
        assert frame_no > 0
        assert frame_no < (5 + 1) * 30  # expect just over 5 frames at 30 fps
        assert isinstance(array, np.ndarray)
        assert array.ndim == 3
        assert array.shape == (2160, 3840, 3)  # 4k colour


@pytest.mark.parametrize("frame_no", [1, 5, 29])
def test_get_frame_by_no(video_path: Path, frame_no: int):
    array = get_frame_by_no(video_path, frame_no)
    assert isinstance(array, np.ndarray)
    assert array.ndim == 3
    assert array.shape == (2160, 3840, 3)  # 4k colour
