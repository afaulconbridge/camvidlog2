from pathlib import Path

import numpy as np
import pytest

from camvidlog2.vid import (
    Colourspace,
    generate_frames_cv2,
    get_frame_by_no,
    get_video_stats,
)


def test_generate_frames_cv2(video_path: Path):
    for frame_no, array in generate_frames_cv2(video_path):
        assert frame_no > 0
        assert frame_no < (5 + 1) * 30  # expect just over 5 frames at 30 fps
        assert isinstance(array, np.ndarray)
        assert array.ndim == 3
        assert array.shape == (2160, 3840, 3)  # 4k colour


@pytest.mark.parametrize("frame_no", [1, 42, 140])
def test_get_frame_by_no(video_path: Path, frame_no: int):
    array = get_frame_by_no(video_path, frame_no)
    assert isinstance(array, np.ndarray)
    assert array.ndim == 3
    assert array.shape == (2160, 3840, 3)  # 4k colour


def test_get_video_stats(video_path: Path):
    stats = get_video_stats(video_path)

    assert stats.x == 3840
    assert stats.y == 2160
    assert stats.colourspace == Colourspace.RGB
    assert stats.fps == 30
    assert stats.frame_count == 150
    assert stats.shape == (2160, 3840, 3)
    assert stats.nbytes == 2160 * 3840 * 3
    assert stats.frame_duration == int(1000 / 30)
