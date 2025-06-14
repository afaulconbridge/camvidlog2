from pathlib import Path

import numpy as np
import pytest

from camvidlog2.vid import (
    Colourspace,
    Region,
    generate_frames_cv2,
    get_frame_by_no,
    get_video_stats,
    slice_frame,
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


def test_slice_frame():
    frame = np.zeros((800, 1000, 3), dtype=np.uint8)
    # image is 800 * 1000
    # slice size is 400*400
    # overlap is 0.25
    # so we should get:
    # if its 1000 wide and slices are 400 with .25 overlap
    #  - first is 0 to 400
    #  - second is 300 to 700
    #  - third is 600 to 1000

    # if its 800 high and the slices are 400 with .25 overlap
    #  - first is 0 to 400
    #  - second is 200 to 600
    #  - third is 400 to 800
    # 3*3 = 9

    slice_size = 400
    slices = slice_frame(frame, slice_size, slice_size, 0.25)

    for i, (region, subframe) in enumerate(slices, 1):
        assert isinstance(region, Region)
        assert isinstance(subframe, np.ndarray)
        assert subframe.shape == (slice_size, slice_size, 3)
        assert region.x1 >= 0 and region.y1 >= 0
        assert region.x2 - region.x1 == slice_size
        assert region.y2 - region.y1 == slice_size

    assert i == 9
