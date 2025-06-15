from pathlib import Path

import numpy as np
import pytest

from camvidlog2.vid import (
    Colourspace,
    Region,
    VideoFileStats,
    generate_frames_cv2,
    generate_frames_cv2_rtsp,
    get_frame_by_no,
    get_video_stats,
    save_video,
    slice_frame,
)


def test_generate_frames_cv2(video_path: Path):
    for frame_no, array in generate_frames_cv2(video_path):
        assert frame_no > 0
        assert frame_no < (5 + 1) * 30  # expect just over 5 frames at 30 fps
        assert isinstance(array, np.ndarray)
        assert array.ndim == 3
        assert array.shape == (2160, 3840, 3)  # 4k colour


def test_generate_frames_cv2_rtsp(rtsp_server: str):
    for frame_no, array in enumerate(generate_frames_cv2_rtsp(rtsp_server)):
        assert isinstance(array, np.ndarray)
        assert array.ndim == 3
        assert array.shape == (2160, 3840, 3)  # 4k colour

        if frame_no > 30:
            # stream will loop by itself indefinately
            break


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


def test_save_video(tmp_path: Path):
    # Create temporary output path
    output_path = tmp_path / "test_output.mp4"

    # Generate sample frames (3 frames of different colors)
    red_frame = np.zeros((256, 256, 3), dtype=np.uint8)
    red_frame[:, :, 2] = 255
    green_frame = np.zeros((256, 256, 3), dtype=np.uint8)
    green_frame[:, :, 1] = 255
    blue_frame = np.zeros((256, 256, 3), dtype=np.uint8)
    blue_frame[:, :, 0] = 255

    # Create video stats for our test frames
    test_video_stats = VideoFileStats(
        fps=30.0, frame_count=3, x=256, y=256, colourspace=Colourspace.RGB
    )

    # Use save_video to write the frames
    with save_video(output_path, test_video_stats) as writer:
        for _ in range(int(test_video_stats.fps)):
            writer.write(red_frame)
        for _ in range(int(test_video_stats.fps)):
            writer.write(green_frame)
        for _ in range(int(test_video_stats.fps)):
            writer.write(blue_frame)

    # Verify the output video file
    stats = get_video_stats(output_path)

    assert stats.frame_count == 89  # rounds down
    assert stats.colourspace == Colourspace.RGB
    assert stats.fps == 30.0

    # Clean up the test file
    output_path.unlink()
