import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generator

import cv2
import numpy as np


class FrameError(Exception):
    pass


class Colourspace(Enum):
    RGB = "rgb"
    greyscale = "greyscale"
    boolean = "boolean"  # mask


class Resolution(Enum):
    # Y,X to match openCV
    VGA = (480, 854)  # FWVGA
    SD = (720, 1280)
    HD = (1080, 1920)  # 2k, full HD
    UHD = (2160, 3840)  # 4K, UHD


@dataclass(frozen=True)
class Region:
    x1: int
    y1: int
    x2: int
    y2: int

    """Represents a rectangular region in a 2D space.

    The region is defined by its top-left (x1, y1) and bottom-right (x2, y2)
    coordinates. It ensures that the region's boundaries are valid, meaning x2
    is greater than x1 and y2 is greater than y1.

    Attributes:
        x1 (int): The x-coordinate of the top-left corner.
        y1 (int): The y-coordinate of the top-left corner.
        x2 (int): The x-coordinate of the bottom-right corner.
        y2 (int): The y-coordinate of the bottom-right corner.
    """

    def __post_init__(self):
        if not (self.x2 > self.x1 and self.y2 > self.y1):
            raise ValueError("Invalid region dimensions. Ensure x2 > x1 and y2 > y1.")


@dataclass(frozen=True)
class VideoFileStats:
    fps: float
    frame_count: int
    x: int
    y: int
    colourspace: Colourspace

    @property
    def shape(self) -> tuple[int, int, int]:
        # Y,X to match openCV
        return (self.y, self.x, 1 if self.colourspace == Colourspace.greyscale else 3)

    @property
    def nbytes(self) -> int:
        match self.colourspace:
            case Colourspace.greyscale:
                return self.x * self.y
            case Colourspace.RGB:
                return self.x * self.y * 3
            case _:
                raise NotImplementedError("Unimplemented colourspace")

    @property
    def frame_duration(self) -> int:
        """Return the duration of a single video frame in milliseconds"""
        return int(1000 / self.fps)


def get_video_stats(filename: str | Path) -> VideoFileStats:
    video_capture = None
    try:
        video_capture = cv2.VideoCapture(str(filename), cv2.CAP_ANY)
        fps = float(video_capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        # frame details are more reliable than capture properties
        # x = cv2.CAP_PROP_FRAME_WIDTH
        # y = cv2.CAP_PROP_FRAME_HEIGHT
        # bw = cv2.CAP_PROP_MONOCHROME
        (success, frame) = video_capture.read()
        if not success:
            msg = f"Unable to read frame from {filename}"
            raise RuntimeError(msg)
        y = frame.shape[0]
        x = frame.shape[1]
        colourspace = Colourspace.greyscale if frame.shape[2] == 1 else Colourspace.RGB

        return VideoFileStats(
            fps=fps,
            frame_count=frame_count,
            x=x,
            y=y,
            colourspace=colourspace,
        )
    finally:
        if video_capture:
            video_capture.release()
            video_capture = None


def generate_frames_cv2(
    filename: str | Path, *, start_ms: float | None = None
) -> Generator[tuple[int, np.ndarray], None, None]:
    video_capture = cv2.VideoCapture(str(filename), cv2.CAP_ANY)

    # cv2.CAP_PROP_POS_FRAMES is unreliable - see https://github.com/opencv/opencv/issues/9053
    if start_ms is not None:
        set_success = video_capture.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        if not set_success:
            raise RuntimeError("Unable to seek video file")

    success = True
    while success:
        success, array = video_capture.read()
        frame_no = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        if success:
            yield frame_no, array
    video_capture.release()


def get_frame_by_no(filename: str | Path, frame_no: int) -> np.ndarray:
    if frame_no <= 0:
        raise FrameError("Frame number must be positive")

    vid_stats = get_video_stats(filename)

    if frame_no > vid_stats.frame_count:
        raise FrameError("Frame number must be in video")

    start_ms = (frame_no - 1) * vid_stats.frame_duration

    frame_generator = generate_frames_cv2(filename, start_ms=start_ms)
    res = next(frame_generator, None)
    if res is None:
        raise FrameError("Unable to read from file")
    frame_no_hit, array = res

    # might need to slip forward a few frames
    while frame_no_hit < frame_no:
        frame_no_hit, array = next(frame_generator)
    if frame_no_hit != frame_no:
        raise FrameError("Hit the wrong frame")
    return array


def save(filename: str | Path, array: np.ndarray) -> None:
    result = cv2.imwrite(str(filename), array)
    if not result:
        raise RuntimeError("Failed to write file")


def slice_frame(
    frame: np.ndarray,
    slice_width: int = 224,
    slice_height: int = 224,
    slice_overlap: float = 0.25,
) -> Generator[tuple[Region, np.ndarray], None, None]:
    """
    Slice a frame into smaller frames
    """
    if slice_width < 1:
        raise ValueError("slcie width must be at least 1")
    if slice_height < 1:
        raise ValueError("slcie height must be at least 1")
    if slice_overlap < 0.0:
        raise ValueError("slice overlap must not be negative")
    if slice_overlap >= 1.0:
        raise ValueError("slice overlap must not be greater or equal to 1.0")

    # get the dimensions of the input image
    frame_height, frame_width = frame.shape[:2]
    # calculate how many slices we need to make based on the desired output size and overlap ratio
    slice_height_overlap = int(slice_height * slice_overlap)
    slice_width_overlap = int(slice_width * slice_overlap)
    slice_height_non_overlap = slice_height - slice_height_overlap
    slice_width_non_overlap = slice_width - slice_width_overlap
    num_slices_height = math.ceil(
        (frame_height - slice_height_overlap) / slice_height_non_overlap
    )
    num_slices_width = math.ceil(
        (frame_width - slice_width_overlap) / slice_width_non_overlap
    )

    # recalculate the overlaps to use based on the number of slices
    extra_width = (num_slices_width * slice_width) - frame_width
    extra_height = (num_slices_height * slice_height) - frame_height
    slice_width_overlap_updated = extra_width / (num_slices_width - 1)
    slice_height_overlap_updated = extra_height / (num_slices_height - 1)
    assert slice_height_overlap_updated >= slice_height_overlap
    assert slice_width_overlap_updated >= slice_width_overlap

    slice_offset_height = slice_height - slice_height_overlap_updated
    slice_offset_width = slice_width - slice_width_overlap_updated

    for yi in range(num_slices_height):
        for xj in range(num_slices_width):
            # calculate the coordinates of the current slice
            y1 = int(yi * slice_offset_height)
            x1 = int(xj * slice_offset_width)
            # calculate the current slice
            x2 = x1 + slice_width
            y2 = y1 + slice_height
            frame_slice = frame[y1:y2, x1:x2]
            region = Region(x1, y1, x2, y2)
            yield region, frame_slice
