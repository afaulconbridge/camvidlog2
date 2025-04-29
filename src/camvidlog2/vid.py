from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generator

import cv2
import numpy as np


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


@dataclass
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
        raise ValueError("Frame number must be positive")

    vid_stats = get_video_stats(filename)

    if frame_no > vid_stats.frame_count:
        raise ValueError("Frame number must be in video")

    start_ms = (frame_no - 1) * vid_stats.frame_duration

    frame_generator = generate_frames_cv2(filename, start_ms=start_ms)
    res = next(frame_generator, None)
    if res is None:
        raise RuntimeError("Unable to read from file")
    frame_no_hit, array = res

    # might need to slip forward a few frames
    while frame_no_hit < frame_no:
        frame_no_hit, array = next(frame_generator)
    if frame_no_hit != frame_no:
        raise RuntimeError("Hit the wrong frame")
    return array


def save(filename: str | Path, array: np.ndarray) -> None:
    result = cv2.imwrite(str(filename), array)
    if not result:
        raise RuntimeError("Failed to write file")
