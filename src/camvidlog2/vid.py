import contextlib
import math
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generator

import cv2
import numpy as np


class FrameError(Exception):
    """Exception raised when related to video frames."""


class RTSPError(Exception):
    """Exception raised when related to RTSP streaming."""


class Colourspace(Enum):
    """Enum representing different color spaces."""

    RGB = "rgb"
    greyscale = "greyscale"
    boolean = "boolean"  # mask


class Resolution(Enum):
    # Y,X to match openCV
    VGA = (480, 854)  # FWVGA
    SD = (720, 1280)
    HD = (1080, 1920)  # 2K, full HD
    UHD = (2160, 3840)  # 4K, UHD


@dataclass(frozen=True)
class Region:
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

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self):
        if not (self.x2 > self.x1 and self.y2 > self.y1):
            raise ValueError("Invalid region dimensions. Ensure x2 > x1 and y2 > y1.")


@dataclass(frozen=True)
class VideoFileStats:
    """Contains metadata about a video file.

    Attributes:
        fps (float): Frames per second of the video
        frame_count (int): Total number of frames in the video
        x (int): Width of the video frames in pixels
        y (int): Height of the video frames in pixels
        colourspace (Colourspace): The color space of the video frames

    Properties:
        shape (tuple[int, int, int]): Shape of the video frames as (height, width, channels)
        nbytes (int): Total bytes required to store one frame
        frame_duration (int): Duration of a single frame in milliseconds
    """

    fps: float
    frame_count: int
    x: int
    y: int
    colourspace: Colourspace

    @property
    def shape(self) -> tuple[int, int, int]:
        """Returns the shape of the video frames as (height, width, channels)."""
        # Y,X to match openCV
        return (self.y, self.x, 1 if self.colourspace == Colourspace.greyscale else 3)

    @property
    def nbytes(self) -> int:
        """Calculates and returns the total bytes required to store one frame."""
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
    """Retrieves metadata about a video file.

    Args:
        filename (str | Path): Path to the video file

    Returns:
        VideoFileStats: Object containing video metadata

    Raises:
        RuntimeError: If unable to read frames from the video
    """
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
        # this doesn't work, always read as colour
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
    """Generates video frames with OpenCV.

    Args:
        filename (str | Path): Path to the video file
        start_ms (float | None): Optional start time in milliseconds

    Yields:
        tuple[int, np.ndarray]: A tuple containing (frame_number, frame_array)

    Raises:
        FrameError: If unable to seek to the requested position
    """
    video_capture = cv2.VideoCapture(str(filename), cv2.CAP_ANY)
    try:
        # cv2.CAP_PROP_POS_FRAMES is unreliable - see https://github.com/opencv/opencv/issues/9053
        if start_ms is not None:
            set_success = video_capture.set(cv2.CAP_PROP_POS_MSEC, start_ms)
            if not set_success:
                raise FrameError("Unable to seek video file")

        success = True
        while success:
            success, array = video_capture.read()
            frame_no = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            if success:
                yield frame_no, array
    finally:
        video_capture.release()


def generate_frames_cv2_rtsp(
    rtsp_url: str,
) -> Generator[np.ndarray, None, None]:
    """
    Generates video frames from an RTSP stream using OpenCV.

    Args:
        rtsp_url (str): The RTSP stream URL to connect to.

    Yields:
        np.ndarray: frame as an array

    Raises:
        RTSPError: If unable to read from the RTSP stream.
    """
    # for networking performance
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

    video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_ANY)
    try:
        if not video_capture.isOpened():
            raise RTSPError(f"Unable to open stream: {rtsp_url}")

        success = True
        while success:
            success, array = video_capture.read()
            if success:
                yield array
    finally:
        video_capture.release()


def _grab_frame(
    video_capture: cv2.VideoCapture,
    lock: threading.Lock,
    terminator: threading.Event,
    sleep: float | None,
) -> None:
    success = True
    while success and not terminator.is_set():
        with lock:
            success = video_capture.grab()
        # share the lock more nicely
        if sleep:
            time.sleep(sleep)


def generate_latest_frames_cv2_rtsp(
    rtsp_url: str,
) -> Generator[np.ndarray, None, None]:
    """
    Generator that always returns the freshest frame.

    This uses a thread in the background to fetch frames from the camera, and might skip frames
    if the consumer can't keep up.
    """

    video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_ANY)
    try:
        if not video_capture.isOpened():
            raise RTSPError(f"Unable to open stream: {rtsp_url}")

        terminator = threading.Event()
        lock = threading.Lock()
        frame_thread = threading.Thread(
            target=_grab_frame,
            args=[video_capture, lock, terminator, 0.01],
            daemon=True,
        )
        try:
            frame_thread.start()
            while frame_thread.is_alive():
                with lock:
                    success, array = video_capture.retrieve()
                    if success:
                        yield array
                    else:
                        break
        finally:
            terminator.set()
            frame_thread.join(5)
            if frame_thread.is_alive():
                raise RuntimeError("Frame thread did not terminate")
    finally:
        video_capture.release()


def get_frame_by_no(filename: str | Path, frame_no: int) -> np.ndarray:
    """Retrieves a specific frame from a video file.

    Args:
        filename (str | Path): Path to the video file
        frame_no (int): The 1-based index of the frame to retrieve

    Returns:
        np.ndarray: The requested video frame as a numpy array

    Raises:
        FrameError: If frame_no is invalid or out of range
    """
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
    os.makedirs(Path(filename).parent, exist_ok=True)
    result = cv2.imwrite(str(filename), array)
    if not result:
        raise RuntimeError("Failed to write file")


@contextlib.contextmanager
def save_video(
    output_path: Path | str, stats: VideoFileStats, codex: str = "mp4v"
) -> Generator[cv2.VideoWriter, None, None]:
    """
    Saves a video by creating a VideoWriter object and yielding it as a context manager.

    Note: VideoWriter expects BGR format frames

    Args:
        output_path (Path | str): Path to the output video file.
        stats (VideoFileStats): Statistics of the input video file (FPS, dimensions).
        codex (str, optional): Codec to use for writing the output video. Defaults to "mp4v".

    Yields:
        cv2.VideoWriter: A VideoWriter object that can be used to write frames to the output video.
    """
    fourcc = cv2.VideoWriter_fourcc(*codex)  # type: ignore
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        stats.fps,
        (stats.x, stats.y),
        isColor=stats.colourspace in {Colourspace.RGB},
    )
    try:
        yield out
    finally:
        out.release()


def slice_frame(
    frame: np.ndarray,
    slice_width: int = 224,
    slice_height: int = 224,
    slice_overlap: float = 0.25,
) -> Generator[tuple[Region, np.ndarray], None, None]:
    """Slices a frame into smaller overlapping regions.

    Args:
        frame (np.ndarray): The input frame to be sliced
        slice_width (int): Width of each slice. Defaults to 224
        slice_height (int): Height of each slice. Defaults to 224
        slice_overlap (float): Proportion of overlap between slices. Defaults to 0.25

    Yields:
        tuple[Region, np.ndarray]: A tuple containing (region, subframe)

    Raises:
        ValueError: If invalid dimensions or overlap values are provided
    """
    if slice_width < 1:
        raise ValueError("slice width must be at least 1")
    if slice_height < 1:
        raise ValueError("slice height must be at least 1")
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

    if num_slices_height == 1 and num_slices_width == 1:
        # one slice covers the whole frame
        slice_height_overlap_updated = slice_height_overlap
        slice_width_overlap_updated = slice_width_overlap
    elif num_slices_height == 1:
        # horizontal slices only
        extra_width = (num_slices_width * slice_width) - frame_width
        slice_width_overlap_updated = extra_width / (num_slices_width - 1)
        slice_height_overlap_updated = slice_height_overlap
    elif num_slices_width == 1:
        # vertical slices only
        extra_height = (num_slices_height * slice_height) - frame_height
        slice_width_overlap_updated = slice_width_overlap
        slice_height_overlap_updated = extra_height / (num_slices_height - 1)
    else:
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


def slice_frame_scaling(
    frame: np.ndarray,
    slice_width: int,
    slice_height: int,
    slice_overlap=0.25,
    slice_scaling_max=2.0,
) -> Generator[tuple[Region, np.ndarray], None, None]:
    """Slices a frame into smaller overlapping regions with scaling.

    Args:
        frame (np.ndarray): The input frame to be sliced
        slice_width (int): Minimum width of each slice in pixels
        slice_height (int): Minimum height of each slice in pixels
        slice_overlap (float): Proportion of overlap between slices. Defaults to 0.25
        slice_scaling_max (float): Maximum scaling factor for slice size. Defaults to 2.0

    Yields:
        tuple[Region, np.ndarray]: A tuple containing (region, subframe)
    """
    if slice_width < 1:
        raise ValueError("slice width must be at least 1")
    if slice_height < 1:
        raise ValueError("slice height must be at least 1")
    if slice_overlap < 0.0:
        raise ValueError("slice overlap must not be negative")
    if slice_overlap >= 1.0:
        raise ValueError("slice overlap must not be greater or equal to 1.0")

    # get the dimensions of the input image
    frame_height, frame_width = frame.shape[:2]

    if slice_width > frame_width:
        raise ValueError("slice width must be no wider than frame")
    if slice_height > frame_height:
        raise ValueError("slice width must be no higher than frame")

    # 1. Determine the maximum scaling factor that fits within the frame
    max_scale_w = frame_width / slice_width
    max_scale_h = frame_height / slice_height
    max_scale = max(max_scale_w, max_scale_h)

    # 2. Calculate the number of scales (steps) to use
    scales = [1.0]
    while ith_scale := slice_scaling_max ** len(scales) < max_scale:
        scales.append(ith_scale)

    # 3. Calculate the scaling factor to use (nth root)
    slice_scaling_updated = max_scale ** (1.0 / len(scales))

    for i in range(len(scales)):
        slice_scaling_factor = slice_scaling_updated**i
        slice_width = int(slice_width * slice_scaling_factor)
        slice_height = int(slice_height * slice_scaling_factor)

        yield from slice_frame(frame, slice_width, slice_height, slice_overlap)

    # also yield the original frame, sliced to keep aspect ratio constant
    if slice_width / frame_width < slice_height / frame_height:
        yield from slice_frame(
            frame,
            int(slice_width * (frame_height / slice_height)),
            frame_height,
            slice_overlap,
        )
    else:
        yield from slice_frame(
            frame,
            frame_width,
            int(slice_height * (frame_width / slice_width)),
            slice_overlap,
        )
