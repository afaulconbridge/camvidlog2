from pathlib import Path
from typing import Generator

import cv2
import numpy as np


def generate_frames_cv2(
    filename: str | Path,
) -> Generator[tuple[int, np.ndarray], None, None]:
    video_capture = cv2.VideoCapture(str(filename), cv2.CAP_ANY)
    success = True
    frame_no = 1
    while success:
        success, array = video_capture.read()
        if success:
            yield frame_no, array
        frame_no += 1
    video_capture.release()
